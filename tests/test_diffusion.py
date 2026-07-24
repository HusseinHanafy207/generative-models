"""Mathematical and shape tests for DDPM scheduler and forward diffusion."""

from pathlib import Path

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.ddpm import (
    DDPM,
    NoiseScheduler,
    TimestepEmbedding,
    UNet,
    forward_diffuse,
    forward_diffuse_trajectory,
    sample_timesteps,
    save_forward_diffusion_grid,
    sinusoidal_time_embedding,
)
from generative_models.losses import DDPMLoss


def test_linear_beta_schedule_bounds_and_shape():
    scheduler = NoiseScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)

    assert scheduler.betas.shape == (1000,)
    assert torch.isclose(scheduler.betas[0], torch.tensor(1e-4, dtype=torch.float64))
    assert torch.isclose(scheduler.betas[-1], torch.tensor(0.02, dtype=torch.float64))
    assert torch.all(scheduler.betas[1:] > scheduler.betas[:-1])


def test_alphas_equal_one_minus_betas():
    scheduler = NoiseScheduler(num_timesteps=1000)

    expected = 1.0 - scheduler.betas
    assert torch.allclose(scheduler.alphas, expected)


def test_alphas_cumprod_definition():
    """ᾱ_t must equal the cumulative product of α_s for s = 0 … t."""
    scheduler = NoiseScheduler(num_timesteps=1000)

    expected = torch.cumprod(scheduler.alphas, dim=0)
    assert torch.allclose(scheduler.alphas_cumprod, expected)


def test_alphas_cumprod_starts_near_one_ends_near_zero():
    """With the DDPM defaults, ᾱ_0 ≈ 1 and ᾱ_{T-1} ≈ 0."""
    scheduler = NoiseScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)

    assert scheduler.alphas_cumprod[0].item() > 0.999
    assert scheduler.alphas_cumprod[-1].item() < 0.01
    assert torch.all(scheduler.alphas_cumprod[1:] < scheduler.alphas_cumprod[:-1])


def test_sqrt_coefficients_match_alphas_cumprod():
    scheduler = NoiseScheduler(num_timesteps=1000)

    assert torch.allclose(
        scheduler.sqrt_alphas_cumprod, torch.sqrt(scheduler.alphas_cumprod)
    )
    assert torch.allclose(
        scheduler.sqrt_one_minus_alphas_cumprod,
        torch.sqrt(1.0 - scheduler.alphas_cumprod),
    )


def test_q_sample_output_shape():
    scheduler = NoiseScheduler(num_timesteps=1000)
    x_0 = torch.randn(8, 1, 28, 28)
    t = torch.randint(0, 1000, (8,))

    x_t = scheduler.q_sample(x_0, t)

    assert x_t.shape == x_0.shape


def test_q_sample_closed_form_identity():
    """x_t must equal √ᾱ_t x_0 + √(1 - ᾱ_t) ε exactly for fixed ε."""
    scheduler = NoiseScheduler(num_timesteps=1000)
    x_0 = torch.randn(4, 1, 28, 28)
    noise = torch.randn_like(x_0)
    t = torch.tensor([0, 10, 500, 999])

    x_t = scheduler.q_sample(x_0, t, noise=noise)

    sqrt_ab = scheduler.sqrt_alphas_cumprod[t].float().view(4, 1, 1, 1)
    sqrt_omb = scheduler.sqrt_one_minus_alphas_cumprod[t].float().view(4, 1, 1, 1)
    expected = sqrt_ab * x_0 + sqrt_omb * noise

    assert torch.allclose(x_t, expected, atol=1e-6)


def test_q_sample_t0_is_nearly_clean():
    """At t = 0, √ᾱ_0 ≈ 1 and √(1 - ᾱ_0) is tiny, so x_t ≈ x_0."""
    scheduler = NoiseScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    x_0 = torch.randn(16, 1, 28, 28)
    t = torch.zeros(16, dtype=torch.long)
    noise = torch.randn_like(x_0)

    x_t = scheduler.q_sample(x_0, t, noise=noise)

    # Residual should be on the order of √β_0 ≈ 0.01
    mse = torch.mean((x_t - x_0) ** 2).item()
    assert mse < 1e-3


def test_q_sample_late_t_is_dominated_by_noise():
    """At t = T-1, ᾱ ≈ 0 so x_t ≈ ε (correlation with x_0 near zero)."""
    scheduler = NoiseScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    torch.manual_seed(0)
    x_0 = torch.randn(64, 1, 28, 28)
    noise = torch.randn_like(x_0)
    t = torch.full((64,), scheduler.num_timesteps - 1, dtype=torch.long)

    x_t = scheduler.q_sample(x_0, t, noise=noise)

    # x_t should match noise far more than x_0
    mse_to_noise = torch.mean((x_t - noise) ** 2).item()
    mse_to_x0 = torch.mean((x_t - x_0) ** 2).item()
    assert mse_to_noise < 0.05
    assert mse_to_x0 > 0.5


def test_q_sample_mean_and_variance_match_gaussian():
    """Monte Carlo: E[x_t] ≈ √ᾱ x_0 and Var(x_t) ≈ (1 - ᾱ) over many ε."""
    scheduler = NoiseScheduler(num_timesteps=1000)
    torch.manual_seed(1)

    x_0 = torch.ones(1, 1, 8, 8)
    t_index = 250
    t = torch.tensor([t_index])
    alpha_bar = scheduler.alphas_cumprod[t_index].float()

    n_samples = 2000
    samples = []
    for _ in range(n_samples):
        samples.append(scheduler.q_sample(x_0, t))
    stacked = torch.stack(samples)  # (N, 1, 1, 8, 8)

    empirical_mean = stacked.mean(dim=0)
    empirical_var = stacked.var(dim=0, unbiased=True)

    expected_mean = torch.sqrt(alpha_bar) * x_0
    expected_var = (1.0 - alpha_bar).expand_as(x_0)

    assert torch.allclose(empirical_mean, expected_mean, atol=0.05)
    assert torch.allclose(empirical_var, expected_var, atol=0.05)


def test_invalid_timestep_raises():
    scheduler = NoiseScheduler(num_timesteps=100)
    x_0 = torch.randn(2, 1, 28, 28)

    try:
        scheduler.q_sample(x_0, torch.tensor([0, 100]))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_unsupported_schedule_raises():
    try:
        NoiseScheduler(schedule="cosine")
        raised = False
    except ValueError:
        raised = True
    assert raised


# --- Stage 2: forward diffusion ---


def test_sample_timesteps_range_and_shape():
    t = sample_timesteps(batch_size=32, num_timesteps=1000)

    assert t.shape == (32,)
    assert t.dtype == torch.int64
    assert int(t.min()) >= 0
    assert int(t.max()) < 1000


def test_forward_diffuse_returns_xt_t_and_noise():
    scheduler = NoiseScheduler(num_timesteps=1000)
    train_loader, _ = get_mnist_dataloaders(batch_size=8, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))

    x_t, t, noise = forward_diffuse(scheduler, x_0)

    assert x_t.shape == x_0.shape
    assert t.shape == (8,)
    assert noise.shape == x_0.shape
    assert int(t.min()) >= 0
    assert int(t.max()) < 1000


def test_forward_diffuse_respects_provided_t_and_noise():
    scheduler = NoiseScheduler(num_timesteps=1000)
    x_0 = torch.randn(4, 1, 28, 28)
    t = torch.tensor([0, 10, 100, 999])
    noise = torch.randn_like(x_0)

    x_t, t_out, noise_out = forward_diffuse(scheduler, x_0, t=t, noise=noise)

    assert torch.equal(t_out, t)
    assert torch.equal(noise_out, noise)
    expected = scheduler.q_sample(x_0, t, noise=noise)
    assert torch.allclose(x_t, expected)


def test_forward_diffuse_trajectory_shape_and_shared_noise():
    scheduler = NoiseScheduler(num_timesteps=1000)
    x_0 = torch.randn(3, 1, 28, 28)
    timesteps = [0, 50, 100, 999]
    noise = torch.randn_like(x_0)

    progression, shared = forward_diffuse_trajectory(
        scheduler, x_0, timesteps, noise=noise
    )

    assert progression.shape == (3, 4, 1, 28, 28)
    assert torch.equal(shared, noise)
    # Column 0 is nearly clean; last column matches nearly-pure noise
    assert torch.mean((progression[:, 0] - x_0) ** 2).item() < 1e-3
    assert torch.mean((progression[:, -1] - noise) ** 2).item() < 0.05


def test_forward_diffuse_trajectory_mse_increases_with_t():
    """With shared ε, mean squared error to x_0 should grow with t."""
    scheduler = NoiseScheduler(num_timesteps=1000)
    torch.manual_seed(0)
    train_loader, _ = get_mnist_dataloaders(batch_size=16, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))
    timesteps = [0, 50, 100, 200, 400, 600, 800, 999]

    progression, _ = forward_diffuse_trajectory(scheduler, x_0, timesteps)

    prev_mse = -1.0
    for i in range(len(timesteps)):
        mse = torch.mean((progression[:, i] - x_0) ** 2).item()
        assert mse >= prev_mse - 1e-6
        prev_mse = mse


def test_save_forward_diffusion_grid(tmp_path: Path):
    scheduler = NoiseScheduler(num_timesteps=1000)
    x_0 = torch.rand(2, 1, 28, 28)
    timesteps = [0, 100, 999]
    progression, _ = forward_diffuse_trajectory(scheduler, x_0, timesteps)

    output = tmp_path / "forward.png"
    path = save_forward_diffusion_grid(progression, timesteps, output_path=output)

    assert path.exists()
    assert path.stat().st_size > 0


# --- Stage 3: time embedding ---


def test_sinusoidal_embedding_shape():
    t = torch.tensor([0, 1, 50, 999])
    emb = sinusoidal_time_embedding(t, embedding_dim=64)

    assert emb.shape == (4, 64)
    assert emb.dtype == torch.float32


def test_sinusoidal_embedding_matches_transformer_formula():
    import math

    embedding_dim = 64
    t = torch.tensor([0, 7, 100, 999])
    emb = sinusoidal_time_embedding(t, embedding_dim=embedding_dim)

    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    expected = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    assert torch.allclose(emb, expected, atol=1e-6)


def test_sinusoidal_embedding_odd_dim_is_padded():
    emb = sinusoidal_time_embedding(torch.tensor([0, 1]), embedding_dim=63)

    assert emb.shape == (2, 63)
    assert torch.all(emb[:, -1] == 0)


def test_sinusoidal_embedding_t0_is_sin0_cos1_pattern():
    """At t=0: sin(0)=0 and cos(0)=1 for every frequency."""
    emb = sinusoidal_time_embedding(torch.tensor([0]), embedding_dim=8)
    half = 4
    assert torch.allclose(emb[0, :half], torch.zeros(half), atol=1e-6)
    assert torch.allclose(emb[0, half:], torch.ones(half), atol=1e-6)


def test_sinusoidal_embeddings_differ_across_timesteps():
    emb = sinusoidal_time_embedding(torch.arange(16), embedding_dim=32)
    dists = torch.cdist(emb, emb)
    off_diag = dists[~torch.eye(16, dtype=torch.bool)]
    assert off_diag.min().item() > 0


def test_timestep_embedding_module_output_shape():
    module = TimestepEmbedding(embedding_dim=128)
    t = torch.randint(0, 1000, (8,))

    out = module(t)

    assert out.shape == (8, 128)


def test_timestep_embedding_is_differentiable():
    module = TimestepEmbedding(embedding_dim=64)
    t = torch.tensor([0, 10, 100, 999])

    out = module(t)
    out.sum().backward()

    assert module.mlp[0].weight.grad is not None


def test_timestep_embedding_rejects_bad_timestep_rank():
    module = TimestepEmbedding(embedding_dim=32)
    try:
        module(torch.tensor([[0, 1]]))
        raised = False
    except ValueError:
        raised = True
    assert raised


# --- Stage 4: U-Net ---


def _small_unet() -> UNet:
    """Faster U-Net for unit tests (still MNIST-shaped)."""
    return UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mult=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(7,),
        dropout=0.0,
    )


def test_unet_output_shape_matches_input():
    model = _small_unet()
    x = torch.randn(4, 1, 28, 28)
    t = torch.tensor([0, 10, 100, 999])

    out = model(x, t)

    assert out.shape == (4, 1, 28, 28)


def test_unet_on_forward_diffused_mnist():
    model = _small_unet()
    scheduler = NoiseScheduler(num_timesteps=1000)
    train_loader, _ = get_mnist_dataloaders(batch_size=4, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))

    x_t, t, noise = forward_diffuse(scheduler, x_0)
    pred = model(x_t, t)

    assert pred.shape == noise.shape == x_0.shape


def test_unet_is_differentiable():
    model = _small_unet()
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([5, 50])

    pred = model(x, t)
    pred.sum().backward()

    assert model.conv_in.weight.grad is not None
    assert model.time_embed.mlp[0].weight.grad is not None


def test_unet_is_conditioned_on_timestep():
    model = _small_unet()
    model.eval()
    x = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        y0 = model(x, torch.tensor([0]))
        y1 = model(x, torch.tensor([999]))

    assert not torch.allclose(y0, y1)


def test_unet_rejects_batch_mismatch():
    model = _small_unet()
    x = torch.randn(2, 1, 28, 28)
    try:
        model(x, torch.tensor([0, 1, 2]))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_default_unet_param_count_is_positive():
    model = UNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 1_000_000


# --- Stage 5: DDPM model ---


def _small_ddpm() -> DDPM:
    return DDPM(
        unet=_small_unet(),
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    )


def test_ddpm_forward_shapes():
    model = _small_ddpm()
    train_loader, _ = get_mnist_dataloaders(batch_size=4, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))

    noise_pred, noise, t = model(x_0)

    assert noise_pred.shape == x_0.shape
    assert noise.shape == x_0.shape
    assert t.shape == (4,)
    assert int(t.min()) >= 0
    assert int(t.max()) < model.num_timesteps


def test_ddpm_forward_is_differentiable():
    model = _small_ddpm()
    x_0 = torch.randn(2, 1, 28, 28)

    noise_pred, noise, _ = model(x_0)
    loss = torch.mean((noise_pred - noise) ** 2)
    loss.backward()

    assert model.unet.conv_in.weight.grad is not None
    assert model.unet.time_embed.mlp[0].weight.grad is not None


def test_ddpm_respects_provided_t_and_noise():
    model = _small_ddpm()
    model.eval()
    x_0 = torch.randn(3, 1, 28, 28)
    t = torch.tensor([0, 50, 999])
    noise = torch.randn_like(x_0)

    with torch.no_grad():
        pred_a, noise_a, t_a = model(x_0, t=t, noise=noise)
        pred_b, noise_b, t_b = model(x_0, t=t, noise=noise)

    assert torch.equal(t_a, t)
    assert torch.equal(noise_a, noise)
    assert torch.allclose(pred_a, pred_b)


def test_ddpm_predict_noise_matches_unet():
    model = _small_ddpm()
    x_t = torch.randn(2, 1, 28, 28)
    t = torch.tensor([10, 100])

    assert torch.allclose(model.predict_noise(x_t, t), model.unet(x_t, t))


def test_ddpm_builds_default_unet_and_scheduler():
    model = DDPM(
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        num_timesteps=100,
    )
    assert model.num_timesteps == 100
    assert isinstance(model.unet, UNet)
    assert isinstance(model.scheduler, NoiseScheduler)

    noise_pred, noise, t = model(torch.randn(2, 1, 28, 28))
    assert noise_pred.shape == noise.shape == (2, 1, 28, 28)
    assert t.shape == (2,)


# --- Stage 6: diffusion loss ---


def test_ddpm_loss_mean_matches_manual_mse():
    criterion = DDPMLoss(reduction="mean")
    noise_pred = torch.randn(4, 1, 28, 28)
    noise = torch.randn(4, 1, 28, 28)

    loss = criterion(noise_pred, noise)
    expected = torch.mean((noise_pred - noise) ** 2)

    assert loss.shape == torch.Size([])
    assert torch.allclose(loss, expected)


def test_ddpm_loss_perfect_prediction_is_zero():
    criterion = DDPMLoss()
    noise = torch.randn(2, 1, 28, 28)

    assert criterion(noise, noise).item() == 0.0


def test_ddpm_loss_with_model_forward_is_differentiable():
    model = _small_ddpm()
    criterion = DDPMLoss()
    train_loader, _ = get_mnist_dataloaders(batch_size=4, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))

    noise_pred, noise, _ = model(x_0)
    loss = criterion(noise_pred, noise)
    loss.backward()

    assert loss.ndim == 0
    assert model.unet.conv_in.weight.grad is not None


def test_ddpm_loss_rejects_shape_mismatch():
    criterion = DDPMLoss()
    try:
        criterion(torch.randn(2, 1, 28, 28), torch.randn(2, 1, 14, 14))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_ddpm_loss_invalid_reduction_raises():
    try:
        DDPMLoss(reduction="batchmean")
        raised = False
    except ValueError:
        raised = True
    assert raised

