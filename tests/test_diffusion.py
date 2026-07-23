"""Mathematical and shape tests for DDPM scheduler and forward diffusion."""

from pathlib import Path

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.ddpm import (
    NoiseScheduler,
    forward_diffuse,
    forward_diffuse_trajectory,
    sample_timesteps,
    save_forward_diffusion_grid,
)


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
