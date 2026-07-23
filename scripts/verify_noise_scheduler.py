"""Verify DDPM noise scheduler math on paper defaults (T=1000, linear β)."""

import torch

from generative_models.ddpm import NoiseScheduler


def main() -> None:
    scheduler = NoiseScheduler(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule="linear",
    )

    print("=== Schedule tensors ===")
    print(f"betas shape:          {tuple(scheduler.betas.shape)}")
    print(f"alphas shape:         {tuple(scheduler.alphas.shape)}")
    print(f"alphas_cumprod shape: {tuple(scheduler.alphas_cumprod.shape)}")
    print(f"beta_0  = {scheduler.betas[0].item():.6f}")
    print(f"beta_T  = {scheduler.betas[-1].item():.6f}")
    print(f"alpha_0 = {scheduler.alphas[0].item():.6f}")
    print(f"alpha_T = {scheduler.alphas[-1].item():.6f}")
    print(f"alpha_bar_0     = {scheduler.alphas_cumprod[0].item():.6f}")
    print(f"alpha_bar_50    = {scheduler.alphas_cumprod[50].item():.6f}")
    print(f"alpha_bar_500   = {scheduler.alphas_cumprod[500].item():.6f}")
    print(f"alpha_bar_999   = {scheduler.alphas_cumprod[-1].item():.6f}")

    # Closed-form identity: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
    torch.manual_seed(42)
    x_0 = torch.randn(4, 1, 28, 28)
    noise = torch.randn_like(x_0)
    t = torch.tensor([0, 50, 500, 999])
    x_t = scheduler.q_sample(x_0, t, noise=noise)

    sqrt_ab = scheduler.sqrt_alphas_cumprod[t].float().view(4, 1, 1, 1)
    sqrt_omb = scheduler.sqrt_one_minus_alphas_cumprod[t].float().view(4, 1, 1, 1)
    expected = sqrt_ab * x_0 + sqrt_omb * noise
    max_err = (x_t - expected).abs().max().item()

    print("\n=== q(x_t | x_0) closed form ===")
    print(f"x_0 shape: {tuple(x_0.shape)}")
    print(f"x_t shape: {tuple(x_t.shape)}")
    print(f"max |x_t - (sqrt(a_bar) x_0 + sqrt(1-a_bar) eps)| = {max_err:.2e}")

    for i, step in enumerate(t.tolist()):
        signal = sqrt_ab[i].item()
        noise_scale = sqrt_omb[i].item()
        print(
            f"t={step:4d}: sqrt(a_bar)={signal:.4f}, "
            f"sqrt(1-a_bar)={noise_scale:.4f}, "
            f"mse(x_t, x_0)={torch.mean((x_t[i] - x_0[i]) ** 2).item():.4f}"
        )

    print("\nScheduler looks correct." if max_err < 1e-5 else "\nScheduler mismatch!")


if __name__ == "__main__":
    main()
