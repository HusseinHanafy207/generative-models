"""Verify forward diffusion shapes and the shared-noise trajectory."""

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.ddpm import NoiseScheduler
from generative_models.ddpm.forward import forward_diffuse, forward_diffuse_trajectory


def main() -> None:
    scheduler = NoiseScheduler(num_timesteps=1000)
    train_loader, _ = get_mnist_dataloaders(batch_size=8, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))

    print("=== Training-style forward step ===")
    torch.manual_seed(0)
    x_t, t, noise = forward_diffuse(scheduler, x_0)
    print(f"x_0 shape: {tuple(x_0.shape)}")
    print(f"t shape:   {tuple(t.shape)}  values={t.tolist()}")
    print(f"eps shape: {tuple(noise.shape)}")
    print(f"x_t shape: {tuple(x_t.shape)}")

    print("\n=== Shared-noise trajectory ===")
    timesteps = [0, 50, 100, 200, 400, 600, 800, 999]
    progression, shared_noise = forward_diffuse_trajectory(scheduler, x_0, timesteps)
    print(f"timesteps: {timesteps}")
    print(f"progression shape: {tuple(progression.shape)}")
    print(f"shared eps shape:  {tuple(shared_noise.shape)}")

    print("\nMSE(x_t, x_0) should increase with t:")
    prev_mse = -1.0
    monotonic = True
    for i, step in enumerate(timesteps):
        mse = torch.mean((progression[:, i] - x_0) ** 2).item()
        marker = "ok" if mse >= prev_mse - 1e-6 else "NOT monotonic"
        if mse < prev_mse - 1e-6:
            monotonic = False
        print(f"  t={step:4d}: mse={mse:.4f}  [{marker}]")
        prev_mse = mse

    print("\nForward diffusion looks correct." if monotonic else "\nUnexpected MSE trend.")


if __name__ == "__main__":
    main()
