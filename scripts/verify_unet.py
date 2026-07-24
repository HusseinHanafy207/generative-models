"""Verify MNIST U-Net shapes, gradients, and timestep conditioning."""

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.ddpm import NoiseScheduler, UNet, forward_diffuse


def main() -> None:
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(7,),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print("=== UNet ===")
    print(f"parameters: {n_params:,}")

    train_loader, _ = get_mnist_dataloaders(batch_size=8, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))
    scheduler = NoiseScheduler(num_timesteps=1000)

    x_t, t, noise = forward_diffuse(scheduler, x_0)
    pred = model(x_t, t)

    print(f"x_0 shape:   {tuple(x_0.shape)}")
    print(f"x_t shape:   {tuple(x_t.shape)}")
    print(f"t shape:     {tuple(t.shape)}  values={t.tolist()}")
    print(f"eps shape:   {tuple(noise.shape)}")
    print(f"eps_hat shape: {tuple(pred.shape)}")

    loss = torch.mean((pred - noise) ** 2)
    loss.backward()
    print(f"mse(eps_hat, eps) = {loss.item():.4f}")
    print(f"grad on conv_in: {model.conv_in.weight.grad is not None}")

    model.eval()
    with torch.no_grad():
        x = x_0[:1]
        y_early = model(x, torch.tensor([0]))
        y_late = model(x, torch.tensor([999]))
        diff = (y_early - y_late).abs().mean().item()
    print(f"mean |f(x,0) - f(x,999)| = {diff:.4f}  (should be > 0)")

    ok = pred.shape == x_0.shape and diff > 0 and model.conv_in.weight.grad is not None
    print("\nUNet looks correct." if ok else "\nUNet mismatch!")


if __name__ == "__main__":
    main()
