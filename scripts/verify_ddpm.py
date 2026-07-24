"""Verify the DDPM training forward: x_0 → t → ε → x_t → ε̂."""

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.ddpm import DDPM, UNet


def main() -> None:
    # Small U-Net keeps the smoke check fast
    unet = UNet(
        base_channels=32,
        channel_mult=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(7,),
        dropout=0.0,
    )
    model = DDPM(unet=unet, num_timesteps=1000)

    train_loader, _ = get_mnist_dataloaders(batch_size=8, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))

    print("=== DDPM forward ===")
    print(f"num_timesteps: {model.num_timesteps}")
    print(f"x_0 shape:     {tuple(x_0.shape)}")

    noise_pred, noise, t = model(x_0)

    print(f"t shape:       {tuple(t.shape)}  values={t.tolist()}")
    print(f"eps shape:     {tuple(noise.shape)}")
    print(f"eps_hat shape: {tuple(noise_pred.shape)}")

    loss = torch.mean((noise_pred - noise) ** 2)
    loss.backward()
    print(f"mse(eps_hat, eps) = {loss.item():.4f}")
    print(f"grad on unet.conv_in: {model.unet.conv_in.weight.grad is not None}")

    # Deterministic path with fixed t and noise
    t_fixed = torch.full((x_0.shape[0],), 100, dtype=torch.long)
    noise_fixed = torch.randn_like(x_0)
    pred_a, noise_a, t_a = model(x_0, t=t_fixed, noise=noise_fixed)
    pred_b, noise_b, t_b = model(x_0, t=t_fixed, noise=noise_fixed)
    same = torch.allclose(pred_a, pred_b) and torch.equal(t_a, t_b) and torch.equal(
        noise_a, noise_b
    )
    print(f"deterministic with fixed (t, eps): {same}")

    ok = (
        noise_pred.shape == x_0.shape
        and model.unet.conv_in.weight.grad is not None
        and same
    )
    print("\nDDPM looks correct." if ok else "\nDDPM mismatch!")


if __name__ == "__main__":
    main()
