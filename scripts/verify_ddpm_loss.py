"""Verify DDPM noise-prediction loss on a real forward pass."""

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.ddpm import DDPM, UNet
from generative_models.losses import DDPMLoss


def main() -> None:
    unet = UNet(
        base_channels=32,
        channel_mult=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(7,),
        dropout=0.0,
    )
    model = DDPM(unet=unet, num_timesteps=1000)
    criterion = DDPMLoss(reduction="mean")

    train_loader, _ = get_mnist_dataloaders(batch_size=8, data_dir="data/raw")
    x_0, _ = next(iter(train_loader))

    noise_pred, noise, t = model(x_0)
    loss = criterion(noise_pred, noise)
    loss.backward()

    print("=== DDPMLoss ===")
    print(f"noise_pred shape: {tuple(noise_pred.shape)}")
    print(f"noise shape:      {tuple(noise.shape)}")
    print(f"t values:         {t.tolist()}")
    print(f"loss shape:       {tuple(loss.shape)}")
    print(f"loss (MSE mean):  {loss.item():.4f}")
    print(f"manual mean sq:   {torch.mean((noise_pred - noise) ** 2).item():.4f}")
    print(f"grad on unet:     {model.unet.conv_in.weight.grad is not None}")

    # Perfect prediction → zero loss
    zero = criterion(noise, noise)
    print(f"MSE(eps, eps):    {zero.item():.2e}")

    ok = (
        loss.ndim == 0
        and abs(loss.item() - torch.mean((noise_pred.detach() - noise) ** 2).item()) < 1e-6
        and zero.item() == 0.0
        and model.unet.conv_in.weight.grad is not None
    )
    print("\nDDPM loss looks correct." if ok else "\nDDPM loss mismatch!")


if __name__ == "__main__":
    main()
