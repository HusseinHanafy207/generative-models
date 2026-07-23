"""Reconstruct test-set MNIST digits and save comparison grids."""

import argparse
from pathlib import Path

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.evaluation import load_vae_checkpoint, reconstruct_test_grid, save_reconstruction_grid
from generative_models.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct test MNIST digits with a trained VAE.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/vae/checkpoints/vae_epoch100.pt"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae/mnist.yaml"),
    )
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vae/figures/test_reconstructions.png"),
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)

    _, test_loader = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    model, _ = load_vae_checkpoint(args.checkpoint, device=device, config_path=args.config)

    originals, reconstructions = reconstruct_test_grid(
        model,
        test_loader,
        num_images=args.num_images,
        device=device,
    )
    output_path = save_reconstruction_grid(
        originals,
        reconstructions,
        args.output,
        nrow=args.num_images,
        title="Test set: Original (top) vs Reconstruction (bottom)",
    )

    print(f"Saved reconstruction grid to {output_path}")


if __name__ == "__main__":
    main()
