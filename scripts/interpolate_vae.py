"""Interpolate between two digit latent vectors and save a transition grid."""

import argparse
from pathlib import Path

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.evaluation import (
    find_digit_image,
    interpolate_latents,
    load_vae_checkpoint,
    save_sample_grid,
)
from generative_models.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent interpolation between two MNIST digits.")
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
    parser.add_argument("--digit-a", type=int, default=3)
    parser.add_argument("--digit-b", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vae/figures/latent_interpolation_3_to_8.png"),
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)

    _, test_loader = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    model, _ = load_vae_checkpoint(args.checkpoint, device=device, config_path=args.config)

    image_a = find_digit_image(test_loader, args.digit_a, device=device)
    image_b = find_digit_image(test_loader, args.digit_b, device=device)

    decoded_steps = interpolate_latents(
        model,
        image_a,
        image_b,
        steps=args.steps,
        device=device,
    )

    output_path = save_sample_grid(
        decoded_steps,
        args.output,
        nrow=args.steps,
        title=f"Latent interpolation: {args.digit_a} → {args.digit_b}",
    )

    print(f"Interpolated {args.digit_a} -> {args.digit_b} in {args.steps} steps")
    print(f"Saved grid to {output_path}")


if __name__ == "__main__":
    main()
