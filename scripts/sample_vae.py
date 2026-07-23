"""Generate new MNIST digits by sampling z ~ N(0, I) from a trained VAE."""

import argparse
from pathlib import Path

import torch
import yaml

from generative_models.evaluation import (
    generate_random_samples,
    load_vae_checkpoint,
    save_sample_grid,
)
from generative_models.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample new digits from a trained VAE.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/vae/checkpoints/latest.pt"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae/mnist.yaml"),
        help="Fallback config if not stored in checkpoint.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=8,
        help="Images per row in the saved grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vae/samples/random_samples.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, config = load_vae_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        config_path=args.config,
    )

    samples = generate_random_samples(model, num_samples=args.num_samples, device=device)
    output_path = save_sample_grid(
        samples,
        output_path=args.output,
        nrow=args.nrow,
        title=f"Random VAE samples (z ~ N(0, I), n={args.num_samples})",
    )

    print(f"Generated {args.num_samples} samples with shape {tuple(samples.shape)}")
    print(f"Saved grid to {output_path}")


if __name__ == "__main__":
    main()
