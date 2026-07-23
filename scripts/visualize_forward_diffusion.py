"""Visualize MNIST digits dissolving into noise under forward diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.ddpm import NoiseScheduler
from generative_models.ddpm.forward import (
    forward_diffuse_trajectory,
    save_forward_diffusion_grid,
)


DEFAULT_TIMESTEPS = [0, 50, 100, 200, 400, 600, 800, 999]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize forward diffusion on MNIST (no neural network)."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of MNIST images (rows in the grid).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=DEFAULT_TIMESTEPS,
        help="Timesteps to show as columns (shared noise across columns).",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Total diffusion steps T in the schedule.",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=1e-4,
        help="Linear schedule beta_start.",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=0.02,
        help="Linear schedule beta_end.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="MNIST download / cache directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ddpm/figures/forward_diffusion.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image choice and shared noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    scheduler = NoiseScheduler(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    # One batch is enough; take the first num_images after shuffling via seed
    train_loader, _ = get_mnist_dataloaders(
        batch_size=max(args.num_images, 64),
        data_dir=str(args.data_dir),
    )
    images, labels = next(iter(train_loader))
    x_0 = images[: args.num_images]
    chosen_labels = labels[: args.num_images].tolist()

    timesteps = args.timesteps
    for step in timesteps:
        if not 0 <= step < scheduler.num_timesteps:
            raise ValueError(
                f"timestep {step} out of range [0, {scheduler.num_timesteps - 1}]"
            )

    progression, _ = forward_diffuse_trajectory(scheduler, x_0, timesteps)

    # Sanity: MSE to x_0 should grow with t when ε is shared
    mse_by_t = []
    for i, step in enumerate(timesteps):
        mse = torch.mean((progression[:, i] - x_0) ** 2).item()
        mse_by_t.append((step, mse))

    output_path = save_forward_diffusion_grid(
        progression,
        timesteps,
        output_path=args.output,
        title="Forward diffusion on MNIST (shared ε per row)",
    )

    print(f"Images (labels): {chosen_labels}")
    print(f"progression shape: {tuple(progression.shape)}  # (B, S, C, H, W)")
    print("Mean MSE(x_t, x_0) by timestep:")
    for step, mse in mse_by_t:
        print(f"  t={step:4d}: {mse:.4f}")
    print(f"Saved grid to {output_path}")


if __name__ == "__main__":
    main()
