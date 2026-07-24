"""Visualize sinusoidal timestep embeddings as a heatmap over t."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from generative_models.ddpm import sinusoidal_time_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sinusoidal timestep embeddings (no neural network yet)."
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Sinusoidal feature dimension.",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Number of timesteps to plot along the vertical axis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ddpm/figures/time_embedding.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timesteps = torch.arange(args.num_timesteps)
    emb = sinusoidal_time_embedding(timesteps, embedding_dim=args.embedding_dim)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(emb.numpy(), aspect="auto", interpolation="nearest", cmap="coolwarm")
    plt.colorbar(label="Activation")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Timestep t")
    plt.title(
        f"Sinusoidal timestep embeddings "
        f"(T={args.num_timesteps}, dim={args.embedding_dim})"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"embedding shape: {tuple(emb.shape)}")
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
