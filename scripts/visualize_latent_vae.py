"""Visualize the VAE latent space with PCA and t-SNE."""

import argparse
from pathlib import Path

import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.evaluation import (
    collect_test_latents,
    load_vae_checkpoint,
    reduce_latents_pca,
    reduce_latents_tsne,
    save_latent_scatter,
)
from generative_models.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VAE latent space in 2D.")
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
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vae/figures"),
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)

    _, test_loader = get_mnist_dataloaders(batch_size=256, data_dir="data/raw")
    model, _ = load_vae_checkpoint(args.checkpoint, device=device, config_path=args.config)

    latents, labels = collect_test_latents(
        model,
        test_loader,
        device=device,
        max_samples=args.max_samples,
    )

    pca_coords = reduce_latents_pca(latents)
    pca_path = save_latent_scatter(
        pca_coords,
        labels,
        args.output_dir / "latent_space_pca.png",
        title="VAE Latent Space (PCA)",
    )

    tsne_coords = reduce_latents_tsne(latents)
    tsne_path = save_latent_scatter(
        tsne_coords,
        labels,
        args.output_dir / "latent_space_tsne.png",
        title="VAE Latent Space (t-SNE)",
    )

    print(f"Collected {len(labels)} latent vectors from the test set")
    print(f"Saved PCA plot to {pca_path}")
    print(f"Saved t-SNE plot to {tsne_path}")


if __name__ == "__main__":
    main()
