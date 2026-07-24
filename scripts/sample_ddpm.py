"""Generate MNIST digits from a trained DDPM via reverse diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from generative_models.ddpm.checkpointing import load_ddpm_checkpoint
from generative_models.ddpm.sampler import (
    sample,
    sample_trajectory,
    save_denoising_gif,
    save_sample_grid,
)
from generative_models.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample new digits from a trained DDPM (reverse process)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/ddpm/checkpoints/latest.pt"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ddpm/mnist.yaml"),
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
        default=None,
        help="Output PNG path (default: outputs/ddpm/samples/samples_epoch_XXX.png).",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Optional path for a denoising GIF of one sample.",
    )
    parser.add_argument(
        "--gif-save-every",
        type=int,
        default=50,
        help="Save a GIF frame every N reverse steps.",
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
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    model, config = load_ddpm_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        config_path=args.config,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", "?")

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Device: {device} | T={model.num_timesteps} | samples={args.num_samples}")

    samples = sample(
        model,
        num_samples=args.num_samples,
        device=device,
        show_progress=True,
    )

    output = args.output
    if output is None:
        output = Path(config.get("sample_dir", "outputs/ddpm/samples")) / (
            f"samples_epoch_{epoch:03d}.png"
            if isinstance(epoch, int)
            else "samples.png"
        )

    output_path = save_sample_grid(
        samples,
        output_path=output,
        nrow=args.nrow,
        title=f"DDPM samples (epoch {epoch}, n={args.num_samples})",
    )
    print(f"Generated samples shape: {tuple(samples.shape)}")
    print(f"Saved grid to {output_path}")

    if args.gif is not None:
        frames, frame_ts = sample_trajectory(
            model,
            num_samples=1,
            save_every=args.gif_save_every,
            device=device,
            show_progress=True,
        )
        gif_path = save_denoising_gif(frames, args.gif, sample_index=0)
        print(f"Saved denoising GIF ({len(frame_ts)} frames) to {gif_path}")
        print(f"Frame timesteps: {frame_ts[:5]} ... {frame_ts[-3:]}")


if __name__ == "__main__":
    main()
