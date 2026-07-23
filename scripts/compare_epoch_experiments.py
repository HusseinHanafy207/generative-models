"""Snapshot and compare VAE experiments at different epoch counts."""

import argparse
import csv
import shutil
from pathlib import Path

import torch
import yaml

from generative_models.datasets import get_mnist_dataloaders
from generative_models.evaluation.vae_sampling import (
    generate_random_samples,
    load_vae_checkpoint,
    reconstruct_images,
    save_reconstruction_grid,
    save_sample_grid,
)
from generative_models.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare epoch-50 vs epoch-100 VAE experiments.")
    parser.add_argument(
        "--checkpoint-50",
        type=Path,
        default=Path("outputs/vae/checkpoints/checkpoint_epoch_050.pt"),
    )
    parser.add_argument(
        "--checkpoint-100",
        type=Path,
        default=Path("outputs/vae/checkpoints/checkpoint_epoch_100.pt"),
    )
    parser.add_argument(
        "--train-metrics",
        type=Path,
        default=Path("outputs/vae/logs/train_metrics_sum.csv"),
    )
    parser.add_argument(
        "--val-metrics",
        type=Path,
        default=Path("outputs/vae/logs/val_metrics_sum.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vae/experiments"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def read_epoch_row(metrics_path: Path, epoch: int) -> dict[str, str]:
    with metrics_path.open("r", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if int(row["epoch"]) == epoch]
    if not rows:
        raise ValueError(f"No metrics found for epoch {epoch} in {metrics_path}")
    return rows[-1]


def snapshot_experiment(
    label: str,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> None:
    exp_dir = output_dir / label
    exp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, exp_dir / checkpoint_path.name)

    model, _ = load_vae_checkpoint(checkpoint_path, device=device)
    torch.manual_seed(seed)

    samples = generate_random_samples(model, num_samples=64, device=device)
    save_sample_grid(
        samples,
        exp_dir / "random_samples.png",
        title=f"{label} — random samples (seed={seed})",
    )

    _, test_loader = get_mnist_dataloaders(batch_size=8, data_dir="data/raw")
    images, _ = next(iter(test_loader))
    reconstructions = reconstruct_images(model, images, device=device)
    save_reconstruction_grid(
        images,
        reconstructions,
        exp_dir / "reconstructions.png",
        title=f"{label} — reconstructions",
    )


def write_comparison_table(
    output_dir: Path,
    row_50: dict[str, str],
    row_100: dict[str, str],
    val_50: dict[str, str] | None,
    val_100: dict[str, str] | None,
) -> Path:
    comparison_path = output_dir / "comparison.md"
    lines = [
        "# VAE Experiment Comparison: 50 vs 100 Epochs",
        "",
        "| Metric | 50 Epochs | 100 Epochs | Change |",
        "|--------|-----------|------------|--------|",
    ]

    def fmt_change(a: float, b: float) -> str:
        delta = b - a
        pct = (delta / a * 100) if a else 0.0
        return f"{delta:+.2f} ({pct:+.1f}%)"

    for key, label in [
        ("total_loss", "Train Loss"),
        ("recon_loss", "Recon Loss"),
        ("kl_loss", "KL Loss"),
    ]:
        a = float(row_50[key])
        b = float(row_100[key])
        lines.append(f"| {label} | {a:.2f} | {b:.2f} | {fmt_change(a, b)} |")

    if val_50 and val_100:
        a = float(val_50["total_loss"])
        b = float(val_100["total_loss"])
        lines.append(f"| Val Loss | {a:.2f} | {b:.2f} | {fmt_change(a, b)} |")

    lines.extend(
        [
            "",
            "## Visual comparisons",
            "",
            "- `epoch_50/random_samples.png` vs `epoch_100/random_samples.png`",
            "- `epoch_50/reconstructions.png` vs `epoch_100/reconstructions.png`",
            "",
            "Samples use the same random seed for a fair comparison.",
        ]
    )

    comparison_path.write_text("\n".join(lines), encoding="utf-8")
    return comparison_path


def main() -> None:
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)

    snapshot_experiment("epoch_50", args.checkpoint_50, args.output_dir, args.seed, device)
    snapshot_experiment("epoch_100", args.checkpoint_100, args.output_dir, args.seed, device)

    row_50 = read_epoch_row(args.train_metrics, 50)
    row_100 = read_epoch_row(args.train_metrics, 100)

    val_50 = val_100 = None
    if args.val_metrics.exists():
        val_50 = read_epoch_row(args.val_metrics, 50)
        val_100 = read_epoch_row(args.val_metrics, 100)

    comparison_path = write_comparison_table(
        args.output_dir, row_50, row_100, val_50, val_100
    )

    print(f"Saved experiment snapshots to {args.output_dir}")
    print(f"Saved comparison table to {comparison_path}")


if __name__ == "__main__":
    main()
