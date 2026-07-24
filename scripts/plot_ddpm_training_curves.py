"""Plot DDPM training curves from CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DDPM training curves.")
    parser.add_argument(
        "--train-metrics",
        type=Path,
        default=Path("outputs/ddpm/logs/train_metrics.csv"),
    )
    parser.add_argument(
        "--val-metrics",
        type=Path,
        default=Path("outputs/ddpm/logs/val_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ddpm/figures/training_curves.png"),
    )
    parser.add_argument(
        "--readme-output",
        type=Path,
        default=Path("docs/assets/ddpm/training_curves.png"),
    )
    return parser.parse_args()


def load_latest_epoch_rows(metrics_path: Path) -> list[dict[str, str]]:
    """Keep the last logged row for each epoch (handles resumed runs)."""
    rows_by_epoch: dict[int, dict[str, str]] = {}
    with metrics_path.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows_by_epoch[int(row["epoch"])] = row
    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def plot_curves(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]] | None,
    output_path: Path,
) -> None:
    epochs = [int(row["epoch"]) for row in train_rows]
    train_loss = [float(row["loss"]) for row in train_rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_loss, label="Train MSE", linewidth=2)

    if val_rows:
        val_epochs = [int(row["epoch"]) for row in val_rows]
        val_loss = [float(row["loss"]) for row in val_rows]
        plt.plot(val_epochs, val_loss, label="Val MSE", linewidth=2)

    plt.title("DDPM training curves (noise-prediction MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    train_rows = load_latest_epoch_rows(args.train_metrics)
    val_rows = (
        load_latest_epoch_rows(args.val_metrics) if args.val_metrics.exists() else None
    )

    plot_curves(train_rows, val_rows, args.output)
    print(f"Saved {args.output}")

    args.readme_output.parent.mkdir(parents=True, exist_ok=True)
    plot_curves(train_rows, val_rows, args.readme_output)
    print(f"Saved {args.readme_output}")


if __name__ == "__main__":
    main()
