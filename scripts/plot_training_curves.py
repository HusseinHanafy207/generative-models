"""Plot VAE training curves from CSV logs."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VAE training curves.")
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
        "--output",
        type=Path,
        default=Path("outputs/vae/figures/training_curves.png"),
    )
    parser.add_argument(
        "--readme-output",
        type=Path,
        default=Path("docs/assets/vae/training_curves.png"),
    )
    return parser.parse_args()


def load_latest_epoch_rows(metrics_path: Path) -> list[dict[str, str]]:
    """Keep the last logged row for each epoch (handles re-runs in the same CSV)."""
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
    train_total = [float(row["total_loss"]) for row in train_rows]
    train_recon = [float(row["recon_loss"]) for row in train_rows]
    train_kl = [float(row["kl_loss"]) for row in train_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_total, label="Total", linewidth=2)
    axes[0].plot(epochs, train_recon, label="Reconstruction")
    axes[0].plot(epochs, train_kl, label="KL")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (sum reduction)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if val_rows:
        val_epochs = [int(row["epoch"]) for row in val_rows]
        val_total = [float(row["total_loss"]) for row in val_rows]
        axes[1].plot(val_epochs, val_total, label="Validation", color="tab:orange", linewidth=2)
    else:
        axes[1].text(0.5, 0.5, "No validation metrics found", ha="center", va="center")

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (sum reduction)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    train_rows = load_latest_epoch_rows(args.train_metrics)
    val_rows = load_latest_epoch_rows(args.val_metrics) if args.val_metrics.exists() else None

    plot_curves(train_rows, val_rows, args.output)
    plot_curves(train_rows, val_rows, args.readme_output)

    print(f"Saved training curves to {args.output}")
    print(f"Saved README copy to {args.readme_output}")


if __name__ == "__main__":
    main()
