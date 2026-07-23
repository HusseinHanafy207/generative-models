"""Train a VAE using configs/vae/mnist.yaml.

Phased training (recommended):
  Phase 1 sanity check:  python scripts/train_vae.py --epochs 1
  Phase 2 short run:     python scripts/train_vae.py --epochs 10
  Phase 3 full run:      python scripts/train_vae.py --epochs 50
"""

import argparse
import math
from pathlib import Path

import torch
import yaml

from generative_models.datasets import get_mnist_dataloaders
from generative_models.losses import VAELoss
from generative_models.models import VAE
from generative_models.trainers import VAETrainer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae/mnist.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto, cpu, cuda).",
    )
    return parser.parse_args()


def print_phase1_checks(train_metrics: dict[str, float], config: dict) -> None:
    print("\nPhase 1 sanity checks")
    print("-" * 24)

    finite = all(
        math.isfinite(train_metrics[key])
        for key in ("total_loss", "recon_loss", "kl_loss")
    )
    print(f"[{'OK' if finite else 'FAIL'}] Loss is finite (no NaN/Inf)")

    batch_improved = train_metrics["last_batch_loss"] <= train_metrics["first_batch_loss"]
    print(
        f"[{'OK' if batch_improved else 'WARN'}] Batch loss trend: "
        f"{train_metrics['first_batch_loss']:.2f} -> {train_metrics['last_batch_loss']:.2f}"
    )

    checkpoint_path = Path(config["checkpoint_dir"]) / "latest.pt"
    print(f"[{'OK' if checkpoint_path.exists() else 'FAIL'}] Checkpoint saved")

    train_csv = Path(config["log_dir"]) / "train_metrics.csv"
    print(f"[{'OK' if train_csv.exists() else 'FAIL'}] Train metrics CSV written")

    sample_glob = list(Path(config["sample_dir"]).glob("reconstruction_epoch_*.png"))
    print(f"[{'OK' if sample_glob else 'FAIL'}] Reconstruction image saved")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device

    config.setdefault("train_metrics_file", "train_metrics_sum.csv")
    config.setdefault("val_metrics_file", "val_metrics_sum.csv")

    if config.get("seed") is not None:
        torch.manual_seed(config["seed"])

    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=config["batch_size"],
        data_dir=config["data_dir"],
    )

    model = VAE(
        input_dim=784,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        output_dim=784,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = VAELoss()

    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
    )
    trainer.train()

    checkpoint = torch.load(
        Path(config["checkpoint_dir"]) / "latest.pt",
        weights_only=False,
    )
    train_metrics = checkpoint["metrics"]

    train_metrics_path = Path(config["log_dir"]) / "train_metrics.csv"
    print(f"\nFinished training for {config['epochs']} epoch(s).")
    print(f"Train metrics CSV: {train_metrics_path}")
    print(f"Latest checkpoint: {Path(config['checkpoint_dir']) / 'latest.pt'}")

    if config["epochs"] == 1:
        print_phase1_checks(train_metrics, config)


if __name__ == "__main__":
    main()
