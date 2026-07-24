"""Train a DDPM using configs/ddpm/mnist.yaml.

Phased training (recommended):
  Phase 1 sanity check:  python scripts/train_ddpm.py --epochs 1
  Phase 2 short run:     python scripts/train_ddpm.py --epochs 20
  Resume later:          python scripts/train_ddpm.py --resume outputs/ddpm/checkpoints/latest.pt --epochs 50
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml

from generative_models.datasets import get_mnist_dataloaders
from generative_models.ddpm import DDPM
from generative_models.losses import DDPMLoss
from generative_models.trainers import DDPMTrainer
from generative_models.ddpm.checkpointing import build_ddpm_from_config


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_ddpm(config: dict) -> DDPM:
    return build_ddpm_from_config(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM on MNIST.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ddpm/mnist.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume training from.",
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

    finite = math.isfinite(train_metrics["loss"])
    print(f"[{'OK' if finite else 'FAIL'}] Loss is finite (no NaN/Inf)")

    batch_improved = train_metrics["last_batch_loss"] <= train_metrics["first_batch_loss"]
    print(
        f"[{'OK' if batch_improved else 'WARN'}] Batch loss trend: "
        f"{train_metrics['first_batch_loss']:.4f} -> {train_metrics['last_batch_loss']:.4f}"
    )

    checkpoint_path = Path(config["checkpoint_dir"]) / "latest.pt"
    print(f"[{'OK' if checkpoint_path.exists() else 'FAIL'}] Checkpoint saved")

    train_csv = Path(config["log_dir"]) / config.get(
        "train_metrics_file", "train_metrics.csv"
    )
    print(f"[{'OK' if train_csv.exists() else 'FAIL'}] Train metrics CSV written")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device

    if args.resume is None and config.get("seed") is not None:
        torch.manual_seed(config["seed"])

    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=config["batch_size"],
        data_dir=config["data_dir"],
    )

    model = build_ddpm(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = DDPMLoss(reduction="mean")

    trainer = DDPMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
    )

    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        config["start_epoch"] = checkpoint["epoch"]
        if config["epochs"] <= checkpoint["epoch"]:
            raise ValueError(
                f"--epochs {config['epochs']} must be greater than resumed epoch "
                f"{checkpoint['epoch']}."
            )
        print(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"training to epoch {config['epochs']}"
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DDPM parameters: {n_params:,}")
    print(
        f"Training for {config['epochs']} epochs | "
        f"batch_size={config['batch_size']} | lr={config['learning_rate']}"
    )
    print()

    trainer.train()

    checkpoint = torch.load(
        Path(config["checkpoint_dir"]) / "latest.pt",
        weights_only=False,
    )
    train_metrics = checkpoint["metrics"]

    metrics_file = Path(config["log_dir"]) / config["train_metrics_file"]
    print(f"\nFinished training through epoch {config['epochs']}.")
    print(f"Train metrics CSV: {metrics_file}")
    print(f"Latest checkpoint: {Path(config['checkpoint_dir']) / 'latest.pt'}")

    if config["epochs"] == 1 and args.resume is None:
        print_phase1_checks(train_metrics, config)


if __name__ == "__main__":
    main()
