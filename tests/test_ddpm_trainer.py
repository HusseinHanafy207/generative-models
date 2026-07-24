"""Smoke tests for DDPMTrainer (tiny model + tiny data subset)."""

import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from generative_models.ddpm import DDPM
from generative_models.losses import DDPMLoss
from generative_models.trainers import DDPMTrainer


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _tiny_loaders(data_dir: str, batch_size: int = 8) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(Subset(train_ds, range(32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(test_ds, range(16)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_ddpm_trainer_runs_one_epoch(tmp_path: Path):
    config = _load_config(Path("configs/ddpm/mnist.yaml"))
    config.update(
        {
            "epochs": 1,
            "device": "cpu",
            "batch_size": 8,
            "base_channels": 16,
            "channel_mult": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.0,
            "num_timesteps": 100,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "sample_dir": str(tmp_path / "samples"),
            "log_dir": str(tmp_path / "logs"),
            "seed": 0,
        }
    )

    train_loader, val_loader = _tiny_loaders(config["data_dir"], batch_size=8)

    model = DDPM(
        num_timesteps=config["num_timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        base_channels=config["base_channels"],
        channel_mult=tuple(config["channel_mult"]),
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=tuple(config["attention_resolutions"]),
        dropout=config["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = DDPMLoss()

    trainer = DDPMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    trainer.train()

    train_metrics_path = tmp_path / "logs" / "train_metrics.csv"
    val_metrics_path = tmp_path / "logs" / "val_metrics.csv"
    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"

    assert train_metrics_path.exists()
    assert val_metrics_path.exists()
    assert checkpoint_path.exists()

    with train_metrics_path.open("r", encoding="utf-8") as file:
        train_rows = list(csv.DictReader(file))

    assert len(train_rows) == 1
    assert torch.isfinite(torch.tensor(float(train_rows[0]["loss"])))

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "config" in checkpoint
