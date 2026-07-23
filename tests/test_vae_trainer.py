import csv
from pathlib import Path

import torch
import yaml

from generative_models.datasets import get_mnist_dataloaders
from generative_models.losses import VAELoss
from generative_models.models import VAE
from generative_models.trainers import VAETrainer


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_vae_trainer_runs_one_epoch(tmp_path: Path):
    config = _load_config(Path("configs/vae/mnist.yaml"))
    config.update(
        {
            "epochs": 1,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "sample_dir": str(tmp_path / "samples"),
            "log_dir": str(tmp_path / "logs"),
        }
    )

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

    train_metrics_path = tmp_path / "logs" / "train_metrics.csv"
    val_metrics_path = tmp_path / "logs" / "val_metrics.csv"
    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"
    sample_path = tmp_path / "samples" / "reconstruction_epoch_001.png"

    assert train_metrics_path.exists()
    assert val_metrics_path.exists()
    assert checkpoint_path.exists()
    assert sample_path.exists()

    with train_metrics_path.open("r", encoding="utf-8") as file:
        train_rows = list(csv.DictReader(file))

    assert len(train_rows) == 1
    assert all(torch.isfinite(torch.tensor(float(train_rows[0][key]))) for key in ("total_loss", "recon_loss", "kl_loss"))
    assert torch.allclose(
        torch.tensor(float(train_rows[0]["total_loss"])),
        torch.tensor(float(train_rows[0]["recon_loss"])) + torch.tensor(float(train_rows[0]["kl_loss"])),
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
