from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torchvision.utils import make_grid

from generative_models.models import VAE
from generative_models.utils.device import get_device


def build_vae(config: dict) -> VAE:
    return VAE(
        input_dim=784,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        output_dim=784,
    )


def load_vae_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    config_path: str | Path | None = None,
) -> tuple[VAE, dict]:
    """Load a trained VAE from a checkpoint file."""
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if config is None and config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    if config is None:
        raise ValueError("Config not found in checkpoint and no config_path provided.")

    model = build_vae(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def generate_random_samples(
    model: VAE,
    num_samples: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample z ~ N(0, I) and decode to new images."""
    device = device or next(model.parameters()).device
    with torch.no_grad():
        return model.sample(num_samples, device=device)


def save_sample_grid(
    samples: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    title: str = "Random VAE samples",
) -> Path:
    """Save a grid of generated images."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_grid(samples.cpu(), nrow=nrow)
    plt.figure(figsize=(nrow * 1.2, nrow * 1.2 // 2 + 1))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def reconstruct_images(
    model: VAE,
    images: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode and decode input images."""
    device = device or next(model.parameters()).device
    images = images.to(device)
    with torch.no_grad():
        reconstruction, _, _ = model(images)
    return reconstruction


def save_reconstruction_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    title: str = "Original (top) vs Reconstruction (bottom)",
) -> Path:
    """Save originals and reconstructions in one grid."""
    comparison = torch.cat([originals.cpu(), reconstructions.cpu()], dim=0)
    grid = make_grid(comparison, nrow=nrow)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(nrow * 1.2, 2.4))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
