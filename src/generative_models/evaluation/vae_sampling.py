from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
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


def encode_mu(
    model: VAE,
    images: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return latent means mu for each image."""
    device = device or next(model.parameters()).device
    images = images.to(device)
    with torch.no_grad():
        mu, _ = model.encoder(images)
    return mu


def generate_random_samples(
    model: VAE,
    num_samples: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample z ~ N(0, I) and decode to new images."""
    device = device or next(model.parameters()).device
    with torch.no_grad():
        return model.sample(num_samples, device=device)


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


def reconstruct_test_grid(
    model: VAE,
    test_loader: DataLoader,
    num_images: int = 8,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take test images and return originals with reconstructions."""
    images, _ = next(iter(test_loader))
    images = images[:num_images]
    reconstructions = reconstruct_images(model, images, device=device)
    return images, reconstructions


def find_digit_image(
    loader: DataLoader,
    digit: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the first image of a given digit from a dataloader."""
    for images, labels in loader:
        matches = images[labels == digit]
        if len(matches) > 0:
            return matches[0:1].to(device) if device else matches[0:1]
    raise ValueError(f"Digit {digit} not found in loader.")


def interpolate_latents(
    model: VAE,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    steps: int = 10,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Linearly interpolate between latent means and decode each step."""
    device = device or next(model.parameters()).device
    z_a = encode_mu(model, image_a, device=device)
    z_b = encode_mu(model, image_b, device=device)

    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    interpolated = torch.stack(
        [(1 - alpha) * z_a + alpha * z_b for alpha in alphas],
        dim=0,
    ).squeeze(1)

    with torch.no_grad():
        decoded = model.decoder(interpolated.view(steps, -1))
    return decoded


def collect_test_latents(
    model: VAE,
    test_loader: DataLoader,
    device: torch.device | None = None,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect latent vectors and labels from the test set."""
    device = device or next(model.parameters()).device
    latents: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    total = 0

    for images, batch_labels in test_loader:
        mu = encode_mu(model, images, device=device)
        latents.append(mu.cpu())
        labels.append(batch_labels)
        total += len(batch_labels)
        if max_samples is not None and total >= max_samples:
            break

    latent_array = torch.cat(latents, dim=0).numpy()
    label_array = torch.cat(labels, dim=0).numpy()

    if max_samples is not None:
        latent_array = latent_array[:max_samples]
        label_array = label_array[:max_samples]

    return latent_array, label_array


def reduce_latents_pca(latents: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(latents)


def reduce_latents_tsne(latents: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(latents)


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
    plt.figure(figsize=(nrow * 1.2, max(nrow // 2, 2)))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


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


def save_latent_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a 2D latent space scatter plot colored by digit label."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.7,
    )
    plt.colorbar(scatter, ticks=range(10), label="Digit")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
