"""DDPM reverse-process sampling (Ho et al., 2020).

Start from pure Gaussian noise and iteratively denoise:

    x_T → x_{T-1} → … → x_0
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm

from generative_models.ddpm.diffusion import DDPM


@torch.no_grad()
def p_sample(
    model: DDPM,
    x_t: torch.Tensor,
    t: int,
) -> torch.Tensor:
    """Denoise ``x_t`` for a single shared integer timestep ``t``."""
    batch = x_t.shape[0]
    t_batch = torch.full((batch,), t, device=x_t.device, dtype=torch.long)
    noise_pred = model.predict_noise(x_t, t_batch)
    return model.scheduler.p_sample_step(x_t, t_batch, noise_pred)


@torch.no_grad()
def sample(
    model: DDPM,
    num_samples: int,
    image_size: int = 28,
    channels: int = 1,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Generate images by running the full reverse diffusion chain.

    Returns:
        Tensor of shape ``(num_samples, channels, image_size, image_size)``
        in roughly the same value range as training data (clamped to ``[0, 1]``
        for display convenience when training used ``[0, 1]`` images).
    """
    device = device or next(model.unet.parameters()).device
    model.eval()

    x = torch.randn(num_samples, channels, image_size, image_size, device=device)
    timesteps = range(model.num_timesteps - 1, -1, -1)
    iterator = tqdm(timesteps, desc="sampling", leave=False) if show_progress else timesteps

    for t in iterator:
        x = p_sample(model, x, t)

    return x.clamp(0.0, 1.0)


@torch.no_grad()
def sample_trajectory(
    model: DDPM,
    num_samples: int = 1,
    image_size: int = 28,
    channels: int = 1,
    save_every: int = 50,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """Generate samples and keep intermediate frames for visualization.

    Frames are recorded at ``t = T-1, …`` every ``save_every`` steps, plus
    the final ``t = 0`` result.

    Returns:
        ``frames`` of shape ``(num_frames, num_samples, C, H, W)`` and the
        list of timesteps corresponding to each frame.
    """
    device = device or next(model.unet.parameters()).device
    model.eval()

    x = torch.randn(num_samples, channels, image_size, image_size, device=device)
    frames: list[torch.Tensor] = [x.clamp(0.0, 1.0).cpu()]
    frame_timesteps: list[int] = [model.num_timesteps - 1]

    timesteps = range(model.num_timesteps - 1, -1, -1)
    iterator = tqdm(timesteps, desc="sampling", leave=False) if show_progress else timesteps

    for t in iterator:
        x = p_sample(model, x, t)
        if t % save_every == 0 or t == 0:
            frames.append(x.clamp(0.0, 1.0).cpu())
            frame_timesteps.append(t)

    return torch.stack(frames, dim=0), frame_timesteps


def save_sample_grid(
    samples: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    title: str = "DDPM samples",
) -> Path:
    """Save a grid of generated images."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_grid(samples.cpu().clamp(0.0, 1.0), nrow=nrow, padding=2)
    plt.figure(figsize=(nrow * 1.2, max((samples.shape[0] + nrow - 1) // nrow, 1) * 1.2 + 0.8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_denoising_gif(
    frames: torch.Tensor,
    output_path: str | Path,
    sample_index: int = 0,
    duration_ms: int = 80,
) -> Path:
    """Save a GIF of one sample's reverse trajectory.

    Args:
        frames: Tensor ``(num_frames, num_samples, C, H, W)``.
        sample_index: Which sample in the batch to animate.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images: list[Image.Image] = []
    for frame in frames:
        img = frame[sample_index].clamp(0.0, 1.0)
        if img.shape[0] == 1:
            arr = (img[0].numpy() * 255).astype("uint8")
            images.append(Image.fromarray(arr, mode="L"))
        else:
            arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
            images.append(Image.fromarray(arr))

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path
