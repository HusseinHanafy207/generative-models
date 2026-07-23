"""Forward diffusion helpers built on ``NoiseScheduler.q_sample``.

Training-style step:
    x_0 → sample t → sample ε → x_t

Visualization trajectory (shared ε across timesteps):
    x_0 → {x_{t0}, x_{t1}, …, x_{tS}}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from generative_models.ddpm.scheduler import NoiseScheduler


def sample_timesteps(
    batch_size: int,
    num_timesteps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample a batch of integer timesteps uniformly from ``[0, T)``."""
    return torch.randint(0, num_timesteps, (batch_size,), device=device)


def forward_diffuse(
    scheduler: NoiseScheduler,
    x_0: torch.Tensor,
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one forward-diffusion step: ``(x_0, t, ε) → x_t``.

    If ``t`` or ``noise`` is omitted, they are sampled (uniform timestep,
    ``ε ~ N(0, I)``). Returns ``(x_t, t, noise)`` so callers can reuse the
    noise target for the DDPM training loss later.
    """
    if t is None:
        t = sample_timesteps(x_0.shape[0], scheduler.num_timesteps, device=x_0.device)
    if noise is None:
        noise = torch.randn_like(x_0)

    x_t = scheduler.q_sample(x_0, t, noise=noise)
    return x_t, t, noise


def forward_diffuse_trajectory(
    scheduler: NoiseScheduler,
    x_0: torch.Tensor,
    timesteps: torch.Tensor | list[int],
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffuse the same ``x_0`` (and shared ``ε``) at every requested ``t``.

    Args:
        scheduler: Noise schedule.
        x_0: Clean images, shape ``(B, C, H, W)``.
        timesteps: 1D integer timesteps of length ``S``.
        noise: Optional shared ε with the same shape as ``x_0``.

    Returns:
        ``progression`` of shape ``(B, S, C, H, W)`` and the shared ``noise``.
    """
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, dtype=torch.long, device=x_0.device)
    else:
        timesteps = timesteps.to(device=x_0.device, dtype=torch.long)

    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be 1D, got shape {tuple(timesteps.shape)}")

    if noise is None:
        noise = torch.randn_like(x_0)

    frames: list[torch.Tensor] = []
    batch_size = x_0.shape[0]
    for step in timesteps:
        t = step.expand(batch_size)
        x_t = scheduler.q_sample(x_0, t, noise=noise)
        frames.append(x_t)

    # (S, B, C, H, W) → (B, S, C, H, W)
    progression = torch.stack(frames, dim=0).transpose(0, 1)
    return progression, noise


def save_forward_diffusion_grid(
    progression: torch.Tensor,
    timesteps: torch.Tensor | list[int],
    output_path: str | Path,
    title: str = "Forward diffusion: x_0 to x_T",
) -> Path:
    """Save a grid with one image per row and increasing ``t`` across columns.

    Args:
        progression: Tensor ``(B, S, C, H, W)`` from ``forward_diffuse_trajectory``.
        timesteps: Length-``S`` list/tensor used as column labels in the title.
        output_path: Where to write the PNG.
    """
    if progression.ndim != 5:
        raise ValueError(
            f"progression must have shape (B, S, C, H, W), got {tuple(progression.shape)}"
        )

    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, dtype=torch.long)
    num_steps = timesteps.numel()
    if progression.shape[1] != num_steps:
        raise ValueError(
            f"progression has S={progression.shape[1]} but len(timesteps)={num_steps}"
        )

    batch_size, _, channels, height, width = progression.shape
    # Clamp for display only: noisy samples can leave [0, 1]
    display = progression.detach().cpu().clamp(0.0, 1.0)
    flat = display.reshape(batch_size * num_steps, channels, height, width)
    grid = make_grid(flat, nrow=num_steps, padding=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_labels = ", ".join(str(int(t)) for t in timesteps.tolist())
    fig_w = max(num_steps * 1.1, 6)
    fig_h = max(batch_size * 1.1, 2.5)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title(f"{title}\nt = [{t_labels}]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
