"""Sinusoidal timestep embeddings for conditioning the DDPM U-Net.

Follows the Transformer positional encoding used in Ho et al. (2020):

    PE(t, 2i)     = sin(t / 10000^(2i / d))
    PE(t, 2i + 1) = cos(t / 10000^(2i / d))

A small MLP then maps the fixed sinusoids into the channel width expected by
the network (same pattern as the original DDPM / Improved DDPM codebases).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """Build Transformer-style sinusoidal embeddings for integer timesteps.

    Args:
        timesteps: 1D integer (or float) tensor of shape ``(B,)``.
        embedding_dim: Output feature size ``d`` (even preferred; odd dims are
            zero-padded on the last channel).
        max_period: Controls the minimum frequency (Transformer default 10000).

    Returns:
        Tensor of shape ``(B, embedding_dim)``.
    """
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be 1D (B,), got shape {tuple(timesteps.shape)}")
    if embedding_dim < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")

    half_dim = embedding_dim // 2
    # Frequencies: 1 / max_period^(i / (half_dim - 1)) for i = 0 … half_dim-1
    # When half_dim == 1, the denominator would be 0; use a single frequency.
    if half_dim > 1:
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / (half_dim - 1)
        )
    else:
        freqs = torch.ones(half_dim, device=timesteps.device, dtype=torch.float32)

    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if embedding_dim % 2 == 1:
        embedding = nn.functional.pad(embedding, (0, 1))

    return embedding


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding followed by a 2-layer MLP.

    Input:  integer timesteps ``t`` of shape ``(B,)``
    Output: conditioning vector of shape ``(B, embedding_dim)``
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int | None = None,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")

        self.embedding_dim = embedding_dim
        self.max_period = max_period
        hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must be 1D (B,), got shape {tuple(timesteps.shape)}"
            )
        sinusoid = sinusoidal_time_embedding(
            timesteps,
            embedding_dim=self.embedding_dim,
            max_period=self.max_period,
        )
        return self.mlp(sinusoid)
