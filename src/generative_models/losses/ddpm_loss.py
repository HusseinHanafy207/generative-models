"""DDPM simplified training objective: noise-prediction MSE.

Ho et al. (2020) show that the variational bound can be rewritten so the
network predicts the forward-process noise ε. The simplified loss is:

    L_simple = E_{t, x_0, ε} [ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||² ]

No reconstruction / KL balancing — just mean squared error between the true
noise and the U-Net prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPMLoss(nn.Module):
    """Mean squared error between predicted noise ε̂ and true noise ε."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}"
            )
        self.reduction = reduction

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``||ε̂ - ε||²`` with the configured reduction.

        Args:
            noise_pred: U-Net prediction ε̂, shape ``(B, C, H, W)``.
            noise: Ground-truth ε from the forward process, same shape.

        Returns:
            Scalar loss for ``mean`` / ``sum``, or per-element tensor for ``none``.
        """
        if noise_pred.shape != noise.shape:
            raise ValueError(
                f"Shape mismatch: noise_pred {tuple(noise_pred.shape)} vs "
                f"noise {tuple(noise.shape)}"
            )
        return F.mse_loss(noise_pred, noise, reduction=self.reduction)
