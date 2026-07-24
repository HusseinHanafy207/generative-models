"""DDPM model: compose noise scheduler + U-Net denoiser.

Training forward pass (Ho et al., 2020):

    x_0 → sample t ~ Uniform{0…T-1}
        → sample ε ~ N(0, I)
        → x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε
        → ε̂ = UNet(x_t, t)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from generative_models.ddpm.forward import forward_diffuse
from generative_models.ddpm.scheduler import NoiseScheduler
from generative_models.ddpm.unet import UNet


class DDPM(nn.Module):
    """Denoising diffusion model that predicts the noise added to ``x_0``."""

    def __init__(
        self,
        unet: UNet | None = None,
        scheduler: NoiseScheduler | None = None,
        *,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        **unet_kwargs,
    ) -> None:
        super().__init__()
        self.scheduler = scheduler or NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.unet = unet or UNet(**unet_kwargs)

    @property
    def num_timesteps(self) -> int:
        return self.scheduler.num_timesteps

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict ε̂ from a noisy image at timestep ``t``."""
        return self.unet(x_t, t)

    def forward(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the training forward pass.

        Args:
            x_0: Clean images, shape ``(B, C, H, W)``.
            t: Optional timesteps ``(B,)``. Sampled uniformly when omitted.
            noise: Optional ε with the same shape as ``x_0``. Sampled when omitted.

        Returns:
            ``(noise_pred, noise, t)`` — predicted ε̂, true ε, and timesteps used.
            The Stage 6 loss is ``MSE(noise_pred, noise)``.
        """
        x_t, t, noise = forward_diffuse(self.scheduler, x_0, t=t, noise=noise)
        noise_pred = self.predict_noise(x_t, t)
        return noise_pred, noise, t
