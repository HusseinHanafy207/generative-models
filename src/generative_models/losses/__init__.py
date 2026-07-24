"""Loss functions for generative models."""

from generative_models.losses.ddpm_loss import DDPMLoss
from generative_models.losses.vae_loss import VAELoss

__all__ = ["DDPMLoss", "VAELoss"]
