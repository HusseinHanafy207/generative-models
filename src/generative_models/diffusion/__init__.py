"""Diffusion model implementations.

DDPM components live in ``generative_models.ddpm``. This package is kept as a
stable import path alias.
"""

from generative_models.ddpm import NoiseScheduler

__all__ = ["NoiseScheduler"]
