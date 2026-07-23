"""Noise schedule and closed-form forward diffusion q(x_t | x_0).

Follows Ho et al. (2020), "Denoising Diffusion Probabilistic Models".

Timestep indexing is 0-based in code: t ∈ {0, …, T-1}, where t = 0 is the
least-noisy step and t = T-1 is the noisiest. This matches the paper's
t = 1 … T after a shift of one.
"""

from __future__ import annotations

import torch


class NoiseScheduler:
    """Linear β schedule and closed-form sampling from q(x_t | x_0).

    Defines:
        β_t              linear from ``beta_start`` to ``beta_end``
        α_t = 1 - β_t
        ᾱ_t = ∏_{s=0}^{t} α_s

    Closed form (paper Eq. 4):
        q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t) I)
        x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε,   ε ~ N(0, I)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        if num_timesteps < 1:
            raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError(
                f"Require 0 < beta_start < beta_end < 1, "
                f"got beta_start={beta_start}, beta_end={beta_end}"
            )
        if schedule != "linear":
            raise ValueError(
                f"Unsupported schedule '{schedule}'. "
                "Only 'linear' is implemented (DDPM paper default)."
            )

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        # β_t, α_t, ᾱ_t  — stored as float64 for stable cumprod, cast on use
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def _extract(self, coeffs: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
        """Gather schedule coefficients at timesteps ``t`` and reshape for broadcast."""
        if t.ndim != 1:
            raise ValueError(f"t must be a 1D tensor of shape (batch,), got {tuple(t.shape)}")
        if t.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError(f"t must be an integer tensor, got dtype={t.dtype}")
        if torch.any(t < 0) or torch.any(t >= self.num_timesteps):
            raise ValueError(
                f"t values must be in [0, {self.num_timesteps - 1}], "
                f"got min={int(t.min())}, max={int(t.max())}"
            )

        out = coeffs.to(device=t.device, dtype=torch.float32).gather(0, t.long())
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``x_t`` from ``q(x_t | x_0)`` via the closed-form expression.

        Args:
            x_0: Clean images, shape ``(B, …)``.
            t: Integer timesteps per batch item, shape ``(B,)``, in ``[0, T)``.
            noise: Optional ε with the same shape as ``x_0``. Drawn as
                ``N(0, I)`` when omitted.

        Returns:
            Noisy images ``x_t`` with the same shape as ``x_0``.
        """
        if t.shape[0] != x_0.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x_0 has batch {x_0.shape[0]}, t has {t.shape[0]}"
            )
        if noise is None:
            noise = torch.randn_like(x_0)
        elif noise.shape != x_0.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match x_0 shape {tuple(x_0.shape)}"
            )

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
