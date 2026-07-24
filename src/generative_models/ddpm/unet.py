"""Small U-Net denoiser for MNIST, conditioned on diffusion timestep.

Architecture follows the spirit of Ho et al. (2020): an encoder–decoder with
residual blocks, skip connections, and Transformer sinusoidal time embeddings
injected into every residual block. Scaled down for 28×28 grayscale images:

    28×28 (base) → 14×14 (2×) → 7×7 (4× + self-attention) → up again
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_models.ddpm.time_embedding import TimestepEmbedding


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with as many groups as possible up to ``max_groups``."""
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResidualBlock(nn.Module):
    """Conv residual block with additive time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial locations."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, height, width = x.shape
        head_dim = c // self.num_heads

        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.num_heads, head_dim, height * width).unbind(1)
        scale = head_dim**-0.5
        attn = torch.softmax(torch.einsum("bhdi,bhdj->bhij", q * scale, k), dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(b, c, height, width)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """Time-conditioned U-Net that predicts noise ε for a noisy image x_t.

    Default config targets MNIST (1×28×28):
        base_channels=64, channel_mult=(1, 2, 4) → 28 → 14 → 7,
        self-attention at 7×7.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (7,),
        dropout: float = 0.1,
        time_embedding_dim: int | None = None,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        if not channel_mult:
            raise ValueError("channel_mult must be non-empty")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.image_size = image_size

        time_dim = time_embedding_dim or base_channels * 4
        self.time_dim = time_dim
        self.time_embed = TimestepEmbedding(embedding_dim=time_dim)

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder levels: each is a list of (ResidualBlock, attention_or_identity)
        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        skip_channels: list[int] = [base_channels]
        ch = base_channels
        resolution = image_size

        for level, mult in enumerate(self.channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                block = ResidualBlock(ch, out_ch, time_dim=time_dim, dropout=dropout)
                attn: nn.Module
                if resolution in self.attention_resolutions:
                    attn = AttentionBlock(out_ch)
                else:
                    attn = nn.Identity()
                blocks.append(nn.ModuleList([block, attn]))
                ch = out_ch
                skip_channels.append(ch)
            self.encoder.append(blocks)

            if level != len(self.channel_mult) - 1:
                self.downs.append(Downsample(ch))
                skip_channels.append(ch)
                resolution //= 2
            else:
                self.downs.append(nn.Identity())

        self.mid_block1 = ResidualBlock(ch, ch, time_dim=time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_dim=time_dim, dropout=dropout)

        # Decoder levels (deep → shallow), mirrored multipliers
        self.decoder = nn.ModuleList()
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block = ResidualBlock(
                    ch + skip_ch, out_ch, time_dim=time_dim, dropout=dropout
                )
                # resolution at this decoder level:
                # deepest level starts at image_size // 2**(L-1)
                attn = (
                    AttentionBlock(out_ch)
                    if resolution in self.attention_resolutions
                    else nn.Identity()
                )
                blocks.append(nn.ModuleList([block, attn]))
                ch = out_ch
            self.decoder.append(blocks)

            if level != 0:
                self.ups.append(Upsample(ch))
                resolution *= 2
            else:
                self.ups.append(nn.Identity())

        self.conv_out = nn.Sequential(
            _group_norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict noise for noisy images ``x`` at ``timesteps``.

        Args:
            x: Noisy images, shape ``(B, C, H, W)``.
            timesteps: Integer timesteps, shape ``(B,)``.

        Returns:
            Predicted noise ε̂ with the same shape as ``x``.
        """
        if x.ndim != 4:
            raise ValueError(f"x must be (B, C, H, W), got {tuple(x.shape)}")
        if timesteps.shape != (x.shape[0],):
            raise ValueError(
                f"timesteps shape {tuple(timesteps.shape)} must be (batch,) "
                f"matching x batch {x.shape[0]}"
            )

        time_emb = self.time_embed(timesteps)
        h = self.conv_in(x)
        skips = [h]

        for level, blocks in enumerate(self.encoder):
            for block, attn in blocks:
                h = block(h, time_emb)
                h = attn(h)
                skips.append(h)
            h = self.downs[level](h)
            if not isinstance(self.downs[level], nn.Identity):
                skips.append(h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        for level, blocks in enumerate(self.decoder):
            for block, attn in blocks:
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)
                h = attn(h)
            h = self.ups[level](h)

        return self.conv_out(h)
