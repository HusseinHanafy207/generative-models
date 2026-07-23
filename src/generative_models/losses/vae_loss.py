import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """VAE loss: reconstruction (BCE) + KL divergence.

    Uses sum reduction for both terms so reconstruction and KL stay on
    comparable scales during training (Pattern A from the VAE literature).
    """

    def forward(
        self,
        reconstruction: torch.Tensor,
        images: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction_loss = F.binary_cross_entropy(
            reconstruction,
            images,
            reduction="sum",
        )

        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
        )

        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss
