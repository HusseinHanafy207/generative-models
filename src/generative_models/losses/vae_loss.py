import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """VAE loss: reconstruction (BCE) + KL divergence."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

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
            reduction=self.reduction,
        )

        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        )
        kl_loss = torch.mean(kl_per_sample)

        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss
