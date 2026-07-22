import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Maps input images to Gaussian parameters (mu, logvar) of q(z|x)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        feature_dim = hidden_dim // 2

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        features = self.backbone(x)
        mu = self.mu(features)
        logvar = self.logvar(features)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
