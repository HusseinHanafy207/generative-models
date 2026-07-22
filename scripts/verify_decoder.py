"""Verify decoder output shape with random latent vectors."""

import torch

from generative_models.models import Decoder


def main() -> None:
    batch_size = 128
    latent_dim = 64

    decoder = Decoder(latent_dim=latent_dim, hidden_dim=512, output_dim=784)
    z = torch.randn(batch_size, latent_dim)

    output = decoder(z)
    print(output.shape)


if __name__ == "__main__":
    main()
