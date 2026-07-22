"""Verify VAE encoder output shapes on a real MNIST batch."""

from generative_models.datasets import get_mnist_dataloaders
from generative_models.models import Encoder


def main() -> None:
    batch_size = 128
    train_loader, _ = get_mnist_dataloaders(batch_size=batch_size, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    encoder = Encoder(input_dim=784, hidden_dim=512, latent_dim=64)
    mu, logvar = encoder(images)

    print(mu.shape)
    print(logvar.shape)

    z = encoder.reparameterize(mu, logvar)
    print(z.shape)


if __name__ == "__main__":
    main()
