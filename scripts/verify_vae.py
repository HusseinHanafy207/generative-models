"""Verify the full VAE pipeline on a real MNIST batch."""

from generative_models.datasets import get_mnist_dataloaders
from generative_models.models import VAE


def main() -> None:
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    reconstruction, mu, logvar = vae(images)

    print(images.shape)
    print(reconstruction.shape)
    print(mu.shape)
    print(logvar.shape)


if __name__ == "__main__":
    main()
