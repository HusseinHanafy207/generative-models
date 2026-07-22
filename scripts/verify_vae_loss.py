"""Verify VAE loss on a real forward pass."""

from generative_models.datasets import get_mnist_dataloaders
from generative_models.losses import VAELoss
from generative_models.models import VAE


def main() -> None:
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    reconstruction, mu, logvar = vae(images)

    criterion = VAELoss()
    loss, recon_loss, kl_loss = criterion(reconstruction, images, mu, logvar)

    print(loss.shape)
    print(recon_loss.shape)
    print(kl_loss.shape)
    print(f"loss = {loss.item():.4f}")
    print(f"recon_loss = {recon_loss.item():.4f}")
    print(f"kl_loss = {kl_loss.item():.4f}")
    print(f"recon + kl = {(recon_loss + kl_loss).item():.4f}")


if __name__ == "__main__":
    main()
