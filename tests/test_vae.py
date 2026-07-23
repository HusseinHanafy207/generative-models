import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.losses import VAELoss
from generative_models.models import Decoder, Encoder, VAE


def test_encoder_output_shapes():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    encoder = Encoder(input_dim=784, hidden_dim=512, latent_dim=64)
    mu, logvar = encoder(images)

    assert mu.shape == (128, 64)
    assert logvar.shape == (128, 64)


def test_encoder_reparameterize_shape():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    encoder = Encoder(input_dim=784, hidden_dim=512, latent_dim=64)
    mu, logvar = encoder(images)
    z = encoder.reparameterize(mu, logvar)

    assert z.shape == (128, 64)


def test_encoder_reparameterize_is_differentiable():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    encoder = Encoder(input_dim=784, hidden_dim=512, latent_dim=64)
    mu, logvar = encoder(images)
    z = encoder.reparameterize(mu, logvar)

    assert z.requires_grad
    z.sum().backward()
    assert encoder.mu.weight.grad is not None


def test_decoder_output_shape():
    decoder = Decoder(latent_dim=64, hidden_dim=512, output_dim=784)
    z = torch.randn(128, 64)

    output = decoder(z)

    assert output.shape == (128, 1, 28, 28)


def test_decoder_output_in_valid_range():
    decoder = Decoder(latent_dim=64, hidden_dim=512, output_dim=784)
    z = torch.randn(128, 64)

    output = decoder(z)

    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_vae_forward_pipeline():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    reconstruction, mu, logvar = vae(images)

    assert images.shape == (128, 1, 28, 28)
    assert reconstruction.shape == (128, 1, 28, 28)
    assert mu.shape == (128, 64)
    assert logvar.shape == (128, 64)


def test_vae_forward_is_differentiable():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    reconstruction, mu, logvar = vae(images)

    loss = reconstruction.sum() + mu.sum() + logvar.sum()
    loss.backward()

    assert vae.encoder.mu.weight.grad is not None
    assert vae.decoder.output[0].weight.grad is not None


def test_vae_loss_output_shapes():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    reconstruction, mu, logvar = vae(images)

    criterion = VAELoss()
    loss, recon_loss, kl_loss = criterion(reconstruction, images, mu, logvar)

    assert loss.shape == torch.Size([])
    assert recon_loss.shape == torch.Size([])
    assert kl_loss.shape == torch.Size([])


def test_vae_loss_is_finite_and_decomposable():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    reconstruction, mu, logvar = vae(images)

    criterion = VAELoss()
    loss, recon_loss, kl_loss = criterion(reconstruction, images, mu, logvar)

    assert torch.isfinite(loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert torch.allclose(loss, recon_loss + kl_loss)
    assert kl_loss > 0.0


def test_vae_loss_kl_is_comparable_to_reconstruction():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    reconstruction, mu, logvar = vae(images)

    criterion = VAELoss()
    _, recon_loss, kl_loss = criterion(reconstruction, images, mu, logvar)

    ratio = kl_loss.item() / recon_loss.item()
    assert ratio > 1e-4


def test_vae_loss_is_differentiable():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    reconstruction, mu, logvar = vae(images)

    criterion = VAELoss()
    loss, _, _ = criterion(reconstruction, images, mu, logvar)
    loss.backward()

    assert vae.encoder.mu.weight.grad is not None
    assert vae.decoder.output[0].weight.grad is not None
