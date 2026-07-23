import torch

from generative_models.datasets import get_mnist_dataloaders
from generative_models.evaluation import (
    collect_test_latents,
    encode_mu,
    find_digit_image,
    interpolate_latents,
    reconstruct_test_grid,
    reduce_latents_pca,
)
from generative_models.models import VAE


def test_encode_mu_shape():
    model = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    images = torch.randn(4, 1, 28, 28)

    mu = encode_mu(model, images)

    assert mu.shape == (4, 64)


def test_reconstruct_test_grid_shape():
    model = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    _, test_loader = get_mnist_dataloaders(batch_size=16, data_dir="data/raw")

    originals, reconstructions = reconstruct_test_grid(model, test_loader, num_images=8)

    assert originals.shape == (8, 1, 28, 28)
    assert reconstructions.shape == (8, 1, 28, 28)


def test_interpolate_latents_shape():
    model = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    _, test_loader = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")

    image_a = find_digit_image(test_loader, digit=3)
    image_b = find_digit_image(test_loader, digit=8)
    decoded = interpolate_latents(model, image_a, image_b, steps=10)

    assert decoded.shape == (10, 1, 28, 28)


def test_collect_and_reduce_latents():
    model = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    _, test_loader = get_mnist_dataloaders(batch_size=256, data_dir="data/raw")

    latents, labels = collect_test_latents(model, test_loader, max_samples=512)
    coords = reduce_latents_pca(latents)

    assert latents.shape == (512, 64)
    assert labels.shape == (512,)
    assert coords.shape == (512, 2)
