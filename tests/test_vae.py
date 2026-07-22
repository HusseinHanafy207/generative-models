from generative_models.datasets import get_mnist_dataloaders
from generative_models.models import Encoder


def test_encoder_output_shapes():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")
    images, _ = next(iter(train_loader))

    encoder = Encoder(input_dim=784, hidden_dim=512, latent_dim=64)
    mu, logvar = encoder(images)

    assert mu.shape == (128, 64)
    assert logvar.shape == (128, 64)
