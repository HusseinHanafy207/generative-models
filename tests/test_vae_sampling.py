import torch

from generative_models.evaluation import generate_random_samples
from generative_models.models import VAE


def test_vae_sample_output_shape():
    model = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    model.eval()

    samples = generate_random_samples(model, num_samples=16)

    assert samples.shape == (16, 1, 28, 28)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_vae_sample_is_stochastic():
    model = VAE(input_dim=784, hidden_dim=512, latent_dim=64, output_dim=784)
    model.eval()

    torch.manual_seed(0)
    samples_a = generate_random_samples(model, num_samples=4)
    torch.manual_seed(1)
    samples_b = generate_random_samples(model, num_samples=4)

    assert not torch.allclose(samples_a, samples_b)
