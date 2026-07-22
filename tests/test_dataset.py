from generative_models.datasets import get_mnist_dataloaders


def test_mnist_train_loader_shapes():
    train_loader, _ = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")

    images, labels = next(iter(train_loader))

    assert images.shape == (128, 1, 28, 28)
    assert labels.shape == (128,)


def test_mnist_test_loader_shapes():
    _, test_loader = get_mnist_dataloaders(batch_size=128, data_dir="data/raw")

    images, labels = next(iter(test_loader))

    assert images.shape == (128, 1, 28, 28)
    assert labels.shape == (128,)
