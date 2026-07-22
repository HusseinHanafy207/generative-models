from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    batch_size: int,
    data_dir: str,
) -> tuple[DataLoader, DataLoader]:
    """Return MNIST train and test DataLoaders."""
    transform = transforms.ToTensor()
    root = Path(data_dir)

    train_dataset = datasets.MNIST(
        root=str(root),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(root),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
