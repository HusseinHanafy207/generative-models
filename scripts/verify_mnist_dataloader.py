"""Verify MNIST dataloaders and visualize one training batch."""

from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from generative_models.datasets import get_mnist_dataloaders


def main() -> None:
    batch_size = 128
    data_dir = "data/raw"

    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size,
        data_dir=data_dir,
    )

    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)

    grid = make_grid(images[:64], nrow=8)
    figure_path = Path("outputs/figures/mnist_batch.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("MNIST training batch (first 64 images)")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    print(f"Saved visualization to {figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
