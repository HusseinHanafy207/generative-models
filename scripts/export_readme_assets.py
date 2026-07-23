"""Copy key VAE figures into docs/assets/vae for the README."""

import shutil
from pathlib import Path

ASSET_MAP = {
    Path("outputs/vae/figures/training_curves.png"): Path("docs/assets/vae/training_curves.png"),
    Path("outputs/vae/figures/test_reconstructions.png"): Path("docs/assets/vae/test_reconstructions.png"),
    Path("outputs/vae/samples/random_samples.png"): Path("docs/assets/vae/random_samples.png"),
    Path("outputs/vae/figures/latent_interpolation_3_to_8.png"): Path(
        "docs/assets/vae/latent_interpolation_3_to_8.png"
    ),
    Path("outputs/vae/figures/latent_space_tsne.png"): Path("docs/assets/vae/latent_space_tsne.png"),
    Path("outputs/vae/figures/latent_space_pca.png"): Path("docs/assets/vae/latent_space_pca.png"),
    Path("outputs/vae/experiments/comparison.md"): Path("docs/assets/vae/epoch_comparison.md"),
}


def main() -> None:
    copied = 0
    missing = []

    for source, destination in ASSET_MAP.items():
        if not source.exists():
            missing.append(str(source))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
        print(f"Copied {source} -> {destination}")

    print(f"\nCopied {copied} asset(s) to docs/assets/vae/")
    if missing:
        print("Missing (run the experiment scripts first):")
        for path in missing:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
