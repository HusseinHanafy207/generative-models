"""Copy key DDPM figures into docs/assets/ddpm for the README."""

from __future__ import annotations

import shutil
from pathlib import Path

ASSET_MAP = {
    Path("outputs/ddpm/figures/training_curves.png"): Path(
        "docs/assets/ddpm/training_curves.png"
    ),
    Path("outputs/ddpm/samples/samples_epoch_001.png"): Path(
        "docs/assets/ddpm/samples_epoch_001.png"
    ),
    Path("outputs/ddpm/samples/samples_epoch_010.png"): Path(
        "docs/assets/ddpm/samples_epoch_010.png"
    ),
    Path("outputs/ddpm/samples/samples_epoch_020.png"): Path(
        "docs/assets/ddpm/samples_epoch_020.png"
    ),
    Path("outputs/ddpm/samples/samples_epoch_050.png"): Path(
        "docs/assets/ddpm/samples_epoch_050.png"
    ),
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

    print(f"\nCopied {copied} asset(s) to docs/assets/ddpm/")
    if missing:
        print("Missing (download from Drive / Colab outputs first):")
        for path in missing:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
