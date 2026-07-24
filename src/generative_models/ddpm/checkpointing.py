"""Load a trained DDPM and run reverse-process sampling."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from generative_models.ddpm.diffusion import DDPM
from generative_models.utils.device import get_device


def build_ddpm_from_config(config: dict) -> DDPM:
    return DDPM(
        num_timesteps=config["num_timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        in_channels=1,
        out_channels=1,
        base_channels=config["base_channels"],
        channel_mult=tuple(config["channel_mult"]),
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=tuple(config["attention_resolutions"]),
        dropout=config.get("dropout", 0.1),
    )


def load_ddpm_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    config_path: str | Path | None = None,
) -> tuple[DDPM, dict]:
    """Load a trained DDPM from a checkpoint file."""
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if config is None and config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    if config is None:
        raise ValueError("Config not found in checkpoint and no config_path provided.")

    model = build_ddpm_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config
