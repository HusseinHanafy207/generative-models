# generative-models

PyTorch implementations of generative models from scratch.

## Project structure

```
generative-models/
│
├── configs/
│   ├── vae/
│   └── ddpm/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── samples/
│   └── figures/
│
├── src/
│   └── generative_models/
│       ├── __init__.py
│       ├── datasets/
│       ├── models/
│       ├── diffusion/
│       ├── losses/
│       ├── trainers/
│       ├── evaluation/
│       └── utils/
│
├── tests/
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
pip install -e ".[dev]"
```

## Verify installation

```bash
python -c "import generative_models; print(generative_models.__version__)"
pytest
```

## Notes

- `data/` and `outputs/` directories are tracked in git, but their contents are ignored locally (`data/raw/` for originals, `data/processed/` for preprocessed data).
- Add model implementations under `src/generative_models/models/` or `src/generative_models/diffusion/`.
- Place experiment configs in `configs/vae/` or `configs/ddpm/`.
