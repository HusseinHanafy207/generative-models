"""Verify sinusoidal timestep embeddings (shapes + Transformer identities)."""

import math

import torch

from generative_models.ddpm import TimestepEmbedding, sinusoidal_time_embedding


def main() -> None:
    embedding_dim = 64
    timesteps = torch.tensor([0, 1, 50, 100, 500, 999])

    print("=== Sinusoidal PE ===")
    emb = sinusoidal_time_embedding(timesteps, embedding_dim=embedding_dim)
    print(f"t shape:          {tuple(timesteps.shape)}")
    print(f"embedding shape:  {tuple(emb.shape)}")
    print(f"embedding dtype:  {emb.dtype}")

    # Spot-check first frequency pair against the closed-form definition
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
    )
    t = timesteps.float().unsqueeze(1)
    expected_sin = torch.sin(t * freqs)
    expected_cos = torch.cos(t * freqs)
    expected = torch.cat([expected_sin, expected_cos], dim=-1)
    max_err = (emb - expected).abs().max().item()
    print(f"max |PE - (sin, cos)| = {max_err:.2e}")

    # Distinct timesteps should produce distinct embeddings
    pairwise = torch.cdist(emb, emb)
    off_diag = pairwise[~torch.eye(len(timesteps), dtype=torch.bool)]
    print(f"min pairwise L2 (off-diag) = {off_diag.min().item():.4f}")

    print("\n=== TimestepEmbedding MLP ===")
    module = TimestepEmbedding(embedding_dim=embedding_dim)
    out = module(timesteps)
    print(f"module out shape: {tuple(out.shape)}")
    print(f"trainable params: {sum(p.numel() for p in module.parameters())}")

    # Gradients should flow through the MLP
    loss = out.sum()
    loss.backward()
    has_grad = module.mlp[0].weight.grad is not None
    print(f"grad on first Linear: {has_grad}")

    ok = max_err < 1e-5 and has_grad and off_diag.min().item() > 0
    print("\nTime embedding looks correct." if ok else "\nTime embedding mismatch!")


if __name__ == "__main__":
    main()
