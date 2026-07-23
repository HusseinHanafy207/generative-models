# VAE Experiment Comparison: 50 vs 100 Epochs

| Metric | 50 Epochs | 100 Epochs | Change |
|--------|-----------|------------|--------|
| Train Loss | 12519.25 | 12296.67 | -222.59 (-1.8%) |
| Recon Loss | 10068.42 | 9816.44 | -251.98 (-2.5%) |
| KL Loss | 2450.83 | 2480.22 | +29.39 (+1.2%) |
| Val Loss | 12685.67 | 12637.46 | -48.21 (-0.4%) |

## Visual comparisons

- `epoch_50/random_samples.png` vs `epoch_100/random_samples.png`
- `epoch_50/reconstructions.png` vs `epoch_100/reconstructions.png`

Samples use the same random seed for a fair comparison.