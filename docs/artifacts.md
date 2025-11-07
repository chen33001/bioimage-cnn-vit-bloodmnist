# BloodMNIST CNN vs ViT – Artifact Index

This document lists the canonical outputs produced by `scripts/run_all.ps1` to help reviewers locate metrics, models, and visuals quickly.

## Directories

- `models/cnn/` – Best-performing CNN checkpoints (`*_best.pt`) plus optimizer metadata.
- `models/vit/` – Best-performing ViT/Swin checkpoints.
- `results/`
  - `cnn_metrics.csv`, `vit_metrics.csv` – Per-epoch training/validation logs.
  - `*_val_metrics.json`, `*_test_metrics.json` – Final split-level metrics for reproducibility.
  - `*_val_predictions.csv`, `*_test_predictions.csv` – Raw predictions + probabilities for downstream analysis.
  - `final_comparison.csv` – Aggregated CNN vs ViT table covering both validation and test splits.
  - `final_summary.md` – Narrative comparison and mode label (full vs reduced compute).
- `figures/`
  - Training curves, bar charts, and confusion matrices for each architecture.
  - `figures/gradcam/` – Grad-CAM overlays (≥10 samples/class).
  - `figures/attention/` – ViT attention rollout overlays (≥10 samples/class).
- Confusion matrices can be regenerated without rerunning training via `python -m src.plot_confusions --model {cnn|vit} --split {val|test}` (uses saved prediction CSVs).
- `notebooks/results.ipynb` – Lightweight notebook that loads the artefacts above for interactive review.

Reduced-compute runs mirror this structure under `models/reduced/`, `results/reduced/`, and `figures/reduced/`.
