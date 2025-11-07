# Task Breakdown

Feature: Comparative Analysis of CNN vs ViT on BloodMNIST  
Branch: 001-compare-cnn-vit-bloodmnist

Assumptions applied:
- Primary reporting: include both validation and test; highlight test as primary.
- Time budget: full <= 120 min; reduced <= 30 min.
- Interactive demo: deferred (not in current scope).

## Legend
- ID format: TASK-###
- Estimates: S (<= 2h), M (<= 1d), L (<= 2d), XL (> 2d)

## Setup & Data

- [X] TASK-001 — Create reproducible environment docs (venv + Docker)
  - Description: Write concise setup instructions for local venv and optional GPU Docker image; include seed control notes.
  - Acceptance: README section includes commands; new user can set up in < 30 minutes.
  - Artifacts: README, `requirements.txt` (if missing), Dockerfile validation.
  - Estimate: M

- [X] TASK-002 — Data acquisition and integrity checks
  - Description: Script dataset download via MedMNIST API; verify file sizes/checksums; cache path configurable.
  - Acceptance: Command downloads or verifies data; logs dataset version and split counts.
  - Artifacts: `/src/data_loader.py`, `/data/` structure (ignored by VCS), log sample.
  - Estimate: M

- [X] TASK-003 — Unified preprocessing and DataLoader
  - Description: Implement normalization, augmentation, and deterministic seeds; expose train/val/test loaders.
  - Acceptance: Unit run iterates one batch per split; prints shapes and class distribution; seed yields consistent shuffles.
  - Artifacts: `/src/data_loader.py` (extended), config stub.
  - Estimate: L

- [X] TASK-004 — Data exploration notebook
  - Description: Visualize samples per class; basic stats; augmentation preview.
  - Acceptance: Notebook executes top-to-bottom without manual edits.
  - Artifacts: `/notebooks/explore_data.ipynb`, saved preview images.
  - Estimate: M

## CNN Baselines

- [X] TASK-010 CNN training script
  - Description: Implement `/src/train_cnn.py` (ResNet-18, EfficientNet-B0) sharing preprocessing; log metrics via torchmetrics-equivalent pattern.
  - Acceptance: Runs for 10 epochs with defaults; saves best checkpoints; resumes from checkpoint.
  - Artifacts: `/src/train_cnn.py`, `/models/cnn/*.pt`, run logs.
  - Estimate: L

- [X] TASK-011 CNN metrics export and curves
  - Description: Export per-epoch metrics CSV; plot accuracy/loss curves.
  - Acceptance: CSV at `/results/cnn_metrics.csv`; figures saved; confusion matrix for validation/test.
  - Artifacts: `/results/cnn_metrics.csv`, `/figures/cnn_*.png`.
  - Estimate: M

- [X] TASK-012 CNN Grad-CAM overlays
  - Description: Generate Grad-CAM overlays for >= 10 samples per class.
  - Acceptance: Images saved with clear overlays and labels.
  - Artifacts: `/figures/gradcam/*.png`.
  - Estimate: M

## ViT Baselines

- [X] TASK-020 ViT training script
  - Description: Implement `/src/train_vit.py` (ViT-B/16, Swin-Tiny) using same loaders and training loop structure.
  - Acceptance: Runs for 10 epochs; saves best checkpoints; logs GPU memory and wall time.
  - Artifacts: `/src/train_vit.py`, `/models/vit/*.pt`, logs.
  - Estimate: L

- [X] TASK-021 ViT attention visualizations
  - Description: Extract and render attention maps for selected samples.
  - Acceptance: Saved overlays for >= 10 samples per class; legible heatmaps.
  - Artifacts: `/figures/attention/*.png`.
  - Estimate: M

- [X] TASK-022 ViT metrics export and curves
  - Description: Export per-epoch metrics CSV; plot curves; confusion matrices.
  - Acceptance: CSV at `/results/vit_metrics.csv`; figures saved.
  - Artifacts: `/results/vit_metrics.csv`, `/figures/vit_*.png`.
  - Estimate: M

## Evaluation & Reporting

- [X] TASK-030 Comparative metrics aggregation
  - Description: Aggregate CNN and ViT metrics into one table; include Accuracy, Precision, Recall, F1 (macro/micro), AUC (macro/micro).
  - Acceptance: `/results/final_comparison.csv` contains both val and test; highlights test as primary.
  - Artifacts: `/results/final_comparison.csv`.
  - Estimate: M

- [X] TASK-031 Confusion matrices and bar charts
  - Description: Generate per-model confusion matrices and metric bar charts.
  - Acceptance: Figures saved and referenced in notebook.
  - Artifacts: `/figures/*.png`.
  - Estimate: M

- [X] TASK-032 Results notebook and narrative summary
  - Description: Create a concise notebook to load artifacts, render tables/figures, and draft a short narrative.
  - Acceptance: Notebook executes end-to-end; exports selected figures.
  - Artifacts: `/notebooks/results.ipynb`.
  - Estimate: L

- [X] TASK-033 Final report (non-technical)
  - Description: Summarize goals, methods, key results, and recommendations; non-technical tone.
  - Acceptance: `/docs/report.pdf` produced; includes figures and comparison table; conclusions highlight trade-offs.
  - Artifacts: `/docs/report.pdf` (and source if applicable).
  - Estimate: L

## Reproducibility & Operations

- [X] TASK-040 — Single-command workflow
  - Description: Create a wrapper script/CLI to run data -> train -> evaluate -> visualize -> aggregate.
  - Acceptance: One command produces models and artifacts in expected locations within time budget.
  - Artifacts: `/scripts/run_all.(sh|ps1)` or CLI entry.
  - Estimate: L

- [X] TASK-041 — Reduced-compute mode
  - Description: Add config to run a shortened training schedule that completes <= 30 minutes while preserving fair comparison logic.
  - Acceptance: Mode flag reduces epochs/augmentations; artifacts still produced; results labeled as "reduced".
  - Artifacts: Config file and doc note.
  - Estimate: M

- [X] TASK-042 — Documentation polish
  - Description: Consolidate docs: setup, how to run, artifact locations, troubleshooting.
  - Acceptance: New teammate can reproduce in one pass without code edits.
  - Artifacts: README sections, `/docs/` notes.
  - Estimate: M

## Dependencies
- [X] TASK-020 ViT training script
- TASK-011 depends on TASK-010; TASK-022 depends on TASK-020.
- TASK-012 depends on TASK-010; TASK-021 depends on TASK-020.
- TASK-030 depends on TASK-011 and TASK-022.
- TASK-031 depends on TASK-030.
- [X] TASK-032 Results notebook and narrative summary
- [X] TASK-032 Results notebook and narrative summary
- [X] TASK-040 depends on core training/eval tasks (TASK-010, TASK-020, TASK-030).
- [X] TASK-041 depends on TASK-040.

## Suggested Sprint Allocation
- Week 1: TASK-001..004
- Week 2: TASK-010..012 + start TASK-040
- Week 3: TASK-020..022 + finish TASK-040, start TASK-041
- Week 4: TASK-030..033 + TASK-041 + TASK-042

