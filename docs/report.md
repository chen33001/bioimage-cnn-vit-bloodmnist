# Comparative Report: CNN vs ViT on BloodMNIST

**Author**: Wei-Fu Chen  
**Date**: 2025-11-06  
**Datasets**: BloodMNIST (MedMNIST v2, official splits)  
**Pipelines**: Shared preprocessing, seed = 42, 10 epochs, batch size 128, ImageNet normalization

---

## 1. Executive Summary

- The ResNet-18 CNN baseline delivered the strongest overall performance with **95.0% test accuracy** and **0.943 macro F1**, outperforming the ViT-B/16 baseline by ~4 percentage points.
- Vision Transformer retained competitive recall (0.914 macro) and offers complementary attention visualizations, but lagged in precision and achieved slightly higher loss due to slower convergence on low-resolution inputs.
- Interpretability overlays demonstrate that both models focus on diagnostically relevant regions; Grad-CAM heatmaps are crisper on cell membranes, while ViT attention spreads across broader tissue context.
- For BloodMNIST-like tasks on constrained hardware, the CNN is the practical default. ViT fine-tuning becomes attractive when additional compute and data augmentation are available or when attention-based explanations are required.

---

## 2. Methodology

1. **Data Handling**
   - Downloaded BloodMNIST via `medmnist`, using the provided train/validation/test splits.
   - Upsampled images to 224×224 with ImageNet normalization for parity between CNN and ViT architectures.
   - Applied deterministic augmentations (horizontal flip) and fixed seeds across PyTorch, NumPy, and CUDA.

2. **Models & Training**
 - **CNN**: torchvision ResNet-18 with ImageNet weights, final layer replaced with 8-class head.
 - **ViT**: timm `vit_base_patch16_224`, pretrained weights, fine-tuned end-to-end.
   - Optimizer: Adam (lr=1e-3), 10 epochs, batch size 128, best checkpoint selected via validation macro F1.
   - Dataloaders default to eight workers with `pin_memory`/persistent workers; automatic mixed precision (AMP) is enabled on CUDA by default.
   - Scripts: `src/train_cnn.py`, `src/train_vit.py`.

3. **Evaluation & Artefacts**
   - Metrics captured via `src/eval_utils.py` (accuracy, precision/recall/F1 macro & micro, ROC-AUC macro & micro, loss).
   - Predictions saved as CSV for reproducibility; final aggregation handled by `src/aggregate_results.py`.
   - Interpretability: Grad-CAM (`src/interpretability.py --model-type cnn`) and attention rollout (`--model-type vit`).

---

## 3. Quantitative Results

| Model | Split | Accuracy | Macro F1 | Macro Precision | Macro Recall | Macro AUC | Loss |
|-------|-------|---------:|---------:|----------------:|-------------:|----------:|-----:|
| CNN (ResNet-18) | Validation | 0.959 | 0.954 | 0.963 | 0.947 | 0.998 | 0.118 |
| ViT-B/16 | Validation | 0.909 | 0.898 | 0.890 | 0.918 | 0.994 | 0.267 |
| **CNN (ResNet-18)** | **Test** | **0.949** | **0.943** | **0.952** | **0.936** | **0.997** | **0.158** |
| ViT-B/16 | Test | 0.906 | 0.896 | 0.887 | 0.914 | 0.993 | 0.278 |

**Observations**
- CNN advantage is consistent across splits, indicating stable generalization.
- ViT recall slightly exceeds precision, suggesting it captures minority classes better but introduces more false positives.
- Both models achieve very high ROC-AUC (>0.99), showing strong ranking ability even when calibration differs.

---

## 4. Interpretability Insights

- **Grad-CAM (CNN)**: Concentrated heatmaps around leukocyte boundaries and cytoplasm; misclassified examples often show diffuse activations or focus on background artifacts.
- **Attention Rollout (ViT)**: Broader receptive fields covering whole cells and surrounding plasma; useful for understanding context-aware decisions.
- Artefacts located in `figures/gradcam/` and `figures/attention/` with filenames encoding split, sample index, label, and prediction for traceability.

---

## 5. Efficiency & Operational Notes

- **Runtime & Logging**:
  - The latest full run on an RTX 3070 Ti paired with a Ryzen 5800 CPU completed in roughly 9 h 47 min (`logs/run_all_20251107_082243.log`).
  - `scripts/run_all.ps1` prints start/end timestamps and total duration for every run, and writes the same details to `logs/run_all_*.log`.
  - Stage timings from that run: CNN training approx.1 h 18 min, ViT training approx.8 h 28 min, aggregation approx.3 s, interpretability overlays approx.1 min 15 s (all logged automatically).
  - Reduced-compute mode (`-ReducedCompute`) keeps runtimes under ~30 minutes on CPU-only machines by lowering image size and epochs.
- **Resource Footprint**:
  - CNN training curves show faster convergence and lower variance; ViT consumes more VRAM (~2x) but benefits from automatic mixed precision on CUDA.
  - DataLoaders default to eight workers with `pin_memory` and persistent worker pools to keep the GPU utilized even when preprocessing remains CPU-bound.
- **Reproducibility**:
  - Seeds fixed at multiple levels, dataset version recorded, metrics exported as JSON/CSV, and predictions stored for downstream audits.
  - Notebook `notebooks/results.ipynb` and `docs/artifacts.md` provide ready-to-run review materials.

---

## 6. Recommendations & Next Steps

1. **Adopt CNN baseline** for production or embedded settings; it hits performance targets with lower cost.
2. **Investigate ViT enhancements** (longer training schedule, stronger augmentations, reduced learning rate) if attention visualizations are a priority.
3. **Extend evaluation** with class-wise metrics charts and calibration plots using saved prediction CSVs.
4. **Finalize publication** by converting this Markdown report into PDF (`pandoc docs/report.md -o docs/report.pdf`) once wording is approved.

---

## 7. Artefact Checklist

- Training scripts: `src/train_cnn.py`, `src/train_vit.py`
- Aggregation CLI: `src/aggregate_results.py`
- Interpretability CLI: `src/interpretability.py`
- Automation: `scripts/run_all.ps1`
- Metrics & predictions: `results/`
- Visuals: `figures/`
- Notebook: `notebooks/results.ipynb`
- Artifact index: `docs/artifacts.md`
- Report (this file): `docs/report.md`
