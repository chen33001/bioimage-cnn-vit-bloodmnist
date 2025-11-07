# Project Report - BloodMNIST CNN vs ViT

**Course**: Bioimage Analysis Final Project (University of Potsdam, 2025)  
**Author**: Wei-Fu Chen  
**Repository**: `bioimage-cnn-vit-bloodmnist`  
**Last updated**: 2025-11-06

---

## 1. Introduction

This report documents the end-to-end comparison between a convolutional neural network (CNN) baseline (ResNet-18) and a Vision Transformer baseline (ViT-B/16) on the BloodMNIST dataset from MedMNIST v2. The project objectives were:

1. Produce an apples-to-apples benchmark using shared preprocessing, seeds, and evaluation criteria.
2. Generate quantitative results plus interpretability artefacts (Grad-CAM vs attention rollout) that explain each model's focus.
3. Package the workflow into a reproducible script with a reduced-compute mode and document all artefacts.

All code referenced below lives in the `src/` directory and can be executed via `scripts/run_all.ps1`.

---

## 2. Dataset and Preprocessing

- **Dataset**: BloodMNIST (17 092 RGB images, 8 diagnostic classes, 28x28 resolution). Official train/validation/test splits from MedMNIST v2 were used without modification.
- **Class balance**: Slightly skewed toward neutrophils and lymphocytes; macro metrics are therefore emphasized alongside micro metrics.
- **Transforms**:
  - Resize every sample to 224x224 so both models can re-use ImageNet pretraining statistics.
  - Normalize with ImageNet mean and standard deviation.
  - Apply random horizontal flips on the training set only; validation and test remain deterministic.
- **Determinism**: `src/data_loader.py` exposes `set_seed`, which locks Python, NumPy, and PyTorch (CPU and CUDA) seeds and disables cuDNN benchmarking to reduce non-determinism.
- **Data access**: `get_dataloaders` downloads BloodMNIST automatically (if needed) into `data/` and returns train/val/test loaders plus the class count.

Edge cases (download failures, missing dependencies, or CPU-only environments) are surfaced early because the loader validates package availability and creates data directories on demand.

---

## 3. Models and Training Setup

### 3.1 Architectures

- **CNN Baseline**: `torchvision.models.resnet18` loaded with ImageNet weights. The final fully connected layer is replaced by an 8-way classifier.
- **ViT Baseline**: `timm.create_model('vit_base_patch16_224')` with pretrained weights and an 8-way classification head. This matches the popular ViT-B/16 configuration (16x16 patches, 12 transformer blocks).

### 3.2 Optimization Regimen

| Hyperparameter | Value |
|----------------|-------|
| Optimizer      | Adam (learning rate 1e-3, default betas) |
| Batch size     | 128 |
| Epochs         | 10 (full mode) / 3 (reduced-compute mode) |
| Loss function  | CrossEntropyLoss |
| Image size     | 224 (full) / 128 (reduced-compute) |
| Mixed precision| Enabled automatically on CUDA via torch.cuda.amp |

### 3.3 Training Workflow

1. Set seeds and instantiate loaders via `get_dataloaders` (shared across both scripts). Loaders now default to eight workers with `pin_memory=True` and persistent worker pools so the RTX 3070 Ti spends less time idle while the Ryzen 5800 CPU prepares batches.
2. Instantiate the selected architecture and move it to the available device (`cuda` if present, else `cpu`). When CUDA is available the training loop automatically wraps forward/backward passes in `torch.cuda.amp.autocast` with a `GradScaler`.
3. For each epoch, log training loss/accuracy and evaluate on the validation split using `src.eval_utils.evaluate_model`.
4. Save the best checkpoint determined by validation macro F1 (`models/{cnn|vit}/*_best.pt`).
5. Reload the best checkpoint, evaluate on the test split, and persist metrics/predictions to `results/`.
6. Generate confusion matrices, metric bar plots, and training curves via helper functions in `src/eval_utils.py`.

Both training scripts (`src/train_cnn.py`, `src/train_vit.py`) share the same logging and persistence pipeline, ensuring that the aggregated metrics downstream are comparable.

---

## 4. Quantitative Results

The `src/aggregate_results.py` CLI consolidates the JSON/CSV metrics emitted by each training run. The table below summarizes the primary metrics (rounded to three decimals). All other metrics and per-epoch logs remain available in `results/`.

### 4.1 Validation Split

- **CNN (ResNet-18)**  
  - Accuracy: 0.959  
  - Macro Precision / Recall / F1: 0.963 / 0.947 / 0.954  
  - Macro ROC-AUC: 0.998  
  - Loss: 0.118
- **ViT-B/16**  
  - Accuracy: 0.909  
  - Macro Precision / Recall / F1: 0.890 / 0.918 / 0.898  
  - Macro ROC-AUC: 0.994  
  - Loss: 0.267

### 4.2 Test Split

- **CNN (ResNet-18)**  
  - Accuracy: 0.949  
  - Macro Precision / Recall / F1: 0.952 / 0.936 / 0.943  
  - Macro ROC-AUC: 0.997  
  - Loss: 0.158
- **ViT-B/16**  
  - Accuracy: 0.906  
  - Macro Precision / Recall / F1: 0.887 / 0.914 / 0.896  
  - Macro ROC-AUC: 0.993  
  - Loss: 0.278

### 4.3 Interpretation

1. **Performance gap**: ResNet-18 leads by roughly four percentage points on both validation and test accuracy, with similar margins in macro F1. This is expected because BloodMNIST images are low resolution, favoring inductive biases of CNNs.
2. **Recall vs precision**: ViT exhibits slightly higher macro recall than precision, indicating it detects additional positives at the expense of more false alarms. This effect is consistent across splits.
3. **AUC saturation**: Both models exceed 0.99 macro ROC-AUC, suggesting ranking quality is excellent even where thresholded accuracy differs. Calibration analysis would be a reasonable next step using the stored probability vectors.
4. **Loss values**: Higher ViT loss reflects slower convergence under the shared 10-epoch budget. Extending the schedule or reducing the learning rate could narrow the gap but would violate the strict comparison protocol used here.

---

## 5. Interpretability Analysis

1. **Grad-CAM (CNN)**  
   - Produced via `src.interpretability --model-type cnn ...`.  
   - Heatmaps highlight leukocyte membranes and cytoplasmic structures. For misclassified samples, activations drift toward background artifacts, signaling the need for additional augmentations or denoising if further improvement is required.  
   - Artefacts stored under `figures/gradcam/`, with filenames encoding split, sample index, ground-truth label, and prediction.

2. **Attention Rollout (ViT)**  
   - Implemented by hooking Vision Transformer QKV projections to rebuild attention matrices (`src/interpretability.py`).  
   - Attention maps cover entire cells and neighboring plasma, illustrating the model's ability to capture contextual clues (e.g., nearby cell morphology).  
   - Outputs saved under `figures/attention/` using the same naming convention.

3. **Notebook Integration**  
   - `notebooks/results.ipynb` displays a sample of Grad-CAM heatmaps and attention overlays alongside metrics and confusion matrices for qualitative review.

---

## 6. Workflow, Artefacts, and Reproducibility

- **Single-command execution**: `pwsh scripts/run_all.ps1` downloads data, trains both models, aggregates metrics, and generates interpretability artefacts. Reduced-compute mode (`-ReducedCompute`) lowers the image size to 128 pixels and caps epochs at three, keeping runtimes under 30 minutes on CPU-only hardware. Each run now prints start/end timestamps, per-stage durations, and writes a log file to `logs/run_all_YYYYMMDD_HHMMSS.log` capturing the full timeline (e.g., `logs/run_all_20251107_082243.log`).
- **Artefact index**: `docs/artifacts.md` lists all outputs: checkpoints (`models/`), metrics (`results/`), plots (`figures/`), interpretability outputs, and the interactive notebook.
- **Confusion matrices**: `src.plot_confusions` can regenerate matrices from prediction CSVs without rerunning training:
  ```powershell
  .\.venv\Scripts\python.exe -m src.plot_confusions --model cnn --split test --normalize
  .\.venv\Scripts\python.exe -m src.plot_confusions --model vit --split test --normalize
  ```
- **Reporting assets**: 
  - `docs/report.md` (concise stakeholder summary).  
  - `docs/project_report.md` (this detailed report).  
  - Both files can be exported to PDF via Pandoc once LaTeX engines are available.
- **Notebook robustness**: The results notebook auto-detects the repository root so it finds artefacts even if Jupyter is launched from another directory.

---

## 7. Discussion and Recommendations

1. **Model selection**: For BloodMNIST-scale imagery, ResNet-18 is the recommended baseline--faster convergence, higher accuracy, and lower compute. Use ViT-B/16 when interpretability via attention maps is critical or when additional training time/augmentation budget is available.
2. **Potential ViT improvements**:
   - Longer training schedules (>=20 epochs) with cosine decay.
   - Stronger augmentation (RandAugment, Mixup, CutMix) to mitigate overfitting.
   - Smaller-patch or distilled ViTs better suited to low-resolution microscopy.
3. **Additional analyses**:
   - Use the stored prediction CSVs to derive per-class PR curves, calibration plots, or cost-sensitive metrics demanded by clinical stakeholders.
   - Investigate explainability overlaps by overlaying Grad-CAM and attention maps on the same samples.
4. **Operational polish**:
   - Automate PDF generation of the report and include metrics summary snapshots for presentations.
   - Integrate optional mixed-precision training (`torch.cuda.amp`) for ViT to reduce wall-clock time.

---

## 8. Reproduction Guide

```powershell
# 1. Environment setup (Windows PowerShell example)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Full pipeline (GPU recommended)
pwsh scripts/run_all.ps1

# 3. Reduced-compute pipeline (CPU-friendly)
pwsh scripts/run_all.ps1 -ReducedCompute

# 4. Inspect outputs (timestamps and logs written to logs/run_all_*.log)
jupyter lab notebooks/results.ipynb
code docs/artifacts.md docs/report.md docs/project_report.md
```

- Latest full run on an RTX 3070 Ti + Ryzen 5800 completed in ~9 h 47 min (`logs/run_all_20251107_082243.log`).  Stage breakdown logged for that run: CNN training approx.1 h 18 min, ViT training approx.8 h 28 min, aggregation approx.3 s, interpretability overlays approx.1 min 15 s.  
- Reduced mode completes in under 30 minutes on CPU-only laptops.  
- Random seeds keep the test metrics within approximately 1 percentage point between reruns, satisfying the reproducibility tolerance in the specification.

---

## 9. References

1. Yang J. et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification," *Scientific Data*, 2023.  
2. Dosovitskiy A. et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *ICLR*, 2021.  
3. Krizhevsky A., Sutskever I., Hinton G., "ImageNet Classification with Deep Convolutional Neural Networks," *NeurIPS*, 2012.

---

## Appendix

- **Metrics**: `results/cnn_metrics.csv`, `results/vit_metrics.csv`
- **Predictions**: `results/*_val_predictions.csv`, `results/*_test_predictions.csv`
- **Figures**: `figures/` (training curves, bar charts, confusion matrices, Grad-CAM, attention maps)
- **Scripts**: `src/` (data loader, training loops, aggregation utilities, interpretability CLI, confusion plotter)
