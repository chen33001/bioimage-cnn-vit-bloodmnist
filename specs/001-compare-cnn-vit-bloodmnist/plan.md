# 4-Week Project Plan

Project: Comparative Analysis of CNN and Vision Transformer on BloodMNIST  
Duration: 4 weeks (Nov → Dec 2025)

## Week 1 – Environment & Dataset Setup

### Goals

- Prepare full reproducible environment (local venv + Docker GPU).
- Load and explore the BloodMNIST dataset.
- Establish a unified preprocessing & dataloader pipeline.

### Tasks

- Create and activate venv

  - `python3.12 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`

- Verify GPU via `torch.cuda.is_available()`.
- Build Docker image: `docker build -t bloodmnist-cnnvit .`
- Download dataset using medmnist.
- Visualize random samples (matplotlib / seaborn).
- Implement normalization, augmentation, and PyTorch DataLoader.

### Deliverables

- `/src/data_loader.py`
- Sample visualization notebook `/notebooks/explore_data.ipynb`
- Screenshot of dataset summary statistics

## Week 2 – CNN Model Implementation & Training

### Goals

- Implement CNN baselines (ResNet-18 & EfficientNet-B0).
- Train & validate on BloodMNIST.
- Record metrics and learning curves.

### Tasks

- Implement training script `/src/train_cnn.py`.
- Use Adam optimizer, LR = 1e-3, batch = 128, 10 epochs.
- Log accuracy, loss, F1 via torchmetrics.
- Save best checkpoints under `/models/cnn/`.
- Plot training/validation curves.
- (Optional) Export Grad-CAM maps for interpretability.

### Deliverables

- `/models/cnn/*.pt` checkpoints
- `/results/cnn_metrics.csv`
- Figure 1: training vs validation accuracy

## Week 3 – Vision Transformer Implementation & Training

### Goals

- Fine-tune ViT-B/16 and Swin-Tiny using the same preprocessing & dataloaders.
- Compare convergence behavior and computational cost to CNNs.

### Tasks

- Implement `/src/train_vit.py` using timm models.
- Train for 10 epochs with same optimizer settings.
- Track GPU memory and training time.
- Visualize transformer attention maps.
- Save best model to `/models/vit/`.

### Deliverables

- `/models/vit/*.pt`
- `/results/vit_metrics.csv`
- Figure 2: attention-map visualization

## Week 4 – Evaluation · Visualization · Report

### Goals

- Quantitatively & qualitatively compare CNN vs ViT.
- Produce final figures, tables, and short written summary.

### Tasks

- Aggregate metrics into a single comparison table.
- Compute: Accuracy, Precision, Recall, F1, AUC.
- Plot bar charts & confusion matrices.
- Summarize results in `/docs/report.pdf`.
- (Optional) Build quick Streamlit app (`src/gui_app.py`) for inference demo.
- Package final code, models, and Docker image.

### Deliverables

- `/results/final_comparison.csv`
- `/figures/*.png` (Grad-CAM vs Attention)
- `/docs/report.pdf`
- (Optional) Streamlit GUI ready for demo

## Milestone Summary

| Week | Focus               | Key Output                             |
|------|---------------------|-----------------------------------------|
| 1    | Setup & Data        | Working env, dataloader                 |
| 2    | CNN Training        | CNN checkpoints + metrics               |
| 3    | ViT Training        | ViT checkpoints + attention maps        |
| 4    | Evaluation & Report | Comparison table + final report (+ GUI optional) |

