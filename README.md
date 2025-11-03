# Comparative Analysis of CNN and Vision Transformer on BloodMNIST

Bioimage Analysis Final Project - University of Potsdam, 2025
Student: **Wei-Fu Chen**

---

## Overview
This repository investigates how Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) perform on BloodMNIST, an eight-class blood-cell microscopy dataset from MedMNIST v2. The project contrasts both architectures across three axes:
- Predictive performance (accuracy, macro F1-score, area under the ROC curve)
- Computational efficiency (training time, parameter count, hardware footprint)
- Interpretability (CNN Grad-CAM explanations versus ViT attention maps)

---

## Project Goals
- Build comparable CNN and ViT baselines tailored to low-resolution biomedical imagery.
- Quantify strengths and weaknesses of each model family on BloodMNIST.
- Provide reproducible training pipelines, experiment artefacts, and visual explanations that can inform biomedical practitioners.

---

## Repository Structure
```
bioimage-cnn-vit-bloodmnist/
|- src/        (training loops, model definitions, data utilities)
|- notebooks/  (exploratory analyses and prototypes)
|- results/    (metrics, Grad-CAM heatmaps, attention maps)
|- docs/       (project report and extended documentation)
|- .github/    (automation prompts and CI configuration)
|- .specify/   (internal tooling and project constitution)
|- Dockerfile  (container definition)
`- README.md   (project overview and guidance)
```

- `src/` - Training loops, model definitions, data modules, and shared utilities.
- `notebooks/` - Exploratory analyses and experiment prototypes.
- `results/` - Saved metrics (`metrics.csv`), Grad-CAM heatmaps, attention maps, and other generated artefacts.
- `docs/` - Project report and extended documentation.
- `.github/` - Repository automation prompts and CI definitions.
- `.specify/` - Internal automation scripts and project constitution (ignored in releases).
- `Dockerfile`, `.dockerignore` - Containerised development and runtime environment.
- `index.md` - Lightweight landing page for publishing the project summary online.

---

## Environment Setup

### Local Python Environment
```bash
git clone https://github.com/chen33001/bioimage-cnn-vit-bloodmnist.git
cd bioimage-cnn-vit-bloodmnist
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Docker Workflow
```bash
docker build -t bioimage-cnn-vit .
docker run -it --rm -p 8888:8888 -v ${PWD}:/app bioimage-cnn-vit
```
After the container starts, open the JupyterLab URL printed in the console. For GPU acceleration, append `--gpus all` when running the container on systems with NVIDIA Container Toolkit.

---

## Dataset
BloodMNIST comprises 17,092 colour blood-cell microscopy images across eight diagnostic classes at 28x28 resolution. The dataset is distributed as part of MedMNIST v2. Follow the MedMNIST license and citation requirements; this project references Yang et al. (2023) for dataset details.

---

## Experiments and Results
- Models share identical training, validation, and test splits with matching augmentation strategies to maintain fairness.
- Preliminary results indicate CNN accuracy around 91 percent and ViT accuracy around 96 percent on the held-out test split.
- Interpretability artefacts (Grad-CAM for CNNs and attention rollout for ViTs) are stored in `results/` and referenced in `index.md`.
- Full experiment logs, hyperparameters, and narrative discussion are tracked in `docs/project_report.md` (in progress).

---

## References
- Yang J. et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. *Scientific Data*, 10(41).
- Dosovitskiy A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
- Krizhevsky A., Sutskever I., and Hinton G. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*.
- Wiley Online Library (2025). Explainable AI for Blood Image Classification with BloodMNIST.
- Nature Scientific Reports (2024). Implementing Vision Transformers for 2D Biomedical Images.
