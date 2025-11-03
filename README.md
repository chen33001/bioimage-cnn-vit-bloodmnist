# 🧠 Comparative Analysis of CNN and Vision Transformer on BloodMNIST

> Bioimage Analysis Final Project – University of Potsdam, 2025  
> Student: **Wei-Fu Chen** 

---
bioimage-cnn-vit-bloodmnist/
│
├── README.md
├── index.md
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── notebooks/
│   └── bloodmnist_comparison.ipynb
├── src/
│   ├── cnn_model.py
│   ├── vit_model.py
│   └── train.py
├── results/
│   ├── metrics.csv
│   ├── gradcam_examples.png
│   └── attention_maps.png
└── docs/
    └── project_report.md


---

## 🎯 Overview
This project compares **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** for **blood-cell image classification** using the [BloodMNIST](https://medmnist.com) dataset from **MedMNIST v2**.  
It aims to analyze and visualize how both architectures differ in:
- performance (accuracy, AUC, F1-score)  
- computational efficiency  
- interpretability (Grad-CAM vs Attention Maps)

---

## 🔬 Background
In biomedical imaging, low-resolution datasets like BloodMNIST present a unique challenge.  
While CNNs extract **local morphological features** (edges, textures, nuclei), Vision Transformers use **global self-attention** to capture long-range dependencies.  
This project investigates which architecture is **more suitable for small-scale, structured biomedical data**.

---

## ⚙️ Environment Setup

### ▶️ Local Setup
```bash
git clone https://github.com/chen33001/bioimage-cnn-vit-bloodmnist.git
cd bioimage-cnn-vit-bloodmnist
python -m venv venv
venv\Scripts\activate        # or: source venv/bin/activate
pip install -r requirements.txt
```bash

🐳 Run with Docker

docker build -t bioimage-cnn-vit .
docker run -it --rm -p 8888:8888 -v ${PWD}:/app bioimage-cnn-vit

Then open the provided Jupyter Lab link in your browser.
Supports GPU (with --gpus all) if NVIDIA Docker toolkit is installed.

---
🧬 Dataset

BloodMNIST – 17,092 color blood-cell microscopy images (8 classes, 28×28 resolution).
MedMNIST v2 Dataset Paper

---
📚 References

Yang J. et al. (2023). MedMNIST v2 — Lightweight Benchmark for Biomedical Image Classification. Scientific Data, 10, 41.

Dosovitskiy A. et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. ICLR.

Krizhevsky A., Sutskever I., & Hinton G. (2012). ImageNet Classification with Deep CNNs. NeurIPS.

Wiley Online Library (2025). Explainable AI for Blood Image Classification With BloodMNIST.

Nature Scientific Reports (2024). Implementing Vision Transformers for 2D Biomedical Images.