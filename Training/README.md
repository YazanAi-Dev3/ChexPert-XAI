# Phase 1: Model Training

This directory contains the code and methodology for the first phase of the project: training the baseline classification models. 

## 🎯 Objective
The primary goal of this phase is to train two identical deep learning models (**DenseNet-121**) on the CheXpert dataset under the exact same conditions, with the only variable being their **pre-trained initialization weights**. This provides the foundation for our comparative interpretability (XAI) analysis.

## 🧠 Models
1. **DenseImageNet**: Initialized with standard `ImageNet` weights.
2. **DenseRadImageNet**: Initialized with `RadImageNet` weights (a large-scale dataset of radiological images).

## ⚙️ Methodology & Training Conditions
Both models were trained using an identical pipeline to ensure a fair comparison:
- **Architecture**: DenseNet-121 (adapted for multi-label classification).
- **Dataset**: CheXpert (resized to 224x224, with CLAHE enhancement).
- **Two-Phase Training Strategy**:
  1. **Specialization Phase**: Uses Asymmetric Loss (ASL) to focus on hard positive cases and handle class imbalance.
  2. **Cooling Phase**: Uses BCE with per-class positive weights to fine-tune and stabilize the learning process.
- **Hardware/Config**: Trained using AMP (Automatic Mixed Precision) on CUDA.

## 📂 Directory Structure
- `DenseImageNet/`: Contains the training pipeline notebook (`training_pipeline.ipynb`) used for the ImageNet-initialized model.
- `DenseRadImageNet/`: Contains the training pipeline notebook (`training_pipeline-RAD.ipynb`) used for the RadImageNet-initialized model.

> **Note:** The actual dataset (`CheXpert-v1.0-small`) and the trained model weights (`.pth` files) are excluded from this repository due to size constraints. They are referenced in the Master `README.md` for download and reproducibility.
