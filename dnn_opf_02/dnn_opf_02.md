# Hasan DNN (M8) Baseline Project Summary

## 1. Implementation Overview

This project implements the "Hasan DNN" (Model M8) baseline for Optimal Power Flow (OPF), designed to serve as a performance benchmark for the Graph Convolutional Neural Network (GCNN) approach.

### Key Components Completed:
*   **Feature Engineering (`compute_topology_means.py`):** 
    *   Implemented the "Delta V" feature logic.
    *   Calculates mean voltage profiles for each topology to serve as a baseline for voltage deviation features.
*   **Dataset Pipeline (`dataset_02.py`):**
    *   Custom PyTorch Dataset that constructs the flattened input vector `[Pd, Qd, Δe, Δf]`.
    *   Handles data loading and normalization using pre-computed statistics.
*   **Model Architecture (`model_02.py`):**
    *   Implemented the M8 architecture: A 4-layer MLP (1000 neurons/layer, ReLU activations).
    *   Dual-head output: Head 1 for Generator targets (PG, VG), Head 2 for Bus Voltages (e, f).
*   **Training Pipeline (`train_02.py`):**
    *   Implemented a **Two-Stage Training** strategy:
        *   **Stage 1:** Supervised Learning ($\kappa=0$) to learn mapping from inputs to labels.
        *   **Stage 2:** Physics-Informed Fine-tuning ($\kappa=1$) incorporating physical power flow constraints.
*   **Evaluation (`evaluate_02.py`):**
    *   Comprehensive evaluation script calculating MSE, RMSE, MAE, MAPE, and R².

## 2. Results

The model was trained for 50 epochs total (25 epochs Stage 1 + 25 epochs Stage 2). The evaluation on the test set demonstrates high accuracy, establishing a strong baseline.

| Metric | Generator Power (PG) | Generator Voltage (VG) |
| :--- | :--- | :--- |
| **MAPE** | **1.03%** | **0.00%** |
| **R²** | **0.976** | **1.000** |
| **RMSE** | **0.0086** | **0.00004** |

## 3. Usage Guide

### Prerequisites
Ensure the `opf311` environment is active and training data is available in `Week3/samples/`.

### Step 1: Setup (One-time)
Compute the topology-specific mean voltage profiles required for the Delta V feature.
```bash
python dnn_opf_02/compute_topology_means.py
```

### Step 2: Training
Run the full two-stage training pipeline.
```bash
python dnn_opf_02/train_02.py --epochs_stage1 25 --epochs_stage2 25 --batch_size 10
```

### Step 3: Evaluation
Evaluate the trained model using the best checkpoint from Stage 2.
```bash
python dnn_opf_02/evaluate_02.py --model_path dnn_opf_02/results/best_model_stage2.pth
```

## 4. Directory Structure
*   `dnn_opf_02/`
    *   `data/`: Stores generated topology means and intermediate data.
    *   `results/`: Stores model checkpoints and training logs.
    *   `src/`: Source code for model, dataset, and training.
