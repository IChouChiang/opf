# Week 7-8 Report: Comprehensive Model Evaluation

## Overview
This report consolidates the evaluation of **Model 01 (GCNN)** and **Model 03 (DeepOPF-FT)** across both **Seen** (Standard Test Set) and **Unseen** (Zero-Shot) datasets. It also documents the reproducibility steps for feature development and data generation.

## 1. Model Parameters

| Model | Architecture | Parameters | Description |
| :--- | :--- | :--- | :--- |
| **Model 01** | GCNN (Physics-Guided) | **115,306** | Graph Convolutional Network with physics-guided loss. |
| **Model 03 (Small)** | DeepOPF-FT (Variant) | **83,718** | MLP with 180 neurons (matched capacity to GCNN). |
| **Model 03 (Large)** | DeepOPF-FT (Baseline) | **2,105,018** | MLP with 1000 neurons (flattened admittance embedding). |

*Note: Model 03 (Small) was trained to test if the performance gap was due to parameter count.*

## 2. Evaluation Results

### A. Seen Test Set (Standard Evaluation)
*   **Dataset**: `gcnn_opf_01/data/samples_test.npz` (2000 samples)
*   **Topologies**: 5 fixed topologies (Base + 4 N-1 contingencies) seen during training.

| Metric | Model 01 (GCNN) | Model 03 (Small) | Model 03 (Large) |
| :--- | :--- | :--- | :--- |
| **PG Accuracy (< 1 MW)** | 97.97% | **99.55%** | 99.15% |
| **PG RMSE (p.u.)** | 0.0071 | **0.0063** | 0.0070 |
| **PG R² Score** | 0.9835 | **0.9873** | 0.9840 |
| **VG Accuracy (< 0.001)** | 100.00% | 100.00% | 100.00% |
| **VG RMSE (p.u.)** | 0.000043 | **0.000025** | 0.000034 |

**Analysis**: Surprisingly, the smaller MLP (Model 03 Small) outperforms both the GCNN and the larger MLP on the seen dataset. This suggests that for this specific small-scale problem (Case6ww), a compact MLP with admittance embedding is extremely efficient.

### B. Unseen Test Set (Zero-Shot Generalization)
*   **Dataset**: `gcnn_opf_01/data_unseen` (1200 samples)
*   **Topologies**: 3 new N-1 contingencies never seen during training.

| Metric | Model 01 (GCNN) | Model 03 (Small) | Model 03 (Large) |
| :--- | :--- | :--- | :--- |
| **PG Accuracy (< 1 MW)** | 44.14% | **53.03%** | 49.39% |
| **PG RMSE (p.u.)** | 0.0972 | **0.0292** | 0.0389 |
| **PG R² Score** | -2.00 | **0.7282** | 0.5201 |

**Analysis**: 
*   **Generalization**: The small MLP (Model 03 Small) generalizes significantly better than the others, achieving an R² of 0.73 on unseen topologies.
*   **Overfitting**: The larger Model 03 likely overfitted to the training topologies (hence lower generalization R² of 0.52).
*   **GCNN Issues**: GCNN struggles most with generalization (-2.00 R²), possibly because the graph convolution operation is sensitive to the exact spectral properties of the graph, which change drastically with topology shifts in such a small network.

## 3. Feature Development & Reproducibility

The following commands were used to generate datasets and features for this project.

### Data Generation (Standard Set)
Generates 10k training + 2k test samples across 5 base topologies.
```bash
# Run from project root
python gcnn_opf_01/generate_dataset.py
```
*   **Output**: `gcnn_opf_01/data/samples_train.npz`, `samples_test.npz`, `norm_stats.npz`
*   **Key Logic**: `sample_generator_model_01.py` (RES sampling), `feature_construction_model_01.py` (k-hop features).

### Data Generation (Unseen Set)
Generates 1200 samples across 3 new N-1 topologies for zero-shot testing.
```bash
# Run from project root
python gcnn_opf_01/generate_unseen_dataset.py
```
*   **Output**: `gcnn_opf_01/data_unseen/samples_test.npz`

### Feature Construction
The core feature construction logic (Eqs. 16-25 in the paper) is encapsulated in:
*   **Script**: `gcnn_opf_01/feature_construction_model_01.py`
*   **Function**: `construct_features(ppc_int, pd, qd, k=8)`
*   **Usage**: Automatically called during dataset generation.

### Evaluation Commands
```bash
# Evaluate GCNN on Seen Data
python gcnn_opf_01/evaluate.py --model_path gcnn_opf_01/results/final_1000n_bs24/best_model.pth --data_dir gcnn_opf_01/data --norm_stats_path gcnn_opf_01/data/norm_stats.npz

# Evaluate DeepOPF-FT (Large) on Seen Data
python dnn_opf_03/evaluate_03.py --model_path dnn_opf_03/results/best_model.pth --data_dir gcnn_opf_01/data --norm_stats_path gcnn_opf_01/data/norm_stats.npz

# Evaluate DeepOPF-FT (Small) on Seen Data
python dnn_opf_03/evaluate_03.py --model_path dnn_opf_03/results/exp1_180n/best_model.pth --data_dir gcnn_opf_01/data --norm_stats_path gcnn_opf_01/data/norm_stats.npz --hidden_dim 180
```
