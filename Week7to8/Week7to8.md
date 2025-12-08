# Week 7-8 Report: Comprehensive Model Evaluation

## Overview
This report consolidates the evaluation of **Model 01 (GCNN)** and **Model 03 (DeepOPF-FT)** across both **Seen** (Standard Test Set) and **Unseen** (Zero-Shot) datasets. It also documents the reproducibility steps for feature development and data generation.

## 1. Model Parameters

| Model | Architecture | Parameters | Description |
| :--- | :--- | :--- | :--- |
| **Model 01** | GCNN (Physics-Guided) | **115,306** | Graph Convolutional Network with physics-guided loss. |
| **Model 01 (Light v1)** | GCNN (Reduced) | **46,802** | Reduced capacity GCNN (512 neurons, 6 channels). |
| **Model 01 (Light v2)** | GCNN (Reduced) | **59,186** | Reduced capacity GCNN (512 neurons, 8 channels). |
| **Model 01 (Light v2 - 2Phase)** | GCNN (Reduced) | **59,186** | Same as Light v2, but trained with 2-phase method (Sup -> Phys). |
| **Model 03 (Tiny)** | DeepOPF-FT (Reduced) | **46,226** | MLP with 128 neurons (matched to GCNN Light v1). |
| **Model 03 (Small)** | DeepOPF-FT (Variant) | **83,718** | MLP with 180 neurons (matched capacity to GCNN). |
| **Model 03 (Large)** | DeepOPF-FT (Baseline) | **2,105,018** | MLP with 1000 neurons (flattened admittance embedding). |

*Note: Model 03 (Small) was trained to test if the performance gap was due to parameter count.*

## 2. Evaluation Results

### A. Seen Test Set (Standard Evaluation)
*   **Dataset**: `gcnn_opf_01/data/samples_test.npz` (2000 samples)
*   **Topologies**: 5 fixed topologies (Base + 4 N-1 contingencies) seen during training.

| Metric | Model 01 (GCNN) | Model 01 (Light v1) | Model 01 (Light v2) | Model 01 (Light v2 - 2Phase) | Model 03 (Tiny) | Model 03 (Small) | Model 03 (Large) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PG Accuracy (< 1 MW)** | 97.97% | 85.10% | 94.97% | 96.93% | **98.12%** | 99.55% | 99.15% |
| **PG RMSE (p.u.)** | 0.0071 | 0.0094 | 0.0076 | 0.0073 | **0.0071** | 0.0063 | 0.0070 |
| **PG R² Score** | 0.9835 | 0.9714 | 0.9813 | 0.9829 | **0.9837** | 0.9873 | 0.9840 |
| **VG Accuracy (< 0.001)** | 100.00% | 100.00% | 100.00% | 100.00% | **100.00%** | 100.00% | 100.00% |
| **VG RMSE (p.u.)** | 0.000043 | 0.000202 | 0.000066 | 0.000035 | **0.000045** | 0.000025 | 0.000034 |

**Analysis**: 
*   **Model 03 (Small)** remains the best performer on the seen dataset.
*   **Model 01 (Light v1)** showed a significant drop in accuracy (98% -> 85%) when channels were reduced to 6.
*   *   **Model 01 (Light v2)** recovered most of the performance (95% accuracy) by restoring channels to 8, confirming that matching the channel count to the feature iteration count ($k=8$) is critical for this architecture.
*   **Model 01 (Light v2 - 2Phase)** further improved accuracy to 97% on the seen dataset, showing that the physics-informed loss helps refine the solution within the known topology.

### B. Unseen Test Set (Zero-Shot Generalization)
*   **Dataset**: `gcnn_opf_01/data_unseen` (1200 samples)
*   **Topologies**: 3 new N-1 contingencies never seen during training.

| Metric | Model 01 (GCNN) | Model 01 (Light v1) | Model 01 (Light v2) | Model 01 (Light v2 - 2Phase) | Model 03 (Tiny) | Model 03 (Small) | Model 03 (Large) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PG Accuracy (< 1 MW)** | 44.14% | 40.44% | 44.19% | 48.42% | **56.25%** | 53.03% | 49.39% |
| **PG RMSE (p.u.)** | 0.0972 | 0.0742 | 0.0565 | 0.0838 | **0.0278** | 0.0292 | 0.0389 |
| **PG R² Score** | -2.00 | -0.7482 | -0.0157 | -1.2305 | **0.7543** | 0.7282 | 0.5201 |

**Analysis**: 
*   **Generalization**: The small MLP (Model 03 Small) generalizes significantly better than the others, achieving an R² of 0.73 on unseen topologies.
*   **GCNN Failure**: Both GCNN variants fail to generalize. Reducing the model size did **not** improve generalization. Light v2 improved R² to near zero (-0.01), effectively predicting the mean, but failed to capture the unseen topology physics.
*   **Physics Loss Impact**: The 2-Phase training (Light v2 - 2Phase) improved accuracy on the *seen* set but degraded R² on the *unseen* set (-1.23 vs -0.01). This suggests the physics loss causes the model to overfit to the specific physics correlations of the training topology, making it perform worse (larger errors) when the topology changes.


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

# Train Model 01 (Light v1) - Experimental
# Config: neurons_fc=512, channels_gc_out=6
python gcnn_opf_01/train.py --results_dir gcnn_opf_01/results/exp1_512n_6c --epochs 50 --batch_size 6

# Train Model 01 (Light v2) - Experimental
# Config: neurons_fc=512, channels_gc_out=8
python gcnn_opf_01/train.py --results_dir gcnn_opf_01/results/exp1_512n_8c --epochs 50 --batch_size 6

# Train Model 01 (Light v2 - 2Phase) - Experimental
# Config: neurons_fc=512, channels_gc_out=8, 2-phase training
python gcnn_opf_01/train.py --results_dir gcnn_opf_01/results/exp1_512n_8c_2phase --two_stage --phase1_epochs 25 --phase2_epochs 25 --batch_size 6

# Evaluate Model 01 (Light) on Seen Data
python gcnn_opf_01/evaluate.py --model_path gcnn_opf_01/results/exp1_512n_6c/best_model.pth --data_dir gcnn_opf_01/data --norm_stats_path gcnn_opf_01/data/norm_stats.npz

# Train Model 03 (Tiny) - Experimental
# Config: hidden_dim=128, n_hidden_layers=3 (46k params)
python dnn_opf_03/train_03.py --results_dir dnn_opf_03/results/exp2_128n --epochs 50 --batch_size 6

# Evaluate Model 03 (Tiny) on Seen Data
python dnn_opf_03/evaluate_03.py --model_path dnn_opf_03/results/exp2_128n/best_model.pth --test_file samples_test.npz --hidden_dim 128

# Evaluate Model 03 (Tiny) on Unseen Data
python dnn_opf_03/evaluate_03.py --model_path dnn_opf_03/results/exp2_128n/best_model.pth --data_dir gcnn_opf_01/data_unseen --test_file samples_test.npz --norm_stats_path gcnn_opf_01/data/norm_stats.npz --hidden_dim 128
```
