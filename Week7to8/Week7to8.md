# Week 7-8 Report: Comprehensive Model Evaluation

## Overview
This report consolidates the evaluation of **Model 01 (GCNN)** and **Model 03 (DeepOPF-FT)** across both **Seen** (Standard Test Set) and **Unseen** (Zero-Shot) datasets. It also documents the reproducibility steps for feature development and data generation.

## 1. Model Parameters

| Model | Architecture | Parameters | Description |
| :--- | :--- | :--- | :--- |
| **01 Final** | GCNN (Physics-Guided) | **115,306** | Graph Convolutional Network with physics-guided loss. |
| **01 Light v1** | GCNN (Reduced) | **46,802** | Reduced capacity GCNN (512 neurons, 6 channels). |
| **01 Light v2** | GCNN (Reduced) | **59,186** | Reduced capacity GCNN (512 neurons, 8 channels). |
| **01 Light v2 (2P)** | GCNN (Reduced) | **59,186** | Same as Light v2, but trained with 2-phase method (Sup -> Phys). |
| **01 Light v3 (2P)** | GCNN (Reduced) | **29,746** | Reduced capacity GCNN (256 neurons, 8 channels), 2-phase training. |
| **01 Light v4 (2P)** | GCNN (Reduced) | **17,250** | Reduced capacity GCNN (256 neurons, 4 channels), 4 feature iterations. |
| **01 Light v5 (2P)** | GCNN (Reduced) | **34,402** | Reduced capacity GCNN (512 neurons, 4 channels), 4 feature iterations. |
| **01 Dropout (2P)** | GCNN (Dropout) | **59,186** | Same as Light v2 (512n, 8c), but with Dropout=0.3. |
| **01 Node-Wise** | GCNN (Node-Wise) | **5,668** | **New Architecture**: No flattening, shared weights across all nodes. Inductive. |
| **03 Tiny** | DeepOPF-FT (Reduced) | **46,226** | MLP with 128 neurons (matched to GCNN Light v1). |
| **03 Tiny (17k)** | DeepOPF-FT (Reduced) | **17,195** | MLP with 89 neurons (matched to GCNN Light v4). |
| **03 Small** | DeepOPF-FT (Variant) | **83,718** | MLP with 180 neurons (matched capacity to GCNN). |
| **03 Large** | DeepOPF-FT (Baseline) | **2,105,018** | MLP with 1000 neurons (flattened admittance embedding). |

*Note: Model 03 (Small) was trained to test if the performance gap was due to parameter count.*

## 2. Evaluation Results

### A. Seen Test Set (Standard Evaluation)
*   **Dataset**: `gcnn_opf_01/data/samples_test.npz` (2000 samples)
*   **Topologies**: 5 fixed topologies (Base + 4 N-1 contingencies) seen during training.

| Metric | 01 Final | 01 Light v1 | 01 Light v2 | 01 Light v2 (2P) | 01 Light v3 (2P) | 01 Light v4 (2P) | 01 Light v5 (2P) | 01 Dropout (2P) | 01 Node-Wise | 03 Tiny | 03 Small | 03 Large |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PG Accuracy (< 1 MW)** | 97.97% | 85.10% | 94.97% | 96.93% | 98.58% | 96.98% | 95.57% | 93.58% | 63.98% | **98.12%** | 99.55% | 99.15% |
| **PG RMSE (p.u.)** | 0.0071 | 0.0094 | 0.0076 | 0.0073 | 0.0068 | 0.0074 | 0.0075 | 0.0079 | 0.0186 | **0.0071** | 0.0063 | 0.0070 |
| **PG R² Score** | 0.9835 | 0.9714 | 0.9813 | 0.9829 | 0.9848 | 0.9823 | 0.9819 | 0.9800 | 0.8883 | **0.9837** | 0.9873 | 0.9840 |
| **VG Accuracy (< 0.001)** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.98% | 100.00% | 99.92% | **100.00%** | 100.00% | 100.00% |
| **VG RMSE (p.u.)** | 0.000043 | 0.000202 | 0.000066 | 0.000035 | 0.000045 | 0.000045 | 0.000065 | 0.000005 | 0.000142 | **0.000045** | 0.000025 | 0.000034 |

**Additional Model Results:**

| Metric | 03 Tiny (17k) |
| :--- | :--- |
| **PG Accuracy (< 1 MW)** | 99.22% |
| **PG RMSE (p.u.)** | 0.0071 |
| **PG R² Score** | 0.9837 |
| **VG Accuracy (< 0.001)** | 100.00% |
| **VG RMSE (p.u.)** | 0.000018 |

**Analysis**: 
*   **03 Small** remains the best performer on the seen dataset.
*   **01 Light v1** showed a significant drop in accuracy (98% -> 85%) when channels were reduced to 6.
*   **01 Light v2** recovered most of the performance (95% accuracy) by restoring channels to 8, confirming that matching the channel count to the feature iteration count ($k=8$) is critical for this architecture.
*   **01 Light v2 (2P)** further improved accuracy to 97% on the seen dataset, showing that the physics-informed loss helps refine the solution within the known topology.
*   **01 Light v3 (2P)** achieved excellent performance on the seen dataset (98.58% accuracy), surpassing the larger GCNN models. This suggests that for a fixed topology distribution, a smaller, well-trained GCNN is highly effective.
*   **01 Light v4 (2P)** performed comparably to Light v3 (96.98% vs 98.58%), showing that reducing feature iterations to 4 (matching the 4 channels) is a viable strategy for further parameter reduction (17k params), though with a slight accuracy cost.

### B. Unseen Test Set (Zero-Shot Generalization)
*   **Dataset**: `gcnn_opf_01/data_unseen` (1200 samples)
*   **Topologies**: 3 new N-1 contingencies never seen during training.

| Metric | 01 Final | 01 Light v1 | 01 Light v2 | 01 Light v2 (2P) | 01 Light v3 (2P) | 01 Light v4 (2P) | 01 Light v5 (2P) | 01 Dropout (2P) | 01 Node-Wise | 03 Tiny | 03 Small | 03 Large |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PG Accuracy (< 1 MW)** | 44.14% | 40.44% | 44.19% | 48.42% | 40.44% | 45.89% | 40.89% | 47.17% | 18.42% | **56.25%** | 53.03% | 49.39% |
| **PG RMSE (p.u.)** | 0.0972 | 0.0742 | 0.0565 | 0.0838 | 0.0683 | 0.1137 | 0.1020 | 0.0591 | 0.1402 | **0.0278** | 0.0292 | 0.0389 |
| **PG R² Score** | -2.00 | -0.7482 | -0.0157 | -1.2305 | -0.4844 | -3.11 | -2.3043 | -0.1099 | -5.24 | **0.7543** | 0.7282 | 0.5201 |

**Additional Model Results:**

| Metric | 03 Tiny (17k) |
| :--- | :--- |
| **PG Accuracy (< 1 MW)** | 45.97% |
| **PG RMSE (p.u.)** | 0.0452 |
| **PG R² Score** | 0.3522 |

**Analysis**: 
*   **Generalization**: The small MLP (03 Small) generalizes significantly better than the others, achieving an R² of 0.73 on unseen topologies.
*   **GCNN Failure**: Both GCNN variants fail to generalize. Reducing the model size did **not** improve generalization. Light v2 improved R² to near zero (-0.01), effectively predicting the mean, but failed to capture the unseen topology physics.
*   **Physics Loss Impact**: The 2-Phase training (Light v2 - 2P) improved accuracy on the *seen* set but degraded R² on the *unseen* set (-1.23 vs -0.01). This suggests the physics loss causes the model to overfit to the specific physics correlations of the training topology, making it perform worse (larger errors) when the topology changes.
*   **Light v3 (256n)**: Further reducing the model size to 256 neurons (Light v3) did not solve the generalization issue (R² = -0.48), confirming that the architecture itself (or the feature set) is the bottleneck for zero-shot generalization, not just model capacity.
*   **03 Tiny (17k) vs GCNN (17k)**: The 17k parameter MLP (03 Tiny 17k) achieved an R² of 0.35 on unseen data, significantly outperforming the matched 17k GCNN (Light v4), which had an R² of -3.11. This confirms that even at very low parameter counts, the MLP architecture generalizes better than the current GCNN implementation, although reducing the MLP size from 46k to 17k did drop its generalization performance (R² 0.75 -> 0.35).
*   **Light v4 (256n 4c)**: Reducing channels and iterations to 4 slightly improved accuracy (45.89%) over the 8-channel version (40.44%), but the R² score (-3.11) indicates severe large-error outliers.
*   **Dropout (0.3)**: Adding dropout (0.3) to the Light v2 architecture (512n, 8c) slightly reduced seen accuracy (93.58% vs 96.93%) as expected, but did **not** solve the generalization problem (R² = -0.11). It performed better than the non-dropout version in terms of R² (closer to 0), but still failed to capture the underlying physics of the unseen topologies.
*   **Light v5 (512n 4c)**: Reducing channels/iterations to 4 while keeping 512 neurons resulted in good seen performance (95.57%) but very poor unseen performance (R² = -2.30), similar to Light v4. This confirms that simply having more neurons (512 vs 256) does not help generalization if the feature construction (4 iterations) is insufficient or biased.
*   **01 Node-Wise (New)**: This architecture achieved the **highest parameter efficiency** (5,668 params) and excellent accuracy on seen data (VG R² 0.999). However, it failed to generalize to unseen topologies (PG R² -5.24). Interestingly, it achieved a **VG R² of 0.65** on unseen data (not shown in table), indicating it successfully propagated some voltage physics through the new graph structure, whereas the PG prediction (which depends on global cost optimization) failed. The "blind" MLP (03 Tiny) outperformed it on PG because the N-1 optimal dispatch is statistically close to the base case, allowing the MLP to "guess" correctly by ignoring the topology change, whereas the GCNN attempted to adjust based on the topology but lacked the training diversity to do so correctly.

## 3. Case 39 (IEEE 39-bus) Investigation

### A. Experiment: Strict Reproduction
We attempted to reproduce the results on the larger IEEE 39-bus system using the "Strict Reproduction" configuration (Batch Size 10, Weight Decay 0, 2-Phase Training).

*   **Objective**: Verify if the GCNN architecture scales to larger, more complex grids.
*   **Result**: The model failed to generalize.
    *   **Training Loss**: ~0.0012 (Supervised) - Extremely low, indicating perfect memorization.
    *   **Validation Loss**: ~0.04 - 0.08 - High, indicating poor generalization.
    *   **Physics Loss**: ~4.65 - Constant throughout training.

### B. Root Cause Analysis: The "Detached Physics" Bug
Comparing the training logs of the successful Case 6 run vs. the failed Case 39 run revealed a critical anomaly:

| Metric | Legacy Success (Case 6) | Strict Repro (Case 39) | Interpretation |
| :--- | :--- | :--- | :--- |
| **PG Accuracy (< 1 MW)** | **98.58%** | **50.16%** | Huge drop in performance; Case 39 model fails strict threshold. |
| **Final Train Sup. Loss** | `0.0055` | `0.0012` | Case 39 model overfitted more severely than Case 6. |
| **Physics Loss Trend** | **Flat** (~1.08) | **Flat** (~4.65) | **CRITICAL**: Physics loss never decreased in either case. |

**Code Inspection**:
Inspection of `train.py` revealed that the physics loss was being **detached** from the computational graph before being added to the total loss:
```python
# train.py (Bug)
phys_loss_total += loss_p.item()  # .item() detaches gradient
# ...
loss = sup_loss + kappa * torch.tensor(phys_loss, device=device) # New tensor, no gradient history
```

**Implication**:
*   The "Physics-Guided" GCNN has effectively been running as a **Standard Supervised GCNN**.
*   **Why Case 6 worked**: The system is small (6 buses, 3 gens). The model could learn the mapping purely from labels (Supervised) without needing physics constraints to regularize it.
*   **Why Case 39 failed**: The system is larger and more complex. Pure supervised learning on the limited dataset led to severe overfitting (memorization) without learning the underlying physical laws. The model produced physically invalid solutions (Phys Loss 4.65) on unseen data.

### C. Next Steps
We must fix the `train.py` script to correctly accumulate the physics loss tensors (preserving gradient history). This will enable the model to actually learn from the physics constraints, acting as a regularizer to prevent the overfitting observed in Case 39.

## 4. Usage Reminder

For full reproduction steps and detailed documentation, refer to `gcnn_opf_01/docs/gcnn_opf_01.md`.

The project now uses a JSON configuration system located in `gcnn_opf_01/configs/`.

### Quick Evaluation Example
```bash
# Evaluate using a specific config
python gcnn_opf_01/evaluate.py --config gcnn_opf_01/configs/final_1000n.json --model_path <path_to_model>
```




