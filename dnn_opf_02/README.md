# Hasan DNN (M8) Baseline for OPF

This directory contains the implementation of the "Hasan DNN" (Model M8) baseline for Optimal Power Flow (OPF), as described in the project documentation. This model serves as a baseline to compare against the Graph Convolutional Neural Network (GCNN) approach.

## Model Architecture (M8)

- **Type:** Fully Connected Neural Network (MLP)
- **Structure:** 4 Hidden Layers x 1000 Neurons
- **Activation:** ReLU
- **Inputs:** Flattened vector of size `4 * N_BUS` containing `[Pd, Qd, Δe, Δf]`
  - `Pd`, `Qd`: Active and Reactive Load
  - `Δe`, `Δf`: Voltage deviation from topology-specific mean profile
- **Outputs:**
  - Head 1: Generator Active Power (`PG`) and Voltage Magnitude (`VG`)
  - Head 2: Bus Voltage Components (`e`, `f`) - *Used for physics loss*

## Feature Engineering: Delta V

The model uses a "Delta V" feature, which is the deviation of the voltage from a pre-computed mean voltage profile for each topology.
- **Script:** `compute_topology_means.py`
- **Data:** `data/topology_voltage_means.npz`

## Training Strategy

The training is performed in two stages:
1.  **Stage 1 (Supervised):** Training with pure MSE loss against ground truth labels ($\kappa = 0$).
2.  **Stage 2 (Physics-Informed):** Fine-tuning with a combined loss of MSE + Physics Constraints ($\kappa = 1.0$).

## Directory Structure

```
dnn_opf_02/
├── data/                   # Generated data and statistics
│   ├── topology_voltage_means.npz
│   └── ...
├── results/                # Model checkpoints and evaluation logs
│   ├── best_model_stage1.pth
│   ├── best_model_stage2.pth
│   └── evaluation_results.npz
├── compute_topology_means.py # Script to compute mean voltage profiles
├── dataset_02.py           # Custom PyTorch Dataset with Delta V features
├── model_02.py             # PyTorch Model Definition (M8)
├── train_02.py             # Two-stage training script
├── evaluate_02.py          # Evaluation script
└── README.md               # This file
```

## How to Run

### 1. Prerequisites
Ensure you are in the `opf311` environment and have the necessary data in `Week3/samples/`.

### 2. Compute Topology Means (One-time setup)
This computes the mean voltage profile for each topology to be used as a baseline for the Delta V feature.
```bash
python dnn_opf_02/compute_topology_means.py
```

### 3. Train the Model
Run the two-stage training pipeline.
```bash
python dnn_opf_02/train_02.py --epochs_stage1 25 --epochs_stage2 25 --batch_size 10
```
*Arguments:*
- `--epochs_stage1`: Number of epochs for supervised training (default: 25)
- `--epochs_stage2`: Number of epochs for physics-informed fine-tuning (default: 25)
- `--batch_size`: Batch size (default: 10)

### 4. Evaluate the Model
Evaluate the trained model on the test set.
```bash
python dnn_opf_02/evaluate_02.py --model_path dnn_opf_02/results/best_model_stage2.pth
```

## Results (Example)

Evaluated on Stage 2 Model (25+25 epochs):

| Metric | PG (Active Power) | VG (Voltage Mag) |
| :--- | :--- | :--- |
| **MAPE** | **1.03%** | **0.00%** |
| **R²** | 0.976 | 1.000 |
| **RMSE** | 0.0086 | 0.00004 |

These results indicate a highly accurate baseline model.
