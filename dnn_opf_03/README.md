# DeepOPF-FT Baseline (DNN with Flattened Topology)

This directory contains the implementation of the **DeepOPF-FT** baseline model for AC-OPF. This model uses a standard Deep Neural Network (MLP) that takes flattened admittance matrices (G, B) concatenated with demand (Pd, Qd) as input.

## Architecture
- **Model**: `AdmittanceDNN` (MLP)
- **Input**: Concatenated vector $[P_d, Q_d, \text{vec}(G), \text{vec}(B)]$
  - For Case6ww (6 buses): Input dimension = $6 + 6 + 36 + 36 = 84$
- **Hidden Layers**: 3 layers of 1000 neurons each (ReLU activation)
- **Output**: Generator active power ($P_G$) and voltage magnitude ($V_G$) for all generators.

## Training Strategy
Two-stage training process:
1.  **Phase 1 (Supervised)**: Minimizes MSE between predicted and optimal generation labels.
2.  **Phase 2 (Physics-Informed)**: Minimizes MSE + Lagrangian penalty for power balance violations.

## Results (Case6ww)
Evaluated on 2000 test samples:

| Metric | Value |
| :--- | :--- |
| **PG RMSE** | 0.0070 p.u. |
| **PG MAPE** | 0.50% |
| **PG Accuracy (<1MW)** | 99.15% |
| **VG RMSE** | 0.000034 p.u. |
| **VG Accuracy (<0.001)** | 100.00% |

## Usage

### Training
```bash
python dnn_opf_03/train_03.py --data_dir gcnn_opf_01/data --results_dir dnn_opf_03/results --two_stage
```

### Evaluation
```bash
python dnn_opf_03/evaluate_03.py --model_path dnn_opf_03/results/best_model.pth --data_dir gcnn_opf_01/data
```
