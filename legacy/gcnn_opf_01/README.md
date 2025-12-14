# Physics-Guided GCNN (Model 01)

**Model 01** implements the Physics-Guided Graph Convolutional Neural Network (GCNN) proposed by Gao et al. It leverages the power grid topology and physics equations (Kirchhoff's laws) directly within the neural network architecture.

## 1. Architecture
- **Core Layer**: `GraphConv` (Physics-Guided)
  - Implements the convolution operation using the admittance matrix ($G, B$) as the graph operator.
  - Input: Voltage state ($e^l, f^l$), Demand ($P_d, Q_d$).
  - Output: Updated voltage state ($e^{l+1}, f^{l+1}$).
- **Structure**:
  - $K$ GraphConv layers (typically $K=2$ or matched to channel count).
  - Shared Fully Connected (FC) trunk.
  - **Two Heads**:
    1.  **Generation Head**: Outputs $P_G$ and $V_G$ (Control variables).
    2.  **Voltage Head**: Outputs $e$ and $f$ (State variables) for physics loss calculation.

## 2. Physics-Informed Loss
The model is trained using a **Correlative Loss** function:
$$ L = L_{sup} + \kappa L_{\Delta, PG} $$

- **$L_{sup}$ (Supervised Loss)**: MSE between predicted generation ($P_G, V_G$) and optimal labels.
- **$L_{\Delta, PG}$ (Physics Consistency Loss)**: Measures the discrepancy between the predicted generation $P_G$ and the power flow implied by the predicted voltages $V_{out}$ ($e, f$).
  - $L_{\Delta, PG} = || P_G - f_{PG}(V_{out}) ||^2$
  - This enforces that the predicted control variables ($P_G$) are consistent with the predicted state variables ($V_{out}$) according to the AC power flow equations.

## 3. Key Findings (Week 7-8)
| Metric | Seen Data (Case 6) | Unseen Data (Zero-Shot) |
| :--- | :--- | :--- |
| **PG Accuracy** | ~98% | ~44% |
| **Physics Loss ($L_{phys}$)** | **~0.01** | **~0.08** |
| **Consistency** | **High** | **Moderate** |

*   **High Consistency**: Unlike the MLP baseline (Model 03), the GCNN maintains high physical consistency ($L_{phys} \approx 0.01$ vs $1.5$).
*   **Generalization Challenge**: While physically consistent, the current architecture struggles with zero-shot generalization to unseen topologies (Accuracy drops to ~44%), likely due to the difficulty of learning the global optimization landscape from local graph operations.

## 4. Usage

### Training
```bash
# Train with default config
python gcnn_opf_01/train.py --config gcnn_opf_01/configs/base.json
```

### Evaluation
```bash
# Evaluate on Seen Data
python gcnn_opf_01/evaluate.py --model_path gcnn_opf_01/results/best_model.pth --data_dir gcnn_opf_01/data

# Evaluate on Unseen Data
python gcnn_opf_01/evaluate.py --model_path gcnn_opf_01/results/best_model.pth --data_dir gcnn_opf_01/data_unseen
```

## 5. File Structure
- `model_01.py`: GCNN architecture and `GraphConv` layer.
- `loss_model_01.py`: Implementation of the Correlative Loss ($L_{sup} + \kappa L_{\Delta, PG}$).
- `feature_construction_model_01.py`: Pre-processing to generate initial voltage estimates ($e^0, f^0$).
- `train.py`: Training loop with physics loss accumulation.
- `evaluate.py`: Evaluation script.
- `configs/`: JSON configuration files.
