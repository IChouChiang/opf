# DeepOPF-FT Baseline (Model 03)

**DeepOPF-FT** (Flattened Topology) is a Multi-Layer Perceptron (MLP) baseline for AC-OPF. It treats the grid topology as a flattened input vector, learning a direct mapping from `[Demand, Topology]` to `[Generation]`.

## 1. Architecture
- **Model**: `AdmittanceDNN` (Fully Connected Network)
- **Input**: Concatenated vector $[P_d, Q_d, \text{vec}(G), \text{vec}(B)]$
  - *Note*: Requires fixed input size. Changes in grid size require retraining.
- **Output**: Generator active power ($P_G$) and voltage magnitude ($V_G$).
- **Auxiliary Output**: Full voltage state ($e, f$) for physics loss calculation (optional).

## 2. Model Variants
We explored several capacity variants to match GCNN parameter counts:
- **03 Large**: 1000 neurons, 3 layers (~2.1M params).
- **03 Small**: 180 neurons, 3 layers (~83k params).
- **03 Tiny**: 128 neurons, 3 layers (~46k params).
- **03 Tiny (17k)**: 89 neurons, 2 layers (~17k params).

## 3. Key Findings (Week 7-8)
| Metric | Seen Data (Case 6) | Unseen Data (Zero-Shot) |
| :--- | :--- | :--- |
| **PG Accuracy** | > 99% | ~50% |
| **Physics Loss ($L_{phys}$)** | **~1.5 - 1.8** | **~1.6 - 2.0** |
| **Consistency** | Low | Very Low |

*   **High Accuracy, Low Physics**: The MLP achieves excellent regression accuracy on fixed topologies but has **100x higher physics violation** than GCNNs ($L_{phys} \approx 1.5$ vs $0.01$).
*   **Black Box**: It acts as a curve fitter, memorizing the optimal dispatch without learning the underlying power flow equations.

## 4. Usage

### Training
Use JSON configs for reproducibility:
```bash
# Train Tiny model (17k params)
python dnn_opf_03/train_03.py --config dnn_opf_03/configs/exp_tiny_17k.json

# Train Large baseline
python dnn_opf_03/train_03.py --config dnn_opf_03/configs/case39_baseline.json
```

### Evaluation
Evaluate accuracy and physics consistency:
```bash
# Evaluate on Seen Data
python dnn_opf_03/evaluate_03.py --model_path dnn_opf_03/results/best_model.pth --data_dir gcnn_opf_01/data

# Evaluate on Unseen Data
python dnn_opf_03/evaluate_03.py --model_path dnn_opf_03/results/best_model.pth --data_dir gcnn_opf_01/data_unseen
```

## 5. File Structure
- `model_03.py`: MLP architecture definition.
- `train_03.py`: Training loop (supports 2-phase training).
- `evaluate_03.py`: Evaluation script (calculates MSE, R², and $L_{phys}$).
- `loss_model_03.py`: Physics loss implementation (Correlative Loss).
- `dataset_03.py`: PyTorch dataset loader for flattened inputs.
- `configs/`: JSON hyperparameters.
