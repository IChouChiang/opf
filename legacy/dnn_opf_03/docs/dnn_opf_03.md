# DeepOPF-FT Baseline (dnn_opf_03)

## 1. Project Overview
This project implements a **Deep Neural Network (DNN)** baseline for AC Optimal Power Flow (AC-OPF), inspired by the **DeepOPF-FT** (Zhou et al.) approach. It serves as a rigorous baseline to compare against Graph Convolutional Neural Networks (GCNN).

**Key Strategy:**
Instead of using graph convolutions to handle topology, this model flattens the full admittance matrices ($G, B$) and concatenates them with the load vector ($P_d, Q_d$) to form a single, large input vector. This "continuous admittance embedding" allows a standard MLP to learn topological variations.

---

## 2. Architecture (`model_03.py`)

### Class: `AdmittanceDNN`
A fully connected Multi-Layer Perceptron (MLP).

*   **Input Layer:** Dimension $D_{in} = 2N + 2N^2$
    *   $N$: Number of buses (6 for Case6ww)
    *   Components: $[P_d, Q_d, \text{vec}(G), \text{vec}(B)]$
    *   For Case6ww: $12 + 72 = 84$ inputs.
*   **Hidden Layers:**
    *   3 Hidden Layers
    *   Width: 1000 neurons per layer
    *   Activation: ReLU
*   **Output Heads:**
    1.  `pg_head`: Linear(1000 $\to$ $N_{gen}$) $\to$ Generator Active Power ($P_G$)
    2.  `vg_head`: Linear(1000 $\to$ $N_{gen}$) $\to$ Generator Voltage Magnitude ($V_G$)
    3.  `v_all_head`: Linear(1000 $\to$ $2N_{bus}$) $\to$ Full Bus Voltages ($e, f$)
        *   *Note: `v_all_head` is used only during training for the physics-informed loss.*

**Total Parameters:** ~2.1 Million (for Case6ww)

---

## 3. Hyperparameters (`config_03.py`)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `input_dim` | 84 | $2N + 2N^2$ (Case6ww) |
| `hidden_dim` | 1000 | Neurons per hidden layer |
| `n_hidden_layers` | 3 | Depth of the network |
| `batch_size` | 24 | Optimal batch size (matched to GCNN) |
| `lr` (Phase 1) | 1e-3 | Learning rate for Supervised Phase |
| `lr` (Phase 2) | 1e-4 | Learning rate for Physics Phase |
| `weight_decay` | 1e-5 | L2 Regularization |
| `kappa` ($\kappa$) | 1.0 | Weight for Physics Loss in Phase 2 |

---

## 4. Mathematical Formulation

### 4.1 Input Embedding
For a sample $i$ with topology $k$:
$$x_i = [\mathbf{P}^d_i, \mathbf{Q}^d_i, \text{vec}(\mathbf{G}^{(k)}), \text{vec}(\mathbf{B}^{(k)})]$$

Where:
*   $\mathbf{G}^{(k)}, \mathbf{B}^{(k)}$ are the full $N \times N$ conductance and susceptance matrices.
*   $\text{vec}(\cdot)$ flattens the matrix into a vector of size $N^2$.

### 4.2 Loss Functions (`loss_model_03.py`)

**1. Supervised Loss ($L_{sup}$):**
Standard Mean Squared Error against optimal labels ($P_G^*, V_G^*$):
$$L_{sup} = \frac{1}{N_{gen}} \sum_{g} (P_G - P_G^*)^2 + (V_G - V_G^*)^2$$

**2. Physics-Informed Loss ($L_{phy}$):**
Enforces the AC power balance equations. The model predicts full voltages $V = e + jf$, and we compute the implied generation $P_G^{calc}$:
$$P_{G,i}^{calc} = P_{d,i} + e_i \sum_j (G_{ij}e_j - B_{ij}f_j) + f_i \sum_j (G_{ij}f_j + B_{ij}e_j)$$

The loss penalizes the deviation between the model's direct prediction $P_G^{pred}$ and the physics-calculated $P_G^{calc}$:
$$L_{phy} = || P_G^{pred} - P_G^{calc}(V_{pred}) ||^2$$

**3. Total Loss (Phase 2):**
$$L_{total} = L_{sup} + \kappa \cdot L_{phy}$$

---

## 5. Key Implementation Details

### 5.1 `dataset_03.py`
*   **Function:** `OPFDataset03`
*   **Key Logic:**
    *   Loads sparse operators (`g_diag`, `g_ndiag`, etc.).
    *   Reconstructs full dense matrices: $G = G_{ndiag} + \text{diag}(G_{diag})$.
    *   Flattens and concatenates them with demands to form the input vector.

### 5.2 `model_03.py`
*   **Function:** `AdmittanceDNN`
*   **Key Logic:**
    *   Standard PyTorch `nn.Linear` layers.
    *   Multi-head output design to support both label prediction and physics loss calculation.

### 5.3 `train_03.py`
*   **Function:** `train_epoch`
*   **Key Logic:**
    *   **Two-Stage Training:**
        *   **Phase 1:** $\kappa=0$. Pure supervised learning to initialize weights.
        *   **Phase 2:** $\kappa=1$. Adds physics loss to refine solution feasibility.
    *   **Batch Processing:** Reconstructs $G, B$ matrices for each sample in the batch to compute $L_{phy}$ vectorially.

### 5.4 `evaluate_03.py`
*   **Function:** `evaluate_model`
*   **Key Logic:**
    *   Computes probabilistic accuracy ($P(|error| < \epsilon)$).
    *   Thresholds: 1 MW (0.01 p.u.) for Power, 0.001 p.u. for Voltage.
    *   Denormalizes predictions before metric calculation.

---

## 6. Results (Case6ww)
Evaluated on 2000 test samples:

| Metric | Value |
| :--- | :--- |
| **PG Accuracy (<1MW)** | **99.15%** |
| **VG Accuracy (<0.001)** | **100.00%** |
| **PG MAPE** | 0.50% |
| **PG RMSE** | 0.0070 p.u. |

This baseline demonstrates that for small, fixed-size topologies, a dense MLP with explicit admittance inputs is highly effective.
