# Baseline Implementation Plan: dnn_opf_03 (DeepOPF-FT Style)

**Goal:** Implement a Deep Neural Network baseline that uses the **continuous admittance embedding** strategy proposed in DeepOPF-FT (Zhou et al.), but aligned with our rigorous **Two-Stage Training** and **Physics-Informed Loss**.

**Paper Reference:** *DeepOPF-FT: One Deep Neural Network for Multiple AC-OPF Problems With Flexible Topology*.

---

## 1. Mathematical Formulation

### 1.1 Input Representation (The "Embedding")
The core idea of DeepOPF-FT is to map the discrete topology into a continuous space by using the admittance matrix elements as inputs.

For a system with $N$ buses, the input vector $x$ for sample $i$ with topology $k$ is:

$$x_i = [\mathbf{P}^d_i, \mathbf{Q}^d_i, \text{vec}(\mathbf{G}^{(k)}), \text{vec}(\mathbf{B}^{(k)})]$$

Where:
- $\mathbf{P}^d_i, \mathbf{Q}^d_i \in \mathbb{R}^N$: Active and reactive load vectors.
- $\mathbf{G}^{(k)}, \mathbf{B}^{(k)} \in \mathbb{R}^{N \times N}$: The full Conductance and Susceptance matrices for topology $k$.
- $\text{vec}(\cdot)$: The flattening operation (row-major) that converts an $N \times N$ matrix into an $N^2$ vector.

**Reconstruction from Operators:**
Our dataset (`topology_operators.npz`) stores diagonal ($d$) and off-diagonal ($nd$) parts separately. We reconstruct the full matrices as:

$$G^{(k)} = G_{ndiag}^{(k)} + \text{diag}(G_{diag}^{(k)})$$
$$B^{(k)} = B_{ndiag}^{(k)} + \text{diag}(B_{diag}^{(k)})$$

### 1.2 Architecture Alignment
To ensure a fair "Architecture vs. Architecture" comparison with `gcnn_opf_01`, we deviate from the paper's variable size and strictly match our GCNN capacity:
- **Depth:** 4 Layers (1 Input + 3 Hidden + Heads).
- **Width:** 1000 Neurons per hidden layer.
- **Heads:** Two heads matching GCNN:
    1.  `gen_head`: Predicts Generator Active Power ($P_G$) and Voltage Magnitude ($V_G$).
    2.  `v_head`: Predicts Full Bus Voltages ($e, f$) — *Required for Physics Loss calculation.*

---

## 2. Implementation Steps

### Phase 1: Data Engineering

- [ ] **1.1 Create `dnn_opf_03/dataset_03.py`**
    - **Logic:**
        1. Load `topology_operators.npz`.
        2. Get the sample's `topo_id`.
        3. Retrieve the specific operators for that topology: 
           - `G_topo = operators['g_ndiag'][topo_id] + diag(operators['g_diag'][topo_id])`
           - `B_topo = operators['b_ndiag'][topo_id] + diag(operators['b_diag'][topo_id])`
           *Note: These matrices AUTOMATICALLY contain the "zeros" for N-1 contingencies as described in the Zhou et al. paper.*
        4. **Flatten** these into 1D vectors: `g_flat`, `b_flat`.
        5. **Concatenate Input:** `x = cat([pd, qd, g_flat, b_flat])`.
        6. Return: `{'input': input_vec, 'labels': ..., 'operators': ...}`.

### Phase 2: Model Architecture

- [ ] **2.1 Create `dnn_opf_03/model_03.py`**
    - **Class:** `AdmittanceDNN`.
    - **Input Dimension:** $2N + 2N^2$ (For Case6ww: $12 + 72 = 84$).
    - **Structure:**
        ```python
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1000), nn.ReLU(),
            nn.Linear(1000, 1000),      nn.ReLU(),
            nn.Linear(1000, 1000),      nn.ReLU(),
            nn.Linear(1000, 1000),      nn.ReLU(),
        )
        ```
    - **Heads:**
        - `fc_gen`: Linear(1000 -> N_GEN * 2)  # (PG, VG)
        - `fc_v`:   Linear(1000 -> N_BUS * 2)  # (e, f) for Physics Loss

### Phase 3: Training Pipeline

- [ ] **3.1 Create `dnn_opf_03/train_03.py`**
    - **Setup:** Copy logic from `gcnn_opf_01/train.py`.
    - **Adaptation:**
        - Import `AdmittanceDNN` and `OPFDataset03`.
        - Pass `input_vec` to the model instead of `e_0_k, f_0_k, ...`.
    - **Physics Loss Integration:**
        - The loss function `correlative_loss_pg` requires $(P_G, V_{out}, G, B)$.
        - `AdmittanceDNN` outputs $P_G$ (from gen_head) and $V_{out}$ (from v_head).
        - $G, B$ are available from the dataset batch.
        - **Constraint:** Ensure `v_out` is passed to the loss function correctly to compute $L_{\Delta, PG}$.
    - **Strategy:** Enforce the **Two-Stage Training** (Phase 1 $\kappa=0 \to$ Phase 2 $\kappa=1$).

### Phase 4: Evaluation

- [ ] **4.1 Create `dnn_opf_03/evaluate_03.py`**
    - **Metric Logic:** Use the **Denormalized** evaluation (physical units).
    - **Thresholds:**
        - Power: 1 MW (0.01 p.u.).
        - Voltage: 0.001 p.u.
    - **Goal:** Produce a results file (`evaluation_results.npz`) compatible with the GCNN results for side-by-side comparison.