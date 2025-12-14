# Baseline Implementation Plan: M8 (Hasan DNN)

**Goal:** Implement the Custom Hasan DNN (Model M8) in a dedicated folder `dnn_opf_02`, using the existing data from `gcnn_opf_01`.

**Architecture:**

- **Input:** Flattened vector of Demand ($P_D, Q_D$) + Topology Proxy ($\Delta V$).
- **Structure:** 4 layers × 1000 neurons (ReLU).
- **Output:** Direct $P_G, V_G$.
- **Loss:** $L_{sup} + \kappa L_{phy}$ (Same as GCNN).

---

## Phase 1: Feature Engineering

- [ ] **1.1 Compute Mean Voltage Profiles**
    - **Script:** `dnn_opf_02/compute_topology_means.py`
    - **Logic:**
        1. Load `gcnn_opf_01/data/samples_train.npz`.
        2. Group samples by `topo_id`.
        3. Calculate mean voltage magnitude vector $V_{mean}^{(k)}$ for each topology.
        4. Save to `dnn_opf_02/data/topology_voltage_means.npz`.

- [ ] **1.2 Create Dataset Class**
    - **File:** `dnn_opf_02/dataset_02.py`
    - **Class:** `OPFDataset02` (inherit or clone from `OPFDataset`).
    - **Logic:**
        - Load `topology_voltage_means.npz`.
        - For each sample, compute `delta_v = V_mean[topo_id] - V_mean[0]`.
        - Return `{'pd':..., 'qd':..., 'delta_v':..., 'labels':...}`.

## Phase 2: Model & Training

- [ ] **2.1 Define Model Architecture**
    - **File:** `dnn_opf_02/model_02.py`
    - **Class:** `HasanDNN`
    - **Structure:** MLP with input dim `N_BUS*3` (PD+QD+DeltaV) -> 1000 -> 1000 -> 1000 -> 1000 -> Heads.

- [ ] **2.2 Create Training Script**
    - **File:** `dnn_opf_02/train_02.py`
    - **Logic:**
        - Adapt `gcnn_opf_01/train.py` to use `HasanDNN` and `OPFDataset02`.
        - Implement the **Two-Stage Strategy** (Phase 1 $\kappa=0$, Phase 2 $\kappa=1$).
        - Load physics operators ($G, B$) for the loss function, even though the model doesn't use them as input.

## Phase 3: Evaluation

- [ ] **3.1 Create Evaluation Script**
    - **File:** `dnn_opf_02/evaluate_02.py`
    - **Logic:** Use **Denormalized Metrics** (1 MW / 0.001 p.u. thresholds) to ensure fair comparison with GCNN results.