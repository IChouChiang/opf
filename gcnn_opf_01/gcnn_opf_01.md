# GCNN_OPF_01

## Current Status (2025-11-19)

- Files present:
  - `config_model_01.py`: Dataclasses-based configs (`ModelConfig`, `TrainingConfig`) with convenience instances and legacy constants retained for compatibility.
  - `model_01.py`: GCNN model; fixed concatenation bug (now uses `torch.cat`) and removed unused imports. Shapes verified at module level.
  - `sample_config_model_01.py`: Utilities for IEEE-39 case loading, N-1 topology application, and operator construction:
    - `TOPOLOGY_BRANCH_PAIRS_1BASED` with topo IDs 0–4.
    - `apply_topology(ppc_int, topo_id)` sets `branch[:, 10]` (BR_STATUS) to 0 for outages.
    - `build_G_B_operators(ppc_int)` returns `G, B, g_diag, b_diag, g_ndiag, b_ndiag` as `torch.float32` tensors.
  - `topo_N-1_model_01.html`: Interactive vis-network copy for IEEE-39 with requested edges highlighted.
  - `gcnn_opf_01.md`: Design notes and data loading guidance (this file).
  - `formulas_model_01.md`: Formula references for the GCNN/feature construction.

- Observations:
  - BR_STATUS column index `10` matches MATPOWER/PYPOWER convention.
  - Topology pairs (14–13) and (15–16) exist as direct branches in case39; (26–27) and (1–39) do not appear as direct rows.
  - Operator construction relies on `makeYbus` and cleanly splits diag/off-diag parts into tensors, aligning with `model_01.py` expectations.

- Next steps (suggested):
  - Add a thin adapter to package `PD/QD`, `G/B` into tensors and tile `e_0_k`, `f_0_k` channels for the model.
  - Write a small unit test for `model_01.py` to assert output shapes given synthetic inputs.
  - Decide on Dataset strategy: precompute features to NPZ vs on-the-fly.

## To-do list (in order)

Putting it all together:

### 1. Network & config

- [ ] Write `sample_config_model_01.py` (load `case39`, compute basic metadata).
- [ ] Decide 5 line contingencies and implement `apply_topology(ppc_int, topo_id)`.

---

### 2. Scenario Generator

- [ ] Implement `ScenarioGenerator` with load fluctuation, RES sampling, target penetration, and RES-as-negative-leak.
- [ ] Debug for a handful of samples (check total PD, RES penetration, etc.).

---

### 3. OPF labeling

- [ ] Implement `solve_ac_opf(ppc_base, pd, qd, topo_id)` (Pyomo+Gurobi).
- [ ] Test on a few manually constructed scenarios.
- [ ] Loop over 12k scenarios and fill arrays for `PD_all`, `QD_all`, `topo_all`, `PG_all`, `VG_all`.
- [ ] Save to `opf39_dataset_v1.npz`.

---

### 4. Model-Informed Feature Construction

- [ ] Implement `construct_features(ppc_base, pd, qd, topo_id, num_iter=7)` according to III.C (Eqs. 8-9, 16-22, 23-25).
- [ ] Verify for one sample that features don't diverge and normalization work.
- [ ] Decide: precompute $e\_0\_k$, $f\_0\_k$ for all samples into a new NPZ, or compute on the fly in Dataset.

---

### 5. PyTorch Dataset + DataLoader

- [ ] Implement `OPF39Dataset` that reads NPZ (+ maybe calls `construct_features`).
- [ ] Implement z-score normalization (fit on training set, save mean/std).

---

### 6. Model wiring (GCNN\_OPF\_01)

* [x] `GraphConv` (done conceptually).
* [x] `GCNN_OPF_01` (only minor shape tests left).
* Write a quick unit test to check `model(e_0_k, f_0_k, pd, qd) \rightarrow \text{shape } [N\_GEN, 2]`.

---

### 7. Training loop

- [ ] Implement basic training with $\mathcal{L}_{sup}$ (MSE) only.
- [ ] Monitor loss, maybe simple validation error.

---

### 8. (Optional) Correlative loss

- [ ] Implement $\mathcal{L}_{PG}(\mathbf{v}_{pred})$ and add $k \mathcal{L}_{\Delta, P\mathcal{X}}$; term to loss.
- [ ] Fine-tune using that as in the paper.

## GCNN OPF Data Loading Notes

**Purpose**
- Summarize which functions in `src/helpers_ac_opf.py` help feed `gcnn_opf_01/model_01.py`, and how well they align.

**Useful Helpers**
- `prepare_ac_opf_data(ppc)`: Builds per-unit system data from a PYPOWER case (after `ext2int`). Provides:
  - `PD[i]`, `QD[i]`: Bus active/reactive demand (p.u.).
  - `G[i,j]`, `B[i,j]`: Real/imag parts of Ybus (dense, p.u.).
  - `GEN_BUS[g]`: Mapping generator → bus (internal indexing).
  - `Vmin[i]`, `Vmax[i]`, `PGmin/max[g]`, `QGmin/max[g]`, cost data (if needed later).
  - Returns the internal-numbered case `ppc_int` alongside the Pyomo data dict.
- `initialize_voltage_from_flatstart(instance, ppc_int)`: Shows how to compute Cartesian voltages from case data:
  - Uses `Vm` and `Va_deg` to set `e = Vm*cos(Va)` and `f = Vm*sin(Va)`.
- `solve_ac_opf(...)`: End‑to‑end pipeline; useful as reference for pulling baseMVA, `gen` table, and slack info.

**Alignment with `model_01.py`**
- Model inputs expected:
  - `e_0_k`, `f_0_k`: `[N_BUS, Cin]` initial feature tensors for graph conv.
  - `pd`, `qd`: length‑`N_BUS` (or `[N, 1]`) tensors.
  - Physics operators: `g_ndiag [N,N]`, `b_ndiag [N,N]`, `g_diag [N]`, `b_diag [N]`.
- What helpers already provide:
  - `PD/QD` per bus → directly map to `pd/qd`.
  - Full `G`, `B` → derive `g_diag = diag(G)`, `b_diag = diag(B)`, `g_ndiag = G - diag(diag(G))`, `b_ndiag = B - diag(diag(B))`.
  - `ppc_int['bus'][:,7]` (`Vm`) and `ppc_int['bus'][:,8]` (`Va_deg`) → build `e_0_k`, `f_0_k` via cos/sin (optionally tile across `Cin`).
- Indexing/units:
  - Helpers already run `ext2int` and convert MW/MVAr → p.u., which matches the model’s expectation to work in p.u.
  - Bus indices are 0‑based and consistent for tensor construction.

**Gaps and Quick Adapters**
- Missing: small utilities to transform helper outputs into tensors for the GCNN. Suggested adapters:
  - Build physics operators:
    ```python
    import numpy as np, torch as t
    # from prepare_ac_opf_data outputs
    G_np, B_np = G_matrix, B_matrix  # reconstruct from dicts or capture before dict-ification
    g_diag = t.tensor(np.diag(G_np), dtype=t.float32)
    b_diag = t.tensor(np.diag(B_np), dtype=t.float32)
    g_ndiag = t.tensor(G_np - np.diag(np.diag(G_np)), dtype=t.float32)
    b_ndiag = t.tensor(B_np - np.diag(np.diag(B_np)), dtype=t.float32)
    ```
  - Initial features from case voltages:
    ```python
    Vm = ppc_int['bus'][:,7]
    Va = np.deg2rad(ppc_int['bus'][:,8])
    e0 = t.tensor(Vm*np.cos(Va), dtype=t.float32).unsqueeze(1)  # [N,1]
    f0 = t.tensor(Vm*np.sin(Va), dtype=t.float32).unsqueeze(1)
    # If Cin > 1, tile along channel dim
    e_0_k = e0.repeat(1, Cin)
    f_0_k = f0.repeat(1, Cin)
    ```
  - Demands:
    ```python
    # PD/QD dicts keyed by int bus ids
    pd = t.tensor([PD[i] for i in range(N)], dtype=t.float32)
    qd = t.tensor([QD[i] for i in range(N)], dtype=t.float32)
    ```
- Optional: keep `N_BUS = len(PD)`, `N_GEN = len(GEN_BUS)` in config to sync with case size.

**Takeaways**
- `prepare_ac_opf_data` provides all core quantities to feed `model_01.py` after a light transformation step (diag/off-diag split and tensor conversion).
- Voltage initialization logic is already demonstrated in `initialize_voltage_from_flatstart`; reuse it to create stable `e_0_k`, `f_0_k`.
- No changes needed in helpers; add a thin loader in `gcnn_opf_01` to adapt outputs into PyTorch tensors ready for the GCNN.
