# GCNN_OPF_01

## Current Status (2025-01-19)

- Files present:
  - `config_model_01.py`: Dataclasses-based configs (`ModelConfig`, `TrainingConfig`) with convenience instances and legacy constants retained for compatibility.
  - `model_01.py`: GCNN model; fixed concatenation bug (now uses `torch.cat`) and removed unused imports. Shapes verified at module level.
  - **`sample_config_model_01.py`**: Utilities for **case6ww** (6-bus Wood & Wollenberg system):
    - **Updated from case39 to case6ww** for faster AC-OPF solving (~10-20x speedup).
    - `TOPOLOGY_BRANCH_PAIRS_1BASED` with topo IDs 0–4: N-1 contingencies at branches (5-2), (1-2), (2-3), (5-6) in external 1-based numbering.
    - **RES bus configuration**: Wind at bus 5, PV at buses 4 and 6 (external numbering).
    - `find_branch_indices_for_pairs()` now properly converts external 1-based bus IDs to internal 0-based indices using `e2i` mapping.
    - `apply_topology(ppc_int, topo_id)` sets `branch[:, 10]` (BR_STATUS) to 0 for outages.
    - `load_case6ww_int()` (renamed from `load_case39_int()`) loads case6ww with internal indexing.
    - `build_G_B_operators(ppc_int)` returns `G, B, g_diag, b_diag, g_ndiag, b_ndiag` as `torch.float32` tensors.
  - `sample_generator_model_01.py`: Scenario generator completed with:
    - Gaussian load fluctuation, wind (Weibull + turbine curve), PV (Beta + linear irradiance), per-bus nameplate tied to base PD.
    - Target RES penetration scaling (global), with percent/fraction auto-interpretation.
    - `allow_negative_pd` flag to optionally permit net export (no clipping).
    - **Power factor correction** (line 127, 193): Calculates `power_factor_ratio = QD_base / (PD_base + 1e-8)`, then recalculates `qd = power_factor_ratio * pd` after RES injection to maintain consistent QD/PD ratio.
  - **`tests/test_sample_generator.py`**: Test updated for case6ww:
    - **Migrated from case39 to case6ww** (6 buses, 3 generators, 11 branches).
    - Uses 30% RES penetration (reduced from 50.9% for smaller system stability).
    - Disabled `allow_negative_pd` for case6ww.
    - Report penetration by injected offset (method 1): `(sum(pd_raw) - sum(pd)) / sum(pd_raw)` → matches target (30%).
    - **AC-OPF integration**: Generates 3 scenarios, solves each with `solve_ac_opf()` from `src/helpers_ac_opf.py`.
    - **CPU optimization**: Detects and uses all 20 CPU cores via `multiprocessing.cpu_count()` for Gurobi parallel solving.
    - **Enhanced output**: Shows PD and QD totals, penetration %, and OPF results (generation, cost, demand, losses).
  - **`tests/test_topology_outages.py`**: Topology verification test updated to case6ww; all 4 N-1 topologies verified ✓.
  - `tests/debug_sample2.py`: Standalone debug script for Sample 2 with verbose Gurobi output; used to diagnose time limit issues before QD fix.
  - **`topo_N-1_model_01.html`**: Interactive visualization for case6ww with highlighted features:
    - **Red edges**: N-1 contingency lines (5-2, 1-2, 2-3, 5-6).
    - **Green node**: Wind generation bus (bus 5).
    - **Yellow nodes**: PV generation buses (buses 4, 6).
  - **`create_topo_viz.py`** and **`apply_highlights.py`**: Scripts to generate and customize topology visualization.
  - `gcnn_opf_01.md`: Design notes and data loading guidance (this file).
  - `formulas_model_01.md`: Formula references for the GCNN/feature construction.

- Observations:
  - BR_STATUS column index `10` matches MATPOWER/PYPOWER convention.
  - **case6ww migration**: System reduced from 39 buses/10 gens to 6 buses/3 gens for faster training data generation.
  - **AC-OPF solve time**: case6ww solves in <1 second vs 10-180 seconds for case39 (10-20x speedup).
  - Operator construction relies on `makeYbus` and cleanly splits diag/off-diag parts into tensors, aligning with `model_01.py` expectations.
  - The prior "low penetration" readout was due to using raw availability; penetration is now computed from injected RES and hits the target.
  - **Power factor fix impact**: QD recalculation from final PD ensures physically consistent reactive power in scenarios.

### Quick Notes (2025-11-19)
- **Migrated from case39 (39-bus) to case6ww (6-bus)** for feasible 12k sample generation.
- Penetration reporting now uses injected offset (method 1), not raw availability.
- Known warning: NumPy 2.x vs Torch compiled on 1.x — benign for this test path.
- **AC-OPF test results for case6ww** (3 scenarios, 30% RES penetration):
  - Sample 1: PD=1.357 p.u., QD=1.357 p.u., Objective=2218.18 $/hr, optimal, ~0.5s solve time.
  - Sample 2: PD=1.502 p.u., QD=1.502 p.u., Objective=2391.10 $/hr, optimal, ~0.8s solve time.
  - Sample 3: PD=1.516 p.u., QD=1.516 p.u., Objective=2410.95 $/hr, optimal, ~0.5s solve time.
  - Losses: 0.028-0.040 p.u. (realistic for 6-bus system).
- **Solver fixes in `src/helpers_ac_opf.py`**:
  - Lines 267-290: Warm start initialization with bounds clipping to prevent W1002 warnings.
  - Lines 318-328: Replaced deprecated `pyo.SolverResults()` with custom `ErrorResult` class.
- **case6ww topology visualization** with color-coded highlights created successfully.
- **Dataset generation** (2025-11-19):
  - **COMPLETED**: Full 12k dataset generation successful
  - Script: `generate_dataset.py` with 20 CPU threads
  - Runtime: ~42 minutes total (16:36-17:18)
  - Training: 10,000 samples (96.2% success rate, 394 failures)
  - Test: 2,000 samples (95.7% success rate, 89 failures)
  - Total storage: 4.46 MB (compressed NPZ files)
  - Output files in `gcnn_opf_01/data/`:
    - `samples_train.npz` (3.72 MB)
    - `samples_test.npz` (0.74 MB)
    - `topology_operators.npz` (1.64 KB)
    - `norm_stats.npz` (2 KB)
  - Normalization stats: pd/qd mean=0.246±0.275, pg mean=0.505±0.056, vg mean=1.057±0.009

- Next steps (suggested):
  - Verify generated dataset integrity (shapes, statistics, normalization).
  - Add a thin adapter to package tensors for model input.
  - Write a small unit test for `model_01.py` to assert output shapes given synthetic inputs.
  - Implement PyTorch Dataset class to load NPZ files.
  - Begin training loop implementation.

## Sample Config

`sample_config_model_01.py` (case6ww):
- System: 6 buses / 3 gens / 11 branches / baseMVA=100 (≤1s AC‑OPF).
- Topologies (external buses): 0=[] 1=(5,2) 2=(1,2) 3=(2,3) 4=(5,6).
- RES buses ext→int: Wind [5]→[4]; PV [4,6]→[3,5].
- Load fluctuation σ=0.10 (relative).
- Wind Weibull: λ=5.089, k=2.016; speeds: cut‑in=4, rated=12, cut‑out=25 (m/s).
- PV Beta: α=2.06, β=2.5; G_STC=1000 W/m².
- Branch status column: 10 (BR_STATUS).
- Functions: load_case6ww_int(), get_res_bus_indices(), find_branch_indices_for_pairs(), apply_topology(), build_G_B_operators(), get_operators_for_topology().
- All powers in p.u. (MW/MVAr ÷ baseMVA); cost scaling handled elsewhere.

## Model Config

`model_01.py` (GCNN predictor):
- Inputs: e_0_k,f_0_k [N_BUS,Cin]; pd,qd [N_BUS]; g_diag,b_diag [N_BUS]; g_ndiag,b_ndiag [N_BUS,N_BUS].
- Outputs: gen_out [N_GEN,2]=(PG,VG) and v_out [N_BUS,2]=(e,f) for physics loss.
- Implemented: 2×GraphConv → shared FC → two heads (gen, voltage).
- Pending: feature iteration (III.C eqs 8–25); z‑score normalization; physics losses L_PG, L_{Δ,P𝓧}.
- Next tasks: (1) adapter tensor loader (2) generator mask/A_g2b (3) shape unit test for both heads (4) construct_features loop (5) configurable depth/activation (6) PG bounds clamp.
- Risks: overfit on 6‑bus; scaling to 39‑bus needs sparsity; enforce index alignment assertions.


## To-do list (in order)

### 1. Network & config ✅ COMPLETE

- [x] Write `sample_config_model_01.py` (load case, compute basic metadata).
- [x] Decide 5 line contingencies and implement `apply_topology(ppc_int, topo_id)`.
- [x] **Migrate from case39 to case6ww** for faster AC-OPF solving.
- [x] Add `extract_gen_limits()` for generator bounds extraction.
- [x] Fix `build_G_B_operators()` for PyTorch tensor compatibility.

---

### 2. Scenario Generator ✅ COMPLETE

- [x] Implement scenario generator with fluctuation, RES sampling, target scaling, RES-as-negative-load, and `allow_negative_pd` flag.
- [x] Debug on 3 samples (case6ww): penetration (method 1) stable at ~30%, negative PD disabled, power factor correction applied.

---

### 3. OPF labeling ✅ COMPLETE

- [x] Implement `solve_ac_opf(ppc_base, pd, qd, topo_id)` (Pyomo+Gurobi) — using shared `src/helpers_ac_opf.py`.
- [x] Test on a few manually constructed scenarios — tested with 3 RES scenarios, all solve optimally.
- [x] **Dataset generation script** (`generate_dataset.py`) implemented with full pipeline.
- [x] **CPU optimization**: Configured Gurobi to use all 20 CPU threads for parallel solving.
- [x] **Complete 12k sample generation** (10k train + 2k test) — ✅ COMPLETED 2025-11-19.
- [x] Dataset saved to `gcnn_opf_01/data/` (4.46 MB, 96%+ success rate).

---

### 4. Model-Informed Feature Construction ✅ COMPLETE

- [x] Implement `construct_features()` according to III.C (Eqs. 16-25) — **feature_construction_model_01.py**.
- [x] Implement `construct_features_from_ppc()` wrapper with generator limits extraction.
- [x] Verify for one sample that features don't diverge and normalization works — **test_feature_construction.py** ✓.
- [x] Fix tensor indexing for generator bus operations.
- [ ] Decide: precompute $e\_0\_k$, $f\_0\_k$ for all samples into a new NPZ, or compute on the fly in Dataset.

---

### 5. PyTorch Dataset + DataLoader

- [ ] Implement `OPF6WWDataset` that reads NPZ (+ maybe calls `construct_features`).
- [ ] Implement z-score normalization (fit on training set, save mean/std).

---

### 6. Model wiring (GCNN\_OPF\_01) ✅ ARCHITECTURE COMPLETE

- [x] `GraphConv` implemented with physics-guided convolution.
- [x] `GCNN_OPF_01` with two-head architecture: gen_head [N_GEN,2] and v_head [N_BUS,2].
- [x] Physics-informed loss functions in **loss_model_01.py**:
  - [x] `build_A_g2b()`: generator-to-bus incidence matrix.
  - [x] `f_pg_from_v()`: compute PG from voltages using power flow equations.
  - [x] `correlative_loss_pg()`: L_supervised + κ·L_Δ,PG.
- [ ] Write unit test to check `model(e_0_k, f_0_k, pd, qd, ...)` returns correct shapes.

---

### 7. Training loop

- [ ] Implement basic training with $\mathcal{L}_{sup}$ (MSE) only.
- [ ] Monitor loss, maybe simple validation error.
- [ ] Integrate correlative loss $\mathcal{L}_{\Delta, PG}$ from `loss_model_01.py`.

---

### 8. Correlative loss ✅ IMPLEMENTED

- [x] Implement $\mathcal{L}_{PG}(\mathbf{v}_{pred})$ and $\mathcal{L}_{\Delta, P\mathcal{X}}$ — **loss_model_01.py**.
- [x] API: `correlative_loss_pg(gen_out, v_out, gen_labels, pd, G, B, gen_bus_indices, kappa=0.1)`.
- [ ] Fine-tune using correlative loss in training loop.

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
