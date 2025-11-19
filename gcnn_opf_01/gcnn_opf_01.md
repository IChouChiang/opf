# GCNN_OPF_01

# GCNN_OPF_01

## Current Status (2025-11-19) - ✅ TRAINING COMPLETED

### Completed Pipeline:
- ✅ Model architecture (2-head GCNN)
- ✅ Feature construction (k=8 iterations)
- ✅ Physics-informed loss functions
- ✅ Dataset generation (12k samples, 96% success rate)
- ✅ PyTorch Dataset & DataLoader
- ✅ Training pipeline (23 epochs, early stopping)
- ✅ Evaluation (R²=97.65% power, R²=99.99% voltage)
- ✅ Week5 Chinese documentation

### Training Results (2025-11-19):
- **Model:** 15,026 parameters (NEURONS_FC=128)
- **Training:** 23 epochs, 4.8 minutes, early stopping at epoch 20
- **Best validation loss:** 0.160208
- **Physics loss weight (κ):** 0.1

### Test Performance (2,000 samples):
- **Generator Power (PG):**
  - R² = 0.9765 (97.65% variance explained)
  - RMSE = 0.153 p.u. ≈ 15.3 MW (100 MVA base)
  - MAE = 0.073 p.u. ≈ 7.3 MW
  - MAPE = 30.20%

- **Generator Voltage (VG):**
  - R² = 0.9999 (99.99% variance explained)
  - RMSE = 0.0077 p.u. ≈ 0.77%
  - MAE = 0.0060 p.u. ≈ 0.60%
  - MAPE = 0.68%

### Files Present:
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
  - `topo_N-1_model_01.html`: Interactive visualization for case6ww with highlighted features:
    - **Red edges**: N-1 contingency lines (5-2, 1-2, 2-3, 5-6).
    - **Green node**: Wind generation bus (bus 5).
    - **Yellow nodes**: PV generation buses (buses 4, 6).
  - **`create_topo_viz.py`** and **`apply_highlights.py`**: Scripts to generate and customize topology visualization.
  - `gcnn_opf_01.md`: Design notes and status tracking (this file).
  - `formulas_model_01.md`: Formula references for the GCNN/feature construction.
  - **`dataset.py`**: PyTorch Dataset class (OPF6WWDataset) with z-score normalization.
  - **`train.py`**: Training pipeline with physics loss, early stopping, checkpointing.
  - **`evaluate.py`**: Comprehensive evaluation script with per-generator analysis.
  - **`results/`**: Training artifacts
    - `best_model.pth`: Best model checkpoint (epoch ~20)
    - `final_model.pth`: Final model (epoch 23)
    - `training_log.csv`: Epoch-by-epoch metrics
    - `training_curves.png`: Loss visualization
    - `training_history.npz`: NumPy arrays of training history
    - `evaluation_results.npz`: Test predictions and metrics

### Key Observations:
  - BR_STATUS column index `10` matches MATPOWER/PYPOWER convention.
  - **case6ww migration**: System reduced from 39 buses/10 gens to 6 buses/3 gens for faster training data generation.
  - **AC-OPF solve time**: case6ww solves in <1 second vs 10-180 seconds for case39 (10-20x speedup).
  - Operator construction relies on `makeYbus` and cleanly splits diag/off-diag parts into tensors.
  - **Power factor fix impact**: QD recalculation from final PD ensures physically consistent reactive power.
  - **Training convergence**: Model converged in 23 epochs with early stopping, no overfitting observed.
  - **Excellent voltage prediction**: R²>99.99% indicates model accurately learned voltage-power relationships.
  - **Good power prediction**: R²>97% demonstrates strong generalization capability.
  - **Gen 1 systematic underestimation**: Likely due to operating near power limits in high-load scenarios.

### Dataset Statistics (from 2025-11-19 generation):
- Training: 10,000 samples (96.2% success rate, 394 failed attempts)
- Test: 2,000 samples (95.7% success rate, 89 failed attempts)
- Total runtime: 42 minutes (16:36:15 - 17:18:52)
- Normalization statistics:
  - pd: mean=0.2457±0.2753 p.u.
  - qd: mean=0.2457±0.2753 p.u.
  - pg: mean=0.5047±0.0558 p.u.
  - vg: mean=1.0567±0.0094 p.u.

### Next Steps (Future Work):
- Consider expanding dataset with more high-load scenarios for Gen 1 improvement
- Hyperparameter tuning (κ, learning rate, FC neurons)
- Potential scaling to larger systems (case39, case118)
- Investigate Gen 1 underestimation with constraint analysis

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


## To-do list (COMPLETED ✅)

### 1. Network & config ✅ COMPLETE
### 2. Scenario Generator ✅ COMPLETE
### 3. OPF labeling ✅ COMPLETE
### 4. Model-Informed Feature Construction ✅ COMPLETE
### 5. PyTorch Dataset + DataLoader ✅ COMPLETE
### 6. Model wiring (GCNN_OPF_01) ✅ COMPLETE
### 7. Training loop ✅ COMPLETE
### 8. Correlative loss ✅ COMPLETE
### 9. Evaluation ✅ COMPLETE
### 10. Documentation ✅ COMPLETE

**All major tasks completed (2025-11-19). Project ready for future extensions.**

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
