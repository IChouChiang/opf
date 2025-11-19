# Maintenance Log

## 2025-11-19: GCNN Training & Evaluation Completion

**Achievement:** Completed full GCNN-OPF training pipeline with physics-informed loss and comprehensive evaluation.

**Implementation:**
1. **PyTorch Dataset & DataLoader** (`gcnn_opf_01/dataset.py`, `train.py`):
   - OPF6WWDataset class loads NPZ files with z-score normalization
   - Batch size=10 (paper recommendation)
   - Automatic topology operator loading from precomputed file

2. **Training Pipeline** (`gcnn_opf_01/train.py`):
   - Adam optimizer (lr=1e-3, weight_decay=1e-5)
   - Combined loss: L_supervised + κ·L_Δ,PG (κ=0.1)
   - Early stopping with patience=10
   - Model checkpointing (best and final)
   - Training curves visualization

3. **Training Results:**
   - Total epochs: 23 (early stopping at epoch 20)
   - Training time: ~4.8 minutes
   - Best validation loss: 0.160208
   - Final train loss: 0.176, validation loss: 0.171
   - No overfitting observed

4. **Evaluation** (`gcnn_opf_01/evaluate.py`):
   - Comprehensive metrics: MSE, RMSE, MAE, MAPE, R²
   - Per-generator analysis
   - Test set: 2,000 samples

5. **Test Set Performance:**
   - **Generator Power (PG):**
     - R² = 0.9765 (97.65% variance explained)
     - RMSE = 0.153 p.u. ≈ 15.3 MW
     - MAE = 0.073 p.u. ≈ 7.3 MW
     - MAPE = 30.20%
   - **Generator Voltage (VG):**
     - R² = 0.9999 (99.99% variance explained)
     - RMSE = 0.0077 p.u. ≈ 0.77%
     - MAE = 0.0060 p.u. ≈ 0.60%
     - MAPE = 0.68%

6. **Per-Generator Analysis (PG):**
   - Gen 0: MSE=0.0085 (best performance)
   - Gen 1: MSE=0.0438 (systematic underestimation)
   - Gen 2: MSE=0.0176 (good performance)

7. **Week5 Documentation:**
   - Comprehensive Chinese documentation in `Week5/Week5.md`
   - Covers model architecture, sample generation, training results
   - Includes all parameters, shapes, implementation files

**Files Created:**
- `gcnn_opf_01/dataset.py`: PyTorch Dataset implementation
- `gcnn_opf_01/train.py`: Training loop with physics loss
- `gcnn_opf_01/evaluate.py`: Comprehensive evaluation script
- `gcnn_opf_01/results/`: Training artifacts (best_model.pth, training_log.csv, etc.)
- `Week5/Week5.md`: Complete Chinese documentation

**Technical Notes:**
- Model converged well with physics-informed loss
- Voltage prediction extremely accurate (R²>99.99%)
- Power prediction shows good generalization (R²>97%)
- Gen 1 shows systematic underestimation, likely due to operating near limits
- Training speed: ~12.5 seconds per epoch on GPU

**Next Steps:**
- Consider expanding dataset with more high-load scenarios for Gen 1 improvement
- Potential scaling to larger systems (case39, case118)
- Hyperparameter tuning (κ, learning rate, FC neurons)

---

## 2025-11-19: Dataset Generation Pipeline (12k Samples)

**Achievement:** Implemented full dataset generation pipeline for gcnn_opf_01 with CPU optimization.

**Implementation:**
1. **Dataset Generation Script** (`gcnn_opf_01/generate_dataset.py`):
   - Precomputes topology operators for all 5 topologies (base + 4 N-1 contingencies)
   - Generates 10,000 training + 2,000 test samples
   - Pipeline: topology sampling → RES scenario → feature construction → AC-OPF → label extraction
   - Skip-retry logic ensures exactly 12k feasible solutions (ignores infeasible AC-OPF results)
   - Checkpoints every 500 samples for progress tracking
   - Computes z-score normalization statistics from training set only
   - Output: 4 NPZ files (`samples_train.npz`, `samples_test.npz`, `topology_operators.npz`, `norm_stats.npz`)

2. **CPU Optimization**:
   - Configured Gurobi to use all 20 CPU threads (`SOLVER_THREADS = multiprocessing.cpu_count()`)
   - Previous: half CPU cores → Current: full CPU power
   - Speed: ~5-6 samples/second with 20 threads
   - Estimated runtime: ~30-35 minutes for 12k samples

3. **Dataset Structure**:
   - Features: `e_0_k`, `f_0_k` [N_BUS=6, k=8] from model-informed feature construction
   - Inputs: `pd`, `qd` [N_BUS=6] demand scenarios (30% RES penetration)
   - Labels: `pg_labels` [N_GEN=3], `vg_labels` [N_GEN=3] from AC-OPF solutions
   - Topology: `topo_id` ∈ {0,1,2,3,4} sampled uniformly
   - Normalization: z-score statistics for pd, qd, pg, vg

**Technical Notes:**
- Infeasibility rate: ~3-4% (expected for AC-OPF with N-1 contingencies)
- NumPy 2.x compatibility: Using `.detach().cpu().tolist()` → `np.array()` workaround
- Device-aware feature construction: GPU for k-iteration, CPU for AC-OPF solving
- Power factor preservation: QD maintains consistent ratio with PD after RES injection
- Python executable path: `E:\DevTools\anaconda3\envs\opf311\python.exe`

**Status (2025-11-19 17:18):**
- ✅ Generation completed successfully
- Training: 10,000 samples (96.2% success rate, 394 failed attempts)
- Test: 2,000 samples (95.7% success rate, 89 failed attempts)
- Total runtime: 42 minutes (16:36:15 - 17:18:52)
- Output files:
  - `samples_train.npz`: 3.72 MB
  - `samples_test.npz`: 0.74 MB
  - `topology_operators.npz`: 1.64 KB
  - `norm_stats.npz`: 2.01 KB
- Normalization statistics:
  - pd: mean=0.2457±0.2753 p.u.
  - qd: mean=0.2457±0.2753 p.u.
  - pg: mean=0.5047±0.0558 p.u.
  - vg: mean=1.0567±0.0094 p.u.

**Files Modified:**
- `gcnn_opf_01/generate_dataset.py`: Full pipeline implementation
- `gcnn_opf_01/gcnn_opf_01.md`: Updated to-do list and status
- `README.md`: Updated GCNN status to show in-progress dataset generation

**Next Steps:**
- Monitor generation completion
- Verify dataset integrity (shapes, normalization statistics)
- Implement PyTorch Dataset loader
- Begin training loop with supervised + physics-informed loss

---

## 2025-11-19: GCNN Feature Construction & Physics Loss Implementation

**Achievement:** Completed model-informed feature construction (Section III-C) and physics-informed loss functions for gcnn_opf_01 subproject.

**Implementation:**
1. **Feature Construction Module** (`feature_construction_model_01.py`):
   - Implements k=8 iterations of voltage estimation using power flow equations (Eqs. 16-25)
   - `construct_features()`: Core iterative algorithm with PG/QG computation, generator clamping, voltage updates, and normalization
   - `construct_features_from_ppc()`: Convenience wrapper with automatic generator limits extraction
   - Returns e_0_k, f_0_k [N_BUS, k] tensors for model input

2. **Physics-Informed Loss** (`loss_model_01.py`):
   - `correlative_loss_pg()`: L_supervised + κ·L_Δ,PG (correlative physics loss)
   - `f_pg_from_v()`: Computes PG from predicted voltages using power flow equations (Eq. 8)
   - `build_A_g2b()`: Generator-to-bus incidence matrix for mapping

3. **Model Architecture Update** (`model_01.py`):
   - Two-head GCNN: gen_head [N_GEN,2]=(PG,VG) + v_head [N_BUS,2]=(e,f)
   - Enables supervised loss on generation and physics validation on voltages

4. **Data Utilities** (`sample_config_model_01.py`):
   - `extract_gen_limits()`: Extracts PMIN/PMAX/QMIN/QMAX from ppc_int['gen']
   - Fixed `makeYbus` matrix conversion: `np.asarray(Ybus.todense())` for PyTorch compatibility
   - Fixed tensor indexing for generator bus operations

**Testing:**
- `test_feature_construction.py`: ✓ Validates [6,8] feature shapes and voltage normalization (~1.0)
- `test_sample_generator.py`: ✓ 3 RES scenarios with AC-OPF integration
- `test_topology_outages.py`: ✓ N-1 contingency verification

**Files Added:**
- `gcnn_opf_01/feature_construction_model_01.py`
- `gcnn_opf_01/loss_model_01.py`
- `tests/test_feature_construction.py`
- Documentation: formulas_model_01.md, lossfunction.md, Feature Construction Guide

**Files Modified:**
- `gcnn_opf_01/model_01.py`: Two-head architecture
- `gcnn_opf_01/sample_config_model_01.py`: Generator limits + tensor fixes
- `gcnn_opf_01/gcnn_opf_01.md`: Updated to-do list (tasks 1-2, 4, 6, 8 complete)
- `README.md`: Added GCNN subproject section
- `.github/copilot-instructions.md`: Added GCNN workflow documentation

**Technical Notes:**
- NumPy 2.x compatibility: Use `np.asarray()` instead of `.todense()` for sparse matrix conversion
- Tensor indexing: Convert NumPy int arrays to `torch.tensor(..., dtype=torch.long)` for PyTorch indexing
- Feature normalization: Applied after each iteration to maintain unit voltage magnitude
- Generator clamping: Only applied at generator buses using `gen_bus_indices_tensor`

**Next Steps:**
- Dataset generation: 12k samples with precomputed features
- Training pipeline: Implement basic loop with supervised + correlative loss
- Model validation: Unit tests for forward pass shapes

---

## 2025-11-17: Week3.ipynb Warning Suppression

**Issue:** Week3 notebook produces warnings/errors that distract from current work on other subprojects (e.g., gcnn_opf_01). These include both runtime warnings and Pylance static analysis diagnostics (reportIndexIssue, reportCallIssue, reportOptionalOperand, etc.) - 30+ errors persisting even after pyrightconfig exclusion.

**Solution:** 
1. Added warning suppression cell at the top of `Week3/Week3.ipynb`:
   ```python
   # pyright: reportAttributeAccessIssue=false, reportIndexIssue=false, ...
   import warnings
   warnings.filterwarnings('ignore')
   ```
2. Excluded `Week3/` from Pylance in multiple config layers:
   - `pyrightconfig.json`: Added `"**/Week3/**"` and `"Week3/Week3.ipynb"` to exclude list
   - `.vscode/settings.json`: Added `python.analysis.exclude` and `python.analysis.ignore` for `**/Week3/**`

**Rationale:** Week3 is not the current focus; multi-layer exclusion ensures Pylance diagnostics don't appear regardless of VS Code configuration loading order.

**Files Modified:**
- `Week3/Week3.ipynb`: Added suppression directives and runtime warning filter at top
- `pyrightconfig.json`: Added Week3 paths to exclude list
- `.vscode/settings.json`: Added python.analysis.exclude and ignore rules

**Impact:** Pylance type-checking errors in Week3 should no longer appear in VS Code Problems panel. Requires window reload to take effect.

**Operational Note:** To avoid any remaining noise, do not open `Week3/Week3.ipynb` during current development on other subprojects. Keep VS Code set to show diagnostics for open files only (already configured).
