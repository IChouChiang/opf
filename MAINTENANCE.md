# Maintenance Log

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
