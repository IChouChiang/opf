# AI Agent Instructions for OPF Project

## Project Overview
This is an **Optimal Power Flow (OPF) educational project** using Python for power systems optimization and machine learning. Weekly assignments progress from DC-OPF (Week 2) to ML-based OPF prediction (Week 3) to AC-OPF (Week 4).

**Key Technologies:** Pyomo, PYPOWER/MATPOWER, Gurobi, PyTorch, NumPy

## Critical Architecture Patterns

### 1. PYPOWER/MATPOWER Case Format
- Case data stored as **Python functions returning dict** (`ppc`) with keys: `baseMVA`, `bus`, `gen`, `branch`, `gencost`
- All matrix data uses **MATPOWER column indices** (1-indexed in documentation, 0-indexed in NumPy)
- **Units:** ALWAYS convert MW/MVAr to p.u. by dividing by `baseMVA` (typically 100.0)
- Examples: `Week2/case9.py`, `Week3/case118.py`

### 2. Data Transformation Pipeline
```
PYPOWER case dict → case_to_col() → Uniform p.u. arrays → Pyomo AbstractModel instance
```
- `case_to_col()`: Converts `ppc` to **1-D float64 arrays** with standardized naming (e.g., `bus_Pd`, `gen_Pmax`)
- **Always scale to p.u.:** Generator limits, branch ratings, costs scaled by baseMVA powers
- Polynomial cost coefficients: `c_k_pu = c_k / baseMVA^k` where k is the power degree

### 3. Pyomo Modeling Conventions
- **AbstractModel pattern:** Define model structure once, instantiate with different data
- **BuildAction for derived sets:** Use `pyo.BuildAction` to populate sets like `GENS_AT_BUS` after data loading
- **Flexible cost functions:** Support both polynomial (model=2) and piecewise linear (model=1) from `gencost`
  - Polynomial: `FCOST[g] == sum(cost_coeff[g,k] * PG[g]**k for k in range(n))`
  - Piecewise: `pyo.Piecewise` with SOS2 formulation for convex costs
- **Bounds from Params:** Use functions like `lambda m,g: (m.PGmin[g], m.PGmax[g])` for variable bounds

### 4. AC-OPF Cartesian Form (Week 4)
- **Variables:** `e[i]` (real) and `f[i]` (imag) voltage components instead of polar Vm/Va
  - Conversion: `e = Vm*cos(Va_rad)`, `f = Vm*sin(Va_rad)` via `vm_va_to_e_f.py`
- **Admittance matrix:** Use `ppc_to_g_b()` to extract G (conductance) and B (susceptance) from PYPOWER's `makeYbus`
  - Handles transformers (tap ratio, phase shift) and line charging automatically
- **Power balance:** Nonlinear constraints using `e[i]*sum_j(...)` and `f[i]*sum_j(...)` products
- **Voltage limits:** `Vmin[i]^2 <= e[i]^2 + f[i]^2 <= Vmax[i]^2`

### 5. ML Workflow (Week 3)
- **Training data:** Chunked `.npz` files in `Week3/samples/` containing:
  - `pd_chunk_NNNNNN.npz`: Input demand scenarios (p.u.)
  - `labels_chunk_NNNNNN.npz`: Optimal generator outputs from DCOPF solutions
- **Model:** PyTorch MLP mapping `P_D` → `P_G` (stored in `Week3/results/`)
- **Normalization:** `norm_stats.npz` stores mean/std for input/output scaling

## Development Workflows

### Environment Setup
- **Conda environment:** `opf311` (shared across devices)
- Recreate via: `conda env create -f envs/environment.yml`
- **Key packages:** pyomo, pypower, torch, gurobipy, numpy, matplotlib

### File Organization
```
WeekN/
  ├── caseX.py          # PYPOWER case files (data)
  ├── WeekN.ipynb       # Exploratory notebook
  ├── *.py              # Reusable utility modules
  ├── samples/          # Training data (git-ignored if large)
  └── results/          # Model outputs (git-ignored)
```

### Running DCOPF
1. Import case: `from Week2.case9 import case9; ppc = case9()`
2. Build abstract model: `m = build_dc_opf_model()`
3. Prepare instance data: `data_dict, meta = ppc_to_pyomo_data_for_create_instance(ppc)`
4. Create instance: `instance = m.create_instance(data=data_dict)`
5. Solve: `solver = pyo.SolverFactory('gurobi'); result = solver.solve(instance)`

### Type Checking with Pyright
- Config: `pyrightconfig.json` at root
- **Disable Pyomo type warnings:** Add `# pyright: reportAttributeAccessIssue=false` to avoid false positives on Pyomo dynamic attributes

## Project-Specific Rules

### When Working with Units
- **NEVER mix MW and p.u.** – always convert MW to p.u. using `baseMVA`
- Generator costs in **currency units** (e.g., $/h) stay as-is for polynomial constant terms
- Angles: **MATPOWER uses degrees**, NumPy trig uses **radians** – always convert via `np.deg2rad()`

### When Adding Constraints
- Use Pyomo `Constraint(rule=...)` with functions, NOT inline lambda for complex expressions
- For multi-bus sums (e.g., power balance), iterate over `m.GENS_AT_BUS[i]` not all generators

### When Debugging Solvers
- Check `result.solver.termination_condition` (should be `optimal`)
- Extract values: `pyo.value(instance.PG[g])` NOT direct access
- Infeasible models: Print constraint values with `instance.constraint_name.pprint()`

### When Processing Results
- **Extract voltage from e/f:** `Vm = sqrt(e^2 + f^2)`, `Va_deg = atan2(f, e) * 180/pi`
- Match generator indices to bus indices via `gen_bus` array
- Line flows computed using PTDF (DC) or full power flow equations (AC)

## Common Pitfalls

1. **Off-by-one errors:** Python uses 0-indexing; MATPOWER docs use 1-indexing for column references
2. **Forgetting baseMVA scaling:** Generator `Pmax` in case file is MW, needs `/baseMVA` for p.u.
3. **Sparse vs dense matrices:** `ppc_to_g_b()` returns dense by default; use `return_sparse=True` for large cases
4. **Pyomo instance vs model:** ALWAYS call `create_instance()` before solving; AbstractModel is just a template
5. **Gencost row order:** Must match `ppc["gen"]` row order (one cost per generator)

## Quick Reference

### Extract Data from PPC
```python
from Week4.case_to_col import case_to_col
store = case_to_col(ppc)  # Returns dict of 1-D float64 arrays
Pd = store["bus_Pd"]      # Active demand in p.u. per bus
Pmax = store["gen_Pmax"]  # Generator max in p.u. per generator
```

### Build Admittance Matrix
```python
from Week4.ppc_to_g_b import ppc_to_g_b
G, B = ppc_to_g_b(ppc)  # Dense float64 arrays (n_bus × n_bus)
```

### Initialize Voltage Variables
```python
from Week4.vm_va_to_e_f import vm_va_to_e_f
e_init, f_init = vm_va_to_e_f(store["bus_Vm"], store["bus_Va"])
# Set in Pyomo: instance.e[bus_id].value = e_init[idx]
```

## When to Ask for Clarification
- **Excel data files** mentioned in Week 3 Description but not in repo – ask user for file location
- **Gurobi license** issues – ensure `gurobi.lic` is accessible
- **Chunk size** for training data generation – confirm memory limits before batch processing
