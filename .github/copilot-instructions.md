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

### 2. Data Transformation Pipeline (Week 4 Current Approach)
```
PYPOWER case dict → ext2int → makeYbus → extract G,B → dict params → Pyomo instance
```
- **Direct from PYPOWER:** Use `ext2int()` for internal numbering, `makeYbus()` for admittance matrix
- **Always scale to p.u.:** Generator limits (Pmin/Pmax/Qmin/Qmax), demands (Pd/Qd) divided by `baseMVA`
- **Cost coefficient scaling for p.u. variables:**
  - Original MATPOWER polynomial: `f(P_MW) = c2·P_MW² + c1·P_MW + c0`
  - With PG in p.u.: use `a = c2·baseMVA²`, `b = c1·baseMVA`, `c = c0` to preserve $/hr units
- **Voltage initialization:** `e = Vm·cos(Va_rad)`, `f = Vm·sin(Va_rad)` computed inline

### 3. Pyomo Modeling Conventions
- **AbstractModel pattern:** Define model structure once, instantiate with different data
- **BuildAction for derived sets:** Use `pyo.BuildAction` to populate sets like `GENS_AT_BUS` after data loading
- **Fixed quadratic objective (Week 4):** `min Σ(a[g]·PG[g]² + b[g]·PG[g] + c[g])` where a,b,c are pre-scaled for p.u. variables
- **Bounds from Params:** Use functions like `lambda m,g: (m.PGmin[g], m.PGmax[g])` for variable bounds

### 4. AC-OPF Cartesian Form (Week 4)
- **Variables:** `e[i]` (real) and `f[i]` (imag) voltage components instead of polar Vm/Va
  - Conversion: `e = Vm*cos(Va_rad)`, `f = Vm*sin(Va_rad)` via inline computation
- **Admittance matrix:** Use PYPOWER's `makeYbus` directly to get sparse Ybus, extract G (conductance) and B (susceptance)
  - Handles transformers (tap ratio, phase shift) and line charging automatically
- **Power balance:** Nonlinear constraints using `e[i]*sum_j(...)` and `f[i]*sum_j(...)` products
- **Voltage limits:** `Vmin[i]^2 <= e[i]^2 + f[i]^2 <= Vmax[i]^2`
- **Slack bus anchoring:** Fix slack bus `e[slack]` and `f[slack]` to eliminate rotational symmetry
- **Warm start:** Initialize PG/QG from case data to improve Gurobi convergence

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

### Running AC-OPF (Week 4)
1. Import case: `from pypower.api import case39; ppc = case39()`
2. Build abstract model: `m = ac_opf_create()`
3. Prepare instance data: `data, ppc_int = prepare_ac_opf_data(ppc)`  (see `Week4/test.py`)
4. Create instance: `instance = m.create_instance(data=data)`
5. Initialize voltages and fix slack bus (see test.py for details)
6. Solve: `solver = pyo.SolverFactory('gurobi'); result = solver.solve(instance, tee=True)`
   - Gurobi options: `NonConvex=2`, `Threads=half_cpus`, `TimeLimit=180`, `MIPGap=0.03`

### Running DCOPF (Week 2)
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
3. **Cost coefficient scaling:** Use `a = c2·baseMVA²`, `b = c1·baseMVA`, `c = c0` for quadratic objective with PG in p.u. to preserve $/hr units
4. **Pyomo instance vs model:** ALWAYS call `create_instance()` before solving; AbstractModel is just a template
5. **Gencost row order:** Must match `ppc["gen"]` row order (one cost per generator)
6. **Slack bus symmetry:** Fix slack bus voltage (e,f) to eliminate rotational degree of freedom in AC-OPF

## Quick Reference

### Load PYPOWER Case and Build Ybus
```python
from pypower.api import case39
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus

ppc = case39()
ppc_int = ext2int(ppc)
Ybus, Yf, Yt = makeYbus(ppc_int['baseMVA'], ppc_int['bus'], ppc_int['branch'])
G = Ybus.real.toarray()  # Conductance matrix
B = Ybus.imag.toarray()  # Susceptance matrix
```

### Initialize Voltage Variables
```python
import numpy as np

# From case data (already in ppc_int after ext2int)
Vm = ppc_int['bus'][:, 7]       # Voltage magnitude (p.u.)
Va_deg = ppc_int['bus'][:, 8]   # Voltage angle (degrees)
Va_rad = np.deg2rad(Va_deg)

# Cartesian components
e_init = Vm * np.cos(Va_rad)
f_init = Vm * np.sin(Va_rad)
# Set in Pyomo: instance.e[i].value = e_init[i]
```

### Cost Coefficient Scaling for Quadratic Objective
```python
# Given MATPOWER gencost polynomial: f(P_MW) = c2*P_MW^2 + c1*P_MW + c0
# For Pyomo with PG in p.u., use:
baseMVA = ppc['baseMVA']
a = c2 * (baseMVA ** 2)  # quadratic coefficient
b = c1 * baseMVA         # linear coefficient
c = c0                   # constant term
# Objective: min sum(a[g]*PG[g]^2 + b[g]*PG[g] + c[g])  → yields $/hr
```

## Git Commit Message Convention

Follow **Conventional Commits** style with prefixes:

- `feat:` New feature or functionality (e.g., "feat: add AC-OPF Cartesian voltage formulation")
- `fix:` Bug fix or correction (e.g., "fix: correct cost coefficient scaling for p.u. variables")
- `docs:` Documentation updates (e.g., "docs: update README with Week 4 results")
- `refactor:` Code restructuring without changing behavior (e.g., "refactor: simplify objective to fixed quadratic")
- `chore:` Maintenance, cleanup, dependencies (e.g., "chore: remove unused utility files from Week4")
- `test:` Adding or updating tests (e.g., "test: add baseline validation for case39")
- `perf:` Performance improvements (e.g., "perf: optimize Ybus extraction")
- `style:` Code style/formatting (e.g., "style: fix indentation in test.py")

**Format:** `<type>: <description>` (lowercase, imperative mood, no period)

**Examples:**
- `feat: implement slack bus voltage fixing for AC-OPF`
- `fix: align generator indexing with PYPOWER 1-based convention`
- `docs: add cost scaling formula to copilot instructions`
- `chore: delete case_to_col, ppc_to_g_b, vm_va_to_e_f utilities`

## When to Ask for Clarification
- **Excel data files** mentioned in Week 3 Description but not in repo – ask user for file location
- **Gurobi license** issues – ensure `gurobi.lic` is accessible
- **Chunk size** for training data generation – confirm memory limits before batch processing
