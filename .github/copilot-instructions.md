# AI Agent Instructions for OPF Project

**PLEASE CODE EVERYTHING IN A PROFESSIONAL MANNER. FOLLOW BEST PRACTICES.**

## Project Overview
This is an **Optimal Power Flow (OPF) educational project** using Python for power systems optimization and machine learning. The project has evolved from basic DC-OPF (Week 2) to AC-OPF (Week 4) and now focuses on **Physics-Informed Machine Learning** (Week 7-8).

**Key Technologies:** Pyomo, PYPOWER/MATPOWER, Gurobi, PyTorch, NumPy

## Critical Architecture Patterns

### 1. PYPOWER/MATPOWER Case Format
- Case data stored as **Python functions returning dict** (`ppc`) with keys: `baseMVA`, `bus`, `gen`, `branch`, `gencost`
- All matrix data uses **MATPOWER column indices** (1-indexed in documentation, 0-indexed in NumPy)
- **Units:** ALWAYS convert MW/MVAr to p.u. by dividing by `baseMVA` (typically 100.0)
- Examples: `Week2/case9.py`, `Week3/case118.py`

### 2. Data Transformation Pipeline (AC-OPF)
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
- **Fixed quadratic objective:** `min Σ(a[g]·PG[g]² + b[g]·PG[g] + c[g])` where a,b,c are pre-scaled for p.u. variables
- **Bounds from Params:** Use functions like `lambda m,g: (m.PGmin[g], m.PGmax[g])` for variable bounds

### 4. AC-OPF Cartesian Form
- **Variables:** `e[i]` (real) and `f[i]` (imag) voltage components instead of polar Vm/Va
  - Conversion: `e = Vm*cos(Va_rad)`, `f = Vm*sin(Va_rad)` via inline computation
- **Admittance matrix:** Use PYPOWER's `makeYbus` directly to get sparse Ybus, extract G (conductance) and B (susceptance)
  - Handles transformers (tap ratio, phase shift) and line charging automatically
- **Power balance:** Nonlinear constraints using `e[i]*sum_j(...)` and `f[i]*sum_j(...)` products
- **Voltage limits:** `Vmin[i]^2 <= e[i]^2 + f[i]^2 <= Vmax[i]^2`
- **Slack bus anchoring:** Fix slack bus `e[slack]` and `f[slack]` to eliminate rotational symmetry
- **Warm start:** Initialize PG/QG from case data to improve Gurobi convergence

### 5. ML Models (Week 7-8)
- **Model 01 (GCNN):** Physics-Guided Graph Convolutional Neural Network.
  - **Key Finding:** For larger grids (Case 39), "Detached Physics" (pure supervised training) currently outperforms "Active Physics" (gradient descent on physics loss) due to gradient domination issues.
  - **Files:** `gcnn_opf_01/`
- **Model 03 (DeepOPF-FT):** MLP Baseline.
  - **Architecture:** Fully Connected Network taking flattened `[Pd, Qd, G, B]` as input.
  - **Key Finding:** High accuracy on fixed topologies but poor physics consistency ($L_{phys} \approx 1.5$) and poor generalization to unseen topologies.
  - **Files:** `dnn_opf_03/`

## Development Workflows

### Environment Setup
- **Conda environment:** `opf311`, please always `conda activate opf311` if you found missing packages.
- **Python executable:** `E:\DevTools\anaconda3\envs\opf311\python.exe` (Windows/Alyce)
- Recreate via: `conda env create -f envs/environment.yml`
- **Key packages:** pyomo, pypower, torch, gurobipy, numpy, matplotlib

### File Organization
```
weekly_assignments/
  ├── Week2/
  │   ├── case9.py          # PYPOWER case files (data)
  │   └── Week2.ipynb       # Exploratory notebook
  ├── Week3/
  │   ├── Week3.ipynb       # ML training notebook
  │   ├── samples/          # Training data (git-ignored if large)
  │   └── results/          # Model outputs (git-ignored)
  ├── Week5/
  ├── Week6/
  └── Week7to8/

gcnn_opf_01/            # Model 01: Physics-guided GCNN
  ├── model_01.py                        # 2-head GCNN: gen_head + v_head
  ├── loss_model_01.py                   # Physics-informed loss (correlative)
  ├── feature_construction_model_01.py   # k-iteration voltage estimation
  ├── train.py                           # Training pipeline
  ├── evaluate.py                        # Model evaluation
  ├── data_matlab_npz/                   # Training data (git-ignored)
  └── results/                           # Training artifacts (git-ignored)

dnn_opf_03/             # Model 03: DeepOPF-FT Baseline (MLP)
  ├── model_03.py                        # MLP Architecture
  ├── dataset_03.py                      # Dataset loader
  ├── train_03.py                        # Training pipeline
  ├── evaluate_03.py                     # Evaluation with Physics Loss
  ├── loss_model_03.py                   # Physics loss calculation
  └── results/                           # Training artifacts (git-ignored)

src/
  ├── __init__.py             # Package initialization
  ├── ac_opf_create.py        # Pyomo AbstractModel (Cartesian voltages)
  ├── helpers_ac_opf.py       # AC-OPF helpers (data prep, init, solve)
  ├── topology_viz.py         # Static network visualization
  └── interactive_viz.py      # Interactive network visualization (PyVis)

tests/
  ├── test_case39.py                # IEEE 39-bus AC-OPF test harness
  ├── test_case57.py                # IEEE 57-bus AC-OPF test harness
  ├── test_feature_construction.py  # Feature construction validation
  ├── test_sample_generator.py      # Scenario generator + AC-OPF integration
  └── test_topology_outages.py      # N-1 contingency verification

outputs/                      # Generated files (git-ignored)
```

### Running AC-OPF (Week 4 / tests/)
**Quick run (uses shared helpers):**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pypower.api import case39
from helpers_ac_opf import solve_ac_opf

ppc = case39()
instance, result = solve_ac_opf(ppc, verbose=True)
```

**From command line:**
```bash
cd tests
python test_case39.py  # IEEE 39-bus
python test_case57.py  # IEEE 57-bus
```

**Shared helpers in `src/helpers_ac_opf.py`:**
- `prepare_ac_opf_data(ppc)`: ext2int → makeYbus → G/B extraction → cost scaling → Pyomo data dict
- `initialize_voltage_from_flatstart(instance, ppc_int)`: set e/f from Vm/Va, return slack bus index
- `solve_ac_opf(ppc, verbose=True, time_limit=180, mip_gap=0.03, threads=None)`: full pipeline with warm start & slack fixing

### Running GCNN (Model 01)
**Training:**
```bash
python gcnn_opf_01/train.py --config gcnn_opf_01/configs/case39_baseline.json
```

**Evaluation:**
```bash
python gcnn_opf_01/evaluate.py --model_path gcnn_opf_01/results/best_model.pth
```

### Running DeepOPF-FT (Model 03)
**Training:**
```bash
python dnn_opf_03/train_03.py --config dnn_opf_03/configs/exp_tiny_17k.json
```

**Evaluation (with Physics Loss):**
```bash
python dnn_opf_03/evaluate_03.py --model_path dnn_opf_03/results/best_model.pth --data_dir gcnn_opf_01/data_matlab_npz
```

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
6. **Missing gencost:** If no `gencost` array provided, `prepare_ac_opf_data()` raises `UserWarning` and uses defaults (c2=0.01, c1=40.0, c0=0.0 in p.u. scaling)
6. **Slack bus symmetry:** Fix slack bus voltage (e,f) to eliminate rotational degree of freedom in AC-OPF
7. **Duplicate helpers:** Use `src/helpers_ac_opf.py` shared functions instead of copy-pasting prepare/init/solve logic

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

Never commit anything without verification or permission. 
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

### When to Ask for Clarification
- **Excel data files** mentioned in Week 3 Description but not in repo – ask user for file location
- **Gurobi license** issues – ensure `gurobi.lic` is accessible
- **Chunk size** for training data generation – confirm memory limits before batch processing

### Long-Running Scripts
- **Progress Bars:** Always include a progress bar (e.g., `tqdm`) for scripts that take a long time to run (e.g., dataset generation, training).
- **Status Updates:** Print periodic status updates to the console to indicate progress and estimated time remaining.
