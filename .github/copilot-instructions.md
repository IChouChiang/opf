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

### 5. ML Models (Unified Pipeline)
- **GCNN:** Physics-Guided Graph Convolutional Neural Network
  - Uses graph structure (adjacency from Ybus) for message passing
  - Two heads: generator head (PG) + voltage head (VG)
  - Physics loss couples predictions via AC power flow equations
- **DNN:** Fully Connected Baseline (DeepOPF-style)
  - Flattened input: `[Pd, Qd, G, B]` concatenated
  - Good on fixed topologies, poor generalization to unseen topologies

**Code Location:** `src/deep_opf/` (unified), `legacy/` (old implementations)

## Development Workflows

### Environment Setup
- **Conda environment:** `opf311`, please always `conda activate opf311` if you found missing packages.
- **Python executable:** `E:\DevTools\anaconda3\envs\opf311\python.exe` (Windows/Alyce)
- Recreate via: `conda env create -f envs/environment.yml`
- **Key packages:** pyomo, pypower, torch, gurobipy, numpy, matplotlib, streamlit

### Experiment Automation (Preferred Workflow)
**Streamlit Dashboard:**
```bash
conda activate opf311
python -m streamlit run app/experiment_dashboard.py
```
- Settings tab: Configure model type (GCNN/DNN), architecture, training hyperparams
- Results tab: View experiment history from CSV logs
- Copy generated command to terminal for execution

**CLI Runner:**
```bash
# GCNN with physics-informed two-phase training
python scripts/run_experiment.py gcnn case39 --channels 8 --two-phase --kappa 0.1

# DNN baseline
python scripts/run_experiment.py dnn case39 --hidden_dim 128 --num_layers 3

# Dry-run to preview command without execution
python scripts/run_experiment.py gcnn case39 --dry-run
```

**Key Files:**
- `app/experiment_dashboard.py` - Streamlit web UI
- `scripts/run_experiment.py` - CLI experiment runner with GCNNConfig/DNNConfig dataclasses
- `src/deep_opf/utils/experiment_logger.py` - CSV logging with GCNN/DNN schemas
- Logs: `outputs/gcnn_experiments.csv`, `outputs/dnn_experiments.csv`

### Data Organization
```
data/
  ├── case39/           # IEEE 39-bus dataset
  │   ├── train.npz     # 10k training samples
  │   ├── seen.npz      # 2k validation (same topology)
  │   └── unseen.npz    # 1.2k test (different topologies)
  └── case6ww/          # Wood & Wollenberg 6-bus
      ├── train.npz     # 10k training samples
      └── seen.npz      # 2k validation
```

### File Organization
```
src/deep_opf/             # Unified ML Pipeline (Primary)
  ├── models/             # GCNN and DNN architectures
  ├── loss/               # Physics-informed loss functions
  ├── data/               # DataModule and loaders
  ├── utils/              # Helpers (logging, plotting)
  └── task.py             # PyTorch Lightning training task

legacy/                   # Old implementations (reference only)
  ├── gcnn_opf_01/        # Original GCNN implementation
  ├── dnn_opf_02/         # DNN variant 02
  ├── dnn_opf_03/         # DeepOPF-FT baseline
  └── weekly_assignments/ # Course notebooks

app/                      # Experiment Dashboard (Streamlit)
  └── experiment_dashboard.py

scripts/                  # Automation Scripts
  ├── run_experiment.py   # CLI experiment runner (recommended)
  ├── train.py            # Hydra-based training
  └── evaluate.py         # Model evaluation

data/                     # Dataset Files (git-ignored)
  ├── case39/             # IEEE 39-bus (10k/2k/1.2k samples)
  └── case6ww/            # 6-bus Wood & Wollenberg

outputs/                  # Experiment outputs
  ├── gcnn_experiments.csv  # GCNN results log (tracked)
  ├── dnn_experiments.csv   # DNN results log (tracked)
  └── */                    # Training artifacts (git-ignored)

lightning_logs/           # PyTorch Lightning checkpoints (git-ignored)
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

### Type Checking with Pyright
- Config: `pyrightconfig.json` at root
- **Disable Pyomo type warnings:** Add `# pyright: reportAttributeAccessIssue=false` to avoid false positives on Pyomo dynamic attributes

## Evaluation Metrics

### Supervised Metrics
- **R² (R-squared):** Coefficient of determination. 1.0 = perfect, 0.0 = predicts mean.
- **RMSE (Root Mean Square Error):** In p.u. units.
- **MAE (Mean Absolute Error):** In p.u. units.

### Probabilistic Accuracy (Pacc) - Eq. 37
$$P_{acc} = P(|pred - label| < \epsilon) \times 100\%$$

| Variable | Threshold ε | Physical Meaning |
|----------|-------------|------------------|
| PG | 0.01 p.u. | 1 MW (BaseMVA=100) |
| VG | 0.01 p.u. | 0.01 p.u. voltage |

**Computed over all elements:** For Case39 (10 gens × 2000 samples = 20,000 comparisons).
**1 MW threshold is strict:** Only ~0.32% of mean generator output (~311 MW).

### Physics Violation (Physics_MW)
$$Physics_{MW} = RMSE(A_{g2b} \cdot PG_{pred}, P_{from\_V}) \times BaseMVA$$

- **What it measures:** Mismatch between predicted PG and PG implied by predicted voltage via AC power flow equations.
- **Good values:** < 50 MW (physically consistent)
- **Bad values:** > 200 MW (predictions violate physics badly)
- **Interpretation:** 300 MW physics violation with 311 MW mean output = ~96% mismatch = very poor!

### Physics Loss Trade-off
Using κ > 0 in training:
- **Decreases supervised accuracy** (R², Pacc) - model compromises to satisfy both objectives
- **Decreases physics violation** - predictions are more physically consistent
- **Recommended:** Small κ (0.01-0.1) for balance; a solution with R²=0.95 + Physics_MW=50 is more useful than R²=0.99 + Physics_MW=300

### Constraint Violation Metrics
Measures percentage of predictions that violate physical operating limits from PYPOWER case data.

| Metric | Description |
|--------|-------------|
| **PG_Violation_Rate** | % of generator outputs outside [Pmin, Pmax] limits |
| **VG_Violation_Rate** | % of voltage predictions outside [Vmin, Vmax] limits |

**Physical Limits by Case:**
- **case39:** PG limits per generator (0 to 508-1100 MW), VG limits [0.94, 1.06] p.u.
- **case6ww:** PG limits per generator (0 to 150-200 MW), VG per-generator setpoints ±0.02 tolerance

**Good values:** < 5% violation rate
**Interpretation:** 0% = all predictions feasible; >10% = model needs constraint enforcement

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
