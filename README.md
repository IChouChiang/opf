# Optimal Power Flow (OPF) Educational Project

Power systems optimization and machine learning study using Python, Pyomo, PYPOWER, Gurobi, and PyTorch.

## 🎯 Project Overview

Educational assignments progressing from DC Optimal Power Flow (Week 2) through ML-based prediction (Week 3) to AC Optimal Power Flow (Week 4).

**Key Technologies:** Pyomo, PYPOWER/MATPOWER, Gurobi, PyTorch, NumPy

**Environment:** `opf311` (Anaconda)

---

## 📁 Project Structure

```
opf/
├─ Week2/              # DC-OPF: linear formulation, case9
├─ Week3/              # ML prediction: DCOPF → MLP, case118
│   ├─ samples/        # Training data (chunked .npz)
│   └─ results/        # Trained models
├─ Week4/              # AC-OPF: nonlinear formulation, case39
│   ├─ ac_opf_create.py       # Pyomo AbstractModel (Cartesian voltages)
│   ├─ test.py                # Solver harness (Gurobi NonConvex)
│   └─ case39_baseline.py     # PYPOWER runopf reference
├─ .github/
│   └─ copilot-instructions.md
├─ pyrightconfig.json
└─ README.md
```

---

## 🚀 Quick Start

### Environment setup
```bash
conda activate opf311
```

### Week 4 AC-OPF (Current)
Run the IEEE 39-bus AC-OPF model:
```bash
python Week4/test.py
```

Baseline comparison (PYPOWER):
```bash
python Week4/case39_baseline.py
```

---

## 📊 Week 4 Highlights (AC-OPF)

### Features
- **Cartesian voltage formulation:** Variables `e[i]` (real) and `f[i]` (imag) instead of polar Vm/Va
- **Fixed quadratic objective:** Minimize Σ(a·PG² + b·PG + c) with cost coefficients scaled for p.u. variables
- **Nonlinear power balance:** Bilinear constraints using admittance matrix G, B from PYPOWER's `makeYbus`
- **Voltage magnitude limits:** (Vmin)² ≤ e² + f² ≤ (Vmax)²
- **Gurobi NonConvex solver:** MIQCP with spatial branching, half CPU cores, 3-minute time limit, 3% MIP gap

### Results (IEEE 39-bus)
- **Objective:** 41872.30 $/hr (vs PYPOWER 41864.18, ~0.02% difference)
- **Total generation:** 62.98 p.u., **Demand:** 62.54 p.u., **Losses:** 0.44 p.u.
- **Voltage range:** 1.010–1.052 p.u. (all within limits)
- **Solve time:** ~2 seconds (optimal within tolerance)

### Technical Notes
- Cost scaling: For PG in per-unit, use `a = c2·baseMVA²`, `b = c1·baseMVA`, `c = c0` to preserve $/hr units
- Slack bus voltage fixed to eliminate rotational symmetry
- Generator PG/QG initialized from case data for warm start
- External 1-based bus/gen numbering in output (matches PYPOWER convention)

---

## 🧩 Dependencies

- `pyomo` — optimization modeling
- `pypower` — power flow cases and reference solver
- `gurobipy` — nonconvex quadratic solver
- `torch` — neural network training (Week 3)
- `numpy`, `matplotlib`

See `.github/copilot-instructions.md` for detailed architecture patterns and workflow.

---

## 📝 Development Notes

- **Type checking:** `pyrightconfig.json` configured; use `# pyright: reportAttributeAccessIssue=false` in Pyomo files
- **Units:** Always convert MW/MVAr to p.u. via `baseMVA` (typically 100.0)
- **MATPOWER compatibility:** Bus/gen/branch matrices follow MATPOWER column indexing (0-based in NumPy)

---

## ✅ Completed Milestones

- [x] Week 2: DC-OPF with linear constraints, PTDF analysis
- [x] Week 3: ML-based OPF prediction (MLP: P_D → P_G), 10k samples
- [x] Week 4: AC-OPF Cartesian formulation, Gurobi nonconvex solve, PYPOWER baseline validation

---

## 📚 References

- MATPOWER documentation: https://matpower.org
- Pyomo: https://www.pyomo.org
- Gurobi NonConvex QCQP: https://www.gurobi.com/documentation/

------

## 🧭 1. The baseline situation

**Devices:**

- 🖥️ *Alyce (Windows 11)* — main workstation, VS Code
- 💻 *Chromebook (Crostini Linux)* — lightweight remote editing (vim / Jupyter)

**Environment:**
 `opf311` (Anaconda) — shared libs for OPF, Pyomo, Gurobi, NumPy, PyTorch, etc.

**Work pattern:**
 Weekly tasks from your tutor, sometimes connected, sometimes independent.

------

## 🗂️ 2. Recommended project layout

Here’s a versioned, sync-friendly structure you can push to GitHub safely:

```
opf/
│
├─ envs/
│   └─ environment.yml           ← conda env spec (recreate opf311)
│
├─ notebooks/
│   ├─ week02/
│   │   └─ week02.ipynb
│   ├─ week03/
│   │   └─ week03.ipynb
│   ├─ shared/
│   │   └─ experiments.ipynb     ← optional common scratchpad
│
├─ src/
│   ├─ __init__.py
│   ├─ dcopf_utils.py            ← reusable helper functions
│   └─ ml_utils.py
│
├─ data/
│   ├─ raw/                      ← never commit heavy data; use .gitignore
│   └─ processed/
│
├─ models/                       ← trained NN checkpoints (usually git-ignored)
│
├─ .vscode/                      ← editor settings (OK to sync)
├─ .gitignore
├─ pyproject.toml or pyrightconfig.json
├─ README.md                     ← short intro, env usage, workflow
└─ requirements.txt or environment.yml
```

🟢 **Good habits**

- Keep each week’s notebook in its own folder, versioned in git.
- Put reusable code (plots, DCOPF solvers, data loaders) in `src/`.
- Large data or model files → `.gitignore` (sync through Drive or Git LFS if needed).
- Use `envs/environment.yml` to reproduce your conda setup on any machine.

------

## 🧩 3. About environment files

### 🧱 Conda (`environment.yml`)

Create it once on Alyce:

```bash
conda env export --name opf311 --no-builds > envs/environment.yml
```

Then on Chromebook:

```bash
conda env create -f envs/environment.yml
```

or update:

```bash
conda env update -f envs/environment.yml
```

This file **is safe and useful to commit** — it only lists package names & versions, no paths.

### 🧾 Alternatively: pip

If you sometimes use plain pip:

```bash
pip freeze > requirements.txt
```

But for multi-platform reproducibility, `environment.yml` is better.

------

## 🌐 4. GitHub synchronization strategy

- **Push/pull workflow**

  - On Alyce: regular development, commit & push
  - On Chromebook: `git pull` to update

- **.gitignore** example:

  ```
  # ignore large or transient data
  data/raw/
  models/
  .ipynb_checkpoints/
  __pycache__/
  *.log
  ```

- Never push sensitive files: license keys, `.env` with API secrets, etc.

- Optionally create branches for bigger tasks (e.g., `feature-week5-nn`).

------

## ⚙️ 5. VS Code + Vim consistency

- Keep `.vscode/settings.json` synced — both machines can reuse lint/formatter rules.
- On Chromebook, lightweight editing via `vim` or `jupyter` is fine; your structure doesn’t rely on VS Code features.

------

## ☁️ 6. Data & model handling

GitHub has size limits (100 MB per file, 1 GB total recommended).
 So:

- Save large simulation results or neural-network checkpoints to Google Drive or your router’s SSD (mounted via SMB/NFS).
- Store only small metadata (e.g., `metadata.json`, logs) in GitHub.

------

## 🧠 7. Long-term best practices

| Goal                 | Tool / Method                            |
| -------------------- | ---------------------------------------- |
| Reproducible env     | `environment.yml` pinned versions        |
| Cross-device editing | GitHub + consistent folder names         |
| Clean code reuse     | move helper functions → `src/`           |
| Weekly progress      | separate `weekXX` folders + README notes |
| Safe syncing         | .gitignore large files                   |
| Documentation        | Markdown readme per week if necessary    |

------

## ✅ TL;DR — Best practice summary

- Keep **one conda env (`opf311`)** shared across devices via `environment.yml`.
- Organize weekly notebooks under `notebooks/weekXX/`.
- Place reusable code in `src/`.
- Commit `.vscode/`, `.gitignore`, `environment.yml`, and `.md` docs to GitHub.
- Exclude large data/models.
- Use Drive or LFS for big outputs.
- Rebuild env on Chromebook via `conda env create -f envs/environment.yml`.

------

Would you like me to show you an **example `.gitignore` and `environment.yml`** tailored for your OPF + Pyomo + Gurobi + NN workflow? It’d fit perfectly with this structure.