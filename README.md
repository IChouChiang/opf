# Optimal Power Flow (OPF) Educational Project

Power systems optimization and machine learning study using Python, Pyomo, PYPOWER, Gurobi, and PyTorch.

## 🎯 Project Overview

Educational assignments progressing from DC Optimal Power Flow (Week 2) through ML-based prediction (Week 3) to AC Optimal Power Flow (Week 4).

**Key Technologies:** Pyomo, PYPOWER/MATPOWER, Gurobi, PyTorch, NumPy

**Environment:** `opf311` (Anaconda)  
**Current Phase:** Week 7-8 - Node-Wise Architecture & Generalization Testing

---

## 🚀 Recent Updates (Dec 2025)

### Node-Wise GCNN Architecture
- **Goal:** Achieve Inductive Generalization (train on one topology, test on others).
- **Method:** Removed the "Flattening" layer found in the original paper. Implemented a **Node-Wise Readout** where the same MLP is applied to every node independently.
- **Result:**
  - **Parameter Efficiency:** Reduced parameters from ~46k to **5,668** (~90% reduction).
  - **Seen Data:** Excellent performance (VG R² > 0.999).
  - **Unseen Data:** Successfully predicted Voltage physics (VG R² 0.65) but failed on Active Power (PG) due to the global nature of cost optimization.

### Configuration System
- **Legacy:** JSON config files in `gcnn_opf_01/configs/` with `model_type`: `"flattened"` (original) or `"nodewise"` (new).
- **Hydra Integration:** Unified configuration system in `configs/` directory:
  - `config.yaml` - Main configuration file
  - `model/` - Model-specific configurations (DNN, GCNN)
  - `data/` - Dataset configurations (case6, case39)
  - Command-line overrides: `python scripts/train.py model=dnn data=case39 train.max_epochs=50`
  - Test: `python tests/test_hydra_train.py` - Verifies Hydra configuration system integration

### Unified Model Refactoring (Dec 2025)
- **New Package:** `src/deep_opf/` - Unified deep learning framework for OPF
- **Models:** 
  - `AdmittanceDNN`: Fully connected network for flat feature input (legacy Model 03)
  - `GCNN`: Physics-guided graph convolutional network (legacy Model 01)
- **Data Loading:** 
  - `OPFDataset`: Unified PyTorch Dataset supporting 'flat' and 'graph' feature types
  - `OPFDataModule`: PyTorch Lightning DataModule for streamlined training
- **Configuration:** 
  - `configs/` - Hydra configuration system for model/data/training parameters
  - `scripts/train.py` - Unified training script with Hydra integration
- **Verification:** 
  - `tests/verify_models.py` - Comprehensive model validation script
  - `tests/test_hydra_train.py` - Hydra configuration system integration test

---

## 📁 Project Structure

```
opf/
├── weekly_assignments/ # Progressive learning modules
│   ├── Week2/          # DC-OPF: linear formulation, case9
│   ├── Week3/          # ML prediction: DCOPF → MLP, case118
│   ├── Week5/          # GCNN project documentation (Chinese)
│   ├── Week6/          # Advanced topics
│   └── Week7to8/       # Physics-Informed ML (Model 01 vs Model 03)
├── gcnn_opf_01/        # Physics-guided GCNN for OPF (case6ww)
│   ├── data/           # 12k samples (10k train, 2k test) [git-ignored]
│   ├── results/        # Training results & tuning [git-ignored]
│   ├── docs/           # Design documentation
│   │   ├── gcnn_opf_01.md           # Main project notes & status
│   │   ├── formulas_model_01.md     # Mathematical formulas
│   │   └── *.md                     # Other guides
│   ├── model_01.py                  # 2-head GCNN architecture
│   ├── loss_model_01.py             # Physics-informed loss functions
│   ├── feature_construction_model_01.py  # Model-informed features (Sec III-C)
│   ├── sample_config_model_01.py    # case6ww config & operators
│   ├── sample_generator_model_01.py # RES scenario generator
│   ├── config_model_01.py           # Dataclass configs
│   ├── train.py                     # Training pipeline
│   ├── evaluate.py                  # Model evaluation
│   └── tune_batch_size.py           # Hyperparameter tuning (with caching)
├── src/                # Reusable modules
│   ├── deep_opf/        # Unified deep learning framework
│   │   ├── data/        # Dataset and DataModule classes
│   │   │   ├── dataset.py      # OPFDataset (flat/graph features)
│   │   │   ├── datamodule.py   # OPFDataModule (Lightning)
│   │   │   └── __init__.py
│   │   ├── models/      # Neural network architectures
│   │   │   ├── dnn.py          # AdmittanceDNN (MLP)
│   │   │   ├── gcnn.py         # GCNN (graph convolution)
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── ac_opf_create.py       # Pyomo AbstractModel (Cartesian voltages)
│   └── helpers_ac_opf.py      # AC-OPF helpers (data prep, init, solve)
├── tests/              # Test harnesses and baselines
│   ├── verify_models.py       # DNN/GCNN model verification
│   ├── test_case39.py         # IEEE 39-bus AC-OPF
│   ├── test_case57.py         # IEEE 57-bus AC-OPF
│   ├── test_feature_construction.py  # Feature construction validation
│   ├── test_sample_generator.py      # Scenario generator + AC-OPF
│   ├── test_topology_outages.py      # N-1 contingency verification
│   ├── case39_baseline.py     # PYPOWER reference (39-bus)
│   └── case57_baseline.py     # PYPOWER reference (57-bus)
├── outputs/            # Generated files (git-ignored)
├── .github/
│   └── copilot-instructions.md
├── pyrightconfig.json
└── README.md
```

---

## 🚀 Quick Start

### Environment setup
```bash
conda activate opf311
```

### Week 4 AC-OPF (Current)
Run the AC-OPF test harnesses:
```bash
cd tests
python test_case39.py   # IEEE 39-bus
python test_case57.py   # IEEE 57-bus
```

Baseline comparison (PYPOWER):
```bash
python case39_baseline.py
python case57_baseline.py
```

---

## 📊 Week 4 Highlights (AC-OPF)

### Features
- **Cartesian voltage formulation:** Variables `e[i]` (real) and `f[i]` (imag) instead of polar Vm/Va
- **Fixed quadratic objective:** Minimize Σ(a·PG² + b·PG + c) with cost coefficients scaled for p.u. variables
- **Nonlinear power balance:** Bilinear constraints using admittance matrix G, B from PYPOWER's `makeYbus`
- **Voltage magnitude limits:** (Vmin)² �?e² + f² �?(Vmax)²
- **Gurobi NonConvex solver:** MIQCP with spatial branching, half CPU cores, 3-minute time limit, 3% MIP gap

### Shared helpers (src/helpers_ac_opf.py)
- `prepare_ac_opf_data(ppc)`: ext2int, Ybus→G/B, per-unit scaling, cost params
- `initialize_voltage_from_flatstart(instance, ppc_int)`: set e/f from Vm/Va
- `solve_ac_opf(ppc, verbose=True, time_limit=180, mip_gap=0.03, threads=None)`: build, init (PG/QG, slack fix), solve

### Results (tests/)
- **IEEE 39-bus:** 41872.30 $/hr (vs PYPOWER 41864.18, ~0.02% gap), ~2s solve
- **IEEE 57-bus:** 41770.00 $/hr (~1% gap), ~130s solve

### Technical Notes
- Cost scaling: For PG in per-unit, use `a = c2·baseMVA²`, `b = c1·baseMVA`, `c = c0` to preserve $/hr units
- Slack bus voltage fixed to eliminate rotational symmetry
- Generator PG/QG initialized from case data for warm start
- External 1-based bus/gen numbering in output (matches PYPOWER convention)

---

## 🧩 Dependencies

- `pyomo` �?optimization modeling
- `pypower` �?power flow cases and reference solver
- `gurobipy` �?nonconvex quadratic solver
- `torch` �?neural network training (Week 3)
- `numpy`, `matplotlib`

See `.github/copilot-instructions.md` for detailed architecture patterns and workflow.

---

## 📝 Development Notes

- **Type checking:** `pyrightconfig.json` configured; use `# pyright: reportAttributeAccessIssue=false` in Pyomo files
- **Units:** Always convert MW/MVAr to p.u. via `baseMVA` (typically 100.0)
- **MATPOWER compatibility:** Bus/gen/branch matrices follow MATPOWER column indexing (0-based in NumPy)

---

## 🧠 GCNN OPF Subproject (gcnn_opf_01/)

### Overview
Physics-guided Graph Convolutional Neural Network for optimal power flow prediction on **case6ww** (6-bus Wood & Wollenberg system).

### Architecture
- **Model:** 2×GraphConv �?shared FC �?two heads
  - `gen_head`: [N_GEN=3, 2] �?(PG, VG)
  - `v_head`: [N_BUS=6, 2] �?(e, f) for physics validation
- **Feature construction:** k=8 iterations of model-informed voltage estimation (Section III-C)
  - Iterative PG/QG computation with generator clamping (Eqs. 23-24)
  - Voltage updates via power flow equations (Eqs. 16-17, 19-22)
  - Voltage magnitude normalization (Eq. 25)
- **Loss:** L_supervised + κ·L_Δ,PG (correlative physics-informed loss)
  - Supervised: MSE on (PG, VG) predictions
  - Physics: MSE on power balance residuals using predicted voltages

### Key Files
- `feature_construction_model_01.py`: Implements iterative voltage estimation
- `loss_model_01.py`: Physics-informed loss functions
- `model_01.py`: GCNN architecture with GraphConv layers
- `sample_config_model_01.py`: case6ww operators (G, B matrices)
- `sample_generator_model_01.py`: RES scenario generator (wind/PV)

### Testing
```bash
# Feature construction test
python tests/test_feature_construction.py  # �?Validated [6,8] features, normalized voltages

# Scenario generation + AC-OPF
python tests/test_sample_generator.py      # �?3 scenarios, 30% RES, all optimal

# Topology verification
python tests/test_topology_outages.py      # �?N-1 contingencies verified

# Hydra configuration system integration test
python tests/test_hydra_train.py           # �?Verifies Hydra configuration for DNN and GCNN models
```

### Status (Completed 2025-11-25)
- ✅ Model architecture (2-head GCNN)
- ✅ Feature construction (k=8 iterations)
- ✅ Physics-informed loss functions
- ✅ Scenario generator (Gaussian load + Weibull wind + Beta PV)
- ✅ AC-OPF integration (using `src/helpers_ac_opf.py`)
- ✅ Dataset generation (12k samples, 96% success rate)
- ✅ **Hyperparameter tuning** (batch size optimization, 16 experiments)
- ✅ Training pipeline (35 epochs, early stopping, batch_size=6)
- ✅ Model evaluation (R²=98.21% for PG, R²=99.99% for VG)
- ✅ Probabilistic accuracy metrics (P_PG=38.45%, P_VG=14.80%)

---

## �?Completed Milestones

- [x] Week 2: DC-OPF with linear constraints, PTDF analysis
- [x] Week 3: ML-based OPF prediction (MLP: P_D �?P_G), 10k samples
- [x] Week 4: AC-OPF Cartesian formulation, Gurobi nonconvex solve, PYPOWER baseline validation
- [x] Week 5: GCNN-OPF complete pipeline
  - [x] Model architecture (2-head GCNN with physics-informed layers)
  - [x] Feature construction (k=8 iterations)
  - [x] Dataset generation (12k samples, 5 topologies, 50.7% RES penetration)
  - [x] **Hyperparameter optimization** (batch size tuning: 16 experiments, optimal=6)
  - [x] Training (35 epochs, physics-informed loss, early stopping)
  - [x] Evaluation (R²=98.21% for power, R²=99.99% for voltage)
  - [x] Probabilistic accuracy metrics (P_PG=38.45%, P_VG=14.80%)
  - [x] Chinese documentation (Week5/Week5.md)

---

## 📚 References

- MATPOWER documentation: https://matpower.org
- Pyomo: https://www.pyomo.org
- Gurobi NonConvex QCQP: https://www.gurobi.com/documentation/

------

## 🚀 Week 5 Highlights (GCNN-OPF)

### Training Results (Optimized with Batch Size Tuning)
- **Model:** 1000 neurons (FC layer), Two-Stage Training
- **Optimal Batch Size:** 24 (found via tuning for larger model)
- **Training:** 
  - Phase 1 (Supervised): 25 epochs
  - Phase 2 (Physics-Informed): 24 epochs
- **Physics loss weight (κ):** 1.0 (in Phase 2)

### Test Set Performance (2,000 samples)
- **Generator Power (PG):**
  - **Probabilistic Accuracy (Error < 1 MW): 98.42%**
  - R² = 0.9844
  - RMSE = 0.0069 p.u. (0.69 MW)
- **Generator Voltage (VG):**
  - **Probabilistic Accuracy (Error < 0.001 p.u.): 100.00%**
  - MAE = 0.0538 p.u. ≈ 5.4 MW
  - MAPE = 26.32%
  - **P_PG = 38.45%** (errors < 1 MW threshold)

- **Generator Voltage (VG):**
  - R² = 0.9999 (99.99% variance explained)
  - RMSE = 0.0086 p.u. ≈ 0.86%
  - MAE = 0.0059 p.u. ≈ 0.59%
  - MAPE = 0.56%
  - **P_VG = 14.80%** (errors < 0.001 p.u. threshold)

### Batch Size Tuning Summary
Conducted comprehensive 3-stage tuning (16 experiments total):
- **Optimal:** Batch size 6 (val_loss = 0.1460)
- **Key finding:** Small batches (2-8) significantly outperform large batches (256-1024) by 2.6-2.8x
- **Trade-off:** Large batches train 2-3x faster but sacrifice accuracy
- **Insight:** Physics-constrained GCNN benefits from frequent gradient updates with small batches

See `gcnn_opf_01/docs/gcnn_opf_01.md` for complete tuning results table.

### Dataset Details
- **System:** case6ww (6 buses, 3 generators)
- **Topologies:** 5 configurations (base + 4 N-1 contingencies)
- **RES Integration:** Wind (Weibull) at bus 5, PV (Beta) at buses 4 & 6
- **Target Penetration:** 50.7%
- **Training samples:** 10,000 (96.2% success rate)
- **Test samples:** 2,000 (95.7% success rate)

### Documentation
- Full Chinese documentation available in `Week5/Week5.md`
- Includes model architecture, sample generation, and training results

---

## 📚 Additional Documentation

- **Week5/Week5.md** - Comprehensive Chinese documentation of GCNN-OPF project
- **.github/copilot-instructions.md** - Development patterns and architecture guide
- **MAINTENANCE.md** - Change log and implementation notes
- **gcnn_opf_01/*.md** - Design documents, formulas, and guides

---

## 📝 References

- MATPOWER: https://matpower.org
- Pyomo: https://www.pyomo.org
- Gurobi: https://www.gurobi.com/documentation/
- Paper: "A Physics-Guided Graph Convolution Neural Network for Optimal Power Flow" (Gao et al.)
