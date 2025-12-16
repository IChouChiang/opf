# Experiment Guide for deep_opf

This guide covers training, evaluation, and best practices for running experiments with the deep_opf library.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training](#training)
   - [Basic Training](#basic-training)
   - [Two-Stage Training](#two-stage-training-recommended)
   - [SSH Mode](#ssh-mode-remote-training)
3. [Evaluation](#evaluation)
   - [Seen vs Unseen Datasets](#seen-vs-unseen-datasets)
   - [Evaluation Commands](#evaluation-commands)
4. [Configuration Reference](#configuration-reference)
5. [Best Practices](#best-practices)
   - [Experiment Naming](#experiment-naming-convention)
   - [Directory Structure](#directory-structure)
   - [Hyperparameter Tracking](#hyperparameter-tracking)
6. [Example Workflows](#example-workflows)

---

## Quick Start

```powershell
# Activate environment
conda activate opf311

# Train a model (Case 39, GCNN, 50 epochs)
python scripts/train.py data=case39 model=gcnn train.max_epochs=50

# Evaluate the model
python scripts/evaluate.py data=case39 model=gcnn
```

---

## Training

### Basic Training

```powershell
python scripts/train.py [OPTIONS]
```

**Common Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `data=<name>` | Dataset config (`case6`, `case39`) | `case6` |
| `model=<name>` | Model config (`dnn`, `gcnn`) | `gcnn` |
| `train.max_epochs=<N>` | Maximum training epochs | `100` |
| `train.batch_size=<N>` | Batch size | `24` |
| `model.task.lr=<float>` | Learning rate | `0.001` |
| `model.task.kappa=<float>` | Physics loss weight (0=supervised only) | `0.1` |
| `hydra.run.dir=<path>` | Output directory | `outputs/<date>/<time>` |

**Example - Train GCNN on Case 39:**

```powershell
python scripts/train.py `
    data=case39 `
    model=gcnn `
    model.architecture.in_channels=10 `
    model.architecture.hidden_channels=32 `
    model.architecture.n_layers=4 `
    train.max_epochs=200 `
    hydra.run.dir=outputs/case39_gcnn_baseline
```

### Two-Stage Training (Recommended)

Two-stage training typically yields better results:

1. **Phase 1 (Supervised):** Train with `kappa=0.0` (no physics loss) using higher learning rate
2. **Phase 2 (Physics-Informed):** Fine-tune with `kappa>0` using lower learning rate

#### Phase 1: Supervised Pre-training

```powershell
python scripts/train.py `
    data=case39 `
    model=gcnn `
    model.architecture.in_channels=10 `
    model.architecture.hidden_channels=32 `
    model.architecture.n_layers=4 `
    model.task.lr=0.001 `
    model.task.kappa=0.0 `
    train.max_epochs=1000 `
    train.mode=ssh `
    hydra.run.dir=outputs/case39_phase1_supervised
```

#### Phase 2: Physics-Informed Fine-tuning

Use the best checkpoint from Phase 1 as warm start:

```powershell
python scripts/train.py `
    data=case39 `
    model=gcnn `
    model.architecture.in_channels=10 `
    model.architecture.hidden_channels=32 `
    model.architecture.n_layers=4 `
    model.task.lr=0.0001 `
    model.task.kappa=0.1 `
    train.max_epochs=200 `
    train.mode=ssh `
    "+train.warm_start_ckpt='lightning_logs/version_XX/checkpoints/epoch=YY-val_loss=0.0000.ckpt'" `
    hydra.run.dir=outputs/case39_phase2_physics
```

> **Note:** The `=` in checkpoint filenames requires quoting. Use `"+key='value'"` syntax.

### SSH Mode (Remote Training)

For remote/headless training without rich progress bars:

```powershell
python scripts/train.py train.mode=ssh [OTHER_OPTIONS]
```

SSH mode:
- Uses `LiteProgressBar` callback (writes to `current_status.txt`)
- Avoids TQDM/rich which can cause issues over SSH
- Logs completion status for monitoring

---

## Evaluation

### Seen vs Unseen Datasets

The Case 39 dataset has three splits:

| File | Samples | Topology | Purpose |
|------|---------|----------|---------|
| `samples_train.npz` | 10,000 | Base | Training |
| `samples_test.npz` | 2,000 | Base | Validation (seen topology) |
| `samples_unseen.npz` | 1,200 | Modified | Generalization test (unseen topology) |

**Default behavior:** `evaluate.py` uses `test_file` from config (defaults to `samples_unseen.npz`)

### Evaluation Commands

**Evaluate on unseen topology (default):**

```powershell
python scripts/evaluate.py `
    data=case39 `
    model=gcnn `
    model.architecture.in_channels=10 `
    model.architecture.hidden_channels=32 `
    model.architecture.n_layers=4 `
    "+ckpt_path='lightning_logs/version_XX/checkpoints/epoch=YY-val_loss=0.0000.ckpt'"
```

**Evaluate on seen topology (override test_file):**

```powershell
python scripts/evaluate.py `
    data=case39 `
    data.test_file=samples_test.npz `
    model=gcnn `
    model.architecture.in_channels=10 `
    model.architecture.hidden_channels=32 `
    model.architecture.n_layers=4 `
    "+ckpt_path='lightning_logs/version_XX/checkpoints/epoch=YY-val_loss=0.0000.ckpt'"
```

**Evaluate DNN model:**

```powershell
python scripts/evaluate.py `
    data=case39 `
    data.test_file=samples_test.npz `
    model=dnn `
    model.architecture.hidden_dim=800 `
    "+ckpt_path='lightning_logs/version_XX/checkpoints/epoch=YY-val_loss=0.0000.ckpt'"
```

**Auto-find best checkpoint:**

```powershell
python scripts/evaluate.py data=case39 model=gcnn
# Searches lightning_logs/ and outputs/ for most recent .ckpt
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Pacc PG** | % of PG predictions within 1 MW | Higher is better |
| **Pacc VG** | % of VG predictions within 0.001 p.u. | Higher is better |
| **R² PG/VG** | Coefficient of determination | → 1.0 |
| **RMSE** | Root mean squared error | → 0 |
| **Physics Violation** | Power balance mismatch (MW) | → 0 |

---

## Viewing Results

### Terminal Display

```powershell
# Show all evaluation results
python scripts/show_results.py

# Filter by dataset
python scripts/show_results.py --dataset case39

# Filter by model
python scripts/show_results.py --model gcnn

# Show training rows too
python scripts/show_results.py --phase all

# Compare models
python scripts/show_results.py --dataset case39 --compare
```

### Interactive HTML Dashboard

```powershell
python scripts/show_results.py --dataset case39 --html outputs/results_case39.html

# Then open in browser (start local server first):
cd outputs; python -m http.server 8080
# Open: http://localhost:8080/results_case39.html
```

**HTML Features:**
- 🔍 **Global Search**: Filter all columns instantly (top-right search box)
- ⬆️⬇️ **Column Sorting**: Click any header to sort
- 🔎 **Per-Column Filters**: Type in footer input boxes to filter specific columns
- 📊 **Export Buttons**: Copy, CSV, Excel, PDF, Print
- 🟢 **Best Values Highlighted**: Green = higher is better, Blue = lower is better

**Note:** Empty columns (training params in evaluation rows, metrics in training rows) are automatically hidden in HTML export.

### Understanding the CSV Structure

`experiments_log.csv` contains two row types:

| Row Type | Contains | Purpose |
|----------|----------|---------|
| **Training** | Architecture (layers, hidden_dim, lr, kappa), duration, best_loss | Track training runs |
| **Evaluation** | Metrics (R², Pacc, RMSE, Physics) | Track model performance |

Training and Evaluation rows share `model`, `dataset`, and `log_dir` columns but have different populated fields.

---

## Configuration Reference

### Model Architecture (GCNN)

```yaml
model:
  architecture:
    in_channels: 10       # Input features per node (typically n_gen)
    hidden_channels: 32   # GCN hidden dimension (8, 16, 32, 64)
    n_layers: 4           # Number of GCN layers (2, 4, 6)
    fc_hidden_dim: 1000   # FC layer dimension
    n_fc_layers: 2        # Number of FC layers
    dropout: 0.0          # Dropout rate
```

### Training Parameters

```yaml
train:
  max_epochs: 100
  batch_size: 24
  mode: standard          # 'standard' or 'ssh'
  warm_start_ckpt: null   # Path to checkpoint for warm start

model:
  task:
    lr: 0.001             # Learning rate
    kappa: 0.1            # Physics loss weight
    weight_decay: 0.0     # L2 regularization
```

### Dataset Configuration

```yaml
data:
  name: case39
  n_bus: 39
  n_gen: 10
  gen_bus_indices: [29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
  data_dir: legacy/gcnn_opf_01/data_matlab_npz
  train_file: samples_train.npz
  val_file: samples_test.npz
  test_file: samples_unseen.npz
```

---

## Best Practices

### Experiment Naming Convention

Use descriptive, hierarchical names:

```
outputs/
├── case39_gcnn_baseline/           # First attempt
├── case39_gcnn_ch32_L4/            # Architecture search
├── case39_gcnn_ch32_L4_phase1/     # Two-stage Phase 1
├── case39_gcnn_ch32_L4_phase2/     # Two-stage Phase 2
├── case39_dnn_hidden256/           # DNN comparison
└── case39_ablation_nokappa/        # Ablation study
```

**Naming pattern:** `{dataset}_{model}_{key_params}_{phase/variant}`

### Directory Structure

```
project/
├── configs/
│   ├── data/
│   │   ├── case6.yaml
│   │   └── case39.yaml
│   ├── model/
│   │   ├── dnn.yaml
│   │   └── gcnn.yaml
│   └── train/
│       └── default.yaml
├── lightning_logs/           # Auto-generated by PyTorch Lightning
│   ├── version_0/
│   │   └── checkpoints/
│   ├── version_1/
│   └── ...
├── outputs/                  # Custom experiment outputs (via hydra.run.dir)
│   ├── case39_phase1/
│   └── case39_phase2/
├── experiments_log.csv       # Experiment tracking (auto-generated)
└── current_status.txt        # SSH mode progress (auto-generated)
```

### Hyperparameter Tracking

All experiments are logged to `experiments_log.csv` with:

**Training columns:**
- timestamp, model, dataset, n_bus, n_gen, params
- hidden_dim, channels, in_channels, layers
- lr, kappa, weight_decay, batch_size, max_epochs
- warm_start, best_loss, duration, log_dir

**Evaluation columns:**
- R2_PG, R2_VG, Pacc_PG, Pacc_VG
- RMSE_PG, RMSE_VG, MAE_PG, MAE_VG
- Physics_Violation_MW, n_samples

### Tips for Reproducibility

1. **Always specify architecture params** when evaluating:
   ```powershell
   # Model was trained with these - must match for evaluation
   model.architecture.in_channels=10 `
   model.architecture.hidden_channels=32 `
   model.architecture.n_layers=4
   ```

2. **Use `hydra.run.dir`** for organized outputs:
   ```powershell
   hydra.run.dir=outputs/meaningful_name
   ```

3. **Track phase 1 checkpoint** for phase 2 warm start

4. **Run both seen and unseen evaluations** to measure generalization

---

## Example Workflows

### Workflow 1: Quick Baseline

```powershell
# Train
python scripts/train.py data=case39 model=gcnn train.max_epochs=100

# Evaluate (auto-finds checkpoint)
python scripts/evaluate.py data=case39 model=gcnn
```

### Workflow 2: Full Two-Stage Training

```powershell
# === PHASE 1: Supervised ===
python scripts/train.py `
    data=case39 `
    model=gcnn `
    model.architecture.in_channels=10 `
    model.architecture.hidden_channels=32 `
    model.architecture.n_layers=4 `
    model.task.lr=0.001 `
    model.task.kappa=0.0 `
    train.max_epochs=1000 `
    train.mode=ssh `
    hydra.run.dir=outputs/exp001_phase1

# Note the best checkpoint path from output, e.g.:
# Best model: lightning_logs/version_24/checkpoints/epoch=99-val_loss=0.0000.ckpt

# === PHASE 2: Physics Fine-tuning ===
python scripts/train.py `
    data=case39 `
    model=gcnn `
    model.architecture.in_channels=10 `
    model.architecture.hidden_channels=32 `
    model.architecture.n_layers=4 `
    model.task.lr=0.0001 `
    model.task.kappa=0.1 `
    train.max_epochs=200 `
    train.mode=ssh `
    "+train.warm_start_ckpt='lightning_logs/version_24/checkpoints/epoch=99-val_loss=0.0000.ckpt'" `
    hydra.run.dir=outputs/exp001_phase2

# === EVALUATION ===
# Seen topology
python scripts/evaluate.py `
    data=case39 data.test_file=samples_test.npz `
    model=gcnn model.architecture.in_channels=10 model.architecture.hidden_channels=32 model.architecture.n_layers=4 `
    "+ckpt_path='lightning_logs/version_25/checkpoints/epoch=67-val_loss=0.0000.ckpt'"

# Unseen topology
python scripts/evaluate.py `
    data=case39 `
    model=gcnn model.architecture.in_channels=10 model.architecture.hidden_channels=32 model.architecture.n_layers=4 `
    "+ckpt_path='lightning_logs/version_25/checkpoints/epoch=67-val_loss=0.0000.ckpt'"
```

### Workflow 3: Architecture Search

```powershell
# Test different channel sizes
foreach ($ch in 8, 16, 32, 64) {
    python scripts/train.py `
        data=case39 model=gcnn `
        model.architecture.in_channels=10 `
        model.architecture.hidden_channels=$ch `
        model.architecture.n_layers=2 `
        train.max_epochs=200 `
        hydra.run.dir="outputs/arch_search_ch${ch}"
}
```

### Workflow 4: Comparing Phase 1 vs Phase 2

```powershell
# After training both phases, compare on seen topology:

# Phase 1 (supervised only)
python scripts/evaluate.py data=case39 data.test_file=samples_test.npz `
    model=gcnn model.architecture.in_channels=10 model.architecture.hidden_channels=32 model.architecture.n_layers=4 `
    "+ckpt_path='lightning_logs/version_24/checkpoints/epoch=99-val_loss=0.0000.ckpt'"

# Phase 2 (physics-informed)  
python scripts/evaluate.py data=case39 data.test_file=samples_test.npz `
    model=gcnn model.architecture.in_channels=10 model.architecture.hidden_channels=32 model.architecture.n_layers=4 `
    "+ckpt_path='lightning_logs/version_25/checkpoints/epoch=67-val_loss=0.0000.ckpt'"

# Check experiments_log.csv for side-by-side comparison
```

---

## Troubleshooting

### Hydra Path Parsing Errors

**Problem:** `mismatched input '=' expecting <EOF>`

**Solution:** Checkpoint filenames contain `=`. Use `+` prefix and quotes:
```powershell
# Wrong
train.warm_start_ckpt=path/epoch=99-val_loss=0.0000.ckpt

# Correct
"+train.warm_start_ckpt='path/epoch=99-val_loss=0.0000.ckpt'"
```

### Model Architecture Mismatch

**Problem:** `size mismatch for gc_layers.0.W1.weight`

**Solution:** Ensure evaluation uses same architecture as training:
```powershell
model.architecture.in_channels=10 `
model.architecture.hidden_channels=32 `
model.architecture.n_layers=4
```

### Wrong Test Dataset

**Problem:** Evaluating on 1200 samples instead of 2000

**Solution:** Override test_file for seen topology:
```powershell
data.test_file=samples_test.npz
```

---

## Reference: Experiment Results Template

| Experiment | Model | Phase | Params | Epochs | LR | κ | R² PG | R² VG | Pacc PG | Physics |
|------------|-------|-------|--------|--------|-----|---|-------|-------|---------|---------|
| baseline | GCNN | 1 | 2.6M | 1000 | 1e-3 | 0.0 | 0.929 | 0.977 | 18.2% | 987 MW |
| baseline | GCNN | 2 | 2.6M | 200 | 1e-4 | 0.1 | 0.943 | 0.983 | 34.3% | 152 MW |
| baseline | DNN | 1 | 2.5M | 200 | 1e-3 | 0.0 | 0.234 | 0.659 | 2.7% | 164 MW |

---

*Last updated: December 2025*
