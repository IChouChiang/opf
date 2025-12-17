# Phase C: Sweep Mode Implementation Plan

## Overview
Add hyperparameter sweep support via comma-separated input values. When user enters `8,16,32` for channels, the system generates all combinations and runs them sequentially with full train→eval→CSV logging.

## Sweepable Parameters

### GCNN
| Parameter | CLI Flag | Example |
|-----------|----------|---------|
| channels | `--channels` | `8,16,32` |
| n_layers | `--n_layers` | `1,2,3` |
| fc_hidden_dim | `--fc_hidden_dim` | `128,256,512` |
| n_fc_layers | `--n_fc_layers` | `1,2` |
| batch_size | `--batch_size` | `32,64` |
| max_epochs | `--max_epochs` | `50,100` |
| kappa | `--kappa` | `0.1,0.5,1.0` |

### DNN
| Parameter | CLI Flag | Example |
|-----------|----------|---------|
| hidden_dim | `--hidden_dim` | `128,256,512` |
| num_layers | `--num_layers` | `2,3,4` |
| batch_size | `--batch_size` | `32,64,128` |
| max_epochs | `--max_epochs` | `50,100` |

## Implementation Tasks

### 1. Update `scripts/run_experiment.py`
- [ ] Add `parse_sweep_param(value_str)` → returns list of values
- [ ] Add `expand_combinations(params_dict)` → returns list of param dicts
- [ ] Add `is_sweep_mode(args)` → checks if any param has commas
- [ ] Modify main flow: if sweep mode, loop over combinations
- [ ] Add progress display: `[1/6] channels=8, batch_size=32`

### 2. Update `app/experiment_dashboard.py`
- [ ] Change sweepable params from `st.number_input` to `st.text_input`
- [ ] Add `parse_sweep_value(text)` for validation
- [ ] Add sweep detection and display: `"⚠️ Sweep: 3×2 = 6 runs"`
- [ ] Update command generation to pass comma-separated values

### 3. Validation Rules
- Split by `,` and optional whitespace
- Each value must be valid number (int or float as appropriate)
- Show error if invalid format

## Example Flow

**User input in dashboard:**
```
channels: 8, 16
batch_size: 32, 64
```

**Generated command:**
```bash
python scripts/run_experiment.py gcnn case39 --channels 8,16 --batch_size 32,64
```

**run_experiment.py execution:**
```
🔄 Sweep mode: 4 experiments
[1/4] channels=8, batch_size=32
  Training... ✓
  Evaluating seen... ✓
  Evaluating unseen... ✓
  Logged to CSV ✓
[2/4] channels=8, batch_size=64
  ...
[3/4] channels=16, batch_size=32
  ...
[4/4] channels=16, batch_size=64
  ...
✅ Sweep complete: 4 experiments logged
```

## Files to Modify
1. `scripts/run_experiment.py` - Core sweep logic
2. `app/experiment_dashboard.py` - UI changes for sweep input

## Testing
```bash
# Single run (existing behavior)
python scripts/run_experiment.py gcnn case39 --channels 8 --dry-run

# Sweep run
python scripts/run_experiment.py gcnn case39 --channels 8,16 --batch_size 32,64 --dry-run
```
