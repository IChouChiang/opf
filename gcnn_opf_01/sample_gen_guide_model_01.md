# Dataset Structure for PG-GCNN (Case6ww, 5 topologies)

We will store the training data in **three NPZ files**:

---

## 1. `samples_train.npz`  
Contains **all training scenarios** after:
- scenario generation (RES + load fluctuation)
- N-1 topology sampling
- feature construction (e₀→eₖ, f₀→fₖ)
- OPF labels (PG_label, VG_label)

### Variables:
- `e_0_k`:        float32, shape **[N_train, N_BUS, k]**
- `f_0_k`:        float32, shape **[N_train, N_BUS, k]**
- `pd`:           float32, shape **[N_train, N_BUS]**
- `qd`:           float32, shape **[N_train, N_BUS]**
- `topo_id`:      int32,   shape **[N_train]**  # 0–4
- `pg_labels`:    float32, shape **[N_train, N_GEN]**
- `vg_labels`:    float32, shape **[N_train, N_GEN]**
- `gen_bus_map`:  int32,   shape **[N_GEN]**  # GEN → BUS index

### Notes:
- `k = 8` for your feature-construction depth.  
- `N_train = 10,000` recommended (paper mini-batch = 10).  
- All values are in **p.u.**

---

## 2. `samples_test.npz`  
Same structure as training set, but without generator masks (optional).

### Variables:
- `e_0_k`  
- `f_0_k`  
- `pd`  
- `qd`  
- `topo_id`  
- `pg_labels`  
- `vg_labels`

### Notes:
- `N_test = 2,000` recommended.

---

## 3. `topology_operators.npz`  
**Shared, topology-dependent physics tensors**, computed directly from `sample_config_model_01.apply_topology()`:

### Variables:
- `g_ndiag`: float32, shape **[5, N_BUS, N_BUS]**  
- `b_ndiag`: float32, shape **[5, N_BUS, N_BUS]**
- `g_diag`:  float32, shape **[5, N_BUS]**
- `b_diag`:  float32, shape **[5, N_BUS]**
- `topo_lines`: list of removed branch indices (len = 5)
- `N_BUS`, `N_GEN` (metadata)

### Notes:
- Index 0 = base topology; 1–4 = N-1 branches (case6ww: (5,2), (1,2), (2,3), (5,6)).
- These operators are fed to **GraphConv** during forward pass.

---

# Mini-batch Rule (From Paper)
- Each mini-batch = **10 samples**
- So DataLoader(batch_size=10, shuffle=True) is recommended.

---

# Summary of File Dimensions (Case6ww)
- N_BUS = 6
- N_GEN = 3
- k = 8 feature channels
- Topologies = 5

### shapes:
- `e_0_k`:  [10,000, 6, 8]
- `f_0_k`:  [10,000, 6, 8]
- `pd`:     [10,000, 6]
- `qd`:     [10,000, 6]
- `pg_labels`: [10,000, 3]
- `vg_labels`: [10,000, 3]
- `g_ndiag`:   [5, 6, 6]
- `gen_bus_map`: [3]

---

# Rationale
- **The model only needs** (e₀ₖ, f₀ₖ, pd, qd, topology operators).  
- **The loss function needs** PG_label and VG_label.  
- **Physics loss (ΔPG)** only uses PG, V_out, and bus indices.  
- **No need to store raw ppc** in the dataset.  
- **No need to store OPF duals, λ, μ, etc.**

This is the **cleanest, minimal, and fully paper-compatible** dataset format.

---

# Implementation Status (2025-11-19)

**Dataset generation script:** `generate_dataset.py` implemented and running.

**Configuration:**
- Training samples: 10,000 (RNG seed: 42)
- Test samples: 2,000 (RNG seed: 123)
- Feature iterations: k=8
- RES penetration: 30% target
- Solver: Gurobi with 20 CPU threads, 180s time limit, 3% MIP gap

**Output files** (saved to `gcnn_opf_01/data/`):
1. `samples_train.npz` — training set with features and labels
2. `samples_test.npz` — test set with features and labels  
3. `topology_operators.npz` — precomputed G/B operators for all 5 topologies
4. `norm_stats.npz` — z-score statistics (pd, qd, pg, vg means/stds)

**Progress:** Generation in progress (~30-35 min estimated runtime).

**Technical notes:**
- Skip-retry logic ensures exactly 12k feasible samples (ignores infeasible AC-OPF)
- Infeasibility rate: ~3-4% (typical for AC-OPF with N-1 contingencies)
- Checkpoints every 500 samples for monitoring
- NumPy 2.x compatibility via `.tolist()` → `np.array()` conversion


