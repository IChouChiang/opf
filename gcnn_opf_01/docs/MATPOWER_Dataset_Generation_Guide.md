# MATPOWER Dataset Generation Guide

> **Purpose**: Complete formula reference and data structure specification for generating GCNN-OPF training datasets using MATPOWER in MATLAB. This document extracts all mathematical formulas from the Python implementation for migration to MATLAB.

---

## Table of Contents

1. [Data Structure Specification](#1-data-structure-specification)
2. [Renewable Energy Modeling](#2-renewable-energy-modeling)
3. [Load Fluctuation Modeling](#3-load-fluctuation-modeling)
4. [Topology Handling (N-1 Contingencies)](#4-topology-handling-n-1-contingencies)
5. [Feature Construction Algorithm](#5-feature-construction-algorithm)
6. [AC-OPF Label Extraction](#6-ac-opf-label-extraction)
7. [Normalization Statistics](#7-normalization-statistics)
8. [Output File Format](#8-output-file-format)
9. [Recommended Parameters](#9-recommended-parameters)

---

## 1. Data Structure Specification

### 1.1 Sample Data Arrays

Each sample requires the following data:

| Field | Shape | Data Type | Unit | Description |
|-------|-------|-----------|------|-------------|
| `e_0_k` | `[N_BUS, k]` | float32 | - | Real voltage features from k iterations |
| `f_0_k` | `[N_BUS, k]` | float32 | - | Imaginary voltage features from k iterations |
| `pd` | `[N_BUS]` | float32 | p.u. | Active power demand (after RES injection) |
| `qd` | `[N_BUS]` | float32 | p.u. | Reactive power demand (after adjustment) |
| `topo_id` | scalar | int32 | - | Topology index (0 = base, 1-4 = N-1 contingencies) |
| `pg_labels` | `[N_GEN]` | float32 | p.u. | OPF solution: generator active power |
| `vg_labels` | `[N_GEN]` | float32 | p.u. | OPF solution: generator voltage magnitude |

### 1.2 Topology Operators (Precomputed Per Topology)

| Field | Shape | Data Type | Description |
|-------|-------|-----------|-------------|
| `g_ndiag` | `[N_TOPO, N_BUS, N_BUS]` | float32 | Off-diagonal conductance matrix |
| `b_ndiag` | `[N_TOPO, N_BUS, N_BUS]` | float32 | Off-diagonal susceptance matrix |
| `g_diag` | `[N_TOPO, N_BUS]` | float32 | Diagonal of conductance matrix |
| `b_diag` | `[N_TOPO, N_BUS]` | float32 | Diagonal of susceptance matrix |
| `gen_bus_map` | `[N_GEN]` | int32 | 0-based bus index for each generator |

### 1.3 Admittance Matrix Decomposition

From MATPOWER's `makeYbus`:

```matlab
[Ybus, Yf, Yt] = makeYbus(baseMVA, bus, branch);
G = real(full(Ybus));   % Conductance matrix [N_BUS, N_BUS]
B = imag(full(Ybus));   % Susceptance matrix [N_BUS, N_BUS]

% Decompose into diagonal and off-diagonal parts
g_diag = diag(G);           % [N_BUS, 1]
b_diag = diag(B);           % [N_BUS, 1]
g_ndiag = G - diag(g_diag); % [N_BUS, N_BUS], zeros on diagonal
b_ndiag = B - diag(b_diag); % [N_BUS, N_BUS], zeros on diagonal
```

---

## 2. Renewable Energy Modeling

### 2.1 Wind Power Model (Weibull Distribution + Turbine Curve)

**Wind Speed Sampling (Weibull Distribution)**:

$$
f(v) = \frac{k}{\lambda} \left( \frac{v}{\lambda} \right)^{k-1} e^{-\left( \frac{v}{\lambda} \right)^k}, \quad v \ge 0
$$

**Default Parameters**:
- $\lambda = 5.089$ (scale parameter, m/s)
- $k = 2.016$ (shape parameter)

**MATLAB Implementation**:
```matlab
% Sample wind speed from Weibull distribution
v = wblrnd(lambda_wind, k_wind, [n_wind_buses, 1]);
```

**Wind Turbine Power Curve**:

$$
CF_{wind}(v) = 
\begin{cases}
0, & v < v_{cut-in} \\
\left( \frac{v - v_{cut-in}}{v_{rated} - v_{cut-in}} \right)^3, & v_{cut-in} \le v < v_{rated} \\
1, & v_{rated} \le v < v_{cut-out} \\
0, & v \ge v_{cut-out}
\end{cases}
$$

**Default Parameters**:
- $v_{cut-in} = 4.0$ m/s
- $v_{rated} = 12.0$ m/s  
- $v_{cut-out} = 25.0$ m/s

**MATLAB Implementation**:
```matlab
function cf = wind_power_curve(v, v_cut_in, v_rated, v_cut_out)
    cf = zeros(size(v));
    
    % Region 2: Cubic power curve
    mask1 = (v >= v_cut_in) & (v < v_rated);
    cf(mask1) = ((v(mask1) - v_cut_in) / (v_rated - v_cut_in)).^3;
    
    % Region 3: Rated power
    mask2 = (v >= v_rated) & (v < v_cut_out);
    cf(mask2) = 1.0;
end
```

### 2.2 PV Power Model (Beta Distribution + Linear Irradiance Curve)

**Solar Irradiance Sampling (Beta Distribution)**:

$$
f(G) = \frac{1}{B(\alpha, \beta)} G^{\alpha - 1} (1 - G)^{\beta - 1}, \quad 0 \le G \le 1
$$

**Default Parameters**:
- $\alpha = 2.06$
- $\beta = 2.5$
- $G_{STC} = 1000$ W/m²

**MATLAB Implementation**:
```matlab
% Sample normalized irradiance from Beta distribution
s_norm = betarnd(alpha_pv, beta_pv, [n_pv_buses, 1]);
s = s_norm * G_STC;  % Scale to W/m²
```

**PV Power Curve** (Linear with irradiance):

$$
CF_{PV}(s) = \min\left( \frac{s}{G_{STC}}, 1.0 \right)
$$

**MATLAB Implementation**:
```matlab
function cf = pv_power_curve(s, G_STC)
    cf = min(s / G_STC, 1.0);
    cf = max(cf, 0.0);  % Ensure non-negative
end
```

### 2.3 RES Penetration Scaling

The RES power is scaled to achieve a target penetration level:

$$
P_{RES,scaled} = P_{RES,avail} \times \frac{\rho_{target} \times P_{load,total}}{P_{RES,avail,total}}
$$

Where:
- $\rho_{target}$ = target RES penetration (e.g., 0.30 for 30%)
- $P_{load,total} = \sum_i PD_i^{raw}$ (total load after fluctuation, before RES)
- $P_{RES,avail,total} = \sum_i P_{RES,avail,i}$ (total available RES power)

**Nameplate Capacity**: The nameplate capacity at each RES bus is set to the base load at that bus:

$$
P_{nameplate,i} = PD_{base,i}
$$

**Available RES Power**:

$$
P_{RES,avail,i} = CF_i \times P_{nameplate,i}
$$

---

## 3. Load Fluctuation Modeling

### 3.1 Active Power Fluctuation

$$
PD_i^{raw} = PD_{base,i} \times (1 + \sigma_{rel} \times \varepsilon_i)
$$

Where:
- $\varepsilon_i \sim \mathcal{N}(0, 1)$ (standard normal)
- $\sigma_{rel} = 0.10$ (10% relative standard deviation)

**MATLAB Implementation**:
```matlab
noise = randn(N_BUS, 1);
pd_raw = PD_base .* (1 + sigma_rel * noise);
pd_raw = max(pd_raw, 0);  % Clip negative loads
```

### 3.2 Reactive Power (Power Factor Preservation)

To maintain consistent power factor after load fluctuation and RES injection:

$$
\text{power\_factor\_ratio}_i = \frac{QD_{base,i}}{PD_{base,i} + \epsilon}
$$

$$
QD_i = \text{power\_factor\_ratio}_i \times PD_i
$$

Where $\epsilon = 10^{-8}$ prevents division by zero.

### 3.3 Final Demand After RES Injection

$$
PD_i = PD_i^{raw} - P_{RES,scaled,i}
$$

$$
QD_i = \text{power\_factor\_ratio}_i \times PD_i
$$

> **Important**: RES is treated as **negative load** (demand reduction), not generation.

---

## 4. Topology Handling (N-1 Contingencies)

### 4.1 N-1 Contingency Definition

For each topology, specify which branch(es) to remove using 1-based bus pairs:

```matlab
% Example for IEEE 39-bus system
topology_pairs = {
    [],           % Topology 0: Base case (no outage)
    [6, 7],       % Topology 1: Remove line between buses 6-7
    [14, 13],     % Topology 2: Remove line between buses 14-13
    [2, 3],       % Topology 3: Remove line between buses 2-3
    [21, 22],     % Topology 4: Remove line between buses 21-22
};

% Unseen Topologies (for Zero-Shot Generalization Test)
topology_unseen = {
    [23, 24],     % Unseen 1: Remove line between buses 23-24
    [26, 27],     % Unseen 2: Remove line between buses 26-27
    [2, 25],      % Unseen 3: Remove line between buses 2-25
};
```

### 4.2 Applying Branch Outage

**MATPOWER**: Set `branch(idx, BR_STATUS) = 0` to disable a line.

```matlab
function mpc_topo = apply_topology(mpc_base, branch_pairs)
    mpc_topo = mpc_base;
    
    for i = 1:size(branch_pairs, 1)
        f_bus = branch_pairs(i, 1);
        t_bus = branch_pairs(i, 2);
        
        % Find branch row (1-based MATPOWER convention)
        mask = (mpc_topo.branch(:, 1) == f_bus & mpc_topo.branch(:, 2) == t_bus) | ...
               (mpc_topo.branch(:, 1) == t_bus & mpc_topo.branch(:, 2) == f_bus);
        
        % Set BR_STATUS (column 11) to 0
        mpc_topo.branch(mask, 11) = 0;
    end
end
```

### 4.3 Rebuilding Admittance Matrix

After topology change, **recompute** `Ybus` using `makeYbus`:

```matlab
mpc_topo = apply_topology(mpc_base, branch_pairs);
[Ybus, ~, ~] = makeYbus(mpc_topo);
% Then decompose into g_diag, b_diag, g_ndiag, b_ndiag
```

---

## 5. Feature Construction Algorithm

### 5.1 Overview

The model-informed feature construction performs **k iterations** of physics-guided voltage updates. This encodes both topology and load information into feature vectors.

**Default**: $k = 10$ iterations

### 5.2 Initialization

$$
e_i^{(0)} = 1.0, \quad f_i^{(0)} = 0.0 \quad \forall i \in \{1, \ldots, N_{bus}\}
$$

### 5.3 Iterative Update (Equations 16-25 from Paper)

For each iteration $l = 1, \ldots, k$:

**Step 1: Compute Power Injections**

Using AC power flow equations in Cartesian form:

$$
PG_i = PD_i + e_i \sum_j G_{ij} e_j - e_i \sum_j B_{ij} f_j + f_i \sum_j G_{ij} f_j + f_i \sum_j B_{ij} e_j
$$

$$
QG_i = QD_i - f_i \sum_j G_{ij} e_j + f_i \sum_j B_{ij} f_j + e_i \sum_j G_{ij} f_j - e_i \sum_j B_{ij} e_j
$$

In matrix form:
```matlab
Ge = G * e;  Gf = G * f;  Be = B * e;  Bf = B * f;
PG_bus = pd + e .* Ge - e .* Bf + f .* Gf + f .* Be;
QG_bus = qd - f .* Ge + f .* Bf + e .* Gf - e .* Be;
```

**Step 2: Clamp Generator Powers (Equations 23-24)**

$$
PG_i = \max\left\{ \min(PG_i, \overline{PG_i}), \underline{PG_i} \right\}
$$

$$
QG_i = \max\left\{ \min(QG_i, \overline{QG_i}), \underline{QG_i} \right\}
$$

> **Important**: Apply clamping **only at generator buses**, using the generator limits from `gen(:, [PMAX, PMIN, QMAX, QMIN])`.

```matlab
% Only clamp at generator buses
PG_gen = PG_bus(gen_bus_indices);
QG_gen = QG_bus(gen_bus_indices);

PG_gen_clamped = max(min(PG_gen, PG_max), PG_min);
QG_gen_clamped = max(min(QG_gen, QG_min), QG_max);

PG_bus(gen_bus_indices) = PG_gen_clamped;
QG_bus(gen_bus_indices) = QG_gen_clamped;
```

**Step 3: Compute Effective Demand**

After clamping, recalculate effective demand:

```matlab
pd_eff = PG_bus - (e .* Ge - e .* Bf + f .* Gf + f .* Be);
qd_eff = QG_bus - (e .* Gf - e .* Be - f .* Ge + f .* Bf);
```

**Step 4: Compute Aggregation Terms (Equations 19-22)**

$$
\alpha_i = \sum_{j \neq i} G_{ij}^{ndiag} e_j - \sum_{j \neq i} B_{ij}^{ndiag} f_j
$$

$$
\beta_i = \sum_{j \neq i} G_{ij}^{ndiag} f_j + \sum_{j \neq i} B_{ij}^{ndiag} e_j
$$

$$
\delta_i = -PD_i^{eff} - (e_i^2 + f_i^2) \cdot G_{ii}
$$

$$
\lambda_i = -QD_i^{eff} - (e_i^2 + f_i^2) \cdot B_{ii}
$$

```matlab
alpha = g_ndiag * e - b_ndiag * f;
beta  = g_ndiag * f + b_ndiag * e;
s = e.^2 + f.^2;
delta = -pd_eff - s .* g_diag;
lambda = -qd_eff - s .* b_diag;
```

**Step 5: Voltage Update (Equations 16-17)**

$$
e_i^{(l+1)} = \frac{\delta_i \alpha_i - \lambda_i \beta_i}{\alpha_i^2 + \beta_i^2 + \epsilon}
$$

$$
f_i^{(l+1)} = \frac{\delta_i \beta_i + \lambda_i \alpha_i}{\alpha_i^2 + \beta_i^2 + \epsilon}
$$

```matlab
denom = alpha.^2 + beta.^2 + eps;  % eps = 1e-8
e_next = (delta .* alpha - lambda .* beta) ./ denom;
f_next = (delta .* beta + lambda .* alpha) ./ denom;
```

**Step 6: Voltage Magnitude Normalization (Equation 25)**

$$
e_i \leftarrow \frac{e_i}{\sqrt{e_i^2 + f_i^2 + \epsilon}}, \quad f_i \leftarrow \frac{f_i}{\sqrt{e_i^2 + f_i^2 + \epsilon}}
$$

```matlab
v_mag = sqrt(e_next.^2 + f_next.^2 + eps);
e_next = e_next ./ v_mag;
f_next = f_next ./ v_mag;
```

**Step 7: Store Features**

```matlab
e_features(:, l) = e_next;
f_features(:, l) = f_next;

% Update for next iteration
e = e_next;
f = f_next;
```

### 5.4 Complete MATLAB Implementation

```matlab
function [e_0_k, f_0_k] = construct_features(pd, qd, G, B, ...
    g_ndiag, b_ndiag, g_diag, b_diag, ...
    gen_bus_indices, PG_min, PG_max, QG_min, QG_max, k, eps)
    
    if nargin < 14, k = 10; end
    if nargin < 15, eps = 1e-8; end
    
    N_BUS = length(pd);
    
    % Initialize
    e = ones(N_BUS, 1);
    f = zeros(N_BUS, 1);
    
    e_0_k = zeros(N_BUS, k);
    f_0_k = zeros(N_BUS, k);
    
    for iter = 1:k
        % Step 1: Power injection
        Ge = G * e;  Gf = G * f;  Be = B * e;  Bf = B * f;
        PG_bus = pd + e .* Ge - e .* Bf + f .* Gf + f .* Be;
        QG_bus = qd - f .* Ge + f .* Bf + e .* Gf - e .* Be;
        
        % Step 2: Clamp generator powers
        PG_bus(gen_bus_indices) = max(min(PG_bus(gen_bus_indices), PG_max), PG_min);
        QG_bus(gen_bus_indices) = max(min(QG_bus(gen_bus_indices), QG_max), QG_min);
        
        % Step 3: Effective demand
        pd_eff = PG_bus - (e .* Ge - e .* Bf + f .* Gf + f .* Be);
        qd_eff = QG_bus - (e .* Gf - e .* Be - f .* Ge + f .* Bf);
        
        % Step 4: Aggregation terms
        alpha = g_ndiag * e - b_ndiag * f;
        beta  = g_ndiag * f + b_ndiag * e;
        s = e.^2 + f.^2;
        delta = -pd_eff - s .* g_diag;
        lambda_ = -qd_eff - s .* b_diag;  % 'lambda' is MATLAB keyword
        
        % Step 5: Voltage update
        denom = alpha.^2 + beta.^2 + eps;
        e_next = (delta .* alpha - lambda_ .* beta) ./ denom;
        f_next = (delta .* beta + lambda_ .* alpha) ./ denom;
        
        % Step 6: Normalize
        v_mag = sqrt(e_next.^2 + f_next.^2 + eps);
        e_next = e_next ./ v_mag;
        f_next = f_next ./ v_mag;
        
        % Step 7: Store
        e_0_k(:, iter) = e_next;
        f_0_k(:, iter) = f_next;
        
        e = e_next;
        f = f_next;
    end
end
```

---

## 6. AC-OPF Label Extraction

### 6.1 Solving AC-OPF with MATPOWER

```matlab
% Prepare case with modified demands
mpc = mpc_topo;
mpc.bus(:, PD) = pd * baseMVA;  % Convert p.u. to MW
mpc.bus(:, QD) = qd * baseMVA;  % Convert p.u. to MVAr

% Solve AC-OPF
results = runopf(mpc, mpoption('verbose', 0));

% Check success
if results.success ~= 1
    % Skip this sample or retry
end
```

### 6.2 Extracting Labels

```matlab
% PG labels (in p.u.)
pg_labels = results.gen(:, PG) / baseMVA;  % [N_GEN, 1]

% VG labels (voltage magnitude at generator buses)
gen_bus_indices = results.gen(:, GEN_BUS);  % 1-based bus indices
vg_labels = results.bus(gen_bus_indices, VM);  % [N_GEN, 1]
```

**Important MATPOWER Column Indices**:
- `GEN_BUS = 1` (generator bus number)
- `PG = 2` (active power output, MW)
- `VM = 8` (voltage magnitude, p.u.)
- `PD = 3`, `QD = 4` (active/reactive demand, MW/MVAr)

---

## 7. Normalization Statistics

### 7.1 Z-Score Normalization

Compute from **training set only**:

$$
y_{norm} = \frac{y - \mu}{\sigma}
$$

Where $\mu$ and $\sigma$ are computed element-wise across all training samples.

### 7.2 Statistics to Save

```matlab
norm_stats.pd_mean = mean(pd_all(:));
norm_stats.pd_std  = std(pd_all(:));
norm_stats.qd_mean = mean(qd_all(:));
norm_stats.qd_std  = std(qd_all(:));
norm_stats.pg_mean = mean(pg_labels_all(:));
norm_stats.pg_std  = std(pg_labels_all(:));
norm_stats.vg_mean = mean(vg_labels_all(:));
norm_stats.vg_std  = std(vg_labels_all(:));
```

---

## 8. Output File Format

### 8.1 NPZ (NumPy Compressed) Format

For Python compatibility, save using `.npz` format. MATLAB can write this using Python interop or save as `.mat` and convert later.

**Training Set** (`samples_train.npz`):
```
e_0_k:      [N_TRAIN, N_BUS, k]
f_0_k:      [N_TRAIN, N_BUS, k]
pd:         [N_TRAIN, N_BUS]
qd:         [N_TRAIN, N_BUS]
topo_id:    [N_TRAIN]
pg_labels:  [N_TRAIN, N_GEN]
vg_labels:  [N_TRAIN, N_GEN]
```

**Topology Operators** (`topology_operators.npz`):
```
g_ndiag:     [N_TOPO, N_BUS, N_BUS]
b_ndiag:     [N_TOPO, N_BUS, N_BUS]
g_diag:      [N_TOPO, N_BUS]
b_diag:      [N_TOPO, N_BUS]
gen_bus_map: [N_GEN]
N_BUS:       scalar
N_GEN:       scalar
```

**Normalization Stats** (`norm_stats.npz`):
```
pd_mean, pd_std, qd_mean, qd_std: scalars
pg_mean, pg_std, vg_mean, vg_std: scalars
```

### 8.2 Alternative: MATLAB .mat Format

Save as `.mat` and use Python's `scipy.io.loadmat` to convert:

```matlab
save('samples_train.mat', 'e_0_k', 'f_0_k', 'pd', 'qd', ...
     'topo_id', 'pg_labels', 'vg_labels', '-v7.3');
```

---

## 9. Recommended Parameters

### 9.1 Dataset Size (from Gao et al. Paper)

| System | Training Samples | Test Samples | Topologies |
|--------|------------------|--------------|------------|
| IEEE 39-bus | 10,000 | 2,000 | 5 (base + 4 N-1) |
| IEEE 57-bus | 10,000 | 2,000 | 5 |
| IEEE 118-bus | 10,000 | 2,000 | 5 |
| IEEE 300-bus | 10,000 | 2,000 | 5 |

### 9.2 Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k` (feature iterations) | 10 | Number of voltage estimation iterations |
| `σ_rel` (load fluctuation) | 0.10 | 10% relative standard deviation |
| `ρ_target` (RES penetration) | 0.30-0.50 | 30-50% RES penetration target |
| `baseMVA` | 100.0 | System power base |

### 9.3 Wind/PV Parameters

| Parameter | Wind | PV | Unit |
|-----------|------|-----|------|
| Distribution | Weibull | Beta | - |
| Shape params | λ=5.089, k=2.016 | α=2.06, β=2.5 | - |
| Cut-in/rated/cut-out | 4/12/25 | - | m/s |
| G_STC | - | 1000 | W/m² |

### 9.4 Unseen Topologies for Generalization Testing

Generate a **separate** test set with N-1 contingencies **not seen during training**:

```matlab
% Example: Train on topologies 0-4, test on 5-7
unseen_topology_pairs = {
    [3, 5],   % Unseen topology 5
    [1, 5],   % Unseen topology 6  
    [2, 4],   % Unseen topology 7
};
```

---

## Appendix A: Quick Reference for Key Equations

| Equation | Name | Formula |
|----------|------|---------|
| Eq. 16 | Voltage Update (e) | $e^{l+1} = (\delta \alpha - \lambda \beta) / (\alpha^2 + \beta^2)$ |
| Eq. 17 | Voltage Update (f) | $f^{l+1} = (\delta \beta + \lambda \alpha) / (\alpha^2 + \beta^2)$ |
| Eq. 19 | Alpha | $\alpha = G_{ndiag} e - B_{ndiag} f$ |
| Eq. 20 | Beta | $\beta = G_{ndiag} f + B_{ndiag} e$ |
| Eq. 21 | Delta | $\delta = -PD - (e^2+f^2) G_{diag}$ |
| Eq. 22 | Lambda | $\lambda = -QD - (e^2+f^2) B_{diag}$ |
| Eq. 23 | PG Clamp | $PG = \text{clamp}(PG, PG_{min}, PG_{max})$ |
| Eq. 24 | QG Clamp | $QG = \text{clamp}(QG, QG_{min}, QG_{max})$ |
| Eq. 25 | V Normalize | $e, f \leftarrow e/\sqrt{e^2+f^2}, f/\sqrt{e^2+f^2}$ |

---

## Appendix B: Validation Checklist

Before using generated data for training:

- [ ] Verify admittance matrix dimensions: `[N_BUS, N_BUS]`
- [ ] Confirm all values are in **per-unit** (p.u.) system
- [ ] Check that feature construction converges (no NaN/Inf values)
- [ ] Validate OPF success rate > 90%
- [ ] Verify topology operators change when contingency is applied
- [ ] Compare `vg_labels` with `bus(:, VM)` at generator buses
- [ ] Confirm `pg_labels` sum ≈ total load + losses

---

## 10. Model 01 vs Model 03 Data Structure Comparison

> **Critical**: Both models can use the **same dataset files** but consume them differently. This section ensures MATLAB-generated data is compatible with both Python models.

### 10.1 Shared Data Files

Both models load data from the **same NPZ files**:

| File | Description |
|------|-------------|
| `samples_train.npz` / `samples_test.npz` | Training/test samples |
| `topology_operators.npz` | Precomputed G/B matrices per topology |
| `norm_stats.npz` | Z-score normalization statistics |

### 10.2 Model 01 (GCNN) Data Usage

**Dataset Class**: `gcnn_opf_01/dataset.py` → `OPFDataset`

**Input to Model**:
| Field | Shape | Source | Notes |
|-------|-------|--------|-------|
| `e_0_k` | `[N_BUS, k]` | `samples.npz['e_0_k']` | **Required** - Feature construction output |
| `f_0_k` | `[N_BUS, k]` | `samples.npz['f_0_k']` | **Required** - Feature construction output |
| `pd` | `[N_BUS]` | `samples.npz['pd']` | Normalized by z-score |
| `qd` | `[N_BUS]` | `samples.npz['qd']` | Normalized by z-score |
| `operators` | dict | `topology_operators.npz` | `{g_ndiag, b_ndiag, g_diag, b_diag}` per topology |

**Output Labels**:
| Field | Shape | Source |
|-------|-------|--------|
| `pg_label` | `[N_GEN]` | `samples.npz['pg_labels']` |
| `vg_label` | `[N_GEN]` | `samples.npz['vg_labels']` |
| `gen_label` | `[N_GEN, 2]` | Stacked `[pg, vg]` |

**Key Characteristics**:
- Uses **graph structure** via `g_ndiag`, `b_ndiag` matrices
- Feature iteration count `k` must match model config (`channels_gc_in`)
- Supports slicing features: `feature_iterations` parameter

---

### 10.3 Model 03 (DeepOPF-FT / MLP) Data Usage

**Dataset Class**: `dnn_opf_03/dataset_03.py` → `OPFDataset03`

**Input to Model** (concatenated vector):
| Field | Shape | Construction | Notes |
|-------|-------|--------------|-------|
| `input` | `[2*N + 2*N²]` | `cat([pd, qd, vec(G), vec(B)])` | **Flattened admittance** |

Where:
- `pd`, `qd` = `[N_BUS]` from `samples.npz`
- `G_full` = `g_ndiag + diag(g_diag)` → `[N_BUS, N_BUS]`
- `B_full` = `b_ndiag + diag(b_diag)` → `[N_BUS, N_BUS]`
- `vec(G)`, `vec(B)` = Flattened to `[N_BUS²]` each

**Dimension Calculation**:
```
For IEEE 39-bus: input_dim = 2×39 + 2×39² = 78 + 3042 = 3120
For IEEE 57-bus: input_dim = 2×57 + 2×57² = 114 + 6498 = 6612
```

**Output Labels**: Same as Model 01

**Key Characteristics**:
- **Does NOT use** `e_0_k`, `f_0_k` features
- Concatenates **full admittance matrix** into input vector
- Topology information encoded in flattened G/B values

---

### 10.4 Data Compatibility Matrix

| Data Field | Model 01 (GCNN) | Model 03 (MLP) | Notes |
|------------|-----------------|----------------|-------|
| `e_0_k` | ✅ Required | ❌ Not used | Only needed for GCNN |
| `f_0_k` | ✅ Required | ❌ Not used | Only needed for GCNN |
| `pd` | ✅ Used directly | ✅ Used in concat | Normalized |
| `qd` | ✅ Used directly | ✅ Used in concat | Normalized |
| `topo_id` | ✅ Index for operators | ✅ Index for operators | Same usage |
| `pg_labels` | ✅ Target | ✅ Target | Same format |
| `vg_labels` | ✅ Target | ✅ Target | Same format |
| `g_ndiag` | ✅ Physics loss | ✅ Reconstruct G | |
| `b_ndiag` | ✅ Physics loss | ✅ Reconstruct B | |
| `g_diag` | ✅ Physics loss | ✅ Reconstruct G | |
| `b_diag` | ✅ Physics loss | ✅ Reconstruct B | |
| `gen_bus_map` | ✅ Label extraction | ✅ Label extraction | Same usage |

---

### 10.5 MATLAB Generation Requirements

To ensure MATLAB-generated data works with **both** models:

#### Required for Model 01 (GCNN):
```matlab
% MUST generate feature construction outputs
[e_0_k, f_0_k] = construct_features(pd, qd, G, B, ...);

% Save with exact field names
save('samples_train.mat', 'e_0_k', 'f_0_k', 'pd', 'qd', ...
     'topo_id', 'pg_labels', 'vg_labels', '-v7.3');
```

#### Required for Model 03 (MLP):
```matlab
% Model 03 reconstructs input from pd, qd, and topology_operators
% No additional generation needed beyond what Model 01 requires
% The Python dataset class handles concatenation automatically
```

#### Topology Operators (Both Models):
```matlab
% Must provide SEPARATE diag and ndiag components
save('topology_operators.mat', ...
     'g_ndiag', 'b_ndiag', 'g_diag', 'b_diag', ...
     'gen_bus_map', 'N_BUS', 'N_GEN', '-v7.3');
```

---

### 10.6 Python Data Loading Example

**Convert MATLAB .mat to NPZ**:
```python
import numpy as np
from scipy.io import loadmat

# Load MATLAB file
data = loadmat('samples_train.mat')

# Convert to NPZ (squeeze removes extra dimensions from MATLAB)
np.savez_compressed('samples_train.npz',
    e_0_k = data['e_0_k'].astype(np.float32),
    f_0_k = data['f_0_k'].astype(np.float32),
    pd = data['pd'].squeeze().astype(np.float32),
    qd = data['qd'].squeeze().astype(np.float32),
    topo_id = data['topo_id'].squeeze().astype(np.int32),
    pg_labels = data['pg_labels'].astype(np.float32),
    vg_labels = data['vg_labels'].astype(np.float32),
)
```

---

### 10.7 Validation: Confirming Data Compatibility

```python
# Test loading for Model 01
from gcnn_opf_01.dataset import OPFDataset
ds1 = OPFDataset('data/samples_test.npz', 'data/topology_operators.npz')
print(f"Model 01 sample: e_0_k={ds1[0]['e_0_k'].shape}, pd={ds1[0]['pd'].shape}")

# Test loading for Model 03
from dnn_opf_03.dataset_03 import OPFDataset03
ds3 = OPFDataset03('data/samples_test.npz', 'data/topology_operators.npz')
print(f"Model 03 sample input dim: {ds3[0]['input'].shape}")
```

Expected output:
```
Model 01 sample: e_0_k=torch.Size([39, 10]), pd=torch.Size([39])
Model 03 sample input dim: torch.Size([3120])
```

---

## Appendix C: Data Array Dimension Summary

### For IEEE 39-bus System (N_BUS=39, N_GEN=10, N_TOPO=5, k=10)

| Array | Model 01 Shape | Model 03 Shape | MATLAB Shape |
|-------|---------------|----------------|---------------|
| `e_0_k` | `[N, 39, 10]` | Not used | `[N, 39, 10]` |
| `f_0_k` | `[N, 39, 10]` | Not used | `[N, 39, 10]` |
| `pd` | `[N, 39]` | `[N, 39]` | `[N, 39]` |
| `qd` | `[N, 39]` | `[N, 39]` | `[N, 39]` |
| `topo_id` | `[N]` | `[N]` | `[N, 1]` |
| `pg_labels` | `[N, 10]` | `[N, 10]` | `[N, 10]` |
| `vg_labels` | `[N, 10]` | `[N, 10]` | `[N, 10]` |
| **Model input** | `[39, 10]` × 2 + ops | `[3120]` | - |

---

*Document Version: 1.1*  
*Last Updated: 2025-12-09*  
*Changes: Updated k=8→10; Added Model 01 vs 03 data structure comparison*

## 6. Implementation for IEEE 39-Bus System (Case39)

### 6.1 Configuration
- **System**: IEEE 39-bus (10 Generators, 39 Buses)
- **RES Integration**:
    - **Wind Buses**: `[4, 7, 8, 15, 16, 18, 21, 25, 26, 28]` (10 buses)
    - **PV Buses**: `[3, 12, 20, 23, 24, 27, 29, 39, 1, 9]` (10 buses)
    - **Penetration**: 50.7% total energy
- **Topologies**:
    - **Seen**: Base case + 4 N-1 contingencies (Lines 6-7, 13-14, 2-3, 21-22)
    - **Unseen**: 3 N-1 contingencies (Lines 23-24, 26-27, 2-25)

### 6.2 Troubleshooting & Workflow
During implementation, the following issues were encountered and resolved:

#### Encoding / Newline Issues in Scripts
**Problem**: Writing complex MATLAB scripts using standard text editors or AI tools resulted in "Invalid Expression" errors in MATLAB, likely due to encoding or line ending corruption.
**Solution**: Use PowerShell's `Set-Content` to robustly write the script files on Windows systems.
```powershell
$code = @" ... MATLAB CODE ... "@
$code | Set-Content -Path script.m -Encoding UTF8
```

#### HDF5 / v7.3 Compatibility
**Problem**: MATLAB's `-v7.3` save format uses HDF5, which `scipy.io.loadmat` cannot read without `h5py`.
**Solution**: Save files using the default MATLAB v7 format (remove the `'-v7.3'` flag) for seamless compatibility with SciPy.

#### PyTorch Memory Layout
**Problem**: Loading flattened arrays from these datasets caused "view() on non-contiguous tensor" errors in Model 03.
**Solution**: Use `.reshape()` instead of `.view()` in PyTorch datasets, or call `.contiguous()` before view.

### 6.3 Final Script Structure
- **Dataset Generation**: `gcnn_opf_01/matlab/generate_dataset_case39_repaired.m` (Script format for batch execution)
- **Conversion**: `gcnn_opf_01/convert_mat_to_npz.py` (Handles 1-based indexing correction)
- **Verification**: `verify_datasets_compatibility.py` (Integration test for both models)
