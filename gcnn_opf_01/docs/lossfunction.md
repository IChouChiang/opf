# Physics-Guided Residual Loss Implementation for GCNN_OPF_01  
This document specifies the **math**, **data flow**, and **APIs** for the
physics-based correlative loss term used in Gao et al.
"A Physics-Guided Graph Convolution Neural Network for Optimal Power Flow."

## Status: ✅ IMPLEMENTED

**Implementation:** `loss_model_01.py`  
**Test:** Validated with 2-head GCNN architecture in `model_01.py`

---

## 1. Overview

We implement the loss function:  
**L = L_supervised + κ · L_Δ,PG**

Where:
- **L_supervised** = MSE(predicted PG vs OPF labels) + MSE(predicted VG vs labels)
- **L_Δ,PG** = physics residual loss enforcing nodal active-power balance
  using predicted voltages (e,f) and admittance matrices (G,B).

This requires:
- Access to **final graph-convolution voltages** e,f from GCNN_OPF_01's v_head.
- Bus-level predicted generation P_G^out (scatter from generators to buses).
- Demand vector PD.
- The full Ybus = G + jB for the topology used in the sample.

We use **Cartesian full-node power balance** (Eq. (8) in the paper).

---

## 2. Implemented Functions

### 2.1. `build_A_g2b(N_BUS, gen_bus_indices)`
Creates generator-to-bus incidence matrix [N_BUS, N_GEN].
- Sets A_g2b[gen_bus_indices[g], g] = 1.0
- Used to scatter generator outputs to bus-level injections

### 2.2. `f_pg_from_v(v_out, pd, G, B)`
Computes PG from predicted voltages using power flow equation (Eq. 8).
- Input: v_out [N_BUS, 2] = (e, f), pd [N_BUS], G/B [N_BUS, N_BUS]
- Computes: PG = PD + e·(G@e - B@f) + f·(G@f + B@e)
- Returns: PG_physics [N_BUS]

### 2.3. `correlative_loss_pg(...)`
Main loss function combining supervised and physics terms.
- Supervised: MSE(gen_out vs gen_labels) for both PG and VG
- Physics: MSE(A_g2b·PG_pred vs f_pg_from_v(v_out))
- Returns: L_supervised + kappa·L_Δ,PG

**Signature:**
```python
correlative_loss_pg(
    gen_out,         # [N_GEN, 2] = (PG, VG) predictions
    v_out,           # [N_BUS, 2] = (e, f) voltage predictions
    gen_labels,      # [N_GEN, 2] = (PG_true, VG_true)
    pd,              # [N_BUS] demand
    G, B,            # [N_BUS, N_BUS] admittance matrices
    gen_bus_indices, # [N_GEN] generator bus locations
    kappa=0.1        # physics loss weight
)
```

---

## 3. Model Integration

### 3.1. GCNN_OPF_01 Forward Pass
Returns two outputs:
```python
gen_out, v_out = model(e_0_k, f_0_k, pd, qd, ...)
# gen_out: [N_GEN, 2] = (PG, VG)
# v_out:   [N_BUS, 2] = (e, f)
```

### 3.2. Training Loop Example
```python
from loss_model_01 import correlative_loss_pg

gen_out, v_out = model(e_0_k, f_0_k, pd, qd, ...)

loss = correlative_loss_pg(
    gen_out, v_out, gen_labels,
    pd, G, B, gen_bus_indices,
    kappa=0.1
)

loss.backward()
optimizer.step()
```

---

## 4. Technical Notes

### Units
- All quantities in **per-unit (p.u.)** — matches PYPOWER convention after `/baseMVA`
- No MW/MVAr mixing

### Voltage Range
- e, f outputs from v_head use `tanh` activation → range [-1, 1]
- Physical interpretation: normalized Cartesian voltage components

### Generator Scattering
- `A_g2b @ gen_out[:, 0]` maps generator PG to bus-level injections
- Ensures physics residual computed at all buses, not just generator buses

### Reactive Power
- Current implementation focuses on active power (PG) physics
- Reactive power (QG) validation can be added via similar f_qg_from_v() function

---

## 5. Implementation Status

- ✅ Generator-to-bus incidence matrix (`build_A_g2b`)
- ✅ Physics PG computation from voltages (`f_pg_from_v`)
- ✅ Correlative loss with supervised + physics terms (`correlative_loss_pg`)
- ✅ Integration with 2-head GCNN architecture
- ⏳ Training loop implementation (next step)
- ⏳ Reactive power physics validation (future enhancement)

---

## 6. References

- Equation (8): Active power balance in Cartesian coordinates
- Section IV.B: Correlative loss formulation
- Paper: Gao et al. "A Physics-Guided Graph Convolution Neural Network for Optimal Power Flow"

## 2. Required Inputs

1. From model forward:
   - `PG_gen_pred`: [N_GEN] predicted active generation from final FC layer
     (usually out[:,0]).
   - `e_last`: [N_BUS, Cout] final "e" output after third GraphConv.
   - `f_last`: [N_BUS, Cout] final "f" output.

2. From sample/scenario:
   - `pd`: [N_BUS] bus active demand (including RES as negative load).

3. From topology configuration:
   - `G`: [N_BUS, N_BUS] conductance matrix z(G) = G_ndiag + diag(G_diag)
   - `B`: [N_BUS, N_BUS] susceptance matrix z(B) = B_ndiag + diag(B_diag)
   - `A_g2b`: [N_BUS, N_GEN] generator-to-bus incidence matrix
       Example:
         A_g2b[bus_of_gen[g], g] = 1

4. One constant:
   - `phys_channel`: integer index selecting which channel in e,f is the
     actual physical voltage (e.g., channel 0).

## 3. Converting Generator PG → Bus PG

We create:
P_G_out_bus = A_g2b · PG_gen_pred

This is a linear scattering from generators to buses.

## 4. Physics-Based Target f_PG(V)

### 4.1. Extract physical voltage channel

Use only the first channel:
e_phys = e_last[:, phys_channel]  
f_phys = f_last[:, phys_channel]  
Shapes: [N_BUS]

### 4.2. Compute the nodal power injection from voltage & admittance

Equation (8) from the paper:

P_inj_i =  
    e_i * Σ_j ( G_{ij} e_j - B_{ij} f_j )  
  + f_i * Σ_j ( G_{ij} f_j + B_{ij} e_j )

We vectorize:

Ge = G @ e_phys   # [N_BUS]  
Bf = B @ f_phys  
Gf = G @ f_phys  
Be = B @ e_phys  

term1 = e_phys * (Ge - Bf)  
term2 = f_phys * (Gf + Be)  

PF_rhs = term1 + term2

### 4.3. Compute “power-flow-implied PG”

f_PG(V)_i = PD_i + PF_rhs_i

We call:
PG_pf = pd + PF_rhs

### 4.4. Final Physics Residual Loss

L_PG_res = MSE(P_G_out_bus , PG_pf)

(MSE over buses, optionally averaged over batch.)

## 5. Model Output Requirements

Modify GCNN_OPF_01.forward as:

def forward(..., return_features=False):
    ...
    if return_features:
        return out, e, f
    else:
        return out

Where:
- out: [N_GEN, 2]
- e,f: final layer outputs [N_BUS, Cout]

## 6. Complete Loss API

Define a standalone function:

physics_pg_residual_loss(
    PG_gen_pred,  # [N_GEN]
    e_last, f_last,  # [N_BUS, Cout]
    pd,  # [N_BUS]
    G, B,  # [N_BUS, N_BUS]
    A_g2b,  # [N_BUS, N_GEN]
    phys_channel=0
)

Return: scalar PyTorch loss.

## 7. Training Loop Integration

loss_sup = mse(PG_gen_pred, PG_label)  
loss_pg = physics_pg_residual_loss(...)  
loss = loss_sup + kappa * loss_pg  
loss.backward()

## 8. Minimal assumptions

- We use p.u. units throughout.
- We assume e,f are in normalized tanh range [-1,1].
- We ignore QG and reactive residuals until later.
