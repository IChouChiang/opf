# Physics-Guided Residual Loss Implementation for GCNN_OPF_01  
This document specifies the **math**, **data flow**, and **APIs** required to
complete the physics-based correlative loss term used in Gao et al.
"A Physics-Guided Graph Convolution Neural Network for Optimal Power Flow."

## 1. Overview

We want to implement the loss function:
L = L_supervised + κ · L_PG_residual.

Where:
- **L_supervised** = MSE(predicted PG vs OPF labels)
- **L_PG_residual** = physics residual loss enforcing nodal active-power balance
  using predicted voltages (e,f) and admittance matrices (G,B).

This requires:
- Access to **final graph-convolution voltages** e,f from GCNN_OPF_01.
- Bus-level predicted generation P_G^out (scatter from generators to buses).
- Demand vector PD.
- The full Ybus = G + jB for the topology used in the sample.

We use **Cartesian full-node power balance** (Eq. (8) in the paper).

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
