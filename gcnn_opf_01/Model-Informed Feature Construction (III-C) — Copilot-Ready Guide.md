# Model-Informed Feature Construction (III-C) — Copilot-Ready Guide
This guide reproduces *exactly* the feature-construction mechanism in Fig. 4 of  
**"A Physics-Guided Graph Convolution Neural Network for Optimal Power Flow"**  
but adapted for **k = 8** iterations (matching our CHANNELS_GC_IN = 8).

Goal: compute  
- **e_0_k ∈ ℝ^{N×8}**  
- **f_0_k ∈ ℝ^{N×8}**  
from PD, QD, G, B, using the update laws (16)–(25).

We set:
- Initial voltages:  
  $$ e^0 = \mathbf{1}, \quad f^0 = \mathbf{0} $$
- Number of feature-construction iterations:  
  $$ k = 8 $$

We repeat the cycle in Fig.4 for **i = 7 … 0**.

---

# 1. Required Equations

## (1) Update equations (16), (17)
These generate the next feature vectors.

$$
e_i^{l+1} = 
\frac{\delta_i \alpha_i - \lambda_i \beta_i}{\alpha_i^2 + \beta_i^2}
$$

$$
f_i^{l+1} = 
\frac{\delta_i \beta_i + \lambda_i \alpha_i}{\alpha_i^2 + \beta_i^2}
$$

---

## (2) Intermediate terms α, β, δ, λ

### (19) α
$$
\alpha = z(G_{\text{ndiag}})\,e^l \;-\; z(B_{\text{ndiag}})\,f^l 
$$

### (20) β
$$
\beta = z(G_{\text{ndiag}})\,f^l \;+\; z(B_{\text{ndiag}})\,e^l
$$

### (21) δ
$$
\delta = -PD \;-\; (e^l\odot e^l + f^l\odot f^l)\;z(G_{\text{diag}})
$$

### (22) λ
$$
\lambda = -QD \;-\; (e^l\odot e^l + f^l\odot f^l)\;z(B_{\text{diag}})
$$

Where:
- $z(\cdot)$ = extraction operator (row-sum for diag, full matvec for off-diag)
- $\odot$ = element-wise multiplication.

---

## (3) Compute PG, QG — needed for limiting (23), (24)

### (23) Active power
$$
PG_i = 
PD_i + 
e_i(Ge)_i - e_i(Bf)_i + 
f_i(Gf)_i + f_i(Be)_i
$$

### (24) Reactive power
$$
QG_i = 
QD_i - 
f_i(Ge)_i + f_i(Bf)_i + 
e_i(Gf)_i - e_i(Be)_i
$$

These PG and QG are **used only for clamping, not output**.

---

## (4) Generator clamping (23), (24)

For each generator bus $i$:

$$
PG_i \leftarrow \min\big(\max(PG_i, PG_i^{\min}),\; PG_i^{\max}\big)
$$

$$
QG_i \leftarrow \min\big(\max(QG_i, QG_i^{\min}),\; QG_i^{\max}\big)
$$

Non-generator buses: **ignore PG/QG entirely**.

---

## (5) Normalization step (25)

After each iteration (except the first):

$$
e^l = \frac{e^l}{\sqrt{e_i^2 + f_i^2}}
\qquad
f^l = \frac{f^l}{\sqrt{e_i^2 + f_i^2}}
$$

This is **element-wise normalized voltage magnitude → 1**.

---

# 2. Symbol Dictionary

| Symbol                               | Meaning                                           | Shape           |
| ------------------------------------ | ------------------------------------------------- | --------------- |
| $N$                                  | number of buses                                   | scalar          |
| $e^l, f^l$                           | cosine & sine voltage components at iteration $l$ | $(N)$           |
| $PD, QD$                             | active/reactive load vectors                      | $(N)$           |
| $G_{\text{diag}}, B_{\text{diag}}$   | diagonal of Y-bus                                 | $(N)$           |
| $G_{\text{ndiag}}, B_{\text{ndiag}}$ | off-diagonal of Y-bus                             | $(N,N)$         |
| $PG_i^{\min}, PG_i^{\max}$           | gen limits                                        | scalars per gen |
| $QG_i^{\min}, QG_i^{\max}$           | gen limits                                        | scalars per gen |
| $z(\cdot)$                           | operator: row-sum for diag, matvec otherwise      | –               |

---

# 3. Interpretation of Fig. 4 (step-by-step)

### Initialization
1. Load $(PD, QD, G, B)$.
2. Set:  
   $$e^0 = \mathbf{1}, \quad f^0 = 0$$
3. Set iteration index $l = k-1 = 7$.

### Loop (while $l ≥ 0$):
1. **Compute PG, QG** using (23), (24).  
2. **Clamp generator powers** using limits.  
3. **Compute α, β, δ, λ** using (19)–(22).  
4. **Update voltage estimates** using (16), (17).  
5. **Normalize** using (25).  
6. **Store** $e^l, f^l$ into the feature stacks.  
7. $l \leftarrow l - 1$.

### Output
Return:
- **e_0_k = stack([e⁰ … e⁷], dim=1)**  
- **f_0_k = stack([f⁰ … f⁷], dim=1)**

---

# 4. Minimal Implementation Skeleton (PyTorch)

```python
import torch

def construct_features(
    pd, qd,
    g_ndiag, b_ndiag,
    g_diag, b_diag,
    gen_mask,          # Boolean mask: shape [N]
    PG_min, PG_max,
    QG_min, QG_max,
    k=8
):
    """
    Returns:
        e_0_k : [N, k]
        f_0_k : [N, k]
    """

    N = pd.shape[0]

    # --- 1) init ---
    e = torch.ones(N, dtype=torch.float32)
    f = torch.zeros(N, dtype=torch.float32)

    e_list = []
    f_list = []

    # --- 2) loop ---
    for _ in range(k):

        # ---------------------------------------------------------
        # TODO: Compute PG_i from eq (23)
        # TODO: Compute QG_i from eq (24)
        # ---------------------------------------------------------

        # TODO: Clamp PG, QG only at generator buses using gen_mask

        # ---------------------------------------------------------
        # TODO: Compute α, β from eq (19), (20)
        # TODO: Compute δ, λ from eq (21), (22)
        # ---------------------------------------------------------

        # TODO: e_next, f_next from eq (16), (17)

        # TODO: Normalize via eq (25)

        # store
        e_list.append(e_next)
        f_list.append(f_next)

        # update for next iteration
        e = e_next
        f = f_next

    # --- 3) stack as features ---
    e_0_k = torch.stack(e_list, dim=1)  # [N, k]
    f_0_k = torch.stack(f_list, dim=1)  # [N, k]

    return e_0_k, f_0_k