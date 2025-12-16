# Week 3 Description

In this conversation we are going to build and `pyomo` Concrete Model following these formulas.
$$
\begin{aligned}
\min_{P_G}\quad & C_G^{\!\top} P_G \qquad\text{(linear objective 线性目标)} \\ \text{s.t.}\quad & \mathbf{1}_G^{\!\top} P_G = \mathbf{1}_D^{\!\top} P_D \quad\text{(system balance 系统功率平衡)} \\ & P_L^{\min} \ \le\ M_G P_G + M_D P_D \ \le\ P_L^{\max} \quad\text{(line limits 线路约束)} \\ & P_G^{\min} \ \le\ P_G \ \le\ P_G^{\max} \quad\text{(generator bounds 机组约束).}
\end{aligned}
$$
This is for samples generating for NN training mapping from $P_D$ to $P_G$. In this OPF problem, the parameters $C_G， P_L^{min}, P_L^{max}, M_G, M_D, P_G^{min}, P_G^{max}$ are all fixed and should be extracted once from `case118.py` accessed via `pypower.api`. To build such a `pyomo` concrete model, should we extracted data first or create the model first? What's the best practice, any tiny code snippet as examples?

---

Take everything very slowly. We are going to build everything line by line.

Firstly let's focus on the index sets. There 118 buses in the system with no doubt. Then there is 54 buses as Gen buses, there index stored at the first col of `ppc["gen"]`. So how to extracted this  data.

---

The we need to extract $C_G$ , we need the c2 c1 and c1 col only since in our case all $n$ here are equal to $3$

```python
# 2 startup shutdown n c(n-1) ... c0
ppc["gencost"] = array([
```

---

Why not just create 118 objects and attribute different propertys like `Demand`, if there is a generator on that bus like $2$, $4$ add property like $Pg$ and $P_G^{max}$, $P_G^{min}$

---

Similarly extract the set $L$

```python
# Extract Set L
nbr = ppc["branch"].shape[0] # number of branches (lines)
L = np.arange(nbr)
```

---

Extract $P_L^{min}$ and $P_L^{max}$, actually we won't use the data in RATEA col in `case118.py`, instead the bounds is from a `.xlsx` file and for every lines, all of the upper bounds equal to $3$ while all the lower bounds equal to $-2.4$ p.u.

---

to extracted $P_G^{min}$  and $ P_G^{max}$, the data is also not from `case118.py`, but in the `.xlsx` file, in the third tab or sheet, looks like this. How to import such a file and use the data? Maybe we need a specific function to perform this task.

---

Here comes the most difficult part, extract $M_G$ and $M_D$		
	the `case118.py` offered useful data `x` and `ratio` in the forth col of 

```python		
# fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
```

