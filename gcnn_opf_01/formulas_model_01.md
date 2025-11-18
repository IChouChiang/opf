# Week 4

## Problem Formulation

### The general OPF problem 

$$
\min C = \sum_{i \in S_G} \left(a_i PG_i^2 + b_i PG_i + c_i \right),
\tag{1}
$$

$$
\begin{cases}
PG_i - PD_i = \Gamma \left( V, \theta, z^{(G)}, z^{(B)} \right) \\
QG_i - QD_i = \Psi \left( V, \theta, z^{(G)}, z^{(B)} \right)
\end{cases}
\quad (i, j \in S_B),
\tag{2}
$$

$$
\underline{PG_i} \le PG_i \le \overline{PG_i} \quad (i \in S_G),
\tag{3}
$$

$$
\underline{QG_i} \le QG_i \le \overline{QG_i} \quad (i \in S_G),
\tag{3}
$$

$$
\underline{V_i} \le V_i \le \overline{V_i} \quad (i \in S_B),
\tag{4}
$$

$$
\begin{cases}
PL_{ij} = V_i V_j \left( z^{(G)}_{ij} \cos \theta_{ij} + z^{(B)}_{ij} \sin \theta_{ij} \right)
 - V_i^2 z^{(G)}_{ij}{} \\
-\overline{PL_{ij}} \le PL_{ij} \le \overline{PL_{ij}}
\end{cases}
\quad (i, j \in S_B),
\tag{5}
$$

- $PG_i$ and $VG_i$ are the control variables for the $i$-th generator, representing the active power generation and voltage magnitude, respectively  
- $a_i$, $b_i$, and $c_i$ are the generation cost coefficients for the $i$-th generator  
- $PD_i$ and $QD_i$ are the active and reactive power demand at bus $i$  
- $V_i$ and $\theta_i$ are the voltage magnitude and phase angle at bus $i$  
- $\theta_{ij}$ is the phase angle difference between the $i$-th and $j$-th bus  
- $G_{ij}$ and $B_{ij}$ denote the conductance and susceptance between the $i$-th and $j$-th bus, respectively  
- $z(\cdot)$ represents the topology change function 
- $\Gamma(\cdot)$ and $\Psi(\cdot)$ are the power flow equations  
- $PL_{ij}$ represents the active branch power between the $i$-th node and $j$-th node  
- $S_G$, $S_B$, and $S_K$ are the index sets of generators, buses, and branches  

### GCNN Basics

$$
Y = f(\phi(X,A)W + B), \tag{6}
$$

- $Y \in \mathbb{R}^{n \times k}$ is the output feature matrix (for $n$ nodes and $k$ output features per node) from a graph convolution layer.
- $X \in \mathbb{R}^{n \times m}$ is the input feature matrix ($m$ features per node).
- $A \in \mathbb{R}^{n \times n}$ is the adjacency matrix of the graph (often fixed and derived from the physical network connectivity).
- $W \in \mathbb{R}^{m \times k}$ is the trainable weight matrix; $B \in \mathbb{R}^{k}$ is the trainable bias vector.
- $f(\cdot)$ denotes the activation function (e.g. $\tanh$), applied element-wise.
- $\phi(X, A)$ represents a (possibly non-linear) neighborhood aggregation function replacing $AX$ when simple linear aggregation is insufficient.

### Physics-Embedded Graph Convolution

$$
P_{Gi} - P_{Di} = e_i \sum_{j\in N(i)}\!\Big(z(G)_{ij} e_j - z(B)_{ij} f_j\Big) + f_i \sum_{j\in N(i)}\!\Big(z(G)_{ij} f_j + z(B)_{ij} e_j\Big), \tag{8}
$$

$$
Q_{Gi} - Q_{Di} = f_i \sum_{j \in N(i)} \left( z^{(G)}_{ij} e_j - z^{(B)}_{ij} f_j \right) 
- e_i \sum_{j \in N(i)} \left( z^{(G)}_{ij} f_j + z^{(B)}_{ij} e_j \right),
\tag{9}
$$

$$
\delta_i = e_i \alpha_i + f_i \beta_i,
\tag{10}
$$

$$
\lambda_i = f_i \alpha_i - e_i \beta_i,
\tag{11}
$$

$$
\alpha_i = \sum_{\substack{j \in N(i) \\ j \ne i}} \left( z^{(G)}_{ij} e_j - z^{(B)}_{ij} f_j \right),
\tag{12}
$$

$$
\beta_i = \sum_{\substack{j \in N(i) \\ j \ne i}} \left( z^{(G)}_{ij} f_j + z^{(B)}_{ij} e_j \right),
\tag{13}
$$

$$
\delta_i = P_{Gi} - P_{Di} - \left( e_i^2 + f_i^2 \right) z^{(G)}_{ii},
\tag{14}
$$

$$
\lambda_i = Q_{Gi} - Q_{Di} + \left( e_i^2 + f_i^2 \right) z^{(B)}_{ii},
\tag{15}
$$

$$
e_i^{l+1} = \frac{ \delta_i \alpha_i - \lambda_i \beta_i }{ \alpha_i^2 + \beta_i^2 },
\tag{16}
$$

$$
f_i^{l+1} = \frac{ \delta_i \beta_i + \lambda_i \alpha_i }{ \alpha_i^2 + \beta_i^2 },
\tag{17}
$$

$$
e_i^{l+1} = f\left( \frac{ \delta_i \alpha_i - \lambda_i \beta_i }{ \alpha_i^2 + \beta_i^2 } W_1 + B_1 \right),
\tag{18a}
$$

$$
f_i^{l+1} = f\left( \frac{ \delta_i \beta_i + \lambda_i \alpha_i }{ \alpha_i^2 + \beta_i^2 } W_2 + B_2 \right),
\tag{18b}
$$


- $i, j$ index the graph nodes; $N(i)$ denotes the set of neighbor nodes of node $i$.
- $e_i = V_i \cos\theta_i$ and $f_i = V_i \sin\theta_i$ are the real (cosine) and imaginary (sine) components of the voltage at node $i$ (taken as features).
- $P_{Gi}, Q_{Gi}$ and $P_{Di}, Q_{Di}$ are the active and reactive power generation and demand at node $i$, respectively.
- $z(G)*{ij}, z(B)*{ij}$ are elements of the modified conductance ($G$) and susceptance ($B$) matrices reflecting the network topology (if a line $ij$ is removed, these entries are zero).
- $\alpha_i, \beta_i$ are the aggregated neighbor contributions to node $i$’s power injections (summing the effects from all adjacent nodes $j \in N(i)$).
- $\delta_i, \lambda_i$ represent the self-transformed features (net power injection) at node i: $\delta_i$ corresponds to active power mismatch (generation minus demand minus self-admittance term), and $\lambda_i$ to reactive power mismatch (generation minus demand plus self-admittance term).
- $e_i^{,l+1}, f_i^{,l+1}$ are the updated node feature components after one physics-embedded neighborhood aggregation iteration (producing the $(l+1)$-th layer features).
- $W_1, W_2$ are trainable weight matrices (for updating $e$ and $f$ feature channels, respectively), and $B_1, B_2$ are the corresponding bias vectors for that graph convolution layer.
- $f(\cdot)$ is the activation function (applied to each updated feature), and $\circ$ denotes the Hadamard (element-wise) product.

### Model-Informed Feature Construction

$$
P_{Gi} = \max\!\Big\{\min(P_{Gi},\, \overline{P_{Gi}}),\; \underline{P_{Gi}}\Big\}, \tag{23}
$$

$$
Q_{Gi} = \max\!\Big\{\min(Q_{Gi},\, \overline{Q_{Gi}}),\; \underline{Q_{Gi}}\Big\}, \tag{24}
$$

$$
e \;=\; \frac{e}{\sqrt{\,e^2 + f^2\,}}, \qquad  f \;=\; \frac{f}{\sqrt{\,e^2 + f^2\,}}. \tag{25}
$$

- In the initial state for feature construction, all node voltages are set to $|V_i|=1$ and $\theta_i=0$, so preliminary $P_{Gi}, Q_{Gi}$ values can be calculated from the power flow equations (8) and (9).

- $\underline{P_{Gi}}, \overline{P_{Gi}}$ are the minimum and maximum active power generation limits for generator $i$ (similarly $\underline{Q_{Gi}}, \overline{Q_{Gi}}$ for reactive power) based on practical operational constraints.

- Equations (23) and (24) “clip” or limit the computed generator outputs within their allowable bounds before using them in the feature vector

- Equation (25) normalizes the node voltage feature components so that $e^2 + f^2 = 1$ for each node (i.e. each voltage is scaled to unit magnitude), helping to prevent divergence in the iterative aggregation process

### Correlative Learning Loss Function


$$
L_{\text{supervised}} = \mathbb{E}\left[ (y_{\text{out}} - y_{\text{label}})^2 \right],
\tag{26}
$$

$$
L_{,\!P_G} = \mathbb{E}\left[ (P_{G,\text{out}} - f_{P_G}(V_{\text{out}}))^2 \right],
\tag{27}
$$

$$
L_{,\!Q_G} = \mathbb{E}\left[ (Q_{G,\text{out}} - f_{Q_G}(V_{\text{out}}))^2 \right],
\tag{28}
$$

$$
L_{,\!P_L} = \mathbb{E}\left[ (P_{L,\text{out}} - f_{P_L}(V_{\text{out}}))^2 \right],
\tag{29}
$$

$$
L_{,\!Q_L} = \mathbb{E}\left[ (Q_{L,\text{out}} - f_{Q_L}(V_{\text{out}}))^2 \right],
\tag{30}
$$

$$
L = L_{\text{supervised}} + \kappa\,L_{,\!P_G}.
\tag{35}
$$



- $y_{\text{out}}$ denotes the predicted output vector of the GCNN (e.g. the set of OPF solution variables), and $y_{\text{label}}$ is the ground-truth target vector (from an actual OPF solution). Equation (26) is the standard supervised mean-squared-error loss.
- $f_{P_G}(V_{\text{out}}), f_{Q_G}(V_{\text{out}}), f_{P_L}(V_{\text{out}}), f_{Q_L}(V_{\text{out}})$ are functions derived from the physical power flow model (variants of the AC power flow equations) that calculate the expected $P_G, Q_G, P_L,$ and $Q_L$ values given the predicted node voltages $V_{\text{out}}$.
- $L_{,!P_G}, L_{,!Q_G}, L_{,!P_L}, L_{,!Q_L}$ are additional loss terms that penalize any inconsistency between the network’s outputs and the values computed from those outputs using the physical equations. These extended terms encourage the predictions to obey the physical coupling relationships (for example, between node voltage and power injection) even without direct supervision on those quantities.
- $\kappa$ is a weighting hyperparameter that emphasizes the active power balance term ($L_{,!P_G}$) in the total loss. The final loss $L$ in (35) combines the usual supervised loss with a scaled correlative loss term for active power, helping the model reduce nodal active power mismatches during training.

## Sample Generating

- The wind farms and PVs are integrated to simulate the uncertainty of renewable energies.  
  - The wind velocity obeys the Weibull distribution with parameters λ = 5.089 and k = 2.016. 
    $$ f(v) = 
    \begin{cases}
      \dfrac{k}{\lambda} \left( \dfrac{v}{\lambda} \right)^{k-1}
      e^{-\left( \dfrac{v}{\lambda} \right)^k}, & v \ge 0,\\[6pt]
      0, & v < 0
    \end{cases} $$
  - The solar irradiance obeys the Beta distribution with parameters α = 2.06 and β = 2.5.  
    $$ f(G) = 
    \begin{cases}
      \dfrac{1}{B(\alpha, \beta)}\, G^{\alpha - 1} (1 - G)^{\beta - 1}, & 0 \le G \le 1,\\[6pt]
      0, & \text{otherwise}
    \end{cases} $$
  - Randomly connecting some wind farms and PVs to different nodes, the renewable … 
- To include various topologies in the training and testing data, the N-1 contingency of some branches is included.

>  10000 and 2000 samples are respectively generated for training and testing under five types of line contingencies.

- The general AC-OPF problem (fixed topology)

$$
\min C = \sum_{i \in S_G} \left(a_i P_{G_i}^2 + b_i P_{G_i} + c_i\right),
\tag{1}
$$

$$
P_{G_i} - P_{D_i}
= e_i \sum_{j \in N(i)} \left(G_{ij} e_j - B_{ij} f_j\right)
+ f_i \sum_{j \in N(i)} \left(G_{ij} f_j + B_{ij} e_j\right),
\tag{2a}
$$

$$
Q_{G_i} - Q_{D_i}
= f_i \sum_{j \in N(i)} \left(G_{ij} e_j - B_{ij} f_j\right)
- e_i \sum_{j \in N(i)} \left(G_{ij} f_j + B_{ij} e_j\right),
\tag{2b}
$$

$$
\underline{P_{G_i}} \le P_{G_i} \le \overline{P_{G_i}}
\quad (i \in S_G),
\tag{3}
$$

$$
\underline{Q_{G_i}} \le Q_{G_i} \le \overline{Q_{G_i}}
\quad (i \in S_G),
\tag{4}
$$

$$
\underline{V_i} \le V_i \le \overline{V_i}
\quad (i \in S_B),
\tag{5}
$$
