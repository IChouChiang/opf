# A Physics-Guided Graph Convolution Neural Network for Optimal Power Flow

**Authors:** Maosheng Gao, Juan Yu, Zhifang Yang, Junbo Zhao  
**Source:** IEEE Transactions on Power Systems, Vol. 39, No. 1, January 2024

## Table of Contents
1. [Abstract](#abstract)
2. [I. Introduction](#i-introduction)
3. [II. Problem Formulation](#ii-problem-formulation)
    - [A. Problem Description](#a-problem-description)
    - [B. Framework of the Proposed Method](#b-framework-of-the-proposed-method)
4. [III. Proposed Physics-Guided GCNN](#iii-proposed-physics-guided-gcnn)
    - [A. A Brief Introduction of GCNN](#a-a-brief-introduction-of-gcnn)
    - [B. Physics-Embedded Graph Convolution](#b-physics-embedded-graph-convolution)
    - [C. Model-Informed Feature Construction](#c-model-informed-feature-construction)
    - [D. Physics-guided Graph Convolution Neural Network](#d-physics-guided-graph-convolution-neural-network)
    - [E. Extension of the Proposed Method](#e-extension-of-the-proposed-method)
5. [IV. The Correlative Learning](#iv-the-correlative-learning)
6. [V. Case Study](#v-case-study)
    - [A. Simulation Settings](#a-simulation-settings)
    - [B. Algorithm Validation Under Fixed Topology](#b-algorithm-validation-under-fixed-topology)
    - [C. Validation Under Varying Topologies](#c-validation-under-varying-topologies)
    - [D. Demonstration for the Extension of the Proposed Method](#d-demonstration-for-the-extension-of-the-proposed-method)
7. [VI. Conclusion](#vi-conclusion)
8. [References](#references)

---

## Abstract
The data-driven method with strong approximation capabilities and high computational efficiency provides a promising tool for optimal power flow (OPF) calculation with stochastic renewable energy. However, the topology change dramatically increases the learning difficulties and the demand for learning samples. In this work, we propose a physics-guided graph convolution neural network (GCNN) for OPF calculation with consideration of varying topologies, including the physics-guided graph convolution kernel, feature construction, and loss function formulation. Specifically, a physics-embedded graph convolution kernel is derived by aggregating the features from local neighborhoods utilizing the nodal OPF model formulation. An iterative feature construction method is also developed that encodes both the physical feature and practical constraints into the node vector. Finally, a correlative learning loss function to optimize the unbalanced power injection is developed. Extensive numerical results carried out on various IEEE test systems show that the prediction accuracy of OPF using the proposed method under varying topology changes can be improved by an average of 13.30% and up to 32.63% compared with state-of-the-art methods.

**Index Terms**—Graph convolution neural network (GCNN), neighborhood aggregation, optimal power flow (OPF), varying topology.

## I. INTRODUCTION

Optimal power flow (OPF) plays an essential role in power system operation. It is widely used for power system planning, marketing clearing, economic dispatch, etc. The OPF problem is a nonconvex optimization problem with NP-hard computational complexity [1], [2]. For application scenarios like the probability analysis, the OPF calculation needs to be executed an enormous number of times on randomly generated samples to get the probability statistics. The huge computational burden makes it difficult for practical applications and this calls for new approach developments. Existing studies propose to accelerate the OPF calculation of a single sample, such as simplifying the OPF model [3], [4], using heuristic algorithms [5], [6], or advanced numerical algorithms [7], [8]. However, the accumulated computational burden in the probabilistic analysis is still high.

With the development of artificial intelligence technologies, there is an increasing interest in using them to efficiently solve the OPF problem with significantly enhanced computational efficiency [9]. For example, the stacked extreme learning machine [10], deep neural network (DNN) [11], [12], convolution neural network (CNN) [13], and graph convolution neural network (GCNN) [14] are exploited as a data-driven proxy to the OPF calculation. Those methods settle the heavy computational burden in two different approaches, the direct learning approach, and the indirect learning approach.

The direct learning approach is to predict the solutions by learning the mapping from the load and renewable energy to the OPF outcomes [15]. The direct learning approach can yield 20-100 times faster than the conventional optimization-based methods [11], [12]. In addition to the neural network (NN), other machine learning methods, such as the Gaussian process [16], are also applied to predict the OPF solutions. To ensure the satisfaction of practical constraints like the limits of power generations and the transmission limit of power lines, [11] and [17] construct a new loss function with the regularization part derived from the Karush-Kuhn-Tucker condition or power flow model.

The indirect learning approach is to speed up the conventional solving algorithm by predicting the active constraints or setting the warm start point. [15] uses the NN to predict the solution of OPF and sets the warm-start point to find the nearby feasible solution rather than directly output the solution. [18] predicts active and inactive inequality constraints before solving AC OPF. Taking advantage of the nature that few constraints could be active, solving the size-reduced OPF problem is much faster than the original one.

However, for those data-driven OPF methods, a key concern is whether they can provide a reliable solution given varying operating conditions, especially changing topologies. This brings discrete changes to the projections between the input and output of the data-driven OPF calculations. To cope with the complexity considering varying topology, model-informed feature engineering, and meta-learning are helpful approaches. Instead of encoding topology via a binary input vector, [19] takes the difference of voltage magnitude between the original topology and a new topology as the input topology feature. [20] regards the diagonal element of the admittance matrix as a topology feature and [13] takes advantage of the magnitude and angle of the diagonal elements of the admittance matrix. [21] treats the learning of mapping under a given topology as an independent task and applies the meta-learning algorithm to find good initialization points to accelerate new topology learning.

Although the topology feature engineering and the meta-leaning approach make it possible to predict the OPF solution or active constraints under varying topologies, there are still potential improvements. Specifically, the aforementioned feature engineering has feature loss and it is impossible to input the complete topology information due to the size of the feature dimension. Besides, directly concatenating of load vector and engineered topology feature vector does not reveal the complicated physics relationship between load and the topology. As a result, the combinatory explosion of topology exists and only a few topologies can be considered in practical applications. Instead of topology feature engineering, the GCNN uses the built-in topology to extract the topology feature and is promising to achieve the data-driven approximation of OPF considering topology change. The topology embedding and feature extraction are not limited to the input topology information but extracting the valuable information during the training process. Therefore, the GCNN is a promising tool to address the data-driven OPF problem considering varying topologies.

Some studies have built the GCNN to solve the OPF problem. [22] embeds the topology into the GCNN and exploits the adjacent matrix-based graph convolution kernel to extract the topology feature. On account of the topological representation capability of the physical variable like impedance matrix and admittance matrix, the correlated feature between loads and topologies can be effectively learned. [14] also constructs a topology-embedded GCNN to predict the OPF solutions, which replaces the graph convolution kernel with an impedance-based Gaussian kernel. [23] transforms the impedance-based Gaussian kernel in [14] based on spectral convolution theory to a new convolution kernel. However, the linear graph convolution in [14], [22], and [23] ignores the coupling and the non-linear characteristic of the OPF problem. As a result, their prediction accuracy cannot be guaranteed. To conclude, the characteristics and potential improvements of the related works are summarized in Table I.

To this end, a physics-guided GCNN for OPF considering topology feature learning is proposed and the main contributions are summarized as follows:

1) A novel physics-embedded aggregating method is proposed to construct the graph convolution kernel, which improves the prediction accuracy. This is achieved by decomposing the AC power flow equations based on Gaussian-Seidel iteration, followed by the physics-embedded neighborhood aggregation formulation. This allows the feature propagation in the NNs to follow the physical law, yielding reduced learning difficulty. Thanks to the iteration of the physics-embedded aggregation, the nodal feature vector can be obtained by encoding the topology and inter-relationship between loads. In addition, to sufficiently consider the physical constraints, the clipping technique is used during the feature construction.

2) A new constrained loss function is proposed to consider the physical correlations among the outputs. It is an extended-term that describes the coupling relationship among physical variables, e.g., node voltage and power generation. This allows us to restrict the outliers that do not obey the physical relationship and therefore improve the overall accuracy effectively.

The effectiveness of the proposed method is demonstrated in several IEEE and Polish test systems. Compared to other data-driven OPF methods, the proposed method has stronger topology adaptability and can more accurately predict the OPF solution. The average probabilistic prediction accuracy can be improved by an average of 13.30% and up to 32.63% compared to the state-of-art data-driven methods, like DNN and CNN.

## II. PROBLEM FORMULATION

### A. Problem Description

The OPF problem is to decide the optimal control of power equipment in the power system, like the generation, voltage magnitude, and operational state (on/off) of power generators. It would involve mixed-integer decisions or multiple time-step optimizations. While adding new technologies such as the management of renewable resources, the OPF problem would be more complicated and bring unseen injection distribution and topology. But they share the same modeling basis with the power flow constraints and branch flow limits, which is named the general OPF problem. In this paper, we mainly consider the general OPF problem and study a basic framework for the complicated one. The general OPF problem is to find the low-cost and feasible set-points of generators while satisfying the equality and inequality constraints. The mathematical formulation is shown as:

$$ \min C = \sum_{i \in S_G} (a_i P_{Gi}^2 + b_i P_{Gi} + c_i) $$

Subject to:
$$ P_{Gi} - P_{Di} = \Gamma(V, \theta, z(G), z(B)) \quad (i \in S_B) $$
$$ Q_{Gi} - Q_{Di} = \Psi(V, \theta, z(G), z(B)) \quad (i \in S_B) $$
$$ \underline{P_{Gi}} \le P_{Gi} \le \overline{P_{Gi}} \quad (i \in S_G) $$
$$ \underline{Q_{Gi}} \le Q_{Gi} \le \overline{Q_{Gi}} \quad (i \in S_G) $$
$$ \underline{V_i} \le V_i \le \overline{V_i} \quad (i \in S_B) $$
$$ P_{Lij} = V_i V_j (z(G)_{ij} \cos \theta_{ij} + z(B)_{ij} \sin \theta_{ij}) - V_i^2 z(G)_{ij} \quad (i, j \in S_B) $$
$$ -\underline{P_{Lij}} \le P_{Lij} \le \overline{P_{Lij}} \quad (i, j \in S_B) $$

where $P_{Gi}$ and $V_{Gi}$ are the control variables for the $i$-th generator representing the active power generation and voltage magnitude, respectively; $a_i, b_i$ and $c_i$ are the generation cost coefficients for $i$-th generator; $P_{Di}$ and $Q_{Di}$ are the active and reactive power demand at bus $i$; $V_i$ and $\theta_i$ are the voltage magnitude and phase angle at bus $i$; $\theta_{ij}$ is the phase angle difference between $i$-th and $j$-th bus; $G_{ij}$ and $B_{ij}$ denote the conductance and susceptance between the $i$-th and $j$-th bus, respectively; $z(\cdot)$ represents the topology change function; $\Gamma(\cdot), \Psi(\cdot)$ are the power flow equations; $P_{Lij}$ represents the active branch power between $i$-th node and $j$-th node; $S_G, S_B$ and $S_K$ are the index set of generators, buses and branches.

Differing from the general OPF model, the varying topology based on the default one is considered in the formulation. The topology changes function $z(\cdot)$ indicates the topology of the power system can be occasionally different because of the power line contingency or regular maintenance. It operates on the original conductance or susceptance matrix and the result is still the conductance or susceptance matrix referring to the varying topology. Hence, in (5), $z(G)_{ij}$ and $z(B)_{ij}$ represent the element in the $i$-th row and $j$-th column of $z(G)$ and $z(B)$.

### B. Framework of the Proposed Method

The data-driven methods in solving the OPF problem can be viewed as identifying a mapping from the load fluctuation, uncertain renewable energy, and topology to the set-points of generators. They approximate this mapping from the historic or simulated data. Hence, when solving the OPF problem considering varying topology, the conventional data-driven methods input the feature vector containing the topology representation to a NN or embed the topology with a linear relationship into the NN structure. After the projection from the input via the NN, the final OPF solution is obtained. However, since the nonlinearity and coupling physical relationship under different topologies are not input or embedded into the NN, there could be large biases in the OPF solutions.

In the proposed method, the feature construction, NN structure, and learning loss are improved according to the OPF formulation to reduce learning complexity, see Fig. 1. Firstly, the model-informed feature construction method is proposed. It encodes the loads, renewable energy, and topology into a unique feature space, where the physical relationship and the correlated topology are included. Secondly, the physics-embedded NN is designed to embed the topology and include the coupling physical relationships. While the unique features are input to the NN, the physics-embedded graph convolution can further aggregate neighborhoods’ features and convolute them to a new high-dimensional space. This embedding of non-linear and coupling physical relationships can significantly reduce the learning complexity. Lastly, the correlative learning loss function is used after the input feature and NN structure are determined. It calculates the unbalance power injection at each node and back-propagates them to enhance the physical relationship with the outputs.

## III. PROPOSED PHYSICS-GUIDED GCNN

The section briefly introduces the GCNN. Then, the physics-embedded graph convolution and the physics-guided GCNN structure with physics-embedded graph convolution are proposed. Besides, the model-informed features are constructed.

### A. A Brief Introduction of GCNN

GCNN is one kind of NN operating over graph-structured data $G = (V, E)$ where $G$ represents the graph data; $V \subset R^n$ and $E \subset R^b$ are the vertex set and edge set. To describe the relationship among different nodes, matrix $A \in R^{n \times n}$ denotes the weighted adjacent matrix. Let $a_{ij} \neq 0$ while node $i$ and node $j$ are connected, and otherwise, $a_{ij} = 0$. The input feature of each node is $X \in R^{n \times m}$. Then the graph convolution in a layer can be shown as:

$$ Y = f(AXW + B) $$

where $Y \in R^{n \times k}$ is the output matrix of a graph convolution layer; $W \in R^{m \times k}$ is the trainable parameter matrix; $B \in R^k$ is the trainable bias matrix; $f(\cdot)$ is the activation function.

In (6), $A$ can be constant or trainable, but to decrease the size of trainable parameters, $A$ usually utilizes the constant form derived from the adjacent matrix. For the power system problems, there are physical variables in the branch, and $A$ can be defined based on the physical impedance. $A$ presents the weights that extract the information from adjacent nodes and $AX$ denotes the extraction process named neighborhood aggregation. When the relationship of adjacent nodes is non-linear and cannot be accurately extracted by matrix multiplication, the neighborhood aggregation can be represented by a function $\phi(\cdot)$. So (6) can be rewritten as:

$$ Y = f(\phi(X, A)W + B) $$

In the OPF problem, the nature of non-linearity and non-convexity indicates that the neighborhood aggregation with a single weighted matrix cannot effectively extract features from adjacent nodes. Besides, the physical variables in branches like conductance and susceptance are coupled with node voltage and power. It makes feature extraction difficult but also provides an opportunity to extract features by utilizing the coupled relationship and integrating the physical law into the neighborhood aggregation process.

### B. Physics-Embedded Graph Convolution

For the OPF model in Section II, (1) is the objective function and (2)–(5) are the constraints. Especially, (2) is the AC power flow equations that lead to complex non-linearity. The power flow equations also include topology information and reveal the internal physical relationship between the adjacent nodes. Taking (2) as the feature transformation function, if the power flow equations can be embedded into the neighborhood aggregation, the NN will be capable of efficiently extracting the topology information. By setting $e_i = V_i \cos \theta_i$ and $f_i = V_i \sin \theta_i$ as the nodal feature in the graph and writing the power flow equations in the Cartesian coordinate system, (2) can be rewritten as:

$$ P_{Gi} - P_{Di} = e_i \sum_{j \in N(i)} (z(G)_{ij} e_j - z(B)_{ij} f_j) + f_i \sum_{j \in N(i)} (z(G)_{ij} f_j + z(B)_{ij} e_j) $$
$$ Q_{Gi} - Q_{Di} = f_i \sum_{j \in N(i)} (z(G)_{ij} e_j - z(B)_{ij} f_j) - e_i \sum_{j \in N(i)} (z(G)_{ij} f_j + z(B)_{ij} e_j) $$

Based on the Gaussian-Seidel iteration that separates the diagonal elements of the admittance matrix, there are two kinds of features in the feature transformation function. One is the self-transformed feature which is only affected by the feature at the central node $i$, such as $M_i = (e_i^2 + f_i^2)z(G)_{ii}$ which is the sum of the components with $G_{ii}$ and $B_{ii}$ in (8). The others are the extracted features from adjacent nodes, such as $M_j = e_i(z(G)_{ij} e_j - z(B)_{ij} f_j)$ which indicates the extracted feature from the node $j$. Hence, by moving the self-transformed feature to the left side and keeping the extracted features on the right side, the feature transformation function can be rewritten as:

$$ \delta_i = e_i \alpha_i + f_i \beta_i $$
$$ \lambda_i = f_i \alpha_i - e_i \beta_i $$

where:
$$ \alpha_i = \sum_{j \in N(i), j \neq i} (z(G)_{ij} e_j - z(B)_{ij} f_j) $$
$$ \beta_i = \sum_{j \in N(i), j \neq i} (z(G)_{ij} f_j + z(B)_{ij} e_j) $$
$$ \delta_i = P_{Gi} - P_{Di} - (e_i^2 + f_i^2) z(G)_{ii} $$
$$ \lambda_i = Q_{Gi} - Q_{Di} + (e_i^2 + f_i^2) z(B)_{ii} $$

In (12) and (13), $\alpha_i, \beta_i$ are the aggregated feature from all the neighbors. $\delta_i, \lambda_i$ are the total self-transformed features at node $i$. Hence, to complete the aggregation and obtain a new feature state, the coupled relationship between (10) and (11) can be simplified by solving the linear equations while assuming $e_i^{l+1}, f_i^{l+1}$ are the new feature state at node $i$. Eventually, the function to aggregate features from adjacent nodes under a given topology is obtained as:

$$ e_i^{l+1} = \frac{\delta_i \alpha_i - \lambda_i \beta_i}{\alpha_i^2 + \beta_i^2} $$
$$ f_i^{l+1} = \frac{\delta_i \beta_i + \lambda_i \alpha_i}{\alpha_i^2 + \beta_i^2} $$

In the OPF problem, the generation of active power and reactive power are unknown. In the meanwhile, the graph convolution is to extract the important feature. Hence, the generations of active power and reactive power at all nodes are set to be zeros. By substituting (16) and (17) into (7), the physic-embedded graph convolution is obtained.

$$ Y = f \left( \begin{bmatrix} \frac{\delta \circ \alpha - \lambda \circ \beta}{\alpha \circ \alpha + \beta \circ \beta} \\ \frac{\delta \circ \beta + \lambda \circ \alpha}{\alpha \circ \alpha + \beta \circ \beta} \end{bmatrix} W_1 + B_1 \right) + f \left( \begin{bmatrix} 0 \\ 0 \end{bmatrix} W_2 + B_2 \right) $$

where $\circ$ is the Hadamard (entry-wise) product; $W_1, W_2, B_1$ and $B_2$ are the trainable parameters in the graph convolution layer; $e^{l+1}, f^{l+1}$ are the features matrix in the $l+1$ layer; and:

$$ \alpha = z(G_{ndiag}) e^l - z(B_{ndiag}) f^l $$
$$ \beta = z(G_{ndiag}) f^l + z(B_{ndiag}) e^l $$
$$ \delta = -P_D - (e^l \circ e^l + f^l \circ f^l) z(G_{diag}) $$
$$ \lambda = -Q_D - (e^l \circ e^l + f^l \circ f^l) z(B_{diag}) $$

where $G_{ndiag}, B_{ndiag}$ are the admittance and susceptance matrices without diagonal elements; $e^l, f^l$ are the features matrices in the $l$ layer; $G_{diag}, B_{diag}$ are the diagonal matrices of the admittance and susceptance matrices.

In (18), the non-linear power flow and topology features are embedded in the forward-propagation of a graph convolution layer. Therefore, the physic-embedded graph convolution that can sufficiently extract topology features and physical features, is developed.

### C. Model-Informed Feature Construction

In the data-driven OPF method, despite the physics-embedded graph convolution can significantly reduce the learning difficulty, the data processing including feature construction also affects the prediction performance. Hence, this paper presents a model-informed feature construction strategy.

Based on the physics-embedded neighborhood aggregation method in the graph convolution, the proposed feature construction strategy encodes both the topology and the physical features including the physical relationship as well as the practical constraints like the limitation of the generators. In the solving process of the OPF problem, the voltage magnitude, phase angle, and active and reactive power generation are unknown. But in the aggregation function, all the variables of the initial state should be known. Hence, based on the property of the power system, the voltage magnitude and phase angle for all nodes are set to be 1 and 0. Then, the $P_G$ and $Q_G$ can be calculated by (8) and (9).

However, when directly substituting the $P_G$ and $Q_G$ calculated by (8) and (9) into the neighborhood aggregation function, the main feature of load and renewable energy can be omitted because of the identical relationship of $P_G$ and $Q_G$ in both the aggregation function and power flow equations. Therefore, before substituting calculated $P_G$ and $Q_G$ into the aggregation function, the practical limitations of the power generation in all nodes are considered via (23) and (24).

$$ P_{Gi} = \max(\min(P_{Gi}, P_{Gi}^{\max}), P_{Gi}^{\min}) $$
$$ Q_{Gi} = \max(\min(Q_{Gi}, Q_{Gi}^{\max}), Q_{Gi}^{\min}) $$

After aggregation using the limited $P_G$ and $Q_G$, the new nodal features are obtained. But, a single neighborhood aggregation merely assembles the adjacent features. To assemble the feature from the entire graph, an iterative feature construction strategy is proposed in Fig. 2. With the increase in iteration number, the features extracted can reflect the influence of other nodes far away from the central node in the graph.

While implementing the iteration of neighborhood aggregation, the feature extraction process may be divergent. For instance, when applying the proposed feature construction strategy in the IEEE 57-bus system and visualizing the features at buses 2, 3, and 4, the results are shown in Fig. 3. It can be seen that when the number of iterations is greater than 9, the features obtained by aggregation diverge, resulting in the inapplicability of the features constructed by this method. Hence, it is necessary to normalize the nodal features and avoid diverging.

According to the fact that $e^2 + f^2 \approx 1$, the normalization can be formulated as:

$$ e = \frac{e}{\sqrt{e^2 + f^2}}, \quad f = \frac{f}{\sqrt{e^2 + f^2}} $$

Eventually, the model-informed feature construction is developed and the implementation is shown in Fig. 4.

### D. Physics-guided Graph Convolution Neural Network

Setting the constructed node feature $e^0, f^0 \subset R^{N \times k}$, power demand $P_D, Q_D \subset R^{N \times 1}$, and physic topology information $G_{ndiag}, B_{ndiag}, G_{diag}, B_{diag}$, the physics-guided GCNN is developed to predict the OPF solution. It consists of two blocks, one is the feature extraction block and the other is the prediction block, as depicted in Fig. 5.

The feature extraction block composed of several graph convolution layers plays an important role in extracting the topological and physical features. Due to the assumption in the derivation of the physics-guided neighborhood aggregation, the constructed feature in Section-C cannot accurately indicate the distribution of the OPF solution. Besides, the OPF model does not include the equality constraints of power flow equations. The inequality constraints of power flow limits in power lines based on the practical and engineering conditions also have a big impact. Therefore, the enhanced learning ability with multiple graph convolution layers is necessary to ensure sufficient features extraction. The mathematical formulation of the graph convolution is presented in (18), and the activation function $f(\cdot)$ is the tanh so that $e_i$ and $f_i$ are in $[-1, 1]$. Because of the strong dependence on the physical model, the pooling layers that change the physical topology are not added in the feature extraction block.

The prediction block makes up with several fully connected layers. Through the model-informed feature construction and the trainable feature extraction block, the topology feature and physical relationship can be transformed into a general and high dimension space. Then, the final step is to approximate the OPF solution using the extracted features. Specifically, the feature matrix of all nodes is flattened to 1 dimension and there are several fully connected layers to predict the final OPF solution after the flattened layer.

Eventually, combining the two blocks, the complete architecture of physics-guided GCNN for the OPF problem is designed, as shown in Fig. 5. In the architecture, the topology of the IEEE 6-bus system is taken as an example. There are 2 trainable graph convolution layers and 2 fully connected layers.

### E. Extension of the Proposed Method

Here, we discuss the potential of embedding the proposed method into other learning techniques, such as the reinforcement learning (RL) method. The RL approach is a straightforward solution for optimal power flow. It explores the action space based on environmental rewards and does not require any labels to supervise the learning of neural networks. However, even though the RL approach is model-free, the feature extraction ability of neural networks that are used as actors and critics has a considerable impact on their performance. Therefore, the proposed GCNN could be exploited as the actor and critic neural network in the actor-critic RL approaches to strengthen the feature extraction capability. The detailed structure is shown in Fig. 6.

Besides, RL is also an ideal technique to address the multiple time period dispatch problem [25], [26]. Its goal is to maximize the long-term reward and it can consider the coupling relationship among different time steps. In this circumstance, a robust neural network is required to extract the complex features, where the proposed physics-guided GCNN with robust feature extraction capability can be a desired choice.

## IV. THE CORRELATIVE LEARNING

Essentially, the learning algorithm for NN is a process to approximate the mapping of the OPF problem. In the NN, many trainable variables $\theta = \{w, b\}$ need to be optimized based on the learning algorithm. Hence, the training of NN is an optimization problem and the common loss function is the mean squared error for the approximating problem, shown as

$$ L_{supervised} = E((y_{out} - y_{label})^2) $$

where $y_{out}$ is the predicted value of the NN; $y_{label}$ is the corresponding label.

However, (26) ignores the relationship along errors of all outputs. For instance, the predicted voltage and power generation should satisfy the power flow equation. Therefore, the extended terms to describe the physical relationship are added. Based on the OPF model, the extended-terms in the loss function are defined as:

$$ L_{\phi, PG} = E((P_{Gout} - f_{PG}(V_{out}))^2) $$
$$ L_{\phi, QG} = E((Q_{Gout} - f_{QG}(V_{out}))^2) $$
$$ L_{\phi, PL} = E((P_{Lout} - f_{PL}(V_{out}))^2) $$
$$ L_{\phi, QL} = E((Q_{Lout} - f_{QL}(V_{out}))^2) $$

where $f_{PG}, f_{QG}, f_{PL}, f_{QL}$ are the variants of power flow equations to calculate $P_G, Q_G, P_L, Q_L$ based on the predicted node voltage magnitude and angle.

The extended terms are to optimize the power difference between the predictions and the one calculated by the predicted voltage via the physical equations. Compared with (26), no labels are needed. It can strengthen the physical relationship learning rather than being limited to minimizing errors. Using $L_{\phi, PG}$ at the node $i$ and updating the $w_{mn}$ in the hidden layer which indicates the weight from $m$ neuron to $n$ neuron, which leads to $\Delta w_{mn}$:

$$ \Delta w_{mn} = - \frac{\partial (P_{Gout,i} - f_{PG}(V_{out,i}))^2}{\partial w_{mn}} $$

where the unbalanced power between the directly predicted $P_{Gout,i}$ and $f_{PG}(V_{out,i})$ is used to calculate the gradient and determine $\Delta w_{mn}$. To demonstrate the improvements of this proposed loss function, adding the labels $P_{Glabel,i}$ in (31) and after simplification, we get:

$$ \Delta w_{mn} = - 2e_i \frac{\partial PG_{out,i}}{\partial w_{mn}} - 2e'_i \frac{\partial f_{PG}(V_{out,i})}{\partial w_{mn}} + 2e'_i \frac{\partial PG_{out,i}}{\partial w_{mn}} + 2e_i \frac{\partial f_{PG}(V_{out,i})}{\partial w_{mn}} $$

where:
$$ e_i = P_{Gout,i} - P_{Glabel,i} $$
$$ e'_i = f_{PG}(V_{out,i}) - P_{Glabel,i} $$

Then, the first and second components in (32) are the parameter change computed by the back-propagation algorithm. The two components guide the NN to learn the error directly related to the outputs. The rest two components update $w_{mn}$ according to the physical correlation, which is shown in Fig. 7. For instance, $e'_i$ calculated by $f_{PG}(V_{out,i})$ can also train $w_{mn}$ through $P_{Gout,i}$ in the back-propagation process. Therefore, the proposed method does not only optimize the errors but also mine the physics correlation.

In the power system, the main unbalance term is the nodal active power. Therefore, to improve the training efficiency, the correlative learning loss of PG is added with a weight $\kappa$ to the final loss function, that is:

$$ L = L_{supervised} + \kappa L_{\phi, PG} $$

## V. CASE STUDY

To demonstrate the effectiveness of the proposed OPF method, various case studies have been carried out on the IEEE 39-bus, 57-bus, 118-bus, and 300-bus systems.

### A. Simulation Settings

The settings of different cases are how to simulate the uncertain renewable energy, load fluctuations, and varying topologies. Since it is hard to ensure the full exploration of the possible state, the sample generation is still a bottleneck for the current data-driven methods. In the simulation, certain conditions of the renewable energy, load and topology changes are given for training and testing sample generation. The wind farms and PVs are integrated to simulate the uncertainty of renewable energies. The wind velocity obeys the Weibull distribution with parameters $\lambda = 5.089$ and $k = 2.016$, and the solar irradiance obeys Beta distribution with parameters $\alpha = 2.06$ and $\beta = 2.5$. Randomly connecting some wind farms and PVs to different nodes, the renewable energy penetration rates of each testing system are shown in Table II. The load fluctuation obeys the normal distribution where the mean equals the case’s default value and the standard deviation is 0.1. To include various topologies in the training and testing data, the N-1 contingency of some branches is included. Then, the training and testing data are generated using PYPOWER library on a PC with Intel(R) Core (TM) i7-10700K CPU@ 3.80GHz 3.80 GHz, 16 GB RAM, and NIVIDIA GeForce RTX 2080Ti. The comparison methods and corresponding descriptions are shown in Table III. The GCNNs (M1-M5) have three graph convolution layers with 8 channels and three fully connected layers with 1000 neurons each layer. The CNNs (M6-M7) consist of four convolution layers with 3 × 3 kernels and [64, 128, 256, 256] channels and three fully connected layers with 1000 neurons in each layer. The stride of each convolution is (2, 1). The DNNs (M8-M9) have three layers and 1000 neurons per layer. All of the neural networks are implemented in Tensorflow 2.8.0 python platform. The Adam optimizer is utilized to train all the NNs.

To improve the training efficiency and avoid the impact of outliers, the z-score normalization method is applied in all the testing cases. It is formulated as:

$$ y_{out} = \begin{cases} (y - y_{mean}) / y_{std}, & \text{if } y_{std} \neq 0 \\ y - y_{mean}, & \text{if } y_{std} = 0 \end{cases} $$

where $y_{mean}, y_{std}$ are the mean value and standard deviation, respectively.

In the case study, we mainly discuss the prediction of the set-points of the generators. The performance is measured by the probabilistic accuracy, as shown in (37). It reflects the probability of the prediction error less than a threshold. The thresholds for $P_G$ and $V_G$ are 1MW and 0.001 p.u., respectively.

$$ p = P(|T - \hat{T}| < thr) $$

where $T$ and $\hat{T}$ are the predicted value and actual value, respectively, $thr$ is the threshold.

### B. Algorithm Validation Under Fixed Topology

In this subsection, the effectiveness of the proposed method under a fixed topology is demonstrated. Five GCNNs constructed with identical structures but different graph convolution kernels are compared. Besides, the prediction block in GCNN is composed of three fully connected layers. They are tested in the IEEE 57-bus system under the default topology. After generating 10000 samples for training and 2000 samples for testing, we train the four neural networks with 2000 epochs. The learning rate is 0.001 in the first 1000 epochs and changes to 0.0001 in the remaining epochs. The loss curve is shown in Fig. 8 and the probability accuracies are displayed in Table IV. As for M5, it utilizes the trained model of M4 and fine-tunes the model with $\kappa = 1$ in 200 epochs. Besides, the mean absolute errors of generator voltage are plotted in Fig. 9.

Compared with M1, M3 can converge faster and has smaller errors at the end of training. It indicates that the physics-embedded graph convolution has better convergence. Besides, the probabilistic accuracy in the testing samples increases by 3.37% for the prediction of generator voltage. Therefore, the physics-embedded graph convolution can improve the convergence as well as the prediction accuracy as compared to M1. By contrast, utilizing the impedance matrix to construct the graph convolution kernel in M2 decreases the convergence and the prediction accuracy as compared to M3. Hence, by simply embedding topology with the impedance matrix, the physical relationship cannot be effectively extracted. The reason is that the physical variable, such as the impedance matrix, reveals a coupled physical relationship while ignoring the self-relationship that misleads the NN’s feature extraction. As for M1, although the graph convolution kernel of M1 is a single matrix derived from the adjacent matrix, it is the theoretical kernel for a general graph based on the spectral graph theory.

The comparison between M3 and M4 indicates that the model-informed feature construction provides a better input feature for the physics-guided GCNN and promotes the convergence of training. In Fig. 9, it can be observed that M5 has the least outliers in the testing samples even though the prediction accuracies are very close to M4.

According to the test results, our proposed method can achieve desired performance in fixed topology scenarios. Moreover, the correlative learning loss function to fine-tune the physics-guided GCNN is helpful to avoid outliers by strengthening the physical correlation.

### C. Validation Under Varying Topologies

In this subsection, the effectiveness of the proposed method under varying topologies is demonstrated. There are two kinds of NNs considering the varying topologies in the literature, i.e., DNN and CNN. So the methods M5-M9 are compared in IEEE 39-bus, 57-bus, 118-bus, and 300-bus systems with multiple topologies. Note that M5 is the proposed method. In those cases, the statuses of branches are changed to generate varying topologies. 10000 and 2000 samples are respectively generated for training and testing under five types of line contingencies. The prediction accuracies of generator real power and voltage are shown in Table V.

Compared to other methods, M5 has the best prediction accuracy for all systems. In the IEEE 57-bus system, the prediction accuracy of M5 is up to 99% while the other methods are almost lower than 90%. By contrast, in the IEEE 118-bus and 300-bus systems, the prediction accuracies of generator voltage are only 93.49% and 92.72%, but still significantly outperform other methods. In general, the average improvement is 13.30% and the maximum improvement is up to 32.63%. In the meanwhile, it can be observed that the predication accuracies decrease as the size of system increases. The 10000 training samples cannot ensure accurately prediction even though the physics are embedded into the GCNN.

To analyze the prediction errors in each topology, the testing results of IEEE 300-bus systems are displayed in Fig. 10. Samples are plotted as bubbles in the figure, where the x-axis represents the mean absolute error of generator voltage and the y-axis is the mean absolute error of active generation. Their sizes also indicate the magnitudes of errors. It can be observed that the bubbles with a larger size in M5 are almost red, which indicates that the corresponding samples have identical topology. The main errors in M5 come from T1, indicating the trained physics-guided GCNN can accurately predict the set-points of generators under T2-T5 but not T1. However, in other methods, the big-sized bubbles cover almost the five topologies. Those methods are capable of predicting one topology accurately but have weak adaptability to other topologies.

Therefore, the proposed OPF method can predict the solution more accurately and has better topology adaptability as compared to the state-of-the-arts methods. It could be a promising tool to implement probabilistic analysis in data-ahead operation or real-time risk monitoring, while considering the multiple topologies.

### D. Demonstration for the Extension of the Proposed Method

To demonstrate the effectiveness of the proposed GCNN with RL, it is tested in the Proximal Policy Optimization (PPO) RL method to deal with the optimal power flow problem. The environment is built based on the IEEE 30-bus system and the load profile is sampled from a normal distribution whose average is the default value and the standard derivation is 0.5. Besides, there is a branch randomly chosen to be out of service. The observations include active and reactive loads, current generation and voltage magnitude of power plants, the conductance matrix and susceptance matrix. The reward function is shown as (38) which is referring to [13].

$$ r = \begin{cases} -5000, & \text{PF solver is diverged.} \\ -0.01 C_{generators}, & \text{system is normal.} \\ R_{pg,v} + R_{v,v} + R_{br,v}, & \text{constraints violations exist.} \end{cases} $$

where $R_{pg,v}, R_{v,v}, R_{br,v}$ are the total amount of violations as negative rewards; $C_{generators}$ is the total cost of all generators.

To verify the feature extraction ability of the proposed GCNN with the PPO training algorithm, we train the GCNN, CNN, and DNN with a 0.0001 learning rate. DNN consists of two fully connected layers with 256 neurons. There are two convolution layers with a 4 × 4 kernel, a flatten layer, and two fully connected layers with 256 neurons in CNN. The proposed GCNN has two graph convolution layers, a flatten layer, and two fully connected layers with 256 neurons. After 180000 steps of training, the reward curves of different neural networks are shown in Fig. 11. What’s more, to verify effectiveness of the proposed GCNN in OPF problem considering the time-dependent constraints, the ramping rate of generators is added to the simulated environment. We set the ramping rate to be 10% of the maximum active generation and retrain the three kinds of neural networks. Results are shown in Fig. 12.

According to the results, it can be observed that when the PPO RL is combined with the proposed GCNN, it can converge faster than those with the CNN and DNN using the identical training parameters. The final reward of PPO RL with the proposed GCNN is the largest one, which demonstrate that the proposed GCNN still has the robust feature extraction capability when combining with the PPO RL method. When considering ramping rate of generators, the performance of the proposed GCNN is still the best one. To conclude, the proposed GCNN can be applied in the RL approaches and it is a helpful tool to address complicated OPF problems, such as the multiple-period co-optimization.

## VI. CONCLUSION

The paper proposes a data-driven OPF approach considering topology feature learning. Compared to other data-driven OPF methods, taking advantage of the model-informed feature construction and training the neural network with an extended-term loss function to learn the physical correlation, it improves the feature extraction ability by the physics-embedded graph convolution kernel. Meanwhile, because the complex physical relationships like the coupling and physical topology are directly embedded to the graph convolution network, the learning difficulty of the OPF mapping from loads to generation output is reduced with the help of model-driven GCNN architecture. An evidence is that the prediction performance of the proposed model-guided GCNN is improved by an average of 13.30% and up to 32.63% compared with state-of-the-art methods.

## REFERENCES

[1] K. Lehmann, A. Grastien, and P. Van Hentenryck, “AC-feasibility on tree networks is NP-hard,” IEEE Trans. Power Syst., vol. 31, no. 1, pp. 798–801, Jan. 2016.
[2] J. Lavaei and S. H. Low, “Zero duality gap in optimal power flow problem,” IEEE Trans. Power Syst., vol. 27, no. 1, pp. 92–107, Feb. 2012.
[3] Z. Yang, H. Zhong, A. Bose, T. Zheng, Q. Xia, and C. Kang, “A linearized OPF model with reactive power and voltage magnitude: A pathway to improve the MW-only DC OPF,” IEEE Trans. Power Syst., vol. 33, no. 2, pp. 1734–1745, Mar. 2018.
[4] D. Deka and S. Misra, “Learning for DC-OPF: Classifying active sets using neural nets,” in Proc. IEEE Milan PowerTech, 2019, pp. 1–6.
[5] V. J. Gutierrez-Martinez, C. A. Cañizares, C. R. Fuerte-Esquivel, A. Pizano-Martinez, and X. Gu, “Neural-network security-boundary constrained optimal power flow,” IEEE Trans. Power Syst., vol. 26, no. 1, pp. 63–72, Feb. 2011.
[6] M. Niu, C. Wan, and Z. Xu, “A review on applications of heuristic optimization algorithms for optimal power flow in modern power systems,” J. Modern Power Syst. Clean Energy, vol. 2, no. 4, pp. 289–297, Dec. 2014.
[7] M. Yan, M. Shahidehpour, A. Paaso, L. Zhang, A. Alabdulwahab, and A. Abusorrah, “A convex three-stage SCOPF approach to power system flexibility with unified power flow controllers,” IEEE Trans. Power Syst., vol. 36, no. 3, pp. 1947–1960, May 2021.
[8] S. Mhanna and P. Mancarella, “An exact sequential linear programming algorithm for the optimal power flow problem,” IEEE Trans. Power Syst., vol. 37, no. 1, pp. 666–679, Jan. 2022.
[9] F. Hasan, A. Kargarian, and A. Mohammadi, “A survey on applications of machine learning for optimal power flow,” in Proc. IEEE Texas Power Energy Conf., 2020, pp. 1–6.
[10] X. Lei, Z. Yang, J. Yu, J. Zhao, Q. Gao, and H. Yu, “Data-driven optimal power flow: A physics-informed machine learning approach,” IEEE Trans. Power Syst., vol. 36, no. 1, pp. 346–354, Jan. 2021.
[11] X. Pan, T. Zhao, M. Chen, and S. Zhang, “DeepOPF: A deep neural network approach for security-constrained DC optimal power flow,” IEEE Trans. Power Syst., vol. 36, no. 3, pp. 1725–1735, May 2021.
[12] A. S. Zamzam and K. Baker, “Learning optimal solutions for extremely fast AC optimal power flow,” in Proc. IEEE Int. Conf. Commun., Control, Comput. Technol. for Smart Grids (SmartGridComm), 2020, pp. 1–6.
[13] Y. Zhou, W. Lee, R. Diao, and D. Shi, “Deep reinforcement learning based real-time AC optimal power flow considering uncertainties,” J. Modern Power Syst. Clean Energy, vol. 10, no. 5, pp. 1098–1109, Sep. 2022.
[14] D. Owerko, F. Gama, and A. Ribeiro, “Optimal power flow using graph neural networks,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., 2020, pp. 5930–5934.
[15] A. Velloso and P. Van Hentenryck, “Combining deep learning and optimization for preventive security-constrained dc optimal power flow,” IEEE Trans. Power Syst., vol. 36, no. 4, pp. 3618–3628, Jul. 2021.
[16] P. Pareek and H. D. Nguyen, “Gaussian process learning-based probabilistic optimal power flow,” IEEE Trans. Power Syst., vol. 36, no. 1, pp. 541–544, Jan. 2021.
[17] Z. Yan and Y. Xu, “Real-time optimal power flow: A Lagrangian based deep reinforcement learning approach,” IEEE Trans. Power Syst., vol. 35, no. 4, pp. 3270–3273, Jul. 2020.
[18] F. Hasan, A. Kargarian, and J. Mohammadi, “Hybrid learning aided inactive constraints filtering algorithm to enhance ac OPF solution time,” IEEE Trans. Ind. Appl., vol. 57, no. 2, pp. 1325–1334, Mar./Apr. 2021.
[19] M. Xiang, J. Yu, Z. Yang, Y. Yang, H. Yu, and H. He, “Probabilistic power flow with topology changes based on deep neural network,” Int. J. Elect. Power Energy Syst., vol. 117, 2020, Art. no. 105650.
[20] Y. Du, F. Li, J. Li, and T. Zheng, “Achieving 100x acceleration for N-1 contingency screening with uncertain scenarios using deep convolutional neural network,” IEEE Trans. Power Syst., vol. 34, no. 4, pp. 3303–3305, Jul. 2019.
[21] Y. Chen, S. Lakshminarayana, C. Maple, and H. V. Poor, “A meta-learning approach to the optimal power flow problem under topology reconfigurations,” IEEE Open Access J. Power Energy, vol. 9, pp. 109–120, 2022, doi: 10.1109/OAJPE.2022.3140314.
[22] C. Kim, K. Kim, P. Balaprakash, and M. Anitescu, “Graph convolutional neural networks for optimal load shedding under line contingency,” in Proc. IEEE Power Energy Soc. Gen. Meeting (PESGM), 2019, pp. 1–5.
[23] D. Wang, K. Zheng, Q. Chen, G. Luo, and X. Zhang, “Probabilistic power flow solution with graph convolutional network,” in Proc. IEEE PES Innov. Smart Grid Technol. Europe, 2020, pp. 650–654.
[24] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learn. Representations, 2017, pp. 1–14.
[25] C. Hu, Z. Cai, Y. Zhang, R. Yan, Y. Cai, and B. Cen, “A soft actor-critic deep reinforcement learning method for multi-timescale coordinated operation of microgrids,” Protection Control Modern Power Syst., vol. 7, no. 3, pp. 423–432, 2022.
[26] H. Jin, Y. Teng, T. Zhang, Z. Wang, and Z. Chen, “A deep neural network coordination model for electric heating and cooling loads based on IoT data,” CSEE J. Power Energy Syst., vol. 6, no. 1, pp. 22–30, Mar. 2020.
