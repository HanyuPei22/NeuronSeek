# Mathematical Derivation & Workflow: Tensor Interaction Layer

## Method Formulation

### Global Optimization Objective

We formulate the structure discovery as a constrained optimization problem. The goal is to minimize the task-specific loss while imposing sparsity on the polynomial orders via a differentiable $L_0$ regularization term.

$$\min_{\Theta} \mathcal{J} = \underbrace{\frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(\hat{\mathbf{y}}_i, \mathbf{y}_i)}_{\text{Task Loss}} + \lambda \sum_{k=1}^{K} \left( \mathbb{E}[\|z_{\text{pure}}^{(k)}\|_0] + \mathbb{E}[\|z_{\text{int}}^{(k)}\|_0] \right)$$

Variable Definitions:

- $\Theta$: The set of all learnable parameters ($\mathbf{U}, \mathbf{W}, \mathbf{b}, \text{gates}$).
- $\mathcal{L}(\cdot)$: The loss function (e.g., MSE for regression, Cross-Entropy for classification).
- $z \sim \text{HardConcrete}(\alpha)$: The binary stochastic gates.
- $\mathbb{E}[\|\cdot\|_0]$: The expected $L_0$ cost, calculated as the probability of the gate being non-zero: $\sigma(\log \alpha - \beta \log \frac{-l}{r})$.

### Dual-Stream Predictive Model

The network models the output $\hat{\mathbf{y}} \in \mathbb{R}^{C}$ as a gated summation of explicit univariate power terms and implicit multivariate interaction terms up to order $K$:

$$\hat{y}_c = b_c + \sum_{k=1}^{K} \left[ \underbrace{z_{\text{pure}}^{(k)} (\mathbf{w}_{\text{pure}}^{(k, c)})^\top \mathbf{x}^{ k}}_{\text{Explicit Power Stream}} + \underbrace{z_{\text{int}}^{(k)} \sum_{r=1}^{R} \left( \prod_{m=1}^{k} (\mathbf{u}_{k, m, r}^{(c)})^\top \mathbf{x} \right)}_{\text{Implicit Interaction Stream (CP)}} \right]$$

Variable Definitions:
- $c \in \{1, \dots, C\}$: The class index. Each class possesses independent weight parameters but shares the structural gates $z$.
- $b_c \in \mathbb{R}$: The bias term for class $c$.
- $\mathbf{x}^{k} \in \mathbb{R}^{D}$: The element-wise $k$-th power of the input vector.
- $\mathbf{w}_{\text{pure}}^{(k, c)} \in \mathbb{R}^{D}$: The learnable weight vector projecting the $k$-th order power term for class $c$.
- $R$: The fixed rank hyperparameter determining the capacity of the interaction approximation.
- $\mathbf{u}_{k, m, r}^{(c)} \in \mathbb{R}^{D}$: The factor vector corresponding to the $m$-th component of the $r$-th rank term for order $k$ and class $c$.

Operation Logic: The term $\prod_{m=1}^{k} (\mathbf{u}^\top \mathbf{x})$ computes the $r$-th interaction feature as a scalar product of $k$ linear projections, and $\sum_{r=1}^{R}$ aggregates these $R$ components directly, eliminating the need for external projection matrices.

Structural Gates: $z_{\text{pure}}^{(k)}, z_{\text{int}}^{(k)} \in \{0, 1\}$: The differentiable binary gates that determine whether the $k$-th order terms are included in the final model structure.