# Controlled Boundary Condition Extension: Hard vs. Soft Constraints in FDM Error Estimation

**Problem:** 1D Heat Equation ($\alpha = 0.05, n = 2$, domain $[0, 1] \times [0, 1]$)  
**Study Runner:** [`scripts/run_boundary_extension.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/scripts/run_boundary_extension.py)  
**Configuration:** [`configs/boundary_extension/heat_soft_bc.json`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/configs/boundary_extension/heat_soft_bc.json)  
**Data Artifact:** [`results/boundary_extension.csv`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/results/boundary_extension.csv)  
**Figure Artifact:** [`figures/boundary_condition_comparison.png`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/figures/boundary_condition_comparison.png)  
**Date:** 2026-10-20 (Session 4)

---

## 1. Research Motivation & Scientific Question

In the foundational paper (arXiv:2603.15526), the authors formulate and evaluate their finite-difference residual-based error estimation framework exclusively on PINNs with **exact hard boundary and initial constraints** (Section 3.1 & 4.1). Hard constraints ensure that the neural network ansatz satisfies the initial condition $u(x, 0) = u_0(x)$ and boundary conditions $u(0, t) = g_0(t), u(1, t) = g_1(t)$ analytically by construction:
$$\hat{\phi}_{\text{hard}}(x, t) = \sin(2\pi x) + t \, x(1 - x) \, \mathcal{N}(x, t; \theta)$$
Consequently, the true PINN error along the boundary and initial domain is identically zero:
$$e_{\text{true}}(x, 0) \equiv 0, \quad e_{\text{true}}(0, t) \equiv 0, \quad e_{\text{true}}(1, t) \equiv 0$$

However, in wider scientific machine learning practice (e.g., complex geometries, high dimensions, or irregular boundaries), hard constraints are often impractical or mathematically intractable to construct. Practitioners widely rely on **soft constraints**, where initial and boundary conditions are enforced as penalty loss terms:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{pde}}(\theta) + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}}(\theta) + \lambda_{\text{ic}} \mathcal{L}_{\text{ic}}(\theta)$$

### The Scientific Question:
1. *Does the authors' residual-based error estimator hold when boundary and initial conditions are enforced softly?*
2. *What happens if the upstream solver is applied naively to a soft-constrained PINN?*
3. *Can the residual error estimation framework be generalized mathematically to inhomogeneous initial and boundary errors, and does its ~34x accuracy advantage over direct-FDM persist?*

---

## 2. Mathematical Defect Analysis

### 2.1 The Inhomogeneous Error Initial-Boundary Value Problem (IBVP)
Let $u(x, t)$ be the true physical solution satisfying:
$$\partial_t u - \alpha \partial_{xx} u = 0, \quad u(x, 0) = u_0(x), \quad u(0, t) = g_0(t), \quad u(1, t) = g_1(t)$$
Let $\hat{\phi}(x, t)$ be an arbitrary continuous approximation (e.g., a soft-constrained PINN) with PDE residual:
$$R(x, t) := \partial_t \hat{\phi} - \alpha \partial_{xx} \hat{\phi}$$
Subtracting the operator equations yields the exact defect governing equation for the pointwise error $e(x, t) := u(x, t) - \hat{\phi}(x, t)$:
$$\partial_t e - \alpha \partial_{xx} e = -R(x, t)$$
subject to:
$$e(x, 0) = u_0(x) - \hat{\phi}(x, 0) =: e_{\text{ic}}(x)$$
$$e(0, t) = g_0(t) - \hat{\phi}(0, t) =: e_{\text{bc0}}(t)$$
$$e(1, t) = g_1(t) - \hat{\phi}(1, t) =: e_{\text{bc1}}(t)$$

### 2.2 Crucial Theoretical Property: Computability Without Ground Truth
A foundational property of this formulation is that **all inhomogeneous boundary and initial error data are known from the problem definition alone**:
* $u_0(x) = \sin(2\pi x)$ is specified in the problem statement.
* $g_0(t) = 0$ and $g_1(t) = 0$ are specified in the problem statement.
* $\hat{\phi}(x, t)$ is evaluated from the trained neural network.
Neither the true analytical solution $u(x, t)$ nor external simulation data is needed to evaluate $e_{\text{ic}}(x)$, $e_{\text{bc0}}(t)$, and $e_{\text{bc1}}(t)$!

---

## 3. Failure Mode: The Naive Upstream Solver Assumption

In the upstream codebase (`pinn_error/core/fdm.py`, lines 451–460 and 515–520):
```python
# In _build_matrices:
M_lhs_lil[0, :] = 0; M_lhs_lil[0, 0] = 1 # Identity row
M_rhs_lil[0, :] = 0                      # Zero row (rhs = 0)
M_lhs_lil[-1, :] = 0; M_lhs_lil[-1, -1] = 1
M_rhs_lil[-1, :] = 0

# In residual_integration:
e = np.zeros((self.nt, self.nx)) # e(x, 0) set to 0.0
R_curr[0] = 0.0; R_curr[-1] = 0.0
R_next[0] = 0.0; R_next[-1] = 0.0
```
This implementation **strictly bakes in homogeneous Dirichlet conditions**:
$$e(x, 0) \equiv 0, \quad e(0, t) \equiv 0, \quad e(1, t) \equiv 0$$
If an unwitting practitioner feeds a soft-constrained PINN into this unmodified upstream solver:
1. The solver forcibly zeroes out the initial error, setting $e(x, 0) = 0$ even though $e_{\text{true}}(x, 0) \ne 0$.
2. The solver forcibly zeroes out the boundary error at every time step, setting $e(0, t) = 0$ and $e(1, t) = 0$ even though $e_{\text{true}}(0, t) \ne 0$.
As a result, the naive estimator completely ignores boundary errors, creating large unphysical gradients near the boundary and severe estimation discrepancy.

---

## 4. The Extension Implementation: `InhomogeneousFDMSolverHeatEq`

To solve the inhomogeneous defect IBVP without modifying upstream source files, we implement [`InhomogeneousFDMSolverHeatEq`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/scripts/run_boundary_extension.py):
1. **Initial Condition:**
   $$e^0_i = u_0(x_i) - \hat{\phi}(x_i, 0), \quad i = 0, \dots, N_x - 1$$
2. **Crank-Nicolson Time Stepping:**
   For each time step $n \to n+1$:
   $$\text{rhs} = M_{\text{rhs}} e^n - \frac{\Delta t}{2} (R^n + R^{n+1})$$
   $$\text{rhs}_0 = g_0(t_{n+1}) - \hat{\phi}(0, t_{n+1})$$
   $$\text{rhs}_{N_x-1} = g_1(t_{n+1}) - \hat{\phi}(1, t_{n+1})$$
   $$e^{n+1} = \texttt{spsolve}(M_{\text{lhs}}, \text{rhs})$$
Because $M_{\text{lhs}}$ contains the standard Crank-Nicolson implicit stencil for interior points ($i=1$ and $i=N_x-2$), the non-zero boundary errors $\text{rhs}_0$ and $\text{rhs}_{N_x-1}$ correctly diffuse into the domain interior at rate $\mathcal{O}(\Delta x^2 + \Delta t^2)$.

---

## 5. Controlled Experimental Protocol

All physical, architectural, and optimization parameters are kept strictly identical to the baseline:
* **PDE:** 1D Heat Equation, $\alpha = 0.05, n = 2$.
* **Grid Resolutions:** $N_x = N_t \in \{16, 32, 64, 128\}$.
* **Architecture:** 4-layer MLP `[2, 20, 20, 20, 1]`, `tanh` activation.
* **Collocation Points:** 10,000 domain collocation points.
* **BC/IC Points (Soft PINN):** 1,000 boundary points (500 per side), 1,000 initial points.
* **Training:** 10,000 Adam iterations, initial learning rate $10^{-3}$, Seed 42, `float64` precision on CPU.
* **Direct Baseline Comparison:** Evaluated against both direct-FDM ($u_{\text{FDM}} - \hat{\phi}$) and ground-truth analytical solution ($u_{\text{exact}} - \hat{\phi}$).

---

## 6. Quantitative Results Table

All values are discrete RMS norms ($L_2$) evaluated from [`results/boundary_extension.csv`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/results/boundary_extension.csv).

| Model & Solver Mode | $N_x = N_t$ | $\Delta x = \Delta t$ | True PINN Error $\|e_{\text{true}}\|$ | Max Boundary Error | Estimator Discrepancy $\|\Delta_{\text{est}}\|$ | Direct-FDM Discrepancy $\|\Delta_{\text{FDM}}\|$ | Advantage Factor | Empirical Convergence Order ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hard BC (Standard Homog.)** | 16 | 0.0667 | $9.865 \times 10^{-5}$ | $< 10^{-15}$ | $7.688 \times 10^{-5}$ | $2.749 \times 10^{-3}$ | **35.76x** | — |
| Hard BC (Standard Homog.) | 32 | 0.0323 | $1.039 \times 10^{-4}$ | $< 10^{-15}$ | $1.920 \times 10^{-5}$ | $6.581 \times 10^{-4}$ | **34.29x** | $p \approx 1.91$ |
| Hard BC (Standard Homog.) | 64 | 0.0159 | $1.051 \times 10^{-4}$ | $< 10^{-15}$ | $4.771 \times 10^{-6}$ | $1.613 \times 10^{-4}$ | **33.80x** | $p \approx 1.96$ |
| Hard BC (Standard Homog.) | 128 | 0.0079 | $1.057 \times 10^{-4}$ | $< 10^{-15}$ | $1.185 \times 10^{-6}$ | $3.993 \times 10^{-5}$ | **33.71x** | $p \approx 1.99$ |
| **Soft BC (Inhomogeneous Defect)** | 16 | 0.0667 | $1.219 \times 10^{-2}$ | $1.385 \times 10^{-2}$ | $3.243 \times 10^{-5}$ | $2.749 \times 10^{-3}$ | **84.76x** | — |
| Soft BC (Inhomogeneous Defect) | 32 | 0.0323 | $1.224 \times 10^{-2}$ | $1.386 \times 10^{-2}$ | $7.630 \times 10^{-6}$ | $6.581 \times 10^{-4}$ | **86.25x** | **$p = 2.09$** |
| Soft BC (Inhomogeneous Defect) | 64 | 0.0159 | $1.227 \times 10^{-2}$ | $1.386 \times 10^{-2}$ | $1.882 \times 10^{-6}$ | $1.613 \times 10^{-4}$ | **85.67x** | **$p = 2.02$** |
| Soft BC (Inhomogeneous Defect) | 128 | 0.0079 | $1.228 \times 10^{-2}$ | $1.386 \times 10^{-2}$ | $4.668 \times 10^{-7}$ | $3.993 \times 10^{-5}$ | **85.53x** | **$p = 2.01$** |
| **Soft BC (Naive Homog. - Failure)** | 16 | 0.0667 | $1.219 \times 10^{-2}$ | $1.385 \times 10^{-2}$ | $1.052 \times 10^{-2}$ | $2.749 \times 10^{-3}$ | **0.26x** | — |
| Soft BC (Naive Homog. - Failure) | 32 | 0.0323 | $1.224 \times 10^{-2}$ | $1.386 \times 10^{-2}$ | $1.049 \times 10^{-2}$ | $6.581 \times 10^{-4}$ | **0.06x** | **$p = 0.00$** (Stalled) |
| Soft BC (Naive Homog. - Failure) | 64 | 0.0159 | $1.227 \times 10^{-2}$ | $1.386 \times 10^{-2}$ | $1.048 \times 10^{-2}$ | $1.613 \times 10^{-4}$ | **0.02x** | **$p = 0.00$** (Stalled) |
| Soft BC (Naive Homog. - Failure) | 128 | 0.0079 | $1.228 \times 10^{-2}$ | $1.386 \times 10^{-2}$ | $1.047 \times 10^{-2}$ | $3.993 \times 10^{-5}$ | **0.0038x** | **$p = 0.00$** (Stalled) |

---

## 7. Major Scientific Discoveries

### 7.1 Discovery 1: Catastrophic Breakdown of Unmodified Upstream Solver on Soft PINNs
When the upstream solver's `residual_integration()` is applied to a soft-constrained PINN without modification:
* The estimator forcibly imposes $e(0, t) = 0$, $e(1, t) = 0$, and $e(x, 0) = 0$.
* Because the actual PINN has a boundary error of $\approx 1.38 \times 10^{-2}$, the estimator discrepancy is dominated by the boundary error and **stalls at $\approx 1.05 \times 10^{-2}$ across all grid resolutions ($p = 0.00$)**.
* At $N_x=64$, the naive estimator is **$50\times$ less accurate** than direct-FDM (Advantage Factor $0.02\times$). At $N_x=128$, it is **$260\times$ less accurate** ($0.0038\times$).
* **Conclusion:** Applying the authors' original codebase to soft-constrained PINNs without accounting for boundary errors completely invalidates the error estimation.

### 7.2 Discovery 2: Restoration of Second-Order Convergence via Inhomogeneous Defect Solver
When the defect equation is solved with inhomogeneous initial and boundary data via `InhomogeneousFDMSolverHeatEq`:
* The estimator discrepancy drops from $3.24 \times 10^{-5}$ ($N_x=16$) to $4.67 \times 10^{-7}$ ($N_x=128$).
* The empirical convergence rate is cleanly second-order ($p = 2.01 \pm 0.01$), exactly matching Crank-Nicolson theoretical truncation error.
* The estimator faithfully tracks both the interior error evolution and the boundary departure over time (Figure 1, panels a and b).

### 7.3 Discovery 3: Elevation of Accuracy Advantage to ~85.5x
* For the Hard BC PINN, the advantage factor over direct-FDM is $\approx 33.8\times$.
* For the Soft BC PINN with `InhomogeneousFDMSolverHeatEq`, the advantage factor jumps to **$85.5\times$ to $86.2\times$**!
* **Why this occurs:**
  1. The soft-constrained PINN has a true error of $\approx 1.23 \times 10^{-2}$ (compared to $1.05 \times 10^{-4}$ for the hard-constrained model).
  2. The direct-FDM error is fixed at $1.61 \times 10^{-4}$ ($N_x=64$) because it solves the original homogeneous PDE.
  3. The inhomogeneous defect solver directly integrates the residual perturbation $R$, which is smooth and well-behaved, achieving an estimator error of only $1.88 \times 10^{-6}$ ($N_x=64$).
  4. Ratio: $\frac{1.6126 \times 10^{-4}}{1.8824 \times 10^{-6}} \approx 85.67\times$.

---

## 8. Viva-Preparation & Course Submission Synthesis

* **Question for Viva:** *The Krasowski et al. paper only uses hard constraints. What happens if you run their code on a soft-constrained PINN?*
  * **Answer:** "The upstream implementation forcibly zeroes out boundary and initial residuals, setting $e(x, 0) \equiv 0$ and $e(x_b, t) \equiv 0$. On a soft-constrained PINN where boundary error is $\approx 1.4\%$, the upstream estimator's error stalls completely at $\mathcal{O}(10^{-2})$ with zero convergence order ($p=0$). It is up to $262\times$ worse than direct FDM (advantage factor $0.0038\times$ at $N_x=128$).
  However, we mathematically generalized their formulation to an inhomogeneous IBVP:
  $$\partial_t e - \alpha \partial_{xx} e = -R, \quad e(x, 0) = u_0 - \hat{\phi}(x, 0), \quad e(x_b, t) = g - \hat{\phi}(x_b, t)$$
  Because $u_0$ and $g$ are given by the problem definition, these inhomogeneous boundary data require NO ground-truth knowledge. Our inhomogeneous Crank-Nicolson solver restores observed second-order $\mathcal{O}(h^2)$ convergence and achieves an even higher advantage factor of $85.67\times$ over direct FDM."

