# Baseline Validation & Cross-Verification Report

This document presents the detailed mathematical and empirical audit of the 1D Heat Equation baseline reproduction, verifying our measured run against the primary paper (arXiv:2603.15526) and the upstream reference dataset (`all_experiments_results_float64.csv`).

---

## 1. Baseline Configuration Audit

The baseline corresponds to the 1D Heat Equation problem featured in **Figure 2** and **Figure 3(c)** of the primary paper:

* **Governing PDE:** $\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0$ on $\Omega = (0, 1)$, $t \in [0, 1]$.
* **Physical Parameters:** Diffusivity $\alpha = 0.05$, Frequency mode $n = 2$.
* **Boundary & Initial Conditions:**
  * Initial Condition: $u(x, 0) = \sin(2\pi x)$
  * Dirichlet BCs: $u(0, t) = u(1, t) = 0$
* **Numerical Grid:** $N_x = 64$ spatial points, $N_t = 64$ temporal points ($\Delta x = \Delta t = 1/63 \approx 0.015873$).
* **Neural Network Configuration:**
  * Architecture: Multi-Layer Perceptron (MLP) `[2, 20, 20, 20, 1]`
  * Activation: `tanh`
  * Weight Initialization: Glorot normal
  * Optimizer: Adam, initial learning rate $10^{-3}$
  * Collocation Points: 10,000 domain points sampled uniformly
  * Training Iterations: 10,000 steps
  * Random Seed: 42
  * Floating-Point Precision: `float64`
  * Execution Device: CPU

---

## 2. Independent Verification of Direct-FDM Discrepancy

### 2.1 Near-Machine-Precision Direct-FDM Agreement
In Session 2, we reported:
$$\|e_{\text{true}} - e_{\text{FDM}}\|_{\text{RMS}} = 1.612600 \times 10^{-4}$$
We queried all matching records in the authors' reference CSV (`upstream/pinn-fdm-error-estimation/results/all_experiments_results_float64.csv`) for `problem == 'heat'`, `nx == 64`, `nt == 64`, `n_iterations == 10000`, `num_domain == 10000`:

| Seed | Author Record `pinn_time` | Author Record `pinn_error_l2` | Author Record `e_res_vs_true_l2` | Author Record `e_fdm_vs_true_l2` |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 95.42 s | $6.469261 \times 10^{-5}$ | $5.536043 \times 10^{-6}$ | **0.00016126004189332567** |
| 1 | 96.58 s | $7.857617 \times 10^{-5}$ | $5.762010 \times 10^{-6}$ | **0.00016126004189332567** |
| 2 | 94.08 s | $8.899296 \times 10^{-5}$ | $6.431995 \times 10^{-6}$ | **0.00016126004189332567** |
| 3 | 94.09 s | $8.788001 \times 10^{-5}$ | $4.053644 \times 10^{-6}$ | **0.00016126004189332567** |
| 4 | 92.54 s | $7.343985 \times 10^{-5}$ | $6.358266 \times 10^{-6}$ | **0.00016126004189332567** |
| 5 | 92.77 s | $6.350017 \times 10^{-5}$ | $5.383711 \times 10^{-6}$ | **0.00016126004189332567** |
| 6 | 93.11 s | $1.589004 \times 10^{-4}$ | $7.017099 \times 10^{-6}$ | **0.00016126004189332567** |
| 7 | 94.02 s | $1.891474 \times 10^{-3}$ | $7.023291 \times 10^{-6}$ | **0.00016126004189332567** |
| 8 | 94.30 s | $8.036395 \times 10^{-5}$ | $5.095495 \times 10^{-6}$ | **0.00016126004189332567** |
| 9 | 94.48 s | $7.951927 \times 10^{-5}$ | $7.094192 \times 10^{-6}$ | **0.00016126004189332567** |
| **Our Run (Seed 42)** | **161.77 s** | **$1.051475 \times 10^{-4}$** | **$4.771048 \times 10^{-6}$** | **0.00016126004189332567** |

### 2.2 Mathematical Explanation of the Identical Value
Notice that in every single row—across all 10 author seeds and our independent seed 42—`e_fdm_vs_true_l2` is **strictly identical**:
$$\|u_{\text{FDM}} - u_{\text{true}}\|_{\text{RMS}} = 1.6126004189332567 \times 10^{-4}$$
**Why does this happen?**
Because $u_{\text{FDM}}$ is the solution of the Crank-Nicolson finite difference scheme applied directly to the continuous PDE:
$$\left(I - \frac{1}{2}\Delta t L\right) u^{n+1} = \left(I + \frac{1}{2}\Delta t L\right) u^n, \quad u^0 = \sin(2\pi x)$$
This system does **not** query or depend on the neural network $\hat{\phi}$ or the random seed $\theta_0$ in any way. Given the fixed grid ($N_x=64, N_t=64$), spatial operator $L = \alpha A_{xx}$, and initial vector $u^0$, $u_{\text{FDM}}$ is a purely deterministic linear-algebraic computation.
Consequently, our measured value matches all 10 author reference runs down to machine precision:
$$|\|u_{\text{FDM}} - u_{\text{true}}\|_{\text{ours}} - \|u_{\text{FDM}} - u_{\text{true}}\|_{\text{auth}}| = 1.83 \times 10^{-15} \approx 0$$
Across all 6 grid resolutions ($N_x \in \{8, 16, 32, 64, 128, 256\}$), the difference is bounded by $\le 2.31 \times 10^{-14}$. This exact numerical match establishes that our local environment executes the identical numerical stencil and linear solver (`scipy.sparse.linalg.spsolve`) as the authors'.

---

## 3. Residual Error Discrepancy Statistical Validation

Unlike $u_{\text{FDM}}$, the residual error estimate $e_{\text{res}}$ depends on the trained PINN residual:
$$R(x, t) = \frac{\partial \hat{\phi}}{\partial t} - \alpha \frac{\partial^2 \hat{\phi}}{\partial x^2}$$
Because neural network weights $\hat{\theta}$ depend on stochastic initialization and random collocation sampling, the resulting PINN error and residual vary across seeds:

* **Author Distribution (Seeds 0–9):**
  * Mean: $\mu(e_{\text{res}}) = 5.975575 \times 10^{-6}$
  * Standard Deviation: $\sigma(e_{\text{res}}) = 9.914086 \times 10^{-7}$
  * Min: $4.053644 \times 10^{-6}$ (Seed 3)
  * Max: $7.094192 \times 10^{-6}$ (Seed 9)
* **Our Measured Value (Seed 42):**
  $$e_{\text{res}} = 4.771048 \times 10^{-6}$$
  $$\text{Z-score} = \frac{4.771048 \times 10^{-6} - 5.975575 \times 10^{-6}}{9.914086 \times 10^{-7}} = -1.215$$
Our result falls comfortably within the empirical envelope observed by the authors (between Seed 3 and Seed 8), confirming correct PINN training convergence and residual integration.

---

## 4. Recomputation of the 33.80x Accuracy Ratio

$$\text{Ratio} = \frac{\|e_{\text{true}} - e_{\text{FDM}}\|_{\text{RMS}}}{\|e_{\text{true}} - e_{\text{res}}\|_{\text{RMS}}} = \frac{1.6126004189332567 \times 10^{-4}}{4.771048107297839 \times 10^{-6}} = 33.799709\dots \approx 33.80\times$$
* **Significance:**
  * The numerator represents the error when assessing PINN predictions against a standard numerical solver on a $64 \times 64$ grid.
  * The denominator represents the error when estimating PINN predictions via the authors' residual integration method on the identical $64 \times 64$ grid.
  * Both metrics are discrete RMS discrepancies with respect to the exact analytical ground truth $e_{\text{true}}$. The $33.80\times$ ratio is mathematically fair and confirms that solving the defect equation $\mathcal{D}[e] = -R$ is over an order of magnitude more accurate than solving the original PDE $\mathcal{D}[u] = 0$.

---

## 5. Status of the Error-Bound Estimator ($e_{\text{bound}}$)

In our baseline run, `PINNErrorBoundEstimator` executed in $4.43$ seconds.
* **What it computes:** The semigroup-derived upper bound from Hillebrecht & Unger (2023, 2025):
  $$e_{\text{bound}}(t) = M e^{\omega t} \zeta_0 + \int_0^t M e^{\omega(t-s)} \zeta(s) \, ds$$
  where $M = 1$, $\omega = -0.05 \pi^2 \approx -0.49348$.
* **Role in Research Project:**
  * It provides a 1D scalar bound over time (Page 6, Figure 2e).
  * It does **not** provide spatial resolution across $x$.
  * It is conservative (overestimates actual error).
  * It is **not** part of the spatial grid resolution study (Figure 3), which focuses solely on $e_{\text{res}}$ vs. $e_{\text{FDM}}$ across spatial grids.
