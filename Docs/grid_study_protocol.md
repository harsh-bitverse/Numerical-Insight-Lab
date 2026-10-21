# Grid-Resolution Study Protocol

**Research Question:**  
How does the finite-difference grid resolution affect the accuracy of the residual-based error estimate for the 1D heat equation, and how does this behavior differ between trained and randomly initialized PINNs?

---

## 1. Experimental Matrix & Protocol Design

### 1.1 Controlled Variables
* **Governing PDE:** 1D Heat Equation, $\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0$.
* **Physics Parameters:** Diffusivity $\alpha = 0.05$, Frequency mode $n = 2$, Domain $[0, 1] \times [0, 1]$.
* **Hard Constraints:** $u(x, 0) = \sin(2\pi x)$, $u(0, t) = u(1, t) = 0$.
* **Precision & Device:** `float64`, CPU.
* **Random Seed:** Baseline seed `42`.

### 1.2 Independent Variable 1: Estimator Grid Resolution
Candidate grid discretizations:
$$N_x = N_t \in \{8, 16, 32, 64, 128, 256\}$$

| $N_x = N_t$ | Total Points $N$ | Spatial Spacing $\Delta x$ | Temporal Spacing $\Delta t$ | Solver Stability |
| :---: | :---: | :---: | :---: | :---: |
| 8 | 64 | $1/7 \approx 0.142857$ | $1/7 \approx 0.142857$ | Unconditionally Stable (Crank-Nicolson) |
| 16 | 256 | $1/15 \approx 0.066667$ | $1/15 \approx 0.066667$ | Unconditionally Stable (Crank-Nicolson) |
| 32 | 1,024 | $1/31 \approx 0.032258$ | $1/31 \approx 0.032258$ | Unconditionally Stable (Crank-Nicolson) |
| 64 | 4,096 | $1/63 \approx 0.015873$ | $1/63 \approx 0.015873$ | Unconditionally Stable (Crank-Nicolson) |
| 128 | 16,384 | $1/127 \approx 0.007874$ | $1/127 \approx 0.007874$ | Unconditionally Stable (Crank-Nicolson) |
| 256 | 65,536 | $1/255 \approx 0.003922$ | $1/255 \approx 0.003922$ | Unconditionally Stable (Crank-Nicolson) |

### 1.3 Independent Variable 2: PINN Training State
1. **Well-Trained Regime (`n_iter = 10000`):**
   * Collocation points: 10,000 domain points.
   * Model weights trained via Adam ($10^{-3}$) for 10,000 steps.
   * Cached to ensure the **exact same neural network prediction $\hat{\phi}$** is evaluated across all grid resolutions.
2. **Randomly Initialized Regime (`n_iter = 0`):**
   * Collocation points: 10,000 domain points.
   * Model weights initialized with Glorot normal, 0 optimization steps taken.
   * Cached to ensure the **exact same untrained neural network prediction $\hat{\phi}$** is evaluated across all grid resolutions.

### 1.4 Baseline Comparison Method: Direct-FDM Solution
* At each grid resolution, the standard FDM solver computes $u_{\text{FDM}}$ directly from the PDE and initial condition.
* The surrogate error estimate $e_{\text{FDM}} = u_{\text{FDM}} - u_{\text{pinn}}$ has discrepancy $\Delta_{\text{FDM}} = u_{\text{true}} - u_{\text{FDM}}$.

---

## 2. Separation of Experimental Effects

1. **Isolation of Estimator-Grid Effect:**
   The PINN is trained on continuous spatio-temporal coordinates and its weights are frozen. For each resolution $N_x = N_t$, the exact same frozen model is queried at grid points $(x_i, t_n)$ to compute residuals $R(x_i, t_n)$. Changing $N_x$ and $N_t$ does not alter the underlying PINN in any way.
2. **Comparison with Paper's Figure 3(c):**
   * Figure 3(c) plots convergence curves for `e_fdm`, `e_res (Well-trained)`, and `e_res (Random Init.)` against $N_x$.
   * Our protocol executes this exact 12-run experimental design and compares our measured convergence slopes directly with the 10-seed reference statistics from `all_experiments_results_float64.csv`.

---

## 3. Execution & Artifact Plan

* **Script:** `scripts/run_grid_study.py`
* **Configs:** `configs/grid_study/heat_grid_study.json`
* **Output CSV:** `results/grid_study.csv`
* **Raw Artifacts:** `results/grid_study/`
* **Generated Plot:** `figures/grid_resolution_comparison.png`
* **Analysis Document:** `Docs/grid_study_analysis.md`
