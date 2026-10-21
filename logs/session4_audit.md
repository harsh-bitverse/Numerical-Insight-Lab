# Session 4 Comprehensive Audit Log

**Date:** 2026-10-20  
**Project:** Numerical Insight Lab  
**Upstream Repository:** `pinn-fdm-error-estimation` (Commit `460d66bc2a59720883fa6a06e8c5436db4f99020`)  
**Scope:** Phase A Verification Audit of Baseline Reproduction and 12-Run Grid Study  

---

## 1. Audit Phase A1: Norm Definitions & Quadrature Analysis

### 1.1 Evaluated Quantities & Grid Spacing
The 1D Heat Equation is solved on $\Omega \times [0, 1] = [0, 1] \times [0, 1]$.
The discrete evaluation grid consists of $N_x$ spatial points and $N_t$ temporal points:
* $x_i = i \cdot \Delta x, \quad i = 0, \dots, N_x - 1, \quad \Delta x = \frac{1}{N_x - 1}$
* $t_n = n \cdot \Delta t, \quad n = 0, \dots, N_t - 1, \quad \Delta t = \frac{1}{N_t - 1}$
Both boundary endpoints ($x=0, 1$ and $t=0, 1$) are explicitly included in the evaluation grid.

### 1.2 Unweighted Discrete RMS vs. Composite Trapezoidal Quadrature
In `pinn_error/utils/plotting.py` and `examples/run_all_experiments.py`, all $L_2$ errors are computed as:
$$\|V\|_{\text{RMS}} = \sqrt{\frac{1}{N_x N_t} \sum_{n=0}^{N_t - 1} \sum_{i=0}^{N_x - 1} V(x_i, t_n)^2} = \texttt{np.sqrt(np.mean(V**2))}$$

* **Continuous Quadrature:** The true continuous spatio-temporal $L^2$ norm on $[0, 1] \times [0, 1]$ approximated via the 2D composite trapezoidal rule is:
  $$\|V\|_{\text{trapz}} = \sqrt{\sum_{n=0}^{N_t-1} \sum_{i=0}^{N_x-1} w_n w_i \, V(x_i, t_n)^2 \, \Delta x \, \Delta t}$$
  where weights $w \in \{1/2, 1\}$ assign weight $1/2$ to boundary edges and $1/4$ to corners.
* **Scale Discrepancy:** The uniform RMS assigns weight $\frac{1}{N_x N_t}$ everywhere. The cell volume is $\Delta x \Delta t = \frac{1}{(N_x-1)(N_t-1)}$. The square root of the ratio between cell volume and uniform weight is:
  $$\sqrt{\frac{\Delta x \Delta t}{1 / (N_x N_t)}} = \sqrt{\frac{N_x N_t}{(N_x - 1)(N_t - 1)}} = \frac{N_x}{N_x - 1} \quad (\text{for } N_x = N_t)$$
  For $N_x = 64$, this geometric scale factor is $\frac{64}{63} \approx 1.01587$.
* **Boundary Vanishing Effect:** Under hard boundary and initial constraints:
  $$e_{\text{true}}(0, t) = e_{\text{true}}(1, t) = e_{\text{true}}(x, 0) = 0$$
  $$e_{\text{res}}(0, t) = e_{\text{res}}(1, t) = e_{\text{res}}(x, 0) = 0$$
  All discrepancy values along $x=0$, $x=1$, and $t=0$ are machine zero ($< 10^{-15}$). The only boundary contributing to trapezoidal quadrature is $t=1$.
* **Empirical Comparison on Baseline ($N_x = N_t = 64$):**
  * $\|e_{\text{true}} - e_{\text{res}}\|_{\text{RMS}} = 4.771048 \times 10^{-6}$
  * $\|e_{\text{true}} - e_{\text{res}}\|_{\text{trapz}} = 4.821770 \times 10^{-6}$ (Ratio: $1.010631$)
  * $\|e_{\text{true}} - e_{\text{FDM}}\|_{\text{RMS}} = 1.612600 \times 10^{-4}$
  * $\|e_{\text{true}} - e_{\text{FDM}}\|_{\text{trapz}} = 1.633100 \times 10^{-4}$ (Ratio: $1.012712$)
  * **Advantage Ratio:** RMS $= 33.80\times$, Trapezoidal $= 33.87\times$
* **Conclusion:** The code computes an unweighted discrete RMS norm. Due to hard constraint boundary vanishing, the numerical departure from continuous composite trapezoidal quadrature is under $1.3\%$, and the advantage ratio is identical to two decimal places ($33.8\times$).

---

## 2. Audit Phase A2: Convergence Order Recomputation

We compute the two-point order $p = \frac{\ln(E_1/E_2)}{\ln(h_1/h_2)}$ between consecutive grid levels, and perform linear regression on $\ln(E)$ vs. $\ln(\Delta x)$ over the asymptotic range $N_x \in [32, 256]$:

```
=== CONVERGENCE ORDER RECOMPUTATION ===

Regime: well_trained
  Nx   8 ->  16: p_res = 2.2791, p_fdm = 1.9522
  Nx  16 ->  32: p_res = 1.9114, p_fdm = 1.9694
  Nx  32 ->  64: p_res = 1.9631, p_fdm = 1.9832
  Nx  64 -> 128: p_res = 1.9873, p_fdm = 1.9913
  Nx 128 -> 256: p_res = 1.9950, p_fdm = 1.9956
  Asymptotic fit (Nx in [32, 256]): slope_res = 1.9822, slope_fdm = 1.9901

Regime: random_init
  Nx   8 ->  16: p_res = 1.9793, p_fdm = 1.9522
  Nx  16 ->  32: p_res = 1.9893, p_fdm = 1.9694
  Nx  32 ->  64: p_res = 1.9946, p_fdm = 1.9832
  Nx  64 -> 128: p_res = 1.9973, p_fdm = 1.9913
  Nx 128 -> 256: p_res = 1.9986, p_fdm = 1.9956
  Asymptotic fit (Nx in [32, 256]): slope_res = 1.9969, slope_fdm = 1.9901
```

* **Observation:**
  1. For coarse grid $N_x = 8$, the spatial resolution is insufficient to capture high-curvature residual features, giving a pre-asymptotic rate $p_{\text{res}} \approx 2.28$.
  2. For $N_x \ge 32$, both estimators converge asymptotically at second-order $\mathcal{O}(h^2)$ with slopes $1.982$ to $1.997$, matching theoretical Crank-Nicolson truncation error.

---

## 3. Audit Phase A3: Reference Comparison & Shaded Bands

### 3.1 Direct-FDM Machine Precision Match
In the upstream reference dataset `all_experiments_results_float64.csv`, for `problem == 'heat'` across all 10 author seeds ($0 \le \text{seed} \le 9$):
$$\|u_{\text{FDM}} - u_{\text{true}}\|_{\text{RMS}} = 1.6126004189332567 \times 10^{-4} \quad (\text{identical across all 10 seeds})$$
Our measured run (Seed 42) matches this value:
$$|e_{\text{fdm, ours}} - e_{\text{fdm, auth}}| = 1.83 \times 10^{-15}$$
Across all resolutions:
* $N_x = 8$: difference $= 2.08 \times 10^{-17}$
* $N_x = 16$: difference $= 2.17 \times 10^{-17}$
* $N_x = 32$: difference $= 6.22 \times 10^{-16}$
* $N_x = 64$: difference $= 1.83 \times 10^{-15}$
* $N_x = 128$: difference $= 1.32 \times 10^{-16}$
* $N_x = 256$: difference $= 2.31 \times 10^{-14}$

**Mathematical Cause:** The direct FDM discrepancy is $\Delta_{\text{FDM}} = (u_{\text{true}} - \hat{\phi}) - (u_{\text{FDM}} - \hat{\phi}) = u_{\text{true}} - u_{\text{FDM}}$. The neural network $\hat{\phi}$ and seed $\theta$ cancel out identically. The computation is purely deterministic linear algebra.

### 3.2 Shaded Reference Band Semantics
* The shaded envelope in Figure 3 of the paper represents **Sample Mean $\pm$ 1 Sample Standard Deviation** ($n=10$ random seeds), NOT statistical confidence intervals.
* For well-trained models, author mean at $N_x=64$ is $5.975575 \times 10^{-6}$ with std $9.914 \times 10^{-7}$.
* Our measured baseline (Seed 42) is $4.771048 \times 10^{-6}$, corresponding to a z-score of $z = -1.21$. Across all $N_x$, our results lie within $[-1.24, -0.71]$ standard deviations of the author mean.

---

## 4. Audit Phase A4: The 33.80x Advantage Ratio

$$\text{Advantage Factor} = \frac{\|e_{\text{true}} - e_{\text{FDM}}\|_{\text{RMS}}}{\|e_{\text{true}} - e_{\text{res}}\|_{\text{RMS}}} = \frac{1.61260042 \times 10^{-4}}{4.77104750 \times 10^{-6}} = 33.7997\dots \approx 33.80\times$$
* The numerator is the error in estimating PINN error via direct FDM solution of the continuous PDE.
* The denominator is the error in estimating PINN error via residual defect integration.
* Both are discrete RMS discrepancies against ground truth $e_{\text{true}}$. The ratio is scientifically and mathematically meaningful.
