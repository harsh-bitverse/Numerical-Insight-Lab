# Metric Definitions & Mathematical Formulations

This document specifies the exact mathematical definitions of all variables, fields, error metrics, and norms used across the paper, the official implementation (`pinn-fdm-error-estimation`), and our reproduction project.

---

## 1. Fundamental Fields & Discretization

Let the spatio-temporal domain for the 1D Heat Equation be:
$$\Omega = (x_{\min}, x_{\max}) = (0, 1), \quad t \in [0, t_{\max}] = [0, 1]$$
Discretized on a structured uniform grid with $N_x$ spatial points and $N_t$ temporal points:
$$x_i = x_{\min} + i \, \Delta x, \quad i = 0, \dots, N_x - 1, \quad \Delta x = \frac{x_{\max} - x_{\min}}{N_x - 1}$$
$$t_n = t_{\min} + n \, \Delta t, \quad n = 0, \dots, N_t - 1, \quad \Delta t = \frac{t_{\max} - t_{\min}}{N_t - 1}$$
Total grid evaluation points: $N = N_x \times N_t$.

### 1.1 Field Definitions
1. **Exact Analytical Solution ($u_{\text{true}}$):**
   $$u_{\text{true}}(x, t) = e^{-n^2 \pi^2 \alpha t} \sin(n \pi x), \quad \text{for } \alpha = 0.05, \, n = 2$$
2. **PINN Prediction ($\hat{\phi}$ or $u_{\text{pinn}}$):**
   The continuous neural network output satisfying exact hard constraints:
   $$u_{\text{pinn}}(x, t) = \sin(2\pi x) + t \, x (1 - x) \, \mathcal{N}(x, t; \theta)$$
3. **True Pointwise PINN Error ($e_{\text{true}}$):**
   $$e_{\text{true}}(x, t) := u_{\text{true}}(x, t) - u_{\text{pinn}}(x, t)$$
   *Sign Convention:*
   * $e_{\text{true}} > 0$: The PINN underpredicts the true physical state.
   * $e_{\text{true}} < 0$: The PINN overpredicts the true physical state.
4. **PINN PDE Residual ($R$):**
   $$R(x, t) := \frac{\partial u_{\text{pinn}}}{\partial t} - \alpha \frac{\partial^2 u_{\text{pinn}}}{\partial x^2}$$
5. **Residual-Based Error Estimate ($e_{\text{res}}$):**
   The numerical solution to the defect equation:
   $$\frac{\partial e}{\partial t} - \alpha \frac{\partial^2 e}{\partial x^2} = -R(x, t), \quad e(x, 0) = 0, \quad e(0, t) = e(1, t) = 0$$
   integrated forward using Crank-Nicolson finite differences.
6. **Direct-FDM Numerical Solution ($u_{\text{FDM}}$):**
   The numerical solution to the original heat equation:
   $$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0, \quad u(x, 0) = \sin(2\pi x), \quad u(0, t) = u(1, t) = 0$$
   computed on the identical $(N_x, N_t)$ grid using Crank-Nicolson.
7. **Direct-FDM Estimated PINN Error ($e_{\text{FDM}}$):**
   The conventional error estimate obtained by comparing the PINN against a numerical grid solver:
   $$e_{\text{FDM}}(x, t) := u_{\text{FDM}}(x, t) - u_{\text{pinn}}(x, t)$$

---

## 2. Discrepancy & Estimation Error Definitions

To evaluate how well an estimator predicts the true error $e_{\text{true}}$, we define the estimator discrepancies:

### 2.1 Residual Method Discrepancy
$$\Delta_{\text{res}}(x, t) := e_{\text{true}}(x, t) - e_{\text{res}}(x, t)$$
This measures the point-by-point departure of the authors' residual estimate from ground-truth error.

### 2.2 Direct-FDM Method Discrepancy
$$\Delta_{\text{FDM}}(x, t) := e_{\text{true}}(x, t) - e_{\text{FDM}}(x, t)$$
Expanding the definitions:
$$\Delta_{\text{FDM}}(x, t) = \left(u_{\text{true}} - u_{\text{pinn}}\right) - \left(u_{\text{FDM}} - u_{\text{pinn}}\right) = u_{\text{true}}(x, t) - u_{\text{FDM}}(x, t)$$
Notice that the neural network prediction $u_{\text{pinn}}$ cancels out identically!
Therefore, the error of $e_{\text{FDM}}$ as an estimator of $e_{\text{true}}$ is **identically equal** to the truncation/discretization error of the FDM numerical solver:
$$\Delta_{\text{FDM}}(x, t) = u_{\text{true}}(x, t) - u_{\text{FDM}}(x, t)$$

---

## 3. Norms & Summary Metrics

In the code (`pinn_error/utils/plotting.py` and `examples/run_all_experiments.py`), all reported $L_2$ metrics are defined as **Root Mean Square (RMS)** discrete norms:

### 3.1 Discrete RMS $L_2$ Norm vs. Continuous Trapezoidal Quadrature
For any 2D grid array $V \in \mathbb{R}^{N_t \times N_x}$ with flattened length $N = N_x N_t$:
$$\|V\|_{\text{RMS}} := \sqrt{\frac{1}{N} \sum_{i=1}^N V_i^2} = \sqrt{\frac{1}{N_x N_t} \sum_{n=0}^{N_t-1} \sum_{i=0}^{N_x-1} V(x_i, t_n)^2}$$

*Critical Mathematical Clarification (Audit Phase A1):*
* **What the authors compute:** The authors' code (`pinn_error/utils/plotting.py` and `examples/run_all_experiments.py`) and paper evaluate the unweighted **discrete Root Mean Square (RMS)** norm: `np.sqrt(np.mean(V**2))`.
* **Endpoints and Boundaries:** The grid includes all boundary points: $x_0 = 0, x_{N_x-1} = 1$ with $\Delta x = 1/(N_x-1)$, and $t_0 = 0, t_{N_t-1} = 1$ with $\Delta t = 1/(N_t-1)$.
* **Comparison with Continuous $L^2$ Quadrature:**
  The continuous $L^2$ norm on the unit domain $\Omega \times [0, 1] = [0, 1] \times [0, 1]$ is:
  $$\|v\|_{L^2} = \left(\int_0^1 \int_0^1 |v(x, t)|^2 \, dx \, dt\right)^{1/2}$$
  The standard 2D composite trapezoidal quadrature assigns weights $w_i \in \{1/2, 1\}$ and $w_n \in \{1/2, 1\}$, with boundary edges weighted by $1/2$ and the four corner vertices weighted by $1/4$:
  $$\|V\|_{\text{trapz}} := \sqrt{\sum_{n=0}^{N_t-1} \sum_{i=0}^{N_x-1} w_n w_i \, V(x_i, t_n)^2 \, \Delta x \, \Delta t}$$
  The unweighted discrete RMS norm assigns equal weight $\frac{1}{N_x N_t}$ to all points, rather than trapezoidal boundary weights and $\Delta x \Delta t = \frac{1}{(N_x-1)(N_t-1)}$.
* **Why the Numerical Difference is Negligible for Hard Constraints:**
  Under hard boundary constraints, the error along boundaries is identically zero:
  $$e_{\text{true}}(0, t) = e_{\text{true}}(1, t) = e_{\text{true}}(x, 0) = 0$$
  $$e_{\text{res}}(0, t) = e_{\text{res}}(1, t) = e_{\text{res}}(x, 0) = 0$$
  $$e_{\text{FDM}}(0, t) = e_{\text{FDM}}(1, t) = e_{\text{FDM}}(x, 0) = 0$$
  Consequently, boundary discrepancy values are machine zero ($< 10^{-15}$). The only boundary with non-zero error is the final time slice $t=1$.
  The ratio of trapezoidal quadrature to unweighted RMS is primarily driven by the grid spacing scale factor:
  $$\sqrt{\frac{\Delta x \Delta t}{1 / (N_x N_t)}} = \sqrt{\frac{N_x N_t}{(N_x - 1)(N_t - 1)}} = \frac{N_x}{N_x - 1}$$
  For $N_x = N_t = 64$, this factor is $64 / 63 \approx 1.01587$.
  Empirically measured on our baseline:
  * Residual discrepancy: RMS $= 4.771048 \times 10^{-6}$, Trapezoidal $= 4.821770 \times 10^{-6}$ (ratio $1.010631$)
  * Direct-FDM discrepancy: RMS $= 1.612600 \times 10^{-4}$, Trapezoidal $= 1.633100 \times 10^{-4}$ (ratio $1.012712$)
  * Advantage Factor: RMS $= 33.80\times$, Trapezoidal $= 33.87\times$
  Therefore, while the code uses an unweighted discrete RMS norm, its scaling is within $1.3\%$ of the continuous trapezoidal $L^2$ norm, and the relative advantage factor is preserved to within $0.2\%$. Both metrics are scale-invariant across grid refinements.

### 3.2 Maximum Absolute Norm
$$\|V\|_{\max} := \max_{i=1,\dots,N} |V_i|$$

### 3.3 Relative $L_2$ Error
$$\text{Rel}_{L2}(V, U) := \frac{\|V\|_{\text{RMS}}}{\|U\|_{\text{RMS}} + \epsilon}, \quad \epsilon = 10^{-16}$$

---

## 4. Code Implementation Mapping

| Mathematical Symbol | Variable in `plotting.py` | Column in `run_all_experiments.py` CSV | Formula in Code |
| :--- | :--- | :--- | :--- |
| $\|e_{\text{true}}\|_{\text{RMS}}$ | `True PINN Error L2` | `pinn_error_l2` | `np.sqrt(np.mean(e_true**2))` |
| $\|e_{\text{true}}\|_{\max}$ | `True PINN Error Max` | (Not in CSV) | `np.max(np.abs(e_true))` |
| $\|e_{\text{res}}\|_{\text{RMS}}$ | `Est. L2` (Residual) | (Not in CSV) | `np.sqrt(np.mean(err_estimated**2))` |
| $\|\Delta_{\text{res}}\|_{\text{RMS}}$ | `Diff L2` (Residual) | `e_res_vs_true_l2` | `np.sqrt(np.mean((e_true - e_res)**2))` |
| $\|\Delta_{\text{res}}\|_{\max}$ | `Diff Max` (Residual) | (Not in CSV) | `np.max(np.abs(e_true - e_res))` |
| $\|e_{\text{FDM}}\|_{\text{RMS}}$ | `Est. L2` (FDM) | (Not in CSV) | `np.sqrt(np.mean(fdm_based_error**2))` |
| $\|\Delta_{\text{FDM}}\|_{\text{RMS}}$ | `Diff L2` (FDM) | `e_fdm_vs_true_l2` | `np.sqrt(np.mean((e_true - e_fdm)**2))` |
| $\|\Delta_{\text{FDM}}\|_{\max}$ | `Diff Max` (FDM) | (Not in CSV) | `np.max(np.abs(e_true - e_fdm))` |
| **Advantage Factor** | `better by factor` | (Computed as ratio) | `max(res, fdm) / min(res, fdm)` |

---

## 5. The Advantage Factor Ratio

The reported factor:
$$\text{Advantage Factor} := \frac{\|\Delta_{\text{FDM}}\|_{\text{RMS}}}{\|\Delta_{\text{res}}\|_{\text{RMS}}} = \frac{\|u_{\text{true}} - u_{\text{FDM}}\|_{\text{RMS}}}{\|e_{\text{true}} - e_{\text{res}}\|_{\text{RMS}}}$$
In our verified baseline run:
$$\text{Advantage Factor} = \frac{1.61260042 \times 10^{-4}}{4.771048 \times 10^{-6}} = 33.7997\dots \approx 33.80\times$$
This ratio is mathematically and scientifically sound: it compares the error of the direct numerical baseline in estimating PINN error against the error of the proposed residual integration method in estimating PINN error.
