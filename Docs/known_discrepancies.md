# Known Discrepancies & Code-Paper Audit

This document cataloges all mathematical inconsistencies, documentation mismatches, and implementation nuances identified during the source-grounded audit of the primary paper (*Building Trust in PINNs: Error Estimation through Finite Difference Methods*, arXiv:2603.15526) and its official implementation ([`upstream/pinn-fdm-error-estimation`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/)).

---

## 1. Discrepancy 1: Poisson Equation Sign Convention

### Mathematical Audit
* **Paper Text (Sec. 2.2, Page 3):**
  Defines $\mathcal{D}[u] = \frac{\partial^2 u}{\partial x^2}$ and states $\mathcal{D}[u] - f = 0 \implies \frac{\partial^2 u}{\partial x^2} - f = 0$.
* **Paper Table 1 (Page 5):**
  Lists the PDE as $\nabla^2 u - f = 0$.
* **Upstream README.md:**
  Lists 1D Poisson as $-\frac{\partial^2 u}{\partial x^2} - f(x) = 0$, but lists 2D Poisson with a severe typo:
  $$\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} - f(x) = 0 \quad \text{(Hyperbolic / Wave-like typo)}$$
* **Actual Code Implementation:**
  In [`pinn_error/problems/poisson.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/poisson.py):
  * 1D PDE (Line 69):
    ```python
    return -u_xx - f_x
    ```
  * 2D PDE (Line 166):
    ```python
    return -u_xx - u_yy - f_xy
    ```
  * In [`pinn_error/core/fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) (Line 181, Line 311):
    ```python
    spatial_operator = -1 * A_xx  # L = -d^2/dx^2
    ```
### Independent Analytical Verification
Consider exact solution $u(x) = \sin(\pi x / L)$ on $[0, L]$.
$$u''(x) = - \left(\frac{\pi}{L}\right)^2 \sin\left(\frac{\pi x}{L}\right)$$
Defining positive source term $f(x) = (\pi/L)^2 \sin(\pi x / L)$:
$$u''(x) = - f(x) \implies - u''(x) - f(x) = 0$$
Hence, **the code is mathematically correct**, while the paper text (Section 2.2 and Table 1) contains an ambiguous or missing negative sign ($\nabla^2 u = -f$, not $\nabla^2 u = +f$), and the README contains a spurious minus sign between $\partial^2/\partial x^2$ and $\partial^2/\partial y^2$.

---

## 2. Discrepancy 2: Drift-Diffusion Equation Sign Convention

### Mathematical Audit
* **Paper Table 1 (Page 5) & README:**
  $$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} - \beta \frac{\partial u}{\partial x} = 0$$
* **Actual Code Implementation ([`drift_diffusion.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/drift_diffusion.py#L55)):**
  ```python
  return u_t - self.diffusivity * u_xx + self.velocity_x * u_x
  ```
### Independent Analytical Verification
The exact solution implemented in code (Lines 106–110, 169–180) is:
$$u(x, t) = A \sin(\phi + k(x - \beta t)) e^{-\alpha k^2 t}$$
Differentiating with respect to $t$ and $x$:
$$\frac{\partial u}{\partial t} = -k \beta \cos(\dots) e^{-\alpha k^2 t} - \alpha k^2 u$$
$$\frac{\partial u}{\partial x} = k \cos(\dots) e^{-\alpha k^2 t}$$
$$\frac{\partial^2 u}{\partial x^2} = -k^2 u$$
Combining these:
$$\frac{\partial u}{\partial t} + \beta \frac{\partial u}{\partial x} - \alpha \frac{\partial^2 u}{\partial x^2} = (-k\beta \cos + k\beta \cos) e^{-\alpha k^2 t} + (-\alpha k^2 u - \alpha(-k^2 u)) = 0$$
Thus, the PDE that matches the traveling wave $(x - \beta t)$ moving to the right with speed $\beta > 0$ is $u_t - \alpha u_{xx} + \beta u_x = 0$. The paper Table 1 and README written with $-\beta u_x$ would imply motion in the negative direction unless $\beta < 0$. The code's internal formulation is self-consistent with its analytical solution.

---

## 3. Discrepancy 3: Drift-Diffusion Boundary Conditions

* **Paper Table 1 Claim:**
  "Periodic: $u(x_L) = u(x_R)$"
* **Actual Code Implementation:**
  * In [`drift_diffusion.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/drift_diffusion.py) (`output_transform`, lines 105–120), the PINN does **not** enforce periodic boundary conditions; it evaluates the analytical solution at the boundaries $g_{\min}(t)$ and $g_{\max}(t)$ and linearly interpolates them across the domain.
  * In [`fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) (`FDMSolverDriftDiffusion`, lines 648–665), the FDM solver does not use periodic wraparound matrix rows; it explicitly sets Dirichlet boundary values at every time step from `self.problem.exact_solution(x_min, t)`!
* **Impact on Evaluation:**
  This is a significant methodological divergence. In a real application without an exact solution, the FDM solver cannot query `exact_solution` to set time-dependent Dirichlet boundary conditions. This makes Drift-Diffusion the least autonomous benchmark in the paper.

---

## 4. Discrepancy 4: Wave Equation Initial Velocity Hard-Constraint

* **Paper Text (Sec. 2.1, Eq. 4):**
  Presents hard-constraining using a linear factor $(t - t_0)$:
  $$\phi(x, t) = u_0(x) + (t - t_0)(x - x_L)(x - x_R)\varphi(x, t)$$
* **Actual Code Implementation ([`wave.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/wave.py#L76)):**
  ```python
  return u * _t**2 * (_x - x_min) * (x_max - _x) + torch.sin(n * torch.pi * _x / x_max)
  ```
* **Explanation:**
  For the wave equation, there is a second initial condition: zero initial velocity $\frac{\partial u}{\partial t}(x, 0) = 0$.
  Differentiating the ansatz with $t^2$:
  $$\frac{\partial \phi}{\partial t} = 2t(x - x_L)(x_R - x)\varphi + t^2(x - x_L)(x_R - x)\frac{\partial \varphi}{\partial t}$$
  At $t=0$, this derivative vanishes identically, thereby satisfying $\partial_t u(x, 0) = 0$ by construction! The code's $t^2$ factor is an essential physics-informed adaptation unmentioned in Section 2.1.

---

## 5. Discrepancy 5: Platform Dependency & Hardcoded Paths in Plotting Code

* **Issue:**
  In [`examples/run_experiment.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/examples/run_experiment.py#L35-L46), the author hardcoded their local macOS font path:
  ```python
  fm.fontManager.addfont('/Users/krasowski/Library/Fonts/LibertinusSerif-Regular.ttf')
  mpl.rcParams['text.usetex'] = True
  ```
* **Impact:**
  On Windows systems or systems without a full TeX Live / MiKTeX installation, running `run_experiment.py` without `--no_show` or with the error bound plot enabled can crash with LaTeX rendering errors (`FileNotFoundError: [Errno 2] No such file or directory: 'latex'`).
* **Mitigation:**
  All headless runs must pass `--no_show` and/or override `mpl.rcParams['text.usetex'] = False` when generating local artifacts.

---

## 6. Discrepancy 6: Absence of Upstream Test Suite

* **Observation:**
  The upstream repository contains no unit tests or integration tests (`tests/` directory does not exist).
* **Research Risk:**
  Any latent regression or environment incompatibility cannot be detected via upstream tests.
* **Remedy:**
  Our team will implement modular verification tests in a dedicated `tests/` directory at the project root to rigorously validate solver matrices, boundary conditions, and analytical solutions against ground truth.

---

## 7. Unknowns Requiring Experimental Verification

1. **Stiffness in Wave Equation Time-Stepping:**
   Central differences for second-order time derivatives are conditionally stable ($r = c \frac{\Delta t}{\Delta x} \le 1$). We must verify whether finer grid resolutions ($N_x \ge 256$) encounter stability violations in the explicit solver.
2. **Smoothness of Residuals across Training Iterations:**
   The paper notes that $e_{\text{res}}$ degrades for randomly initialized PINNs because the residual $R(x, t)$ is less smooth. We must verify this transition quantitatively across intermediate training checkpoints ($10, 100, 1000, 10000$ iterations).
3. **Behavior under Soft Boundary Constraints:**
   The authors restricted all evaluations to hard-constrained models. As outlined in our team abstract, comparing hard vs. soft boundary constraints is our primary candidate extension once the baseline reproduction is solidified.
