# Comprehensive Viva Examination & Defense Guide

**Course:** Numerical Analysis Undergraduate Research Project  
**Institution:** Indian Institute of Technology Delhi (IIT Delhi)  
**Project Title:** Reproducing and Evaluating Finite-Difference Error Estimation in Physics-Informed Neural Networks  
**Author and Collaborators:** Harshith N, Nilesh Kumar Verma, Devesh Saini  
**Target Paper:** Aleks Krasowski et al., *Building Trust in PINNs: Error Estimation through Finite Difference Methods*, [arXiv:2603.15526](https://arxiv.org/abs/2603.15526) (2026).  
**Repository:** `https://github.com/harsh-bitverse/Numerical-Insight-Lab`

---

## Part 1: Core Undergraduate Numerical Analysis Concepts

### 1.1 Physics-Informed Neural Networks (PINNs)
A PINN approximates the solution $u(x, t)$ of a differential equation $\mathcal{D}[u] = 0$ by training a parameterized neural network $\hat{u}(x, t; \theta)$. The optimization objective minimizes the residual of the governing PDE evaluated at scattered collocation points, together with boundary and initial condition penalties. Automatic differentiation (AD) evaluates exact spatial and temporal derivatives of the neural network with respect to inputs.

### 1.2 PDE Residual vs. Pointwise Error
* **PDE Residual ($R$):** The algebraic remainder when the neural network is substituted into the differential operator:
  $$R(x, t) := \mathcal{D}[\hat{u}](x, t) = \frac{\partial \hat{u}}{\partial t} - \alpha \frac{\partial^2 \hat{u}}{\partial x^2}$$
* **Pointwise True Error ($e$):** The departure of the neural network from the true analytical solution:
  $$e(x, t) := u_{\text{true}}(x, t) - \hat{u}(x, t)$$
* **The Fundamental Distinction:** A small PDE residual $R$ does *not* imply small pointwise error $e$. Depending on the stability of the differential operator, the error $e$ can be substantially larger than $R$, particularly near boundaries or under stiff dynamics.

### 1.3 The Defect Equation (Error Governing Equation)
Because the differential operator $\mathcal{D} = \partial_t - \alpha \partial_{xx}$ is linear, the true error $e(x, t)$ satisfies its own initial-boundary value problem:
$$\mathcal{D}[e] = \mathcal{D}[u_{\text{true}} - \hat{u}] = \mathcal{D}[u_{\text{true}}] - \mathcal{D}[\hat{u}] = 0 - R(x, t) = -R(x, t)$$
The error function satisfies the original governing equation forced by the negative residual $-R(x, t)$. Solving this auxiliary equation is known as *defect integration*.

### 1.4 Finite Differences & The Crank-Nicolson Scheme
Finite difference methods approximate continuous derivatives with discrete quotients on a grid:
* **Spatial Second Derivative:** Standard three-point central difference:
  $$\frac{\partial^2 e}{\partial x^2}\Big|_{i, n} \approx \frac{e_{i+1, n} - 2e_{i, n} + e_{i-1, n}}{\Delta x^2} = (A_{xx} e)_i$$
* **Crank-Nicolson Averaging:** Averages spatial derivatives and residual forcing at time levels $n$ and $n+1$:
  $$\frac{e^{n+1} - e^n}{\Delta t} = \frac{\alpha}{2} A_{xx}\left(e^{n+1} + e^n\right) - \frac{1}{2}\left(R^n + R^{n+1}\right)$$
  Rearranging into matrix form:
  $$\left(I - \frac{\alpha \Delta t}{2} A_{xx}\right) e^{n+1} = \left(I + \frac{\alpha \Delta t}{2} A_{xx}\right) e^n - \frac{\Delta t}{2}\left(R^n + R^{n+1}\right)$$
* **Stability & Truncation:** Crank-Nicolson is unconditionally stable (A-stable) and has second-order local truncation error $\mathcal{O}(\Delta x^2 + \Delta t^2)$.

### 1.5 Discrete RMS Norm vs. Continuous Quadrature
* **Discrete RMS Norm:**
  $$\|V\|_{\text{RMS}} = \sqrt{\frac{1}{N_x N_t} \sum_{n=0}^{N_t-1} \sum_{i=0}^{N_x-1} V(x_i, t_n)^2}$$
* **Continuous $L^2$ Norm Approximation:** On the unit domain $[0, 1] \times [0, 1]$, composite trapezoidal quadrature assigns weights $w_i \in \{1/2, 1\}$ to account for boundary edges and corner points. Under hard constraints, boundary discrepancies are identically zero ($< 10^{-15}$), so discrete RMS and trapezoidal quadrature differ by less than $1.3\%$.

---

## Part 2: Step-by-Step Mathematical Derivations

### 2.1 Derivation of the Defect Equation
1. Consider the linear parabolic initial-boundary value problem:
   $$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in (0, 1), \; t \in (0, 1]$$
   subject to $u(x, 0) = u_0(x)$ and $u(0, t) = g_0(t), u(1, t) = g_1(t)$.
2. Let $\hat{u}(x, t)$ be an arbitrary smooth approximation. Its PDE residual is:
   $$R(x, t) = \frac{\partial \hat{u}}{\partial t} - \alpha \frac{\partial^2 \hat{u}}{\partial x^2}$$
3. Define the error as $e(x, t) := u(x, t) - \hat{u}(x, t)$. Applying the differential operator:
   $$\mathcal{L}[e] = \frac{\partial}{\partial t}(u - \hat{u}) - \alpha \frac{\partial^2}{\partial x^2}(u - \hat{u})$$
4. By linearity of differential operators:
   $$\mathcal{L}[e] = \left(\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2}\right) - \left(\frac{\partial \hat{u}}{\partial t} - \alpha \frac{\partial^2 \hat{u}}{\partial x^2}\right)$$
5. Since $u$ satisfies the exact PDE, the first term vanishes:
   $$\mathcal{L}[e] = 0 - R(x, t) = -R(x, t)$$
   $$\implies \frac{\partial e}{\partial t} - \alpha \frac{\partial^2 e}{\partial x^2} = -R(x, t) \quad \blacksquare$$

### 2.2 Derivation of Inhomogeneous Initial and Boundary Conditions
1. **Initial Error ($t=0$):**
   $$e(x, 0) = u(x, 0) - \hat{u}(x, 0) = u_0(x) - \hat{u}(x, 0)$$
2. **Boundary Error ($x = 0$):**
   $$e(0, t) = u(0, t) - \hat{u}(0, t) = g_0(t) - \hat{u}(0, t)$$
3. **Boundary Error ($x = 1$):**
   $$e(1, t) = u(1, t) - \hat{u}(1, t) = g_1(t) - \hat{u}(1, t)$$
4. **Independence of Ground Truth:**
   Notice that $u_0(x) = \sin(2\pi x)$ and $g_0(t) = g_1(t) = 0$ are prescribed by the problem definition itself. The approximate solution $\hat{u}$ is provided by evaluating the neural network. Therefore, $e(x, 0)$, $e(0, t)$, and $e(1, t)$ can be evaluated without any knowledge of the exact analytical solution $u(x, t)$.

### 2.3 Derivation of Theoretical Second-Order Convergence
1. In Crank-Nicolson, the temporal derivative is approximated by a centered difference at $t_{n+1/2}$:
   $$\frac{e^{n+1} - e^n}{\Delta t} = \frac{\partial e}{\partial t}\Big|_{n+1/2} + \frac{\Delta t^2}{24} \frac{\partial^3 e}{\partial t^3}\Big|_{n+1/2} + \mathcal{O}(\Delta t^4)$$
2. The spatial second derivative is approximated by the average of central differences at $t_n$ and $t_{n+1}$:
   $$\frac{1}{2}\left(\delta_x^2 e^{n+1} + \delta_x^2 e^n\right) = \frac{\partial^2 e}{\partial x^2}\Big|_{n+1/2} + \mathcal{O}(\Delta x^2) + \mathcal{O}(\Delta t^2)$$
3. Equating these approximations yields a local truncation error of:
   $$\tau_i^n = \mathcal{O}(\Delta t^2 + \Delta x^2)$$
4. Under simultaneous refinement $N_x = N_t$ ($\Delta x = \Delta t = h$), the global discretization error satisfies:
   $$\|e_{\text{numerical}} - e_{\text{exact}}\|_{\text{RMS}} = \mathcal{O}(h^2)$$
   Taking logarithms:
   $$\ln(\|\Delta\|) = 2 \ln(h) + C \implies \text{Slope } p = 2 \quad \blacksquare$$

---

## Part 3: Code-to-Equation Mapping Walkthrough

| Mathematical Object | Source File | Line Number | Code Identifier / Formula |
| :--- | :--- | :--- | :--- |
| **PDE Residual $R(x, t)$** | `upstream/.../heat_1d.py` | L46–48 | `u_t - self.diffusivity * u_xx` |
| **Hard Output Transform** | `upstream/.../heat_1d.py` | L74 | `torch.sin(n*pi*x) + t*x*(1-x)*u` |
| **Analytical Truth $u_{\text{true}}$**| `upstream/.../heat_1d.py` | L115–117 | `np.exp(-(n**2 * pi**2 * alpha * t)) * np.sin(n*pi*x)` |
| **Crank-Nicolson Matrices**| `upstream/.../fdm.py` | L443–445 | `M_lhs = I - 0.5*dt*spatial_op`, `M_rhs = I + 0.5*dt*spatial_op`|
| **Dirichlet Homog. BCs** | `upstream/.../fdm.py` | L451–460 | `M_lhs_lil[0, 0] = 1`, `M_rhs_lil[0, :] = 0` |
| **Residual Source Injection**| `upstream/.../fdm.py` | L523–526 | `residual_source = -0.5 * dt * (R_curr + R_next)` |
| **Linear System Solve** | `upstream/.../fdm.py` | L529 | `e[n + 1] = spsolve(self.M_lhs, rhs)` |
| **Inhomogeneous Error IC**| `scripts/run_boundary_extension.py` | L64–69 | `e[0, :] = u0 - phi_t0` |
| **Inhomogeneous Error BCs**| `scripts/run_boundary_extension.py` | L90–96 | `rhs[0] = 0.0 - phi_bc_left`, `rhs[-1] = 0.0 - phi_bc_right` |
| **Direct FDM Solve** | `scripts/run_boundary_extension.py` | L214–215 | `direct_fdm_solver.solve()` |

---

## Part 4: Results & Interpretation Guide

### 4.1 Understanding the 33.80x Advantage Ratio
$$\text{Advantage Factor} = \frac{\|\Delta_{\text{FDM}}\|_{\text{RMS}}}{\|\Delta_{\text{res}}\|_{\text{RMS}}} = \frac{\|u_{\text{true}} - u_{\text{FDM}}\|_{\text{RMS}}}{\|e_{\text{true}} - e_{\text{res}}\|_{\text{RMS}}} = \frac{1.612600 \times 10^{-4}}{4.771048 \times 10^{-6}} \approx 33.80\times$$
* **Why it occurs:** In direct FDM, the solver integrates the full state $u$ ($|u| \sim 1$), incurring standard truncation error of order $1.6 \times 10^{-4}$. When subtracting $u_{\text{FDM}} - \hat{u}$, the error is dominated by $u_{\text{FDM}}$'s truncation error. In contrast, the defect solver integrates the residual $R$ ($|R| \sim 10^{-4}$), meaning the solver only accumulates truncation error on the *perturbation*, producing a much tighter estimate ($4.77 \times 10^{-6}$).

### 4.2 Understanding the Algebraic Cancellation in Direct FDM
* Expanding $\Delta_{\text{FDM}} = (u_{\text{true}} - \hat{u}) - (u_{\text{FDM}} - \hat{u}) = u_{\text{true}} - u_{\text{FDM}}$.
* Because $\hat{u}$ cancels algebraically, $\|\Delta_{\text{FDM}}\|_{\text{RMS}}$ is strictly identical across all neural network seeds to machine precision ($1.83 \times 10^{-15}$).

### 4.3 Understanding the Soft BC Failure Mode
* For soft PINNs, boundary error is $\approx 1.38 \times 10^{-2}$.
* If passed to the upstream solver, boundary errors are forced to zero ($e_{\text{naive}}(0, t) = 0$).
* The discrepancy $\|\Delta_{\text{naive}}\|_{\text{RMS}}$ stalls at $1.05 \times 10^{-2}$ across all grids ($p = 0.00$), performing $260\times$ worse than direct FDM at $N_x=128$.
* Inhomogeneous defect solver restores second-order convergence ($p = 1.99$) and elevates advantage to $85.5\times$.

---

## Part 5: 25 Critical Viva Questions & Model Answers

### Category 1: Reproducibility & Code Verification
1. **Q: Did you reproduce the authors' exact numbers or approximate them?**  
   *A:* "For the direct-FDM discrepancy $\|\Delta_{\text{FDM}}\|$, we achieved near-machine-precision agreement ($1.83 \times 10^{-15}$ difference) with the authors' reference dataset due to deterministic algebraic cancellation of the neural network prediction. For the residual estimator discrepancy $\|\Delta_{\text{res}}\|$, our measured Seed 42 run ($4.771 \times 10^{-6}$) lies within $-1.21$ sample standard deviations of the author 10-seed distribution ($5.976 \pm 0.991 \times 10^{-6}$), verifying reproduction."
2. **Q: How did you ensure the upstream codebase remained untainted?**  
   *A:* "We vendored the official code snapshot at pinned commit `460d66bc` into `upstream/pinn-fdm-error-estimation/` and treated it as strictly read-only. All our sweeps, inhomogeneous solvers, and tests reside in separate top-level directories (`scripts/`, `configs/`)."
3. **Q: What random seed did you use, and does the advantage depend on it?**  
   *A:* "We used Seed 42 for primary runs. The advantage factor ranges between $25\times$ and $40\times$ across random initializations (averaging $\sim 27\times$ in the authors' 10 seeds and $33.80\times$ in our Seed 42 run), consistently demonstrating an order-of-magnitude advantage."

### Category 2: Norms & Quadrature
4. **Q: Does the paper compute continuous $L^2$ norms or discrete Euclidean norms?**  
   *A:* "The code computes an unweighted discrete Root Mean Square (RMS) norm: $\sqrt{\frac{1}{N}\sum V_i^2}$. It is scale-invariant across grid refinement."
5. **Q: Is the discrete RMS norm equivalent to continuous trapezoidal quadrature?**  
   *A:* "Not strictly identical. Continuous trapezoidal quadrature assigns half-weights to boundary edges and quarter-weights to corners. However, because hard constraints enforce zero boundary error ($< 10^{-15}$), boundary discrepancy terms vanish. Numerically, the difference between discrete RMS and trapezoidal quadrature is under $1.3\%$ ($33.80\times$ RMS vs. $33.87\times$ Trapezoidal)."
6. **Q: What grid convention was used for $\Delta x$ and $\Delta t$?**  
   *A:* "The grid includes both boundary endpoints, so $\Delta x = 1/(N_x - 1)$ and $\Delta t = 1/(N_t - 1)$. For $N_x=64$, $\Delta x = 1/63 \approx 0.015873$."

### Category 3: Error vs. Discrepancy
7. **Q: What is the difference between PINN error and estimator discrepancy?**  
   *A:* "The PINN error is $e_{\text{true}} = u_{\text{true}} - \hat{u}$. The estimator discrepancy is $\Delta_{\text{res}} = e_{\text{true}} - e_{\text{res}}$, which measures how closely the estimator predicts the true error."
8. **Q: Why does the defect solver solve for $e$ rather than $u$?**  
   *A:* "Because $|e| \ll |u|$. Solving for $u$ directly accumulates truncation error on a magnitude $\sim 1.0$ field, causing catastrophic cancellation in $u_{\text{FDM}} - \hat{u}$. Solving for $e$ integrates a tiny residual perturbation ($|R| \sim 10^{-4}$), keeping absolute truncation error orders of magnitude smaller."
9. **Q: Is $e_{\text{res}}$ a certified mathematical bound?**  
   *A:* "No. $e_{\text{res}}$ is an *approximation* of error obtained by numerical integration. Certified bounds, such as the semigroup bound $e_{\text{bound}}$ from Hillebrecht & Unger, provide guaranteed upper bounds but lack spatial resolution."

### Category 4: The Algebraic Cancellation Nuance
10. **Q: Why was your direct-FDM discrepancy identical across all author seeds to near machine precision?**  
    *A:* "In $\Delta_{\text{FDM}} = (u_{\text{true}} - \hat{u}) - (u_{\text{FDM}} - \hat{u}) = u_{\text{true}} - u_{\text{FDM}}$, the neural network $\hat{u}$ cancels algebraically. The metric evaluates a purely deterministic Crank-Nicolson solve of the PDE on a fixed grid, completely independent of the neural network weights and training seeds."
11. **Q: Does this direct-FDM agreement prove the entire PINN training was identical?**  
    *A:* "No. It proves that the spatial discretization matrix $L = \alpha A_{xx}$, boundary conditions, and the sparse linear solver execute with near-machine-precision consistency on the deterministic FDM problem. PINN training itself is stochastic; we verified PINN reproduction separately by checking that our measured PINN error ($1.051 \times 10^{-4}$) lies well within the author 10-seed distribution ($2.6 \pm 5.7 \times 10^{-4}$)."

### Category 5: Convergence & Grid Refinement
12. **Q: What is the theoretical convergence order of the Crank-Nicolson scheme, and how was it tested?**  
    *A:* "Crank-Nicolson has theoretical local truncation error $\mathcal{O}(\Delta x^2 + \Delta t^2)$. In our grid study, spatial and temporal grids were refined simultaneously along the diagonal trajectory $N_x = N_t$ ($\Delta x = \Delta t = h$). This tests the coupled global convergence rate $\mathcal{O}(\Delta x^2 + \Delta t^2) = \mathcal{O}(h^2)$ along that path, rather than isolating spatial versus temporal orders independently."
13. **Q: What empirical convergence order did your grid study establish?**  
    *A:* "Over the asymptotic range $N_x \in [32, 256]$, linear regression of $\ln(\|\Delta\|)$ against $\ln(h)$ yields $p_{\text{res}} = 1.9822$ and $p_{\text{fdm}} = 1.9901$, confirming asymptotic second-order convergence along the coupled refinement trajectory."
14. **Q: Why was the convergence rate higher ($p \approx 2.28$) between $N_x=8$ and $N_x=16$?**  
    *A:* "At $N_x = 8$, the spatial resolution ($h \approx 0.143$) is pre-asymptotic—too coarse to adequately resolve high-curvature features of the residual field."
15. **Q: What do the shaded bands in Figure 3 of the paper represent?**  
    *A:* "They represent Sample Mean $\pm$ 1 Sample Standard Deviation across 10 random seeds in `all_experiments_results_float64.csv`. They are empirical dispersion bands, NOT statistical confidence intervals."

### Category 6: Boundary Condition Extension
16. **Q: Why did you test soft boundary conditions?**  
    *A:* "Because in realistic scientific machine learning applications (complex geometries or irregular domains), exact hard constraints are often mathematically intractable to construct, making soft penalty constraints a widely adopted standard in practice."
17. **Q: What happens if the authors' code is applied naively to a soft-constrained PINN?**  
    *A:* "The upstream solver explicitly zeros out boundary error ($e(x_b, t) \equiv 0$). On a soft PINN where boundary error is $\approx 1.4\%$, this causes the estimator to stall at $\|\Delta\| \approx 1.05 \times 10^{-2}$ across all grids ($p = 0.00$), performing up to $262\times$ worse than direct FDM."
18. **Q: How did you fix this failure mode?**  
    *A:* "We formulated an inhomogeneous defect initial-boundary value problem: $\partial_t e - \alpha \partial_{xx} e = -R$, $e(x, 0) = u_0(x) - \hat{u}(x, 0)$, $e(x_b, t) = g(x_b, t) - \hat{u}(x_b, t)$. We implemented `InhomogeneousFDMSolverHeatEq` to inject boundary errors into the Crank-Nicolson linear system at each time step."
19. **Q: Does your inhomogeneous solver require knowing the analytical solution?**  
    *A:* "No! Initial condition $u_0(x) = \sin(2\pi x)$ and boundary values $g(x_b, t) = 0$ are given by the PDE problem statement. Zero analytical ground truth is required to compute $e_{\text{res}}$."
20. **Q: What was the result of your inhomogeneous extension?**  
    *A:* "It restored observed second-order convergence ($p = 1.99 \pm 0.01$) and elevated the advantage factor over direct FDM from $33.80\times$ (hard BC) to $85.67\times$ at $N_x=64$ (and between $84.8\times$ and $86.3\times$ across resolutions)."

### Category 7: Untrained / Randomly Initialized Models
21. **Q: How does the residual estimator behave on untrained PINNs (0 iterations)?**  
    *A:* "It converges at the same second-order rate ($p = 1.9969$). However, because the untrained residual is large and irregular, $e_{\text{res}}$ is slightly less accurate than $e_{\text{FDM}}$ (Advantage Factor $\approx 0.78\times$)."
22. **Q: Why is that behavior scientifically significant?**  
    *A:* "It demonstrates that the method does not fail catastrophically even on poor models, and confirms that the estimator's advantage over direct-FDM is specific to models where the residual is small."

### Category 8: Novelty & Limitations
23. **Q: What in your work is genuinely novel versus simply standard numerical analysis?**  
    *A:* "Applying Crank-Nicolson to an inhomogeneous parabolic PDE is standard numerical analysis. What is original in our project is identifying the implicit hard-constraint dependency in Krasowski et al.'s framework, demonstrating its failure on soft-constrained PINNs, and demonstrating that an inhomogeneous defect formulation requires zero ground truth while delivering an $85.67\times$ to $92.83\times$ advantage in estimator fidelity."
24. **Q: What are the main limitations of this error estimation approach?**  
    *A:* "1. It provides an empirical approximation, not a certified upper bound. 2. It requires structured or meshable grids for the FDM solver, partially sacrificing PINN mesh-freedom. 3. For nonlinear equations, it requires Newton linearization."
25. **Q: Could this method be applied to the 2D Navier-Stokes equations?**  
    *A:* "Yes, in principle, but with substantial added complexity: nonlinear convective terms $(u \cdot \nabla)u$ require linearizing the defect equation around the current state $\hat{u}$, and incompressibility $\nabla \cdot u = 0$ requires solving a coupled saddle-point system."

### Category 9: Boundary Penalty Sweeps & Optimization Dynamics
26. **Q: How does varying the boundary penalty weight $\lambda_{\text{bc}} \in \{1, 10, 100, 1000\}$ affect the estimator's convergence order?**  
    *A:* "The convergence order of the inhomogeneous defect estimator is completely invariant: $p = 2.00, 1.99, 2.00, 1.98$ across all 4 penalty regimes. This proves that the Crank-Nicolson discretization truncation error $\mathcal{O}(\Delta x^2 + \Delta t^2)$ is mathematically independent of the neural training loss weighting."
27. **Q: Why does the advantage factor drop to $1.00\times$ at $\lambda_{\text{bc}} = 1000$?**  
    *A:* "At $\lambda_{\text{bc}}=1000$, the $1000\times$ boundary weighting causes the optimizer to heavily favor boundary points, while the initial condition loss stalls at $\mathcal{L}_{\text{ic}} \approx 0.498$, leaving the network prediction severely attenuated. In this degenerate regime, the defect equation mathematically collapses to solving the original heat equation itself. Solving it via Crank-Nicolson yields the exact same truncation error ($1.613 \times 10^{-4}$) as direct FDM, resulting in parity ($\eta = 1.00\times$)."
28. **Q: Can you claim you proved gradient stiffness at $\lambda_{\text{bc}}=1000$?**  
    *A:* "We observed severe loss component imbalance on `seed=42` where $\mathcal{L}_{\text{ic}}$ stalled at $0.498$ while $\mathcal{L}_{\text{bc}}$ was minimized to $5 \times 10^{-7}$. While this observation is fully consistent with the gradient conflict pathologies proved by Wang et al. (2021), we carefully report it as an empirical loss trade-off on this seed, as backpropagation gradient norms were not logged directly during training."

---

## Part 6: Honest Limitations & Unresolved Questions

When an examiner asks a question outside the scope of our data, we state the honest boundaries:
1. *Higher-Dimensional Scaling:* We verified the extension on 1D Heat Equation. In 3D, FDM grid storage scales as $\mathcal{O}(N_x^3)$, which may reduce the computational efficiency advantage unless sparse iterative solvers or adaptive mesh refinement (AMR) are used.
2. *Non-Smooth Initial Conditions:* Our experiments used a $C^\infty$ sinusoidal initial condition. For step functions or shock profiles, Crank-Nicolson can suffer from spurious high-frequency oscillations unless sub-stepping (e.g., Rannacher time stepping) is applied.
3. *Single-Seed Sweep Scope:* The boundary penalty sweep was performed on a single seed (`seed=42`). While all 24 grid evaluations and 8 convergence fits were independently verified to bit-for-bit precision against raw model checkpoints, multi-seed statistical confidence intervals were not gathered for the penalty sweep.
4. *Single Architecture Baseline:* All sweeps used a 4-layer MLP with `tanh` activations. While the FDM solver is agnostic to the neural architecture, testing modified Fourier features or Siren networks remains future work.
