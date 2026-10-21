# Research Paper Summary: Building Trust in PINNs

## 1. Paper Metadata

* **Title:** Building Trust in PINNs: Error Estimation through Finite Difference Methods
* **Authors:** Aleksander Krasowski$^{1,*}$, René P. Klausen$^1$, Aycan Celik$^1$, Sebastian Lapuschkin$^{1,4}$, Wojciech Samek$^{1,2,3}$, Jonas Naujoks$^{1,*}$
* **Affiliations:**
  1. Department of Artificial Intelligence, Fraunhofer Heinrich Hertz Institute (HHI), Berlin
  2. Department of Electrical Engineering and Computer Science, Technische Universität Berlin
  3. BIFOLD – Berlin Institute for the Foundations of Learning and Data
  4. Centre of Explainable Artificial Intelligence, Technological University Dublin
* **Preprint:** [arXiv:2603.15526](https://arxiv.org/abs/2603.15526) [cs.LG]
* **Target Venue:** Submitted to the XAI World Conference 2026
* **Official Repository:** [https://github.com/aleks-krasowski/pinn-fdm-error-estimation](https://github.com/aleks-krasowski/pinn-fdm-error-estimation)

---

## 2. Research Question & Stated Contribution

### The Core Problem
Physics-Informed Neural Networks (PINNs) train neural networks $\phi(x; \theta)$ to approximate solutions to partial differential equations (PDEs) by minimizing a loss function penalizing PDE residual violations and boundary/initial condition errors. However:
1. Training loss or PDE residual alone does **not** quantify how far the prediction deviates point-by-point from the true ground-truth solution $u(x)$.
2. Small residual values can mask localized large errors, creating a false sense of trust.
3. In safety-critical scientific computing, practitioners require spatially and temporally resolved error maps to decide *where* and *by how much* a PINN prediction is trustworthy without access to an exact analytical solution.

### Stated Contributions
1. **Defect/Error Equation Formulation:** The authors exploit the linearity of the underlying differential operator $\mathcal{D}$. For any linear PDE, the pointwise error satisfies the same operator $\mathcal{D}$ driven by the negative PINN residual $-R(x)$.
2. **Hard-Constrained Boundaries:** By strictly hard-constraining initial and boundary conditions (IBCs), the error along all domain boundaries is identically zero ($e|_{\partial \Omega} = 0, e|_{t=0} = 0$). Hence, the interior PDE residual is the *sole* source of error.
3. **FDM Integration of the Residual:** The authors solve this error equation numerically using classical, lightweight Finite Difference Methods (FDM) on structured grids.
4. **Empirical Superiority:** The resulting residual-based error estimate ($e_{\text{res}}$) is demonstrated across four benchmark PDEs to be orders-of-magnitude more accurate than standard coarse FDM solutions ($e_{\text{FDM}} = u_{\text{FDM}} - \hat{\phi}$), while remaining far more spatially resolved and less conservative than semigroup-derived certified scalar upper bounds ($e_{\text{bound}}$).

---

## 3. Mathematical Derivation of the Error Equation

Let $\Omega \subset \mathbb{R}^n$ be an open, bounded, connected domain with smooth boundary components $\Gamma_k \subseteq \partial \Omega$.
Consider the linear initial-boundary value problem (IBVP):
$$\mathcal{D}[u](x) = 0, \quad x \in \Omega \tag{1}$$
$$\mathcal{B}_k[u](x) = 0, \quad x \in \Gamma_k, \; k=1,\dots,K \tag{2}$$

Let $\hat{\phi}(x) := \phi(x; \hat{\theta})$ denote a trained PINN with parameters $\hat{\theta}$.

### 3.1 Error Definition & Sign Convention
* **Pointwise Error:**
  $$e(x) := u(x) - \hat{\phi}(x) \tag{3}$$
  *(Note: A positive error indicates the PINN underpredicts the true solution; a negative error indicates overprediction).*
* **PINN PDE Residual:**
  $$R(x) := \mathcal{D}[\hat{\phi}](x) \tag{4}$$

### 3.2 Linear Defect Equation
Applying the linear differential operator $\mathcal{D}$ to $e(x)$:
$$\mathcal{D}[e](x) = \mathcal{D}[u - \hat{\phi}](x) = \mathcal{D}[u](x) - \mathcal{D}[\hat{\phi}](x)$$
Since $u$ is the exact solution, $\mathcal{D}[u](x) \equiv 0$. Substituting (4):
$$\mathcal{D}[e](x) = 0 - R(x) = -R(x) \tag{5}$$

### 3.3 Boundary & Initial Conditions for Error
By hard-constraining the PINN representation:
$$\hat{\phi}(x) = u(x) \quad \forall x \in \Gamma_k \implies e(x) = 0 \quad \forall x \in \Gamma_k \tag{6}$$
For time-dependent problems with initial condition $u(x, 0) = u_0(x)$:
$$\hat{\phi}(x, 0) = u_0(x) \implies e(x, 0) = 0 \tag{7}$$
Thus, the error field $e(x)$ satisfies an identical boundary-value problem to the original PDE, but with **homogeneous boundary conditions** and driven by the source term $-R(x)$.

---

## 4. Benchmark PDEs & Mathematical Verification

The paper evaluates four linear benchmark problems (Table 1, Page 5).

### 4.1 1D Heat Equation (Parabolic)
* **Domain:** $x \in [0, 1]$, $t \in [0, 1]$.
* **PDE:**
  $$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0$$
* **Paper Parameters:** $\alpha = 1/20 = 0.05$, initial frequency mode $n = 2$.
* **Initial Condition:** $u(x, 0) = \sin\left(\frac{2\pi x}{x_{\max}}\right) = \sin(2\pi x)$.
* **Boundary Conditions:** Homogeneous Dirichlet: $u(0, t) = u(1, t) = 0$.
* **Analytical Solution:**
  $$u(x, t) = e^{-\frac{n^2 \pi^2 \alpha t}{L^2}} \sin\left(\frac{n\pi x}{L}\right) = e^{-4\pi^2 (0.05) t} \sin(2\pi x) = e^{-0.2\pi^2 t} \sin(2\pi x)$$
* **Verification:**
  $$\frac{\partial u}{\partial t} = -0.2\pi^2 e^{-0.2\pi^2 t} \sin(2\pi x)$$
  $$\frac{\partial^2 u}{\partial x^2} = -4\pi^2 e^{-0.2\pi^2 t} \sin(2\pi x)$$
  $$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = (-0.2\pi^2 + 0.05 \times 4\pi^2) u = 0 \quad \checkmark$$

### 4.2 1D Wave Equation (Hyperbolic)
* **Domain:** $x \in [0, 1]$, $t \in [0, 1]$.
* **PDE:**
  $$\frac{\partial^2 u}{\partial t^2} - c^2 \frac{\partial^2 u}{\partial x^2} = 0$$
* **Paper Parameters:** $c = 0.5$, $n = 2$.
* **Initial Conditions:**
  * Displacement: $u(x, 0) = \sin(2\pi x)$.
  * Velocity: $\frac{\partial u}{\partial t}(x, 0) = 0$.
* **Boundary Conditions:** Homogeneous Dirichlet: $u(0, t) = u(1, t) = 0$.
* **Analytical Solution:**
  $$u(x, t) = \sin\left(\frac{n\pi x}{L}\right) \cos\left(\frac{c n \pi t}{L}\right) = \sin(2\pi x) \cos(\pi t)$$
* **Verification:**
  $$\frac{\partial^2 u}{\partial t^2} = -\pi^2 \sin(2\pi x) \cos(\pi t)$$
  $$\frac{\partial^2 u}{\partial x^2} = -4\pi^2 \sin(2\pi x) \cos(\pi t)$$
  $$\frac{\partial^2 u}{\partial t^2} - c^2 \frac{\partial^2 u}{\partial x^2} = (-\pi^2 - (0.5)^2 (-4\pi^2)) u = (-\pi^2 + \pi^2) u = 0 \quad \checkmark$$

### 4.3 1D & 2D Poisson Equation (Elliptic) — Rigorous Sign Verification
* **1D Case Domain:** $x \in [0, 1]$.
  * **Analytical Solution:** $u(x) = \sin(\pi x / L) = \sin(\pi x)$.
  * Differentiating:
    $$u'(x) = \pi \cos(\pi x), \quad u''(x) = -\pi^2 \sin(\pi x)$$
  * Therefore:
    $$u''(x) = - \pi^2 \sin(\pi x) = - f(x), \quad \text{where } f(x) = \pi^2 \sin(\pi x)$$
  * In the paper's text (Page 3, Section 2.2), the operator is written $\mathcal{D}[u] = \frac{\partial^2 u}{\partial x^2}$ with $\mathcal{D}[u] - f = 0$. But if $u'' - f = 0$, then $u'' = +f$, which contradicts the negative sign of $\sin''$.
  * **Code Verification:** In `pinn_error/problems/poisson.py`:
    $$\text{pde}: \quad -u_{xx} - f(x) = 0 \iff -(-\pi^2 \sin(\pi x)) - \pi^2 \sin(\pi x) = 0 \quad \checkmark$$
  * In `FDMSolverPoisson1D`: the spatial operator is assembled as $L = -A_{xx}$. Thus $L u = f \implies -u'' = f$, which is mathematically self-consistent.
* **2D Case Domain:** $(x, y) \in [0, 1]^2$.
  * **Exact Solution:** $u(x, y) = \sin(\pi x / L_x) \sin(\pi y / L_y) = \sin(\pi x)\sin(\pi y)$.
  * Laplacian: $\nabla^2 u = u_{xx} + u_{yy} = -\pi^2 u - \pi^2 u = -2\pi^2 u$.
  * Source term: $f(x, y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$.
  * The governing equation is $-\nabla^2 u - f(x, y) = 0$.
  * *Note on README typo:* Upstream README states $\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} - f = 0$, which is a wave-like hyperbolic typo. The actual implementation in `Poisson2D` correctly computes $-u_{xx} - u_{yy} - f = 0$.

### 4.4 1D Drift-Diffusion (Convection-Diffusion)
* **Domain:** $x \in [0, 1]$, $t \in [0, 1]$.
* **PDE Form in Code:**
  $$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} + \beta \frac{\partial u}{\partial x} = 0$$
  *(Note: Table 1 in paper writes $-\beta \frac{\partial u}{\partial x}$, which has the opposite sign from code's `+ self.velocity_x * u_x`).*
* **Parameters:** $\alpha = 1/20 = 0.05$, $\beta = 2.0$, $n = 2$, $\phi = 0$, $A = 1.0$.
* **Exact Traveling Wave Solution:**
  $$u(x, t) = \sin(k(x - \beta t)) e^{-\alpha k^2 t}, \quad k = \frac{n\pi}{L} = 2\pi$$

---

## 5. PINN Architecture & Training Specifications

* **Backbone Architecture:** Multi-Layer Perceptron (MLP) implemented via DeepXDE (PyTorch backend).
  * 1D problems: Layer sizes `[2, 20, 20, 20, 1]` for $(x, t)$ inputs, or `[1, 20, 20, 20, 1]` for 1D Poisson.
  * 2D Poisson: Layer sizes `[2, 20, 20, 20, 1]` for $(x, y)$ inputs.
* **Activation Function:** Hyperbolic tangent (`tanh`).
* **Weight Initialization:** Glorot normal (`"Glorot normal"` in DeepXDE).
* **Hard Boundary Enforcement:**
  * Heat: $\phi(x, t) = \sin(n\pi x / L) + t(x - x_{\min})(x_{\max} - x) \mathcal{N}(x, t)$
  * Wave: $\phi(x, t) = \sin(n\pi x / L) + t^2(x - x_{\min})(x_{\max} - x) \mathcal{N}(x, t)$ (using $t^2$ enforces $\partial_t \phi|_{t=0} = 0$)
  * Poisson 1D: $\phi(x) = (x - x_{\min})(x_{\max} - x) \mathcal{N}(x)$
  * Poisson 2D: $\phi(x, y) = (x - x_{\min})(x_{\max} - x)(y - y_{\min})(y_{\max} - y) \mathcal{N}(x, y)$
* **Training Hyperparameters:**
  * Optimizer: Adam.
  * Learning rate: $10^{-3}$.
  * Domain collocation points ($|\mathcal{X}_{\text{pde}}|$): 10,000 uniform random points.
  * Boundary collocation points ($|\mathcal{X}_{\text{bc}}|$): 0 (hard-constrained, boundary loss omitted from objective).
  * Well-Trained Regime: 10,000 iterations.
  * Randomly-Initialized Regime: 0 iterations.
* **Precision:** `float64` (`dde.config.set_default_float("float64")`).
* **Hardware Target:** CPU execution default (`set_default_device("cpu")`).

---

## 6. FDM Discretization & Numerical Schemes

### 6.1 Spatial Stencils
Uniform grid spacing $\Delta x = (x_{\max} - x_{\min}) / (N_x - 1)$.
Central differences:
$$\left.\frac{\partial^2 u}{\partial x^2}\right|_{x_i} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$

### 6.2 Time Stepping Schemes
* **First-Order Time (Heat & Drift-Diffusion): Crank-Nicolson**
  Implicit, unconditionally stable, $\mathcal{O}(\Delta t^2 + \Delta x^2)$:
  $$\left(I - \frac{1}{2}\Delta t L\right) e^{n+1} = \left(I + \frac{1}{2}\Delta t L\right) e^n - \frac{1}{2}\Delta t \left(R^n + R^{n+1}\right)$$
  Solved at each time step using SciPy sparse direct solver `spsolve`.
* **Second-Order Time (Wave): Explicit Central Differences**
  $$\frac{e^{n+1} - 2e^n + e^{n-1}}{\Delta t^2} = L e^n - R^n \implies e^{n+1} = 2e^n - e^{n-1} + \Delta t^2 (L e^n - R^n)$$
  Initial velocity condition $e_t(x, 0) = 0$ yields virtual step $e^{-1} = e^1$, giving:
  $$e^1 = e^0 + \frac{1}{2}\Delta t^2 (L e^0 - R^0) = -\frac{1}{2}\Delta t^2 R^0$$
  Stability governed by CFL condition: Courant number $r = c \frac{\Delta t}{\Delta x} \le 1$.
* **Steady-State (Poisson): Sparse Direct Solve**
  $$L e = -R$$

---

## 7. Metrics & Reported Experiments

1. **Discrete $L_2$ Error Metric:**
   $$\|e_{\text{true}} - e_{\text{method}}\|_2 = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(e_{\text{true}}(x_i) - e_{\text{method}}(x_i)\right)^2}$$
2. **Relative $L_2$ Error Metric:**
   $$\frac{\|e_{\text{true}} - e_{\text{method}}\|_2}{\|e_{\text{true}}\|_2}$$
3. **Primary Paper Experiments:**
   * **Figure 2:** 1D Heat Equation time-slice error profiles ($t \approx 0.25, 0.5, 0.75, 1.0$) and $L_2$ error evolution over time, comparing $e_{\text{true}}$, $e_{\text{res}}$ (ours), $e_{\text{FDM}}$, and $e_{\text{bound}}$ (Hillebrecht & Unger semigroup bound). Discretization: $N_x = 64, N_t = 64$.
   * **Figure 3:** Grid resolution study across $N_x \in \{8, 16, 32, 64, 128, 256, 512, 1024\}$ for all five problems, evaluating well-trained vs. randomly initialized models over 10 random seeds.
