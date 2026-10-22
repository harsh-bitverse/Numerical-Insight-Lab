# Experimental Protocol: Boundary Penalty Parameter Sweep ($\lambda_{\text{bc}}$)

**Course:** Numerical Analysis Undergraduate Research Project  
**Institution:** Indian Institute of Technology Delhi (IIT Delhi)  
**Project:** Reproducing and Evaluating Finite-Difference Error Estimation in Physics-Informed Neural Networks  
**Target Paper:** Aleks Krasowski et al., *Building Trust in PINNs: Error Estimation through Finite Difference Methods*, [arXiv:2603.15526](https://arxiv.org/abs/2603.15526) (2026)  
**Author and Collaborators:** Harshith N, Nilesh Kumar Verma, Devesh Saini  
**Repository:** `https://github.com/harsh-bitverse/Numerical-Insight-Lab`  
**Base Commit:** `76e0974e7660ce7a17552bb642b5acb42ae94ec8`  
**Document Date:** October 2026 (Session 6D)

---

## 1. Research Objectives & Problem Formulation

### 1.1 Objective
To empirically evaluate how the boundary loss penalty weight $\lambda_{\text{bc}} \in \{1, 10, 100, 1000\}$ in soft-constrained Physics-Informed Neural Networks governs:
1. The Pareto trade-off between boundary conformity ($\|\delta_b\|$) and interior PDE adherence ($\|R\|$).
2. The empirical convergence order $p$ and discretization error of the **inhomogeneous defect estimator** (`InhomogeneousFDMSolverHeatEq`).
3. The breakdown mechanism, stagnation floor, and scaling of the **naive homogeneous estimator** (`FDMSolverHeatEq`).
4. The estimator accuracy advantage ratio $\eta := \|\Delta_{\text{direct-FDM}}\|_{\text{RMS}} / \|\Delta_{\text{est}}\|_{\text{RMS}}$ over direct simulation.

### 1.2 Mathematical Formulation

The 1D Heat Equation benchmark:
$$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in (0, 1), \; t \in (0, 1]$$
with $\alpha = 0.05$, initial condition $u(x, 0) = \sin(2\pi x)$, and homogeneous Dirichlet boundary conditions $u(0, t) = u(1, t) \equiv 0$. The exact analytical ground truth is:
$$u(x, t) = e^{-4\pi^2 \alpha t}\sin(2\pi x)$$

#### Soft-Constrained PINN Optimization Objective
In DeepXDE's `dde.data.TimePDE` with `ic_bcs=[bc_l, bc_r, ic]`, the training loss is:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{pde}}(\theta) + \lambda_{\text{bc}} \mathcal{L}_{\text{bc\_l}}(\theta) + \lambda_{\text{bc}} \mathcal{L}_{\text{bc\_r}}(\theta) + \lambda_{\text{ic}} \mathcal{L}_{\text{ic}}(\theta)$$
where:
* $\mathcal{L}_{\text{pde}}(\theta) = \frac{1}{N_{\text{domain}}} \sum_{i=1}^{N_{\text{domain}}} \left| \partial_t \hat{u} - \alpha \partial_{xx} \hat{u} \right|^2$
* $\mathcal{L}_{\text{bc\_l}}(\theta) = \frac{1}{N_{\text{boundary\_l}}} \sum_{i} \left| \hat{u}(0, t_i) \right|^2$
* $\mathcal{L}_{\text{bc\_r}}(\theta) = \frac{1}{N_{\text{boundary\_r}}} \sum_{i} \left| \hat{u}(1, t_i) \right|^2$
* $\mathcal{L}_{\text{ic}}(\theta) = \frac{1}{N_{\text{initial}}} \sum_{i} \left| \hat{u}(x_i, 0) - \sin(2\pi x_i) \right|^2$
* $\lambda_{\text{ic}} = 1.0$ (held constant).

#### Defect Error Integration Formulations
The pointwise error $e(x, t) := u(x, t) - \hat{u}(x, t)$ satisfies:
$$\partial_t e - \alpha \partial_{xx} e = -R(x, t)$$
1. **Inhomogeneous Defect Integration (`InhomogeneousFDMSolverHeatEq`):**
   * Initial condition: $e(x, 0) = \sin(2\pi x) - \hat{u}(x, 0)$
   * Boundary conditions: $e(0, t) = 0 - \hat{u}(0, t)$, $e(1, t) = 0 - \hat{u}(1, t)$
   * Crank-Nicolson load vector injects empirical boundary departures $\delta_0(t_{n+1}), \delta_1(t_{n+1})$ at each time step.
2. **Naive Homogeneous Defect Integration (`FDMSolverHeatEq`):**
   * Assumes $e(x, 0) \equiv 0$ and $e(0, t) = e(1, t) \equiv 0$ (the unstated assumption of Krasowski et al. 2026).

---

## 2. Experimental Controls & Parameter Matrix

| Parameter Category | Configuration Parameter | Fixed Value / Range | Justification / Control Method |
| :--- | :--- | :--- | :--- |
| **Independent Variable** | Boundary Penalty $\lambda_{\text{bc}}$ | $\{1.0, 10.0, 100.0, 1000.0\}$ | Spans 3 orders of magnitude from unit weighting to severe boundary dominance. |
| **Independent Variable** | Grid Resolution $N_x = N_t$ | $\{16, 32, 64\}$ | Tests empirical asymptotic convergence order $p$. |
| **Independent Variable** | Defect Solver Mode | `inhomogeneous_defect`<br>`naive_homogeneous` | Isolates the effect of inhomogeneous boundary injection. |
| **Controlled Setting** | Initial Penalty $\lambda_{\text{ic}}$ | $1.0$ | Fixed across all runs to isolate boundary penalty effects. |
| **Controlled Setting** | Network Architecture | MLP `[2, 20, 20, 20, 1]` | 921 trainable parameters, `tanh` hidden activations, identity output. |
| **Controlled Setting** | Optimizer & Budget | Adam, $\text{lr}=10^{-3}$, 10,000 steps | Fixed optimizer, learning rate, and iteration budget. |
| **Controlled Setting** | Collocation Sampling | 10k domain, 1k boundary, 1k initial | Fixed point counts; seeded uniform / random sampling. |
| **Controlled Setting** | Hardware & Precision | CPU, `float64` (`torch.float64`) | Deterministic arithmetic without GPU nondeterminism. |
| **Controlled Setting** | Seed | `seed = 42` | Single controlled seed for primary sweep (exploratory benchmark). |

---

## 3. Cryptographic Cache Integrity & Isolation Protocol

To eliminate the risk of silent cache collisions across differing penalty weights:

1. **Deterministic Fingerprinting:**  
   Every model checkpoint is indexed by the SHA-256 hash of its canonical configuration:
   `soft_pinn_lbc{lambda_bc}_lic{lambda_ic}_seed{seed}_{hash[:8]}.pt`
2. **Metadata Companion File:**  
   Alongside each `.pt` file, an identical-basename `.json` file records all training hyperparameters, data sizes, package versions, and timestamps.
3. **Strict Metadata Verification:**  
   Before any checkpoint is loaded, [`scripts/model_cache.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/scripts/model_cache.py) validates that `lambda_bc`, `lambda_ic`, `seed`, `n_iterations`, and all problem parameters match requested values within $10^{-9}$ numerical tolerance. Any mismatch halts loading and triggers re-training.
4. **Directory Isolation:**  
   * Baseline Session 5 models: preserved in `results/models/soft_pinn_seed42.pt`.
   * Penalty sweep models: isolated in `results/models/penalty_sweep/`.
   * Results data: written to `results/penalty_sweep/heat_penalty_sweep.csv`.

---

## 4. Execution & Verification Workflow

### Command Sequence:
```powershell
# 1. Run focused unit test suite
.\.venv\Scripts\pytest.exe -v tests/test_reproducibility.py

# 2. Run full boundary penalty experiment sweep
.\.venv\Scripts\python.exe scripts/run_penalty_sweep.py --config configs/boundary_extension/heat_penalty_sweep.json

# 3. Generate training loss progression figures
.\.venv\Scripts\python.exe scripts/plot_loss_curve.py --sweep_dir results/penalty_sweep --output figures/training_loss_progression.png
```

---

## 5. Metric Definitions & Formulas

1. **Root Mean Square (RMS) Norm:**
   $$\|V\|_{\text{RMS}} := \sqrt{\frac{1}{N_x N_t} \sum_{i=1}^{N_x} \sum_{n=1}^{N_t} |V(x_i, t_n)|^2}$$
2. **True PINN Error:**
   $$e_{\text{true}}(x, t) = u_{\text{exact}}(x, t) - \hat{u}(x, t)$$
3. **Boundary Departure RMS Norm:**
   $$\|\delta_b\|_{\text{RMS}} = \sqrt{\frac{1}{2 N_t} \sum_{n=1}^{N_t} \left( |\hat{u}(0, t_n)|^2 + |\hat{u}(1, t_n)|^2 \right)}$$
4. **Estimator Discrepancy:**
   $$\Delta_{\text{est}}(x, t) = e_{\text{true}}(x, t) - e_{\text{est}}(x, t)$$
5. **Estimator Advantage Factor:**
   $$\eta = \frac{\|\Delta_{\text{direct-FDM}}\|_{\text{RMS}}}{\|\Delta_{\text{est}}\|_{\text{RMS}}}$$
6. **Empirical Convergence Order ($p$):**
   $$p = \frac{\ln(\Delta_{h_1} / \Delta_{h_2})}{\ln(h_1 / h_2)}$$
