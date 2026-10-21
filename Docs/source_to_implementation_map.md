# Source-to-Implementation Map

This document establishes the precise mapping between the mathematical formulas, algorithms, and experimental configurations in the paper (*Building Trust in PINNs: Error Estimation through Finite Difference Methods*, arXiv:2603.15526) and the upstream codebase at commit `460d66b`.

---

## 1. Directory Structure of Upstream Source

All upstream code is vendored at [`upstream/pinn-fdm-error-estimation/`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/).

```
upstream/pinn-fdm-error-estimation/
├── pyproject.toml                     # Dependencies & build configuration
├── README.md                          # Upstream instructions & documentation
├── pinn_error/
│   ├── core/
│   │   ├── problem.py                 # Abstract base class BaseProblem & ProblemDomain
│   │   ├── pinn.py                    # PINNConfig & PINNTrainer (DeepXDE model)
│   │   ├── fdm.py                     # FDMMatrixBuilder & FDM solvers (Poisson, Heat, Drift, Wave)
│   │   └── error_bounds.py            # Semigroup-based upper bound baseline (Hillebrecht & Unger)
│   ├── problems/
│   │   ├── heat_1d.py                 # Heat1DProblemSineIC definition
│   │   ├── poisson.py                 # Poisson1D and Poisson2D definitions
│   │   ├── wave.py                    # Wave1D definition
│   │   └── drift_diffusion.py         # DriftDiffusion definition
│   └── utils/
│       ├── experiment_factory.py      # ExperimentFactory & problem bundler
│       ├── plotting.py                # Plotting utilities and metric reporting
│       ├── io.py                      # Data serialization helpers
│       └── setup.py                   # Device and environment configuration
├── examples/
│   ├── run_experiment.py              # Single experiment CLI runner
│   └── run_all_experiments.py          # Multi-problem & grid-resolution evaluation script
├── figures/                           # Pre-rendered paper figures
├── notebooks/                         # Analysis & visualization Jupyter notebooks
└── results/
    └── all_experiments_results_float64.csv # Full experimental benchmark dataset (850 KB)
```

---

## 2. Component-by-Component Mapping

### 2.1 Problem Definitions & Exact Solutions

| Component | Paper Reference | File Path & Class | Key Methods & Lines |
| :--- | :--- | :--- | :--- |
| **Problem Base & Domain** | Sec. 2.1, Eqs. (1)-(2) | [`problem.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/problem.py) | `BaseProblem`, `ProblemDomain` (lines 14–128) |
| **1D Heat Equation** | Tab. 1, Fig. 1(b) | [`heat_1d.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/heat_1d.py) | `Heat1DProblemSineIC`<br>• `pde()`: L31–48 ($u_t - \alpha u_{xx}$)<br>• `output_transform()`: L50–75<br>• `exact_solution()`: L97–117 |
| **1D Poisson Equation** | Tab. 1, Sec. 2.2, Eq. (6) | [`poisson.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/poisson.py) | `Poisson1D`<br>• `source_term()`: L27–49 ($f(x) = \pi^2 \sin(\pi x)$)<br>• `pde()`: L51–70 ($-u_{xx} - f(x) = 0$)<br>• `output_transform()`: L71–92<br>• `exact_solution()`: L93–106 |
| **2D Poisson Equation** | Tab. 1, Fig. 1(a) | [`poisson.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/poisson.py) | `Poisson2D`<br>• `source_term()`: L136–160<br>• `pde()`: L161–167 ($-u_{xx} - u_{yy} - f = 0$)<br>• `output_transform()`: L168–177<br>• `exact_solution()`: L178–185 |
| **1D Wave Equation** | Tab. 1, Fig. 1(d) | [`wave.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/wave.py) | `Wave1D`<br>• `pde()`: L31–48 ($u_{tt} - c^2 u_{xx}$)<br>• `output_transform()`: L68–78 ($t^2$ factor)<br>• `exact_solution()`: L50–58 |
| **1D Drift-Diffusion** | Tab. 1, Fig. 1(c) | [`drift_diffusion.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/problems/drift_diffusion.py) | `DriftDiffusion`<br>• `pde()`: L37–56 ($u_t - \alpha u_{xx} + \beta u_x$)<br>• `output_transform()`: L57–131<br>• `exact_solution()`: L152–180 |

---

### 2.2 Neural Network & PINN Training

| Component | Paper Reference | File Path & Class | Implementation Details |
| :--- | :--- | :--- | :--- |
| **PINN Hyperparameters** | Sec. 4, Par. 1 | [`pinn.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/pinn.py) | `PINNConfig` (L23–52):<br>• `layers = [2, 20, 20, 20, 1]`<br>• `activation = 'tanh'`<br>• `n_iterations = 10` (default, CLI overrides to 10000)<br>• `num_domain = 10000`<br>• `seed = 42` |
| **Model Construction** | Sec. 2.1, Eq. (3)-(4) | [`pinn.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/pinn.py) | `PINNTrainer._init_model()` (L93–132):<br>• `dde.nn.FNN(..., "Glorot normal")`<br>• Hard constraint applied via `apply_output_transform()` (L100)<br>• `dde.data.TimePDE` or `dde.data.PDE` with empty `bcs=[]`<br>• Compiled with `"adam"`, `lr=1e-3` |
| **Training Execution** | Sec. 4 | [`pinn.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/pinn.py) | `PINNTrainer.train()` (L226–253):<br>• Integrates checkpoint caching in `model_zoo/` based on SHA-256 config hash (L133–169). |
| **Residual Evaluation** | Sec. 3, Eq. (4) | [`pinn.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/pinn.py) | `PINNTrainer.residual()` (L265–286):<br>• Calls `model.predict(X, operator=self.problem.pde)` using automatic differentiation. |

---

### 2.3 Finite Difference Solvers & Error Estimation

| Component | Paper Reference | File Path & Class | Implementation Details |
| :--- | :--- | :--- | :--- |
| **1D Derivative Stencils** | Sec. 2.2, Eq. (5) | [`fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) | `FDMMatrixBuilder.get_derivative_matrix_1d()` (L20–80):<br>• Order 1: central difference `diags([-1, 1], [-1, 1]) / (2*dx)`<br>• Order 2: tridiagonal `[1, -2, 1] / dx^2` |
| **2D Laplacian Stencil** | Sec. 2.2 | [`fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) | `FDMMatrixBuilder.get_laplacian_2d()` (L82–129):<br>• 5-point stencil with lexicographic ordering in CSR format |
| **1D Poisson Solver & Error Integration** | Sec. 2.2, Eq. (6); Sec. 3 | [`fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) | `FDMSolverPoisson1D` (L131–250):<br>• Matrix $L = -A_{xx}$ with boundary rows set to identity (L180–192)<br>• `solve()`: `spsolve(L, f)` (L194–216)<br>• `residual_integration()`: `spsolve(L, -R)` (L218–240) |
| **2D Poisson Solver & Error Integration** | Sec. 3 | [`fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) | `FDMSolverPoisson2D` (L251–376):<br>• Matrix $L = -A_{xx,yy}$ (L303–321)<br>• `solve()`: `spsolve(L, f_2d)` (L323–341)<br>• `residual_integration()`: `spsolve(L, -R_2d)` (L343–370) |
| **1D Heat Solver & Error Integration** | Sec. 2.2, Eq. (8); Sec. 3, Eq. (11) | [`fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) | `FDMSolverHeatEq` (L378–543):<br>• Crank-Nicolson matrices $M_{\text{lhs}} = I - \frac{1}{2}\Delta t L$, $M_{\text{rhs}} = I + \frac{1}{2}\Delta t L$ (L432–463)<br>• `solve()`: forward marching from initial condition $u^0$ (L465–485)<br>• `residual_integration()`: time marching with source $-\frac{1}{2}\Delta t(R^n + R^{n+1})$ (L487–534) |
| **1D Wave Solver & Error Integration** | Sec. 2.2, Eq. (9); Sec. 3, Eq. (12) | [`fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) | `FDMSolverWave1D` (L757–882):<br>• Explicit central difference time-stepping (L806–842)<br>• First step using Neumann initial velocity: $e^1 = e^0 + \frac{1}{2}\Delta t^2 (L e^0 - R^0) = -\frac{1}{2}\Delta t^2 R^0$ (L857–866)<br>• General step: $e^{n+1} = 2e^n - e^{n-1} + \Delta t^2 (L e^n - R^n)$ (L868–874) |
| **1D Drift-Diffusion Solver** | Sec. 2.2; Sec. 3 | [`fdm.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/core/fdm.py) | `FDMSolverDriftDiffusion` (L545–755):<br>• Operator $L = D A_{xx} - \beta A_x$ (L603–635)<br>• Dirichlet boundary values pulled from exact solution (L648–665) |

---

### 2.4 Experiment Execution & Pipeline Automation

| Component | Paper Reference | File Path & Class | Implementation Details |
| :--- | :--- | :--- | :--- |
| **Experiment Factory** | Sec. 4 | [`experiment_factory.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/pinn_error/utils/experiment_factory.py) | `ExperimentFactory` (L144–341):<br>• Bundles problem, PINN trainer, and FDM solver via `get_problem_bundle()` (L117–142)<br>• Executes PINN training, residual integration, and FDM solve in `_run()` (L265–286)<br>• Assembles error dict: `e_true = u_exact - u_pinn`, `e_res`, `e_fdm = u_fdm - u_pinn` (L287–341) |
| **Single Experiment CLI** | Sec. 4, Fig. 2 | [`run_experiment.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/examples/run_experiment.py) | CLI flags `--problem`, `--n_iterations`, `--nx`, `--nt`, `--seed`, etc. Runs experiment, prints error metrics, and saves figure. |
| **Full Benchmark Runner** | Sec. 4, Fig. 3 | [`run_all_experiments.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/examples/run_all_experiments.py) | Multi-problem evaluation across grid sizes $\{8, 16, 32, 64, 128, 256\}$, iterations $\{0, 10000\}$, saving results to Polars DataFrame / CSV. |
