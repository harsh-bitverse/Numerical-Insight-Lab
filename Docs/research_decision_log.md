# Research Decision Log

This log records architectural, scientific, and engineering decisions made during the reproduction and evaluation project for the course deadline (November 10, 2026).

---

## Decision 1: Upstream Code Incorporation Strategy
* **Date:** 2026-10-20 (Session 1)
* **Status:** Decided & Implemented
* **Context:**
  The project requires importing the official implementation from `https://github.com/aleks-krasowski/pinn-fdm-error-estimation`. Two candidate approaches were evaluated: Git Submodule vs. Vendored Snapshot.
* **Evaluation:**
  * *Git Submodule:* Maintains direct Git pointer, but introduces friction for 3 undergraduate team members who clone or pull without recursive submodule flags, often leading to empty directories and broken execution paths.
  * *Vendored Snapshot:* Copies the exact code tree at a pinned commit SHA into the repository, accompanied by an explicit provenance file (`Docs/UPSTREAM_PROVENANCE.md`). Guarantees offline self-containment, immunity to upstream repository deletions or force-pushes, and complete provenance tracking.
* **Decision:**
  Adopt a **Vendored Snapshot** placed at `upstream/pinn-fdm-error-estimation/`, pinned to commit `460d66bc2a59720883fa6a06e8c5436db4f99020`. Exclude `.git/` and `.DS_Store` to prevent repository corruption.

---

## Decision 2: Upstream Isolation & Scientific Baseline Integrity
* **Date:** 2026-10-20 (Session 1)
* **Status:** Decided & Implemented
* **Context:**
  Working Rule 5, 9, and 12 require preserving the authors' implementation as the authority and never silently modifying or replacing the baseline.
* **Decision:**
  The directory [`upstream/pinn-fdm-error-estimation/`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/upstream/pinn-fdm-error-estimation/) is designated as strictly **read-only**. No student code, exploratory scripts, or experimental modifications will be committed inside this folder. All wrappers, new tests, grid studies, boundary condition comparison experiments, and extension modules will reside in dedicated root or `src/` directories.

---

## Decision 3: Numerical Precision & Hardware Standardization
* **Date:** 2026-10-20 (Session 1)
* **Status:** Decided
* **Context:**
  FDM schemes and PINN residual evaluations can exhibit subtle numerical drift across single (`float32`) and double (`float64`) precision, as well as GPU vs. CPU floating-point nondeterminism.
* **Evaluation:**
  The upstream code specifies:
  * `dde.config.set_default_float("float64")`
  * `set_default_device("cpu")`
  * Default random seed `42`
* **Decision:**
  Strictly retain `float64` precision and `cpu` execution for all baseline reproduction runs. GPU acceleration may only be explored during extension work if wall-clock time requires it, with side-by-side consistency validation against CPU baselines.

---

## Decision 4: Structure of Research Documentation
* **Date:** 2026-10-20 (Session 1)
* **Status:** Decided & Implemented
* **Context:**
  Existing project literature and proposal documents are housed in `Docs/`.
* **Decision:**
  Standardize all project research documentation under the existing `Docs/` directory:
  * `Docs/UPSTREAM_PROVENANCE.md`: Pinned commit hash, license, retrieval details.
  * `Docs/paper_summary.md`: Theoretical formulation, equations, benchmark definitions.
  * `Docs/source_to_implementation_map.md`: Traceability matrix from equations to code.
  * `Docs/research_decision_log.md`: This decision log.
  * `Docs/experiment_protocol.md`: Execution protocol and verification criteria.
  * `Docs/known_discrepancies.md`: Audit of mathematical and implementation discrepancies.

---

## Decision 5: Initial Reproduction Baseline Selection
* **Date:** 2026-10-20 (Session 1)
* **Status:** Decided (Recommended to Team Leader)
* **Context:**
  The team must select the smallest credible benchmark for immediate reproduction to validate the toolchain by November 10.
* **Options:**
  1. *Poisson 1D / 2D:* Elliptic, steady-state, fast solve, but lacks time-stepping dynamics and does not exercise the Crank-Nicolson residual integration scheme ($R^n + R^{n+1}$) that forms the paper's core methodological contribution.
  2. *Wave 1D:* Hyperbolic, subject to CFL stability constraints ($\Delta t / \Delta x \le 1/c$).
  3. *Drift-Diffusion:* Code contains a mathematical discrepancy (Dirichlet BCs from exact solution rather than periodic BCs as claimed in Table 1).
  4. *Heat 1D Equation:* Has an exact analytical solution, unconditional Crank-Nicolson stability, features prominent in Figure 2 of the primary paper, includes comparative benchmarks against $e_{\text{FDM}}$ and $e_{\text{bound}}$, and trains in ~113 seconds on CPU.
* **Decision:**
  Select the **1D Heat Equation** ($N_x=64, N_t=64$, $\alpha=0.05, n=2$, well-trained 10,000 iterations) as our primary initial reproduction target.

---

## Decision 6: Controlled Grid-Resolution Study Protocol
* **Date:** 2026-10-20 (Session 3)
* **Status:** Decided & Executed
* **Context:**
  Priority 3 requires executing a controlled grid-resolution study ($N_x = N_t \in \{8, 16, 32, 64, 128, 256\}$) to evaluate how estimator grid resolution affects residual error estimation accuracy, separating estimator-grid effects from PINN training state.
* **Evaluation:**
  * By freezing model weights via the caching mechanism (`use_cache=True`), the exact continuous PINN prediction is held fixed across all 6 grid resolutions.
  * We compare two separate training regimes: Well-trained (10,000 iterations) vs. Random initialization (0 iterations), reproducing the setup of Figure 3(c).
  * Scale-invariant RMS discrete $L_2$ norms ensure consistent comparison across differing grid dimensions.
* **Decision:**
  Implement the automated sweep script in `scripts/run_grid_study.py`, output data to `results/grid_study.csv` and individual JSON files in `results/grid_study/`, and plot convergence curves overlaid with author reference bands in `figures/grid_resolution_comparison.png`.

---

## Decision 7: Mathematical Generalization of Residual Estimation to Soft Boundary Conditions
* **Date:** 2026-10-20 (Session 4)
* **Status:** Decided, Implemented & Verified
* **Context:**
  The upstream residual integration solver (`FDMSolverHeatEq.residual_integration()`) explicitly forces $e(x, 0) = 0$, $e(0, t) = 0$, and $e(1, t) = 0$, which assumes exact hard constraints. If fed a soft-constrained PINN, it ignores the boundary and initial condition errors ($\approx 1.4\%$), causing the error estimator to completely stall at $\mathcal{O}(10^{-2})$ with zero convergence ($p = 0$).
* **Mathematical Resolution:**
  Formulate the inhomogeneous defect initial-boundary value problem (IBVP):
  $$\partial_t e - \alpha \partial_{xx} e = -R(x, t), \quad e(x, 0) = u_0(x) - \hat{\phi}(x, 0), \quad e(x_b, t) = g(x_b, t) - \hat{\phi}(x_b, t)$$
  Because $u_0(x)$ and $g(x_b, t)$ are given by the problem definition, this inhomogeneous defect formulation requires **zero ground-truth knowledge**.
* **Decision:**
  Implement `InhomogeneousFDMSolverHeatEq` in [`scripts/run_boundary_extension.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/scripts/run_boundary_extension.py) without touching upstream files. Conduct a controlled side-by-side study across $N_x \in \{16, 32, 64, 128\}$ comparing:
  1. Baseline Hard BC with standard homogeneous solver.
  2. Soft BC with inhomogeneous defect solver (correct generalization).
  3. Soft BC with naive homogeneous solver (demonstrating catastrophic breakdown).
  Save artifacts in `results/boundary_extension.csv`, `figures/boundary_condition_comparison.png`, and `Docs/boundary_extension_analysis.md`.

