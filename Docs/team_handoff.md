# Team Handoff & Technical Workstream Allocation

**Project Title:** Reproducing and Evaluating Finite-Difference Error Estimation in Physics-Informed Neural Networks  
**Course:** Numerical Analysis Undergraduate Research Project, IIT Delhi  
**Deadline:** November 10, 2026  
**Target Repository:** `https://github.com/harsh-bitverse/Numerical-Insight-Lab`  
**Date:** October 2026

---

## 1. Project Overview & Current State

We have completed the implementation, audit, baseline reproduction, grid-resolution study, and boundary-condition extension for our Numerical Analysis research project. The project is in a **fully reproducible, audited state**:
* Upstream codebase is safely vendored and pinned (`upstream/pinn-fdm-error-estimation/` at commit `460d66bc`).
* Baseline 1D Heat Equation results are validated against the authors' reference CSV (with near-machine-precision agreement on direct-FDM discrepancy via algebraic cancellation).
* Controlled grid-resolution study confirms asymptotic second-order convergence ($p \approx 1.99$ for $N_x \ge 32$) along the coupled diagonal refinement path ($N_x=N_t$).
* Boundary-condition extension demonstrates that the authors' method fails naively on soft-constrained PINNs ($p = 0.00$), and that our `InhomogeneousFDMSolverHeatEq` restores observed second-order convergence ($p = 1.99 \pm 0.01$) with an elevated $85.67\times$ advantage factor.

---

## 2. Core Mathematical Formulation

### 2.1 Forward Problem & PDE Residual
For the 1D Heat Equation on $[0, 1] \times [0, 1]$:
$$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0, \quad u(x, 0) = \sin(2\pi x), \quad u(0, t) = u(1, t) = 0$$
with $\alpha = 0.05$. For continuous PINN approximation $\hat{u}(x, t)$, the PDE residual is:
$$R(x, t) = \frac{\partial \hat{u}}{\partial t} - \alpha \frac{\partial^2 \hat{u}}{\partial x^2}$$

### 2.2 Error Defect Equation (The Core Method)
Defining true error $e(x, t) := u_{\text{true}}(x, t) - \hat{u}(x, t)$, subtracting the operator equations gives:
$$\frac{\partial e}{\partial t} - \alpha \frac{\partial^2 e}{\partial x^2} = -R(x, t)$$
* **For Hard Constraints:** $e(x, 0) \equiv 0$, $e(0, t) \equiv 0$, $e(1, t) \equiv 0$.
* **For Soft Constraints (Our Extension):**
  $$e(x, 0) = \sin(2\pi x) - \hat{u}(x, 0)$$
  $$e(0, t) = 0 - \hat{u}(0, t), \quad e(1, t) = 0 - \hat{u}(1, t)$$
  *Crucial note:* These boundary and initial error values require **zero analytical ground truth** because the boundary functions are given in the problem statement.

### 2.3 Crank-Nicolson Discretization
$$(I - 0.5 \Delta t \alpha A_{xx}) e^{n+1} = (I + 0.5 \Delta t \alpha A_{xx}) e^n - 0.5 \Delta t (R^n + R^{n+1})$$
Inhomogeneous boundary values at $t_{n+1}$ are injected directly into the boundary rows of the linear system.

---

## 3. Map of Key Project Files

| File Path | Purpose | Workstream Owner |
| :--- | :--- | :--- |
| `README.md` | Top-level repository overview and quickstart | Team Leader |
| `Docs/final_research_report.md` | Authoritative academic report for submission | All Members |
| `Docs/viva_preparation.md` | Comprehensive oral exam defense guide (25 Q&As) | All Members |
| `Docs/metric_definitions.md` | Formal mathematical spec (RMS vs. Quadrature) | Workstream B |
| `Docs/baseline_validation.md` | Proof of algebraic cancellation & numerical agreement | Workstream A |
| `Docs/grid_study_analysis.md` | Empirical vs asymptotic orders & regression | Workstream B |
| `Docs/boundary_extension_analysis.md` | Mathematical derivation & soft BC evaluation | Team Leader |
| `scripts/run_grid_study.py` | Automated grid study runner | Workstream B |
| `scripts/run_boundary_extension.py` | Standalone boundary extension runner | Team Leader |
| `results/grid_study.csv` | Raw metrics for grid study | Workstream B |
| `results/boundary_extension.csv` | Raw metrics for boundary extension | Team Leader |
| `figures/grid_resolution_comparison.png` | Convergence plot vs author reference bands | Workstream B |
| `figures/boundary_condition_comparison.png` | 4-panel soft BC extension diagnostic figure | Team Leader |

---

## 4. Exact Reproduction Commands

```bash
# Activate environment
.venv\Scripts\Activate.ps1

# Run Grid Study (uses cached model, executes in ~3 seconds)
python scripts/run_grid_study.py --config configs/grid_study/heat_grid_study.json

# Run Boundary Extension (uses cached model, executes in ~4 seconds)
python scripts/run_boundary_extension.py --config configs/boundary_extension/heat_soft_bc.json
```

---

## 5. Workstream Ownership & Task Allocation

### Workstream 1: Student Team Leader
* **Scope:** Architectural integrity, code repository maintenance, and boundary extension.
* **Immediate Tasks:**
  1. Review `scripts/run_boundary_extension.py` and verify `InhomogeneousFDMSolverHeatEq`.
  2. Lead final report polishing of Sections 7 (Extension) and 8 (Discussion).
  3. Ensure no uncommitted working tree anomalies or untracked temporary files exist before packaging.

### Workstream 2: Technical Workstream Owner A
* **Scope:** Baseline reproduction, DeepXDE PINN training, and numerical validation.
* **Immediate Tasks:**
  1. Cross-verify the baseline verification table in Section 5 of the final report against `results/run_manifest.json`.
  2. Verify the algebraic cancellation argument in Section 3.3 ($\Delta_{\text{FDM}} = u_{\text{true}} - u_{\text{FDM}}$).
  3. Prepare oral defense notes on PINN loss formulation and model caching.

### Workstream 3: Technical Workstream Owner B
* **Scope:** Grid-resolution analysis, convergence orders, and figure quality.
* **Immediate Tasks:**
  1. Verify the regression slopes ($p_{\text{res}} = 1.982$, $p_{\text{fdm}} = 1.990$) in Table 2.
  2. Confirm formatting and resolution of `figures/grid_resolution_comparison.png` and `figures/boundary_condition_comparison.png` for PDF compilation.
  3. Lead oral defense preparation on norm definitions (unweighted RMS vs continuous trapezoidal quadrature).

---

## 6. Known Limitations & Non-Blockers

1. **Analytical Ground Truth Scope:** Our validation relies on an exact sinusoidal heat solution. For general PDEs without analytical solutions, reference numerical solutions computed on highly refined grids ($N_x \ge 1024$) would be required for evaluation.
2. **Semigroup Error Bound ($e_{\text{bound}}$):** While implemented in upstream code (`error_bounds.py`), $e_{\text{bound}}$ provides a 1D scalar bound over time without spatial resolution. It was excluded from our grid study because it does not resolve spatial error distribution.
3. **Hyperbolic / Stiff Extension Limits:** Extending the defect solver to advection-dominated or wave equations requires CFL stability constraints or upwind discretization to prevent numerical dispersion.
