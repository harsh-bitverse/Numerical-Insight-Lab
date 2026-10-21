# Session 5 Final Quality Assurance & Verification Audit

**Project:** Numerical Insight Lab (IIT Delhi)  
**Evaluator Role:** Skeptical Research Reviewer, Numerical QA Engineer & Viva Examiner  
**Date:** 2026-10-21  
**Scope:** Rigorous independent audit of all repository files, source code, raw data, mathematical formulations, metrics, plots, and reproduction workflows.

---

## 1. Inventory & Provenance of Repository Artifacts

### 1.1 Source Code & Execution Scripts
| File Path | Nature / Origin | Status | Verified Behavior |
| :--- | :--- | :--- | :--- |
| `upstream/pinn-fdm-error-estimation/` | Vendored upstream code (commit `460d66b`) | VERIFIED FROM DOCUMENTATION/SOURCE | Preserved untouched; 405 files, read-only. |
| `scripts/run_grid_study.py` | Local automated sweep script | VERIFIED FROM CODE/DATA | Executes $N_x=N_t \in \{8, \dots, 256\}$ grid sweep, models cached. |
| `scripts/run_boundary_extension.py` | Local boundary extension script | VERIFIED FROM CODE/DATA | Implements `InhomogeneousFDMSolverHeatEq` and runs 3-way comparative study. |

### 1.2 Configuration Files
| File Path | Nature / Origin | Status | Verified Settings |
| :--- | :--- | :--- | :--- |
| `configs/grid_study/heat_grid_study.json` | JSON configuration for grid study | VERIFIED FROM CODE/DATA | $N_x \in [8, 256]$, 10k iters / 0 iters, seed 42, float64, CPU. |
| `configs/boundary_extension/heat_soft_bc.json` | JSON configuration for BC extension | VERIFIED FROM CODE/DATA | $N_x \in [16, 128]$, soft vs hard BC, 10k iters, seed 42, float64, CPU. |

### 1.3 Raw Data & Output Manifests
| File Path | Nature / Origin | Status | Verified Quantities |
| :--- | :--- | :--- | :--- |
| `results/grid_study.csv` | Generated raw CSV (12 runs) | VERIFIED FROM CODE/DATA | Well-trained and random-init records across 6 grid levels. |
| `results/boundary_extension.csv` | Generated raw CSV (12 runs) | VERIFIED FROM CODE/DATA | 4 resolutions $\times$ 3 configurations (Hard, Soft Inhom, Soft Naive). |
| `results/run_manifest.json` | Diagnostic & baseline execution manifest | VERIFIED FROM CODE/DATA | Records hardware, timing, git commit, parameter SHA-256 hashes. |
| `results/grid_study/*.json` | 12 per-run raw JSON manifests | VERIFIED FROM CODE/DATA | Explicit timestamps, timing, and discrete metrics per grid level. |
| `results/models/soft_pinn_seed42.pt` | Saved PyTorch weights (Soft PINN) | VERIFIED FROM CODE/DATA | 10,955 bytes; cached model state for rapid evaluation. |
| `upstream/.../model_zoo/89e73629865126e6.pt` | Saved PyTorch weights (Hard PINN) | VERIFIED FROM CODE/DATA | 10,955 bytes; cached baseline model weights. |

### 1.4 Figures & Visualizations
| File Path | Nature / Origin | Status | Verified Integrity |
| :--- | :--- | :--- | :--- |
| `figures/grid_resolution_comparison.png` | Generated plot from `run_grid_study.py` | VERIFIED FROM CODE/DATA | 347,595 bytes; convergence curves vs author reference bands. |
| `figures/boundary_condition_comparison.png` | Generated plot from `run_boundary_extension.py`| VERIFIED FROM CODE/DATA | 528,911 bytes; 4-panel diagnostic (slices, BC drift, slopes, advantage).|
| `results/baseline_run/baseline_heat_results.png` | Generated baseline diagnostic plot | VERIFIED FROM CODE/DATA | 741,975 bytes; 6-panel solution, error, and bound profiles. |

### 1.5 Research Documentation
| File Path | Nature / Origin | Status | Verified Contents |
| :--- | :--- | :--- | :--- |
| `Docs/UPSTREAM_PROVENANCE.md` | Manually curated provenance | VERIFIED FROM DOCUMENTATION/SOURCE | Documents commit `460d66b`, arXiv paper, no explicit code license. |
| `Docs/metric_definitions.md` | Formal mathematical specification | VERIFIED FROM CODE/DATA | Distinguishes discrete RMS vs composite trapezoidal quadrature. |
| `Docs/baseline_validation.md` | Verification against author CSV | VERIFIED FROM CODE/DATA | Confirms near-machine-precision direct-FDM agreement ($\le 2.31 \times 10^{-14}$) via algebraic cancellation. |
| `Docs/grid_study_analysis.md` | Detailed analysis of grid study | VERIFIED FROM CODE/DATA | Verifies two-point orders, asymptotic slopes, std dev bands. |
| `Docs/boundary_extension_analysis.md` | Mathematical & experimental report | VERIFIED FROM CODE/DATA | Details inhomogeneous defect formulation and empirical restoration. |
| `Docs/research_decision_log.md` | Decisions 1 to 7 | VERIFIED FROM CODE/DATA | Complete provenance of decisions through Session 4. |
| `Docs/reproduction_status.md` | Progress tracker | VERIFIED FROM CODE/DATA | Tracks priorities 1 to 5. |

---

## 2. Scientific QA Checklist & Audit Matrix

| Verification Item | Evaluation Scope | Status | Evidence / Analysis |
| :--- | :--- | :--- | :--- |
| **B1. Governing PDE Signs** | $u_t - \alpha u_{xx} = 0$, $R = \hat{u}_t - \alpha \hat{u}_{xx}$ | VERIFIED FROM CODE/DATA | Code in `heat_1d.py` sets `u_t - diffusivity * u_xx`. Defect equation $\partial_t e - \alpha \partial_{xx} e = -R$ is solved with forcing term $-0.5 \Delta t (R^n + R^{n+1})$ in `fdm.py` line 523 and `run_boundary_extension.py` line 85. Signs are strictly consistent. |
| **B1. Defect Boundary Coupling** | Inhomogeneous BC injection | VERIFIED FROM CODE/DATA | `InhomogeneousFDMSolverHeatEq` injects $e(x, 0) = u_0 - \hat{u}(x, 0)$ and $e(x_b, t_{n+1}) = g - \hat{u}(x_b, t_{n+1})$ directly into Crank-Nicolson RHS. Off-diagonal tridiagonal entries propagate boundary error into interior at $\mathcal{O}(h^2)$. |
| **B1. Ground Truth Independence** | Eliminating ground-truth dependence | VERIFIED FROM CODE/DATA | Inhomogeneous error data $e_{\text{ic}}(x)$ and $e_{\text{bc}}(t)$ rely solely on $u_0(x) = \sin(2\pi x)$ and $g = 0$ (given by problem statement) and $\hat{u}$. Zero analytical ground truth is used to construct $e_{\text{res}}$. Analytical truth is used only for external evaluation. |
| **B2. Norm Definitions** | Discrete RMS vs Trapezoidal Quadrature | VERIFIED FROM CODE/DATA | Code calculates `np.sqrt(np.mean(V**2))`. Domain is $[0, 1] \times [0, 1]$ with $N_x, N_t$ points. Boundary values are machine zero for hard BCs ($< 10^{-15}$). For $N_x=64$, Trapezoidal / RMS ratio is $1.0106$ ($e_{\text{res}}$) and $1.0127$ ($e_{\text{fdm}}$). Advantage factor is $33.80\times$ (RMS) vs $33.87\times$ (Trapezoidal). |
| **B3. Asymptotic Grid Range** | $N_x \in [32, 256]$ vs coarse grids | VERIFIED FROM CODE/DATA | Two-point orders: $N_x=8\to16$ is pre-asymptotic ($p_{\text{res}}=2.2791$); for $N_x \ge 32$, orders approach $p \in [1.96, 1.99]$. Asymptotic regression slope over $N_x \in [32, 256]$ is $p_{\text{res}} = 1.9822$ (well-trained) and $p_{\text{res}} = 1.9969$ (random-init), confirming theoretical $\mathcal{O}(h^2)$ along the coupled refinement path. |
| **B3. Reference Bands Semantics** | Fig. 3 shaded bands | VERIFIED FROM CODE/DATA | Shaded envelopes represent Sample Mean $\pm$ 1 Sample Standard Deviation ($n=10$ random seeds from `all_experiments_results_float64.csv`), NOT confidence intervals. Seed 42 run lies within $[-1.24, -0.61]$ standard deviations across all resolutions. |
| **B4. FDM Algebraic Cancellation** | Deterministic FDM agreement | VERIFIED FROM CODE/DATA | $\Delta_{\text{FDM}} = (u_{\text{true}} - \hat{u}) - (u_{\text{FDM}} - \hat{u}) = u_{\text{true}} - u_{\text{FDM}}$. Neural network $\hat{u}$ cancels algebraically. Computation is deterministic linear algebra. Matches author CSV across all 10 seeds to near-machine-precision ($\le 2.31 \times 10^{-14}$). |
| **B4. Advantage Ratios Recomputed** | Hard BC: 33.80x, Soft BC: 85.67x | VERIFIED FROM CODE/DATA | Baseline: $1.612600\times10^{-4} / 4.771048\times10^{-6} = 33.7997\dots \approx 33.80\times$. Soft BC: $1.612600\times10^{-4} / 1.882412\times10^{-6} = 85.6667\dots \approx 85.67\times$. Directly verified from raw CSV records. |
| **B5. Soft BC Failure Mode** | Naive homogeneous solver breakdown | VERIFIED FROM CODE/DATA | When upstream solver forces $e(x_b, t) = 0$ on soft PINN, discrepancy stalls at $\approx 1.05 \times 10^{-2}$ across all grids ($p = 0.00$). Advantage factor plummets to $0.02\times$ at $N_x=64$ and $0.0038\times$ at $N_x=128$ ($262\times$ worse than direct FDM). |
| **B5. Second-Order Restoration** | Inhomogeneous solver convergence | VERIFIED FROM CODE/DATA | Inhomogeneous defect solver discrepancy scales as $3.24\times10^{-5}$ ($N_x=16$) $\to 7.63\times10^{-6}$ ($N_x=32$) $\to 1.88\times10^{-6}$ ($N_x=64$) $\to 4.67\times10^{-7}$ ($N_x=128$). Two-grid orders are $1.9934, 1.9736, 1.9890$, confirming observed second-order convergence ($p = 1.99 \pm 0.01$). |

---

## 3. Discrepancies, Cleanup & Consistency Actions

1. **Root Directory Clutter:**
   * Found redundant file `heat_results.png` in root directory (741,975 bytes). Confirmed identical to `results/baseline_run/baseline_heat_results.png`. Safe to remove from root to maintain clean repository structure.
2. **Model Weight Caching Traceability:**
   * Both `hard_pinn` and `soft_pinn` weights are cleanly cached (`model_zoo/89e73629865126e6.pt` and `results/models/soft_pinn_seed42.pt`), enabling instantaneous re-evaluation of all FDM sweeps in under 5 seconds.
3. **Working Tree State:**
   * Working branch is `main`. No tracked files modified. Working tree clean of uncommitted changes to tracked files.
