# Numerical Insight Lab: Reproducing and Evaluating Finite-Difference Error Estimation in PINNs

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14](https://img.shields.io/badge/pytorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![DeepXDE 1.15](https://img.shields.io/badge/deepxde-1.15-orange.svg)](https://deepxde.readthedocs.io/)
[![Status: Verified](https://img.shields.io/badge/reproduction-verified-brightgreen.svg)]()

Undergraduate Numerical Analysis Research Project  
**Institution:** Indian Institute of Technology Delhi (IIT Delhi)  
**Primary Paper:** *Building Trust in PINNs: Error Estimation through Finite Difference Methods* ([arXiv:2603.15526](https://arxiv.org/abs/2603.15526), 2026)  
**Upstream Repository:** [`aleks-krasowski/pinn-fdm-error-estimation`](https://github.com/aleks-krasowski/pinn-fdm-error-estimation) (Pinned Snapshot Commit `460d66bc`)  
**Submission Deadline:** November 10, 2026

---

## 1. Project Overview

Physics-Informed Neural Networks (PINNs) provide a mesh-free approach to differential equations, but evaluating their pointwise prediction error without analytical solutions remains an open challenge. This repository contains the complete, reproducible research package independently auditing and reproducing the residual-based finite-difference error estimation framework proposed by Krasowski et al. (2026), and contributing an original mathematical generalization to **soft boundary condition enforcement**.

### Key Scientific Results
* **Baseline Reproduction:** Verified the $33.80\times$ accuracy advantage in estimator discrepancy for the residual-based estimator over direct-FDM comparison on the 1D Heat Equation benchmark ($N_x=64, N_t=64$, 10,000 Adam iterations, float64). Direct-FDM error matched the author reference CSV within near-machine-precision ($1.83 \times 10^{-15}$) via algebraic cancellation.
* **Grid Resolution Convergence:** Confirmed asymptotic second-order convergence ($\mathcal{O}(h^2)$ with $p_{\text{res}} = 1.982$, $p_{\text{fdm}} = 1.990$ for $N_x \ge 32$) along a coupled spatial-temporal refinement sweep ($N_x = N_t \in [8, 256]$).
* **Boundary Condition Extension:** Discovered that applying the authors' code unmodified to soft-constrained PINNs results in breakdown ($p = 0.00$, error stalling at $\mathcal{O}(10^{-2})$). Formulated and implemented `InhomogeneousFDMSolverHeatEq`, requiring zero analytical ground truth, which restores observed second-order convergence ($p = 1.99 \pm 0.01$) and elevates estimator advantage to $85.67\times$ over direct FDM.

---

## 2. Repository Layout

```
Numerical-Insight-Lab/
├── Docs/                              # Authoritative documentation & reports
│   ├── final_research_report.md       # Full academic paper & results narrative
│   ├── penalty_sweep_analysis.md      # Detailed empirical analysis of boundary penalty sweep
│   ├── research_gap_assessment.md     # Literature review & research design
│   ├── solo_project_gap_audit.md      # Workstream consolidation & gap audit
│   ├── team_handoff.md                # Technical workstream mapping & handoff
│   ├── viva_preparation.md            # Comprehensive viva examination guide (28 Q&As)
│   ├── metric_definitions.md          # Formal mathematical definitions & norm specs
│   ├── baseline_validation.md         # Exact numerical audit against reference CSV
│   ├── grid_study_analysis.md         # Grid study convergence tables & regression fits
│   ├── boundary_extension_analysis.md # Hard vs Soft BC mathematical derivation & data
│   ├── research_decision_log.md       # Chronological log of research decisions (1–8)
│   ├── reproduction_status.md         # Project tracking against priority goals
│   └── UPSTREAM_PROVENANCE.md         # Upstream commit hash, licenses & retrieval log
├── configs/                           # Experiment configuration files
│   ├── grid_study/heat_grid_study.json
│   ├── boundary_extension/heat_soft_bc.json
│   └── boundary_extension/heat_penalty_sweep.json
├── figures/                           # Generated diagnostic figures
│   ├── grid_resolution_comparison.png # 6-level convergence plot vs author bands
│   ├── boundary_condition_comparison.png # 4-panel soft BC diagnostic figure
│   ├── penalty_sweep_comparison.png   # 4-panel boundary penalty sweep comparison
│   └── training_loss_progression.png  # Loss component evolution across penalty regimes
├── results/                           # Raw experimental metrics & run manifests
│   ├── grid_study.csv                 # 12-run raw CSV metrics
│   ├── boundary_extension.csv         # 12-run comparative extension CSV
│   ├── run_manifest.json              # Baseline execution manifest & hardware hashes
│   ├── grid_study/*.json              # Per-run execution manifests
│   ├── models/soft_pinn_seed42.pt     # Cached PyTorch weights for baseline soft PINN
│   ├── models/penalty_sweep/          # Cached PyTorch weights & JSON metadata for penalty sweep
│   └── penalty_sweep/                 # Penalty sweep CSVs, loss histories, and manifests
├── scripts/                           # Standalone reproduction & extension runners
│   ├── run_grid_study.py              # Automated grid-resolution study runner
│   ├── run_boundary_extension.py      # Hard vs. Soft BC extension runner
│   ├── run_penalty_sweep.py           # Controlled boundary penalty sweep runner
│   ├── plot_loss_curve.py             # Training loss progression visualization
│   ├── audit_penalty_sweep.py         # Independent raw data & convergence order audit tool
│   └── model_cache.py                 # Cryptographic model caching & metadata validator
├── tests/                             # Standalone verification test suite
│   └── test_reproducibility.py        # 8 mathematical and implementation invariant tests
├── upstream/                          # Vendored upstream code (strictly read-only)
│   └── pinn-fdm-error-estimation/     # Upstream snapshot at commit 460d66b
└── logs/                              # Audit trails & quality assurance logs
    ├── session4_audit.md
    └── session5_final_qa.md
```

---

## 3. Environment Setup

### Prerequisites
* Operating System: Windows 10/11, macOS, or Linux
* Python: Version 3.12 (tested on Python 3.12.10)
* Hardware: Standard CPU execution (no GPU required)

### Step-by-Step Installation
```bash
# 1. Clone the repository
git clone https://github.com/harsh-bitverse/Numerical-Insight-Lab.git
cd "Numerical-Insight-Lab"

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install deepxde polars scipy matplotlib pytest
```

---

## 4. Reproducing the Experiments

All model weights for the extension are cached in `results/models/` using SHA-256 configuration hashes. Once trained or loaded from committed checkpoints, evaluations complete in **under 10 seconds**.

### A. Run Invariant Test Suite (< 3 seconds)
Executes all 8 formal mathematical and implementation invariant checks:
```bash
pytest -v tests/test_reproducibility.py
```

### B. Run Grid-Resolution Study ($N_x \in [8, 256]$)
Executes all 12 runs across well-trained (10,000 iters) and random-init (0 iters) regimes:
```bash
python scripts/run_grid_study.py --config configs/grid_study/heat_grid_study.json
```
* **Execution Note on Fresh Clones:** If `upstream/.../model_zoo/` cache is absent, `run_grid_study.py` will train the baseline hard-PINN from scratch (~2.5 minutes on CPU) before caching and evaluating across all grid sizes. With existing cache, it evaluates in ~3 seconds.
* **Outputs:** `results/grid_study.csv`, `figures/grid_resolution_comparison.png`, `results/grid_study/*.json`.

### C. Run Boundary Condition Extension Study
Executes the 3-way comparative evaluation (Hard BC Baseline vs. Soft BC Inhomogeneous vs. Soft BC Naive Homogeneous) across resolutions $N_x \in \{16, 32, 64, 128\}$:
```bash
python scripts/run_boundary_extension.py --config configs/boundary_extension/heat_soft_bc.json
```
* **Outputs:** `results/boundary_extension.csv`, `figures/boundary_condition_comparison.png`, `results/models/soft_pinn_seed42.pt`.

### D. Run Controlled Boundary Penalty Sweep ($\lambda_{\text{bc}} \in \{1, 10, 100, 1000\}$)
Executes the parametric study evaluating boundary penalty scaling on soft PINNs:
```bash
python scripts/run_penalty_sweep.py --config configs/boundary_extension/heat_penalty_sweep.json
```
* **Outputs:** `results/penalty_sweep/heat_penalty_sweep.csv`, `results/penalty_sweep/convergence_orders.csv`, `figures/penalty_sweep_comparison.png`.

### E. Independent Raw Data Audit (< 5 seconds)
Recomputes all 24 grid evaluations and 8 convergence fits directly from checkpoint `.pt` files without relying on stored CSVs:
```bash
python scripts/audit_penalty_sweep.py
```

### F. Generate Training Loss Curves (< 2 seconds)
Visualizes loss component evolution across training steps:
```bash
python scripts/plot_loss_curve.py
```
* **Outputs:** `figures/training_loss_progression.png`.

---

## 5. Traceability & Hardware Details

* **Baseline Platform:** Intel Core i7-11800H @ 2.30 GHz, Windows 11.
* **Precision:** IEEE 754 double precision (`float64`) across all torch tensors, numpy arrays, and scipy sparse solvers.
* **Random Seed:** Standardized to seed `42`.
* **Headless Matplotlib:** All scripts configure `matplotlib.use("Agg")` to support headless terminal and continuous integration execution.

---

## 6. Project Ownership & Academic Attribution

* **Author & Collaborators:** Harshith N, Nilesh Kumar Verma, Devesh Saini (Chemical Engineering, IIT Delhi)
* **Transition Context:** The project was originally structured around three collaborative workstreams (Architecture/Solvers, Training/Residuals, Analysis/Validation). Following the administrative withdrawal of both student teammates from the course, all implementation, validation, research extension, unit testing, and documentation responsibilities were consolidated and completed independently.
* **Primary Reference:** Aleks Krasowski et al., *Building Trust in PINNs: Error Estimation through Finite Difference Methods*, [arXiv:2603.15526](https://arxiv.org/abs/2603.15526) (2026).
* **Licensing Notice:** The research paper text is governed by CC BY 4.0. The upstream code repository has no public software license (all rights reserved by the original authors; academic citation requested). Custom solvers, extensions, tests, and documentation are original contributions of the Numerical Insight Lab.
