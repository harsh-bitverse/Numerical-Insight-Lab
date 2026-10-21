# Project Reproduction Status

**Project Title:** Reproducing and Evaluating Finite-Difference Error Estimation in Physics-Informed Neural Networks  
**Target Paper:** *Building Trust in PINNs: Error Estimation through Finite Difference Methods* ([arXiv:2603.15526](https://arxiv.org/abs/2603.15526))  
**Official Codebase:** Pinned at commit [`460d66b`](https://github.com/aleks-krasowski/pinn-fdm-error-estimation)  
**Course Deadline:** November 10, 2026  
**Current Milestone:** Session 5 Completed (Final Research Package, Scientific QA Audit, Viva Prep & Course Submission Package Ready)

---

## 1. Executive Summary & Validation Verdict

* **Reproduction Verdict:** **Passed Meaningful Numerical Validation.**
* **Scope of Completion:**
  1. The isolated project environment (`.venv/`, Python 3.12.10, PyTorch 2.14.1, DeepXDE 1.15.0) was configured and verified without global pollution.
  2. The diagnostic run passed in 4.88 s.
  3. The primary published benchmark—the **1D Heat Equation** at resolution $N_x=64, N_t=64$ with 10,000 training iterations—was executed and fully validated against both the paper's reported behavior and the author's reference experimental dataset.
  4. The residual-based error estimator achieved an accuracy advantage of **33.80x** over the classical FDM baseline ($\|e_{\text{true}} - e_{\text{res}}\|_2 = 4.771 \times 10^{-6}$ vs. $\|e_{\text{true}} - e_{\text{FDM}}\|_2 = 1.613 \times 10^{-4}$), quantitatively reproducing the core claim of the primary paper.
  5. The controlled grid-resolution study across $N_x \in [8, 256]$ demonstrated theoretical second-order convergence ($p \approx 1.99$).
  6. The boundary condition extension demonstrated the breakdown of the upstream code on soft PINNs ($p = 0.00$) and restored observed second-order convergence ($p = 1.99 \pm 0.01$) with an elevated $85.67\times$ advantage factor via `InhomogeneousFDMSolverHeatEq`.
  7. Final research package, authoritative paper (`Docs/final_research_report.md`), repository README, team handoffs, and comprehensive viva preparation material (25 questions) are delivered and verified.

---

## 2. Quantitative Verification Table (Heat Equation, 64x64)

| Metric | Paper Reported (Fig. 2) | Author Reference CSV (`all_experiments_results_float64.csv`) | Our Measured Run (Seed 42) | Status / Alignment |
| :--- | :--- | :--- | :--- | :--- |
| **Grid Resolution** | $64 \times 64$ | $64 \times 64$ | $64 \times 64$ | Exact match |
| **Iterations / Points** | 10k iters / 10k points | 10k iters / 10k points | 10k iters / 10k points | Exact match |
| **Diffusivity $\alpha$ / Mode $n$** | $\alpha = 0.05, n = 2$ | $\alpha = 0.05, n = 2$ | $\alpha = 0.05, n = 2$ | Exact match |
| **PINN Training Time** | ~113.68 s (Author CPU) | $94.13 \pm 1.25$ s (10 seeds) | $161.77$ s (Windows CPU) | Consistent (hardware difference) |
| **FDM Solve Time** | ~0.002 s | $0.0016 \pm 0.0003$ s | $0.0040$ s | $< 5$ ms (negligible) |
| **$e_{\text{res}}$ Solve Time** | ~0.025 s | $0.106 \pm 0.008$ s | $0.1167$ s | Order of 0.1 s |
| **$e_{\text{bound}}$ Time** | ~1.87 s | Not in CSV | $4.43$ s | Fast ($< 5$ s) |
| **Final PINN Loss** | $\sim 10^{-5}$ | $1.03 \times 10^{-5}$ (approx) | Train: $1.25 \times 10^{-5}$, Test: $8.59 \times 10^{-6}$ | Consistent |
| **True PINN Error $\|u - \hat{\phi}\|_2$** | $\sim 10^{-4}$ | $(2.6 \pm 5.7) \times 10^{-4}$ | $\mathbf{1.051475 \times 10^{-4}}$ | Within author distribution |
| **$e_{\text{FDM}}$ Error ($\|e_{\text{true}} - e_{\text{FDM}}\|_2$)** | $\sim 1.6 \times 10^{-4}$ | $\mathbf{1.612600 \times 10^{-4}}$ ($\text{std} = 0.0$) | $\mathbf{1.612600 \times 10^{-4}}$ | **Near-machine-precision agreement ($1.83 \times 10^{-15}$)** |
| **$e_{\text{res}}$ Error ($\|e_{\text{true}} - e_{\text{res}}\|_2$)** | $\sim 5 \times 10^{-6}$ | $(5.975575 \pm 0.991409) \times 10^{-6}$ | $\mathbf{4.771048 \times 10^{-6}}$ | **Within $1.2\sigma$ of author mean** |
| **Accuracy Advantage Factor** | $\sim 30\times$ | $\sim 27\times$ (mean) | $\mathbf{33.80\times}$ | **Reproduced successfully** |

---

## 3. Project Priorities Progress

| Priority | Description | Target | Status |
| :--- :---: | :--- | :---: | :--- |
| **1** | Reproduce authors' method correctly | Baseline Heat 1D | **Completed & Verified** |
| **2** | Validate numerical results against authors' data | Fig. 2 / CSV | **Completed & Verified** |
| **3** | Perform small, controlled grid-resolution study | $N_x \in [8, 256]$ | **Completed & Verified** |
| **4** | Attempt one meaningful extension (Hard vs Soft BCs) | Heat 1D Inhomogeneous Defect | **Completed & Verified** |
| **5** | Final report, reproducible repository, figures & viva prep | Course Deliverable | **Completed & Delivered** |

---

## 4. Observations & Non-Blocking Technical Nuances

1. **Deterministic FDM Solution:** The FDM solution error ($\|e_{\text{true}} - e_{\text{FDM}}\|_2$) on the $64 \times 64$ grid evaluated to exactly $1.612600 \times 10^{-4}$, matching the author's reference CSV across all 10 random seeds to machine precision ($|e_{\text{ours}} - e_{\text{auth}}| \le 2.31 \times 10^{-14}$). This verifies that our spatial discretization matrix ($L = \alpha A_{xx}$) and Crank-Nicolson solver implementation execute with near-machine-precision consistency relative to the upstream reference on the deterministic FDM problem.
2. **Platform Plotting Setting:** Setting `$env:MPLBACKEND = "Agg"` and providing `./figures/` allowed headless generation of all figures without interactive GUI display issues.
3. **Model Caching Behavior:** Checkpoints are cached automatically in `upstream/pinn-fdm-error-estimation/model_zoo/` based on SHA-256 config hashes, enabling instant model re-evaluation across grid sizes without retraining.
4. **Second-Order Convergence Observed:** Empirical convergence rates of both $e_{\text{FDM}}$ and $e_{\text{res}}$ across $N_x \in [32, 256]$ match the asymptotic theoretical rate $\mathcal{O}(h^2)$ with $p_{\text{res}} = 1.9822$ and $p_{\text{fdm}} = 1.9901$ along the coupled diagonal refinement path ($N_x = N_t$).
5. **Boundary Extension Discovery:** Applying the authors' original code naively to soft-constrained PINNs results in zero convergence ($p = 0.00$) because boundary errors are forcibly ignored. Our custom `InhomogeneousFDMSolverHeatEq` restores observed second-order convergence ($p = 1.99 \pm 0.01$) and elevates the advantage factor over direct-FDM to **$85.67\times$** (and $84.8\times$ to $86.3\times$ across resolutions).

---

## 5. Course Submission Package Status

* **Final Research Report:** [`Docs/final_research_report.md`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/Docs/final_research_report.md)
* **Reproducibility README:** [`README.md`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/README.md)
* **Team Handoff Document:** [`Docs/team_handoff.md`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/Docs/team_handoff.md)
* **Viva Examination Guide:** [`Docs/viva_preparation.md`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/Docs/viva_preparation.md)
* **Quality Assurance Audit:** [`logs/session5_final_qa.md`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/logs/session5_final_qa.md)
* **Submission Readiness:** **Ready for team leader and team review.**

