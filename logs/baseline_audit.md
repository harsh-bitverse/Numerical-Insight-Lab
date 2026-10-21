# Baseline Audit Findings & Verification Log

**Audit Conducted:** 2026-10-20 (Session 3)  
**Audited Artifacts:**
- `logs/first_run.md`
- `results/run_manifest.json`
- `upstream/pinn-fdm-error-estimation/pinn_error/utils/plotting.py`
- `upstream/pinn-fdm-error-estimation/examples/run_all_experiments.py`
- `upstream/pinn-fdm-error-estimation/notebooks/comparison_plot.ipynb`
- `upstream/pinn-fdm-error-estimation/results/all_experiments_results_float64.csv`

---

## 1. Audit Responses to Phase A Checklist

### 1.1 Error Definitions & Sign Conventions
* **True PINN Error:** $e_{\text{true}} = u_{\text{true}} - u_{\text{pinn}}$.
* **Signed vs. Absolute Error:**
  * Slices and heatmaps plot signed error ($e = u_{\text{true}} - u_{\text{pinn}}$).
  * Summary metrics compute root-mean-square of signed differences $\sqrt{\text{mean}(\Delta^2)}$ and max absolute error $\max(|\Delta|)$.
* **Residual-Based Error:** $e_{\text{res}}$ obtained by solving Crank-Nicolson defect system with source $-R$.
* **Direct-FDM Discrepancy:** $e_{\text{FDM}} = u_{\text{FDM}} - u_{\text{pinn}}$, giving discrepancy $\Delta_{\text{FDM}} = e_{\text{true}} - e_{\text{FDM}} = u_{\text{true}} - u_{\text{FDM}}$.

### 1.2 Norm Characterization
* **Norm Type:** Discrete Root Mean Square (RMS) $L_2$ norm:
  $$\|V\|_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{i=1}^N V_i^2}$$
* **Normalization:** The sum of squared errors is normalized by the total number of grid points $N = N_x \times N_t$ before taking the square root.
* **Continuum Analogy:** Because the domain is the unit square $[0, 1] \times [0, 1]$ (area = 1.0), this RMS norm directly approximates the continuous $L^2([0, 1]^2)$ norm without blowing up as $\mathcal{O}(\sqrt{N})$.

### 1.3 Boundary Point Treatment & Grid Compatibility
* In `Heat1DProblemSineIC`, hard-constraining guarantees that $u_{\text{pinn}}(0, t) = u_{\text{true}}(0, t) = 0$ and $u_{\text{pinn}}(1, t) = u_{\text{true}}(1, t) = 0$.
* In `FDMSolverHeatEq`, boundary rows of the spatial matrices explicitly set boundary errors to zero.
* All arrays ($u_{\text{pinn}}, u_{\text{true}}, u_{\text{FDM}}, e_{\text{res}}$) are computed on identical collocated tensor grids of shape $(N_t, N_x)$.

### 1.4 Production of Upstream CSV Dataset
* Script: `examples/run_all_experiments.py`.
* Aggregation: Evaluated across 10 random seeds ($0 \dots 9$) for each problem, grid size, and training regime.
* Precision: `dde.config.set_default_float("float64")`.
* Problem parameters for Heat: $\alpha=0.05, n=2, x \in [0, 1], t \in [0, 1]$.

### 1.5 The 7-Significant-Figure Direct-FDM Match
* Computed value: `0.00016126004189332567`.
* Upstream CSV value: `0.00016126004189332567` across all 10 seeds.
* **Audit Confirmation:** This is an exact mathematical consequence of $u_{\text{FDM}}$ being a deterministic finite-difference solution of the heat equation independent of PINN training.

### 1.6 The 33.80x Ratio Verification
* Ratio: $\frac{1.61260042 \times 10^{-4}}{4.771048 \times 10^{-6}} = 33.7997\dots \approx 33.80\times$.
* Both terms represent discrepancies against the ground truth $e_{\text{true}}$. The ratio is dimensionless and rigorously valid.

### 1.7 Error Bound ($e_{\text{bound}}$) Role
* Semigroup bound over time from Hillebrecht & Unger.
* Highly conservative upper bound on $\|e(\cdot, t)\|_{L^2}$.
* Offers no spatial resolution and is not part of the spatial grid study (Figure 3).

### 1.8 Figure 3(c) Source Protocol
* Source file: `upstream/pinn-fdm-error-estimation/notebooks/comparison_plot.ipynb` (Cells 3 and 4).
* Three curves:
  1. `e_fdm`: $N_x=N_t \in [8, 256]$, $n_{\text{iter}}=10000$, metric `e_fdm_vs_true_l2`.
  2. `e_res_good`: $N_x=N_t \in [8, 256]$, $n_{\text{iter}}=10000$, metric `e_res_vs_true_l2`.
  3. `e_res_random`: $N_x=N_t \in [8, 256]$, $n_{\text{iter}}=0$, metric `e_res_vs_true_l2`.
* Replicates: 10 seeds ($0 \dots 9$) with standard deviation error bands.

---

## 2. Audit Conclusion

The baseline implementation and evaluation code are verified to be mathematically consistent, numerically stable, and reproducible. We are ready to proceed with Phase B (Grid Study Protocol).
