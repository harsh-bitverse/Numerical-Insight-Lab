# Session 2 Execution Log: Diagnostic & First Baseline Runs

## 1. System & Execution Environment

* **Date:** 2026-10-20
* **Platform:** Windows (PowerShell)
* **Python Executable:** `.\.venv\Scripts\python.exe`
* **Python Version:** 3.12.10
* **Key Dependencies:**
  * `torch`: 2.14.1 (CPU)
  * `deepxde`: 1.15.0 (PyTorch backend)
  * `numpy`: 2.5.4
  * `scipy`: 1.18.1
  * `matplotlib`: 3.11.2 (Backend: `Agg`)
  * `seaborn`: 0.13.2
  * `polars`: 2.0.0
* **Precision:** `float64`
* **Device:** CPU (`set_default_device("cpu")`)

---

## 2. Run 1: Smoke Diagnostic Run

* **Objective:** Verify end-to-end execution of PINN training, residual autograd evaluation, FDM solve, and bound estimator on low-resolution grid without runtime errors.
* **Exact Command:**
  ```powershell
  $env:MPLBACKEND = "Agg"
  .\.venv\Scripts\python.exe upstream/pinn-fdm-error-estimation/examples/run_experiment.py --problem heat --n_iterations 10 --num_domain 100 --nx 16 --nt 16 --seed 42 --diffusivity 0.05 --frequency 2 --no_show
  ```
* **Start Time:** 2026-10-20T18:21:22.0614153+05:30
* **End Time:** 2026-10-20T18:21:26.9562920+05:30
* **Wall-Clock Elapsed:** 4.885 seconds
* **Exit Code:** 0
* **Console Output Summary:**
  * Initial training loss: `1.98e+00` &rarr; step 10: `1.91e+00`
  * Model saved to cache: `model_zoo/aaeda4025be66124.pt`
  * PINN training runtime: 0.0507 s
  * FDM error estimation runtime: 0.0225 s
  * FDM solution runtime: 0.0010 s
  * Error bound runtime: 0.29 s
* **Measured Error Metrics (Diagnostic, Not Baseline):**
  * True PINN Error: $L_2 = 4.131827 \times 10^{-1}$, $\text{Max} = 9.004861 \times 10^{-1}$
  * Residual Method ($e_{\text{res}}$): Est. $L_2 = 4.166780 \times 10^{-1}$, Diff $L_2 = 3.585240 \times 10^{-3}$
  * FDM Method ($e_{\text{FDM}}$): Est. $L_2 = 4.105422 \times 10^{-1}$, Diff $L_2 = 2.749211 \times 10^{-3}$
  * Outcome: FDM method temporarily better by 1.30x (expected for untrained 10-iteration model due to non-smooth initial residual).
* **Generated & Preserved Artifacts:**
  * `results/diagnostic_run/diagnostic_heat_results.png` (651 KB)
  * `results/diagnostic_run/diagnostic_heat_eq_err_comparison_time.pdf` (21 KB)

---

## 3. Run 2: Baseline Reproduction Run (Heat Equation, 64x64)

* **Objective:** Reproduce published paper configuration (Section 4, Figure 2) for the 1D Heat Equation.
* **Exact Command:**
  ```powershell
  $env:MPLBACKEND = "Agg"
  .\.venv\Scripts\python.exe upstream/pinn-fdm-error-estimation/examples/run_experiment.py --problem heat --n_iterations 10000 --num_domain 10000 --nx 64 --nt 64 --seed 42 --diffusivity 0.05 --frequency 2 --no_show
  ```
* **Start Time:** 2026-10-20T18:22:08.5998836+05:30
* **End Time:** 2026-10-20T18:24:59.5933374+05:30
* **Wall-Clock Elapsed:** 170.986 seconds (2 min 51 s)
* **Exit Code:** 0
* **Training Loss Progression:**
  * Step 0: `[1.95e+00]`
  * Step 1000: `[5.96e-02]`
  * Step 3000: `[6.61e-04]`
  * Step 5000: `[1.10e-04]`
  * Step 7000: `[3.62e-05]`
  * Step 10000: `[1.25e-05]` (Test loss: `[8.59e-06]`)
* **Execution Timing:**
  * Compilation: 0.671 s
  * PINN Training: 161.773 s
  * FDM Error Estimation ($e_{\text{res}}$): 0.1167 s
  * FDM Solution ($u_{\text{FDM}}$): 0.0040 s
  * Error Bound Estimation ($e_{\text{bound}}$): 4.43 s
* **Measured Error Metrics:**
  * **True PINN Error ($e_{\text{true}} = u_{\text{true}} - u_{\text{pinn}}$):**
    * $L_2$ Error: $1.051475 \times 10^{-4}$
    * Max Error: $4.801462 \times 10^{-4}$
  * **Residual Integration Method ($e_{\text{res}}$ — Authors' Proposal):**
    * Estimated $L_2$ Error: $1.069686 \times 10^{-4}$
    * Estimated Max Error: $4.960294 \times 10^{-4}$
    * **Diff $L_2$ ($\|e_{\text{true}} - e_{\text{res}}\|_2$):** $\mathbf{4.771048 \times 10^{-6}}$
    * Diff Max: $2.121148 \times 10^{-5}$
  * **FDM-Based Baseline ($e_{\text{FDM}} = u_{\text{FDM}} - u_{\text{pinn}}$):**
    * Estimated $L_2$ Error: $1.817874 \times 10^{-4}$
    * Estimated Max Error: $4.509696 \times 10^{-4}$
    * **Diff $L_2$ ($\|e_{\text{true}} - e_{\text{FDM}}\|_2$):** $\mathbf{1.612600 \times 10^{-4}}$
    * Diff Max: $2.748222 \times 10^{-4}$
  * **Accuracy Advantage:**
    * $\frac{\|e_{\text{true}} - e_{\text{FDM}}\|_2}{\|e_{\text{true}} - e_{\text{res}}\|_2} = \mathbf{33.80\times}$
* **Comparison with Author Reference Dataset (`all_experiments_results_float64.csv`):**
  * Author 10-seed mean for $e_{\text{res}}$ Diff $L_2$: $(5.975575 \pm 0.991409) \times 10^{-6}$
  * Our measured $e_{\text{res}}$ Diff $L_2$: $4.771048 \times 10^{-6}$ (within $1.2\sigma$ of author mean)
  * Author $e_{\text{FDM}}$ Diff $L_2$: $1.612600 \times 10^{-4}$ (exact bitwise agreement with our measured $1.612600 \times 10^{-4}$)
* **Generated & Preserved Artifacts:**
  * `results/baseline_run/baseline_heat_results.png`
  * `results/baseline_run/baseline_heat_eq_err_comparison_time.pdf`
  * `upstream/pinn-fdm-error-estimation/model_zoo/cb2ee0641f1f8be8.pt`
  * `upstream/pinn-fdm-error-estimation/model_zoo/cb2ee0641f1f8be8.json`
