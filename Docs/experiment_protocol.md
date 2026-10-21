# Experiment Execution Protocol

This protocol governs all experimental runs, computational reproductions, and parameter studies conducted for the undergraduate Numerical Analysis project at IIT Delhi.

---

## 1. Pre-Execution Verification Checklist

Before launching any numerical run, verify:
- [ ] Working tree is clean or has only approved, reviewable edits.
- [ ] Python environment has required packages installed (`torch`, `deepxde`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `polars`, `tqdm`).
- [ ] Execution device is explicitly pinned (`CPU` for baseline reproduction).
- [ ] Default floating-point precision is set to `float64` (`dde.config.set_default_float("float64")`).
- [ ] Random seed is explicitly recorded (default baseline: `seed = 42`).
- [ ] Output directory exists and will not overwrite previous baseline runs without timestamping.

---

## 2. Experimental Hierarchy & Workflows

### Phase 1: Diagnostic & Toolchain Sanity Check
* **Objective:** Ensure all components (geometry creation, hard-constraint output transform, network forward/backward pass, residual evaluation via autograd, and FDM sparse linear solve) execute without runtime errors.
* **Configuration:**
  * Problem: `heat`
  * Grid: $N_x = 16, N_t = 16$
  * Iterations: `n_iterations = 10`
  * Collocation Points: `num_domain = 100`
  * Expected runtime: $< 5$ seconds.

### Phase 2: Published Paper Baseline Reproduction (Figure 2)
* **Objective:** Reproduce the exact numerical error curves and time slices reported in the primary paper (Page 6, Figure 2).
* **Configuration:**
  * Problem: `heat`
  * Parameters: Diffusivity $\alpha = 0.05$, Frequency mode $n = 2$, Domain $x \in [0, 1], t \in [0, 1]$
  * Grid: $N_x = 64, N_t = 64$
  * Model Configuration:
    * Architecture: MLP with 3 hidden layers, 20 neurons each, `tanh` activation
    * Initializer: Glorot normal
    * Optimizer: Adam, learning rate $10^{-3}$
    * Domain collocation points: 10,000 uniform points
    * Training iterations: 10,000
    * Seed: `42`
  * Execution Command:
    ```bash
    python examples/run_experiment.py --problem heat --n_iterations 10000 --num_domain 10000 --nx 64 --nt 64 --seed 42 --diffusivity 0.05 --frequency 2 --no_show
    ```

### Phase 3: Grid Resolution Study (Figure 3)
* **Objective:** Verify spatial convergence behavior across grid discretizations for both well-trained and randomly initialized models.
* **Configurations:**
  * Grid sizes: $N_x = N_t \in \{8, 16, 32, 64, 128, 256\}$
  * Model regimes:
    * Well-Trained: 10,000 iterations
    * Randomly Initialized: 0 iterations
  * Replicates: 10 seeds ($seed \in \{0, \dots, 9\}$) to compute mean and standard deviation bands.

---

## 3. Evidence Logging & Output Artifact Standards

For every experimental run, the following evidence must be saved to a structured output directory (`results/experiments/<timestamp>_<problem>_<config>/`):
1. **Stdout & Stderr Log:** Full command console output including PINN training loss, solve times, and metric printouts.
2. **Metadata JSON (`meta.json`):**
   * Problem name and all physics kwargs ($\alpha, \beta, c, n, \text{domain}$)
   * FDM parameters ($N_x, N_t, \Delta x, \Delta t$)
   * PINN architecture, iteration count, seed, training time
   * Timestamps and platform hardware info
3. **Metrics Table (`metrics.csv`):**
   * Absolute discrete $L_2$ error of PINN: $\|u_{\text{true}} - u_{\text{pinn}}\|_2$
   * Residual-based error accuracy: $\|e_{\text{true}} - e_{\text{res}}\|_2$
   * FDM baseline error accuracy: $\|e_{\text{true}} - e_{\text{FDM}}\|_2$
   * Relative error percentages
   * Solver runtimes: $t_{\text{pinn}}, t_{e_{\text{res}}}, t_{u_{\text{FDM}}}$
4. **Visual Artifacts:**
   * Spatial error slices at $t \in \{0.25, 0.50, 0.75, 1.00\}$
   * Error over time plot (log-scale $L_2$ error comparison)
   * 2D Spatio-temporal error heatmap ($e_{\text{true}}$, $e_{\text{res}}$, $e_{\text{FDM}}$, $|e_{\text{true}} - e_{\text{res}}|$)

---

## 4. Quantitative Acceptance Criteria for Reproduction

Reproduction of the baseline Heat equation experiment is defined as successful if and only if:
1. **Order of Magnitude Advantage:** The residual-based error estimate achieves $\|e_{\text{true}} - e_{\text{res}}\|_2 \ll \|e_{\text{true}} - e_{\text{FDM}}\|_2$ by at least 1–2 orders of magnitude at $N_x=N_t=64$.
2. **Spatial Correlation:** The pointwise shape and zero-crossings of $e_{\text{res}}(x, t)$ closely match $e_{\text{true}}(x, t)$ across all time slices ($t \approx 0.25, 0.50, 0.75, 1.00$).
3. **Computational Efficiency:** The residual solve time $t_{e_{\text{res}}}$ is under 0.1 seconds on CPU (orders of magnitude faster than the ~100s PINN training time).
4. **Consistency with Author Dataset:** Computed metrics fall within $1\sigma$ of the values recorded in upstream `results/all_experiments_results_float64.csv`.
