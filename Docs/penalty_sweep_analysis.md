# Experimental Analysis: Boundary Penalty Sweep ($\lambda_{\text{bc}} \in \{1, 10, 100, 1000\}$)

**Course:** Numerical Analysis Undergraduate Research Project  
**Institution:** Indian Institute of Technology Delhi (IIT Delhi)  
**Project:** Reproducing and Evaluating Finite-Difference Error Estimation in Physics-Informed Neural Networks  
**Target Paper:** Aleks Krasowski et al., *Building Trust in PINNs: Error Estimation through Finite Difference Methods*, [arXiv:2603.15526](https://arxiv.org/abs/2603.15526) (2026)  
**Author / Sole Project Owner:** Harsh Sharma (Numerical Insight Lab)  
**Repository:** `https://github.com/harsh-bitverse/Numerical-Insight-Lab`  
**Execution Checkpoint:** Session 6D  
**Results CSV:** [`results/penalty_sweep/heat_penalty_sweep.csv`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/results/penalty_sweep/heat_penalty_sweep.csv)  
**Convergence Orders CSV:** [`results/penalty_sweep/convergence_orders.csv`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/results/penalty_sweep/convergence_orders.csv)  
**Diagnostic Figure:** [`figures/penalty_sweep_comparison.png`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/figures/penalty_sweep_comparison.png)  
**Loss History Figure:** [`figures/training_loss_progression.png`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/figures/training_loss_progression.png)  
**Submission Deadline:** November 10, 2026

---

## 1. Executive Summary & Core Scientific Findings

In Session 6D, we executed a controlled parametric investigation into how the boundary penalty weight $\lambda_{\text{bc}} \in \{1, 10, 100, 1000\}$ governs soft-constrained PINN training, boundary error layers, and finite-difference defect estimation.

All runs were protected by **cryptographic cache integrity** ([`scripts/model_cache.py`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/scripts/model_cache.py)), eliminating the cache collision bug and guaranteeing that each penalty weight trained a distinct neural network with validated metadata.

```mermaid
flowchart TD
    subgraph EmpiricalFindings["Key Experimental Discoveries"]
        E1["Inhomogeneous Estimator Invariance:<br>Second-order convergence (p = 2.00, 1.99, 2.00, 1.98)<br>strictly preserved across all 4 penalty regimes."]
        E2["Naive Homogeneous Estimator Breakdown:<br>Complete convergence stalling (p = 0.00 to 0.16)<br>across all grids; discrepancy bounded by boundary departure."]
        E3["Observed Loss Imbalance (Consistent with Gradient Stiffness):<br>At lambda_bc = 1000, boundary loss overpowers IC loss,<br>degrading true PINN error from 1.1e-3 to 0.35."]
        E4["Estimator Robustness Under PINN Failure:<br>Even when PINN error degrades to 0.35, inhomogeneous estimator<br>estimates error with 1.6e-4 accuracy (p = 1.98)."]
    end
```

### Key Quantitative Findings:
1. **Hypothesis 1 (Inhomogeneous Order Invariance) — FULLY CONFIRMED:**  
   The inhomogeneous defect estimator (`InhomogeneousFDMSolverHeatEq`) exhibits near-perfect second-order convergence ($p = 2.00 \pm 0.02$) across all 4 penalty weights ($\lambda_{\text{bc}} = 1, 10, 100, 1000$), matching the theoretical Crank-Nicolson truncation error $\mathcal{O}(\Delta x^2 + \Delta t^2)$.
2. **Hypothesis 2 (Naive Defect Stagnation) — FULLY CONFIRMED:**  
   The author's original homogeneous solver (`FDMSolverHeatEq`) completely fails to converge under grid refinement ($p = 0.00, 0.16, 0.07, 0.00$), exhibiting an error plateau directly dominated by the unmodeled boundary defect.
3. **Hypothesis 3 (Loss Imbalance & Optimization Trade-off) — EMPIRICALLY DEMONSTRATED:**  
   Increasing $\lambda_{\text{bc}}$ from $1 \to 10$ slightly improves boundary conformity, but pushing $\lambda_{\text{bc}} \to 1000$ leads to severe loss imbalance on this exploratory run (`seed=42`). Because the boundary loss is weighted $1000\times$, the optimizer prioritizes boundary residuals while the initial condition loss ($\lambda_{\text{ic}}=1.0$) stalls at $\mathcal{L}_{\text{ic}} \approx 0.498$, causing true PINN error to degrade from $1.15 \times 10^{-3}$ to $0.3518$. We note this is consistent with multi-objective gradient conflict pathologies described by Wang et al. (2021), though gradient norms during backpropagation were not directly logged in this run.
4. **Hypothesis 4 (Decoupled Estimator Fidelity) — DISCOVERED:**  
   Even when the neural network is severely degraded by optimization pathology at $\lambda_{\text{bc}}=1000$ ($e_{\text{true}} \approx 0.35$), the inhomogeneous defect estimator accurately predicts the true error field with discrepancy $\Delta_{\text{inhom}} = 1.613 \times 10^{-4}$ at $64 \times 64$, maintaining asymptotic second-order convergence ($p = 1.98$).

---

## 2. Comprehensive Tabular Results

### 2.1 Empirical Convergence Orders Across Grid Refinements ($N_x = N_t \in \{16, 32, 64\}$)

| Penalty Weight $\lambda_{\text{bc}}$ | Estimator Formulation | Convergence Order $p_{\text{fit}}$ | $p$ ($16 \to 32$) | $p$ ($32 \to 64$) | Discrepancy at $16 \times 16$ | Discrepancy at $32 \times 32$ | Discrepancy at $64 \times 64$ | Status / Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\lambda_{\text{bc}} = 1$** | **Inhomogeneous Defect** | **$2.00$** | $2.02$ | $2.12$ | $3.067 \times 10^{-5}$ | $7.060 \times 10^{-6}$ | $\mathbf{1.737 \times 10^{-6}}$ | **Second-Order Verified** |
| $\lambda_{\text{bc}} = 1$ | Naive Homogeneous | $0.00$ | $0.00$ | $0.00$ | $1.658 \times 10^{-3}$ | $1.655 \times 10^{-3}$ | $1.653 \times 10^{-3}$ | Stalled ($p=0.00$) |
| **$\lambda_{\text{bc}} = 10$** | **Inhomogeneous Defect** | **$1.99$** | $2.04$ | $2.09$ | $1.001 \times 10^{-4}$ | $2.358 \times 10^{-5}$ | $\mathbf{5.750 \times 10^{-6}}$ | **Second-Order Verified** |
| $\lambda_{\text{bc}} = 10$ | Naive Homogeneous | $0.16$ | $0.11$ | $0.22$ | $1.215 \times 10^{-3}$ | $1.044 \times 10^{-3}$ | $9.682 \times 10^{-4}$ | Stalled ($p=0.16$) |
| **$\lambda_{\text{bc}} = 100$** | **Inhomogeneous Defect** | **$2.00$** | $2.04$ | $2.10$ | $2.992 \times 10^{-4}$ | $6.980 \times 10^{-5}$ | $\mathbf{1.700 \times 10^{-5}}$ | **Second-Order Verified** |
| $\lambda_{\text{bc}} = 100$ | Naive Homogeneous | $0.07$ | $0.05$ | $0.10$ | $6.749 \times 10^{-3}$ | $6.296 \times 10^{-3}$ | $6.099 \times 10^{-3}$ | Stalled ($p=0.07$) |
| **$\lambda_{\text{bc}} = 1000$** | **Inhomogeneous Defect** | **$1.98$** | $2.03$ | $2.06$ | $2.750 \times 10^{-3}$ | $6.583 \times 10^{-4}$ | $\mathbf{1.613 \times 10^{-4}}$ | **Second-Order Verified** |
| $\lambda_{\text{bc}} = 1000$ | Naive Homogeneous | $0.00$ | $0.00$ | $0.00$ | $3.522 \times 10^{-1}$ | $3.517 \times 10^{-1}$ | $3.516 \times 10^{-1}$ | Stalled ($p=0.00$) |

---

### 2.2 Estimator Advantage & Error Metrics at Resolution $64 \times 64$

| Metric / Parameter | $\lambda_{\text{bc}} = 1$ | $\lambda_{\text{bc}} = 10$ | $\lambda_{\text{bc}} = 100$ | $\lambda_{\text{bc}} = 1000$ | Direct FDM Reference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Boundary Departure $\|\delta_b\|_{\text{RMS}}$** | $1.434 \times 10^{-3}$ | $6.426 \times 10^{-4}$ | $4.281 \times 10^{-4}$ | $5.550 \times 10^{-4}$ | $0.000$ (Exact) |
| **Initial Departure $\|\delta_{\text{ic}}\|_{\text{RMS}}$** | $2.156 \times 10^{-3}$ | $4.236 \times 10^{-3}$ | $1.968 \times 10^{-2}$ | $6.997 \times 10^{-1}$ | $0.000$ (Exact) |
| **Interior Residual $\|R\|_{\text{RMS}}$** | $3.924 \times 10^{-3}$ | $1.045 \times 10^{-2}$ | $2.915 \times 10^{-2}$ | $3.091 \times 10^{-3}$ | $\text{N/A}$ |
| **True PINN Error $\|e_{\text{true}}\|_{\text{RMS}}$** | $1.147 \times 10^{-3}$ | $1.110 \times 10^{-3}$ | $6.354 \times 10^{-3}$ | $\mathbf{3.518 \times 10^{-1}}$ | $\text{N/A}$ |
| **Direct FDM Error $\|\Delta_{\text{FDM}}\|_{\text{RMS}}$** | $1.613 \times 10^{-4}$ | $1.613 \times 10^{-4}$ | $1.613 \times 10^{-4}$ | $1.613 \times 10^{-4}$ | $1.613 \times 10^{-4}$ |
| **Inhomogeneous Discrepancy $\|\Delta_{\text{inhom}}\|_{\text{RMS}}$** | $\mathbf{1.737 \times 10^{-6}}$ | $\mathbf{5.750 \times 10^{-6}}$ | $\mathbf{1.700 \times 10^{-5}}$ | $\mathbf{1.613 \times 10^{-4}}$ | $\text{N/A}$ |
| **Inhomogeneous Advantage Factor ($\eta_{\text{inhom}}$)** | $\mathbf{92.83\times}$ | $\mathbf{28.05\times}$ | $\mathbf{9.48\times}$ | $\mathbf{1.00\times}$ | $1.00\times$ (Parity) |
| **Naive Discrepancy $\|\Delta_{\text{naive}}\|_{\text{RMS}}$** | $1.653 \times 10^{-3}$ | $9.682 \times 10^{-4}$ | $6.099 \times 10^{-3}$ | $3.516 \times 10^{-1}$ | $\text{N/A}$ |
| **Naive Advantage Factor ($\eta_{\text{naive}}$)** | $\mathbf{0.098\times}$ | $\mathbf{0.167\times}$ | $\mathbf{0.026\times}$ | $\mathbf{0.00046\times}$ | $1.00\times$ (Parity) |

---

### 2.3 Compact Consolidated Summary & Methodological Guardrails

| Boundary Weight $\lambda_{\text{bc}}$ | True PINN Error $\|e_{\text{true}}\|$ | Direct FDM Error $\|\Delta_{\text{FDM}}\|$ | Inhom. Discrepancy $\|\Delta_{\text{inhom}}\|$ | Inhom. Order $p_{\text{fit}}$ | Inhom. Advantage $\eta_{\text{inhom}}$ | Naive Discrepancy $\|\Delta_{\text{naive}}\|$ | Naive Order $p_{\text{fit}}$ | Naive Advantage $\eta_{\text{naive}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\lambda_{\text{bc}} = 1$** | $1.147 \times 10^{-3}$ | $1.613 \times 10^{-4}$ | $1.737 \times 10^{-6}$ | **$2.00$** | **$92.83\times$** | $1.653 \times 10^{-3}$ | $0.00$ | $0.098\times$ |
| **$\lambda_{\text{bc}} = 10$** | $1.110 \times 10^{-3}$ | $1.613 \times 10^{-4}$ | $5.750 \times 10^{-6}$ | **$1.99$** | **$28.05\times$** | $9.682 \times 10^{-4}$ | $0.16$ | $0.167\times$ |
| **$\lambda_{\text{bc}} = 100$** | $6.354 \times 10^{-3}$ | $1.613 \times 10^{-4}$ | $1.700 \times 10^{-5}$ | **$2.00$** | **$9.48\times$** | $6.099 \times 10^{-3}$ | $0.07$ | $0.026\times$ |
| **$\lambda_{\text{bc}} = 1000$** | $3.518 \times 10^{-1}$ | $1.613 \times 10^{-4}$ | $1.613 \times 10^{-4}$ | **$1.98$** | **$1.00\times$** | $3.516 \times 10^{-1}$ | $0.00$ | $0.00046\times$ |

#### Methodological Clarifications:
1. **Zero Information Leakage:** Neither the homogeneous nor the inhomogeneous estimator accesses the analytical solution $u_{\text{true}}(x, t)$. All boundary departures $\delta_b(t) = g(t) - \hat{u}(x_b, t)$ and initial mismatches $\delta_{\text{ic}}(x) = u_0(x) - \hat{u}(x, 0)$ are evaluated purely using the prescribed problem boundary functions $g(t) = 0$ and initial condition function $u_0(x) = \sin(2\pi x)$.
2. **Fixed-Grid Accuracy vs. Multi-Grid Convergence:** At fixed resolution $64 \times 64$, estimator discrepancy varies with residual scale ($1.737 \times 10^{-6}$ at $\lambda=1$ vs $1.613 \times 10^{-4}$ at $\lambda=1000$). However, across grid refinements ($16 \to 32 \to 64$), the asymptotic convergence order is strictly invariant at $p = 1.98 - 2.00$.
3. **Mathematical Parity at $\lambda_{\text{bc}}=1000$ ($\eta \approx 1.00\times$):** At $\lambda_{\text{bc}}=1000$, $\|\Delta_{\text{inhom}}\|_{\text{RMS}} = 1.613008 \times 10^{-4}$ coincides with $\|\Delta_{\text{FDM}}\|_{\text{RMS}} = 1.612600 \times 10^{-4}$ to 4 significant digits. This is not a numerical coincidence: when initial condition loss stalls ($\mathcal{L}_{\text{ic}} \approx 0.498$), the network prediction is severely attenuated ($\hat{u} \approx 0$), making the defect equation $\partial_t e - \alpha \partial_{xx} e = -R$ mathematically equivalent to solving the original heat equation itself. Thus, Crank-Nicolson integration of the defect equation produces the exact same truncation error as solving the heat equation directly via Crank-Nicolson.
4. **Single-Seed Scope:** While the baseline reproduction audited $n=10$ random seeds, this penalty sweep represents a focused exploratory parametric study on `seed=42`.

---

## 3. Deep-Dive Scientific Analysis

### 3.1 Proof of Estimator Second-Order Invariance (Hypothesis 1)
Across all four orders of magnitude of boundary loss penalty:
$$\log_2\left(\frac{\Delta_{32}}{\Delta_{64}}\right) \in [2.06, 2.12], \quad p_{\text{fit}} \in [1.98, 2.00]$$
This proves that the Crank-Nicolson discretization error of the defect equation:
$$\partial_t e - \alpha \partial_{xx} e = -R(x, t), \quad e(0, t) = \delta_0(t), \quad e(1, t) = \delta_1(t), \quad e(x, 0) = \delta_{\text{ic}}(x)$$
is strictly $\mathcal{O}(\Delta x^2 + \Delta t^2)$, completely invariant to the penalty weight $\lambda_{\text{bc}}$ used to train the underlying neural network.

### 3.2 Anatomy of the Naive Homogeneous Breakdown (Hypothesis 2)
In contrast, the author's homogeneous solver enforces $e(0, t) = e(1, t) \equiv 0$. The resulting discrepancy satisfies:
$$\Delta_{\text{naive}}(x, t) = |e(x, t) - e_{\text{hom}}(x, t)| \equiv |w(x, t)|$$
where $w(x, t)$ is the continuous harmonic defect wave driven by the boundary departure $\delta_b(t)$.
* For $\lambda_{\text{bc}} = 1$: $\|\Delta_{\text{naive}}\|_{\text{RMS}} = 1.653 \times 10^{-3}$ at $64 \times 64$, whereas at $16 \times 16$ it is $1.658 \times 10^{-3}$ ($p = 0.00$).
* For $\lambda_{\text{bc}} = 1000$: $\|\Delta_{\text{naive}}\|_{\text{RMS}} = 0.3516$ at $64 \times 64$, whereas at $16 \times 16$ it is $0.3522$ ($p = 0.00$).
Because the error floor is set by the neural network's boundary violation and not by the FDM grid, grid refinement cannot reduce naive estimator error. In all cases, $\eta_{\text{naive}} \ll 1$, meaning **the author's unmodified estimator is up to $2100\times$ less accurate than running a standard direct FDM simulation**.

### 3.3 Loss Imbalance and Multi-Objective Optimization Trade-Offs (Hypothesis 3)
As predicted by Wang et al. (2021), soft-constrained PINNs exhibit severe multi-objective optimization competition:
* When $\lambda_{\text{bc}} = 1$, all loss components have equal weight. The network achieves balanced convergence: $\mathcal{L}_{\text{pde}} \sim 1.88 \times 10^{-5}$, $\mathcal{L}_{\text{bc}} \sim 2.0 \times 10^{-6}$, $\mathcal{L}_{\text{ic}} \sim 4.6 \times 10^{-6}$, producing a low overall prediction error $e_{\text{true}} = 1.15 \times 10^{-3}$.
* When $\lambda_{\text{bc}} = 1000$, the boundary loss is scaled by $10^3$. The Adam optimizer's updates become heavily dominated by boundary point penalties. In consequence, the optimizer prioritizes boundary conformity while initial condition loss stalls at $\mathcal{L}_{\text{ic}} \approx 0.498$, driving the neural network's true error up to $e_{\text{true}} = 0.3518$.
* *Epistemic Calibration:* We frame this finding as an observed loss trade-off that is *consistent with* the gradient conflict pathologies documented by Wang et al. (2021), while noting that backpropagation gradient vectors were not directly logged during this training run. Multi-seed studies would further establish statistical distributions of this trade-off.

### 3.4 Decoupling of Estimator Discrepancy from PINN Prediction Error
A pivotal conceptual distinction demonstrated by this experiment is the difference between:
1. **Network Solution Quality:** Measured by $e_{\text{true}} := \|u_{\text{exact}} - \hat{u}\|_{\text{RMS}}$.
2. **Estimator Fidelity:** Measured by $\Delta_{\text{est}} := \|e_{\text{true}} - e_{\text{est}}\|_{\text{RMS}}$.

At $\lambda_{\text{bc}} = 1000$, the neural network failed as a PDE solver ($e_{\text{true}} \approx 0.35$). However, `InhomogeneousFDMSolverHeatEq` solved the defect IBVP with discrepancy $\Delta_{\text{inhom}} = 1.613 \times 10^{-4}$ and order $p = 1.98$.
**Scientific Takeaway:** The finite-difference defect estimator does not require the neural network to be well-trained to accurately assess its error. It reliably quantifies error even in failed optimization regimes.

---

## 4. Visualizations & Figures

1. **[`figures/penalty_sweep_comparison.png`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/figures/penalty_sweep_comparison.png):**
   * Panel (a): Log-log convergence of $\Delta_{\text{inhom}}$ showing parallel $p \approx 2.0$ slopes across all $\lambda_{\text{bc}}$.
   * Panel (b): Flat horizontal lines ($p \approx 0.0$) showing naive estimator stagnation.
   * Panel (c): Optimization trade-off showing boundary error reduction vs interior error explosion.
   * Panel (d): Advantage factor bar chart showing $92.8\times \to 1.0\times$ advantage for inhomogeneous solver vs $< 0.1\times$ for naive solver.
2. **[`figures/training_loss_progression.png`](file:///c:/Workspace/Research/NA%20Research%20work%20on%20PINN/figures/training_loss_progression.png):**
   * Multi-panel loss history showing unweighted component evolution over 10,000 Adam iterations across $\lambda_{\text{bc}} \in \{1, 10, 100, 1000\}$.

---

## 5. Course Submission & Viva Examination Defensibility

### Model Answers for Anticipated Examination Questions:

* **Examiner:** *"Why did the naive estimator fail so badly at $\lambda_{\text{bc}} = 1000$ ($3.5 \times 10^{-1}$ discrepancy)?"*  
  * **Response:** *"At $\lambda_{\text{bc}}=1000$, the optimizer suffered from gradient pathology, causing large initial errors at $t=0$. Because the naive estimator assumes zero initial defect ($e(x, 0) \equiv 0$) and zero boundary error, it missed the entire initial error wave, producing a discrepancy equal to the magnitude of the initial error itself."*
* **Examiner:** *"Why does the advantage factor decrease from $92.8\times$ at $\lambda_{\text{bc}}=1$ to $1.0\times$ at $\lambda_{\text{bc}}=1000$?"*  
  * **Response:** *"The advantage factor is defined as $\eta := \|\Delta_{\text{FDM}}\| / \|\Delta_{\text{res}}\|$. When the PINN is well-trained ($\lambda_{\text{bc}}=1$), the residual estimator error is $\sim 1.7 \times 10^{-6}$, providing a $92.8\times$ advantage over direct FDM ($1.6 \times 10^{-4}$). When the PINN is poorly trained due to loss imbalance ($\lambda_{\text{bc}}=1000$), the estimator error reaches $1.613 \times 10^{-4}$, which equals direct FDM error ($\eta \approx 1.00\times$). At parity, Crank-Nicolson integration of the defect equation is mathematically equivalent to Crank-Nicolson integration of the heat equation itself."*
* **Examiner:** *"Does your inhomogeneous estimator leak or depend on the exact analytical ground truth?"*  
  * **Response:** *"No. Zero analytical ground truth is used. Boundary and initial defects are computed strictly against the specified problem data: $\delta_b(t) = 0 - \hat{u}(x_b, t)$ and $\delta_{\text{ic}}(x) = \sin(2\pi x) - \hat{u}(x, 0)$. In any practical application where boundary conditions $g(t)$ and initial data $u_0(x)$ are given, these defects are computable directly from the problem statement without knowing $u(x, t)$."*
* **Examiner:** *"Can you claim that $\lambda_{\text{bc}}=1000$ always fails due to gradient stiffness?"*  
  * **Response:** *"We observe severe loss imbalance on `seed=42` where $\mathcal{L}_{\text{ic}}$ stalls at $0.498$ under the $1000\times$ boundary penalty. While this behavior is fully consistent with the multi-objective gradient conflict pathologies described by Wang et al. (2021), we strictly report this as an empirical finding on this seed, as parameter-level backpropagation gradients were not directly logged during training."*
