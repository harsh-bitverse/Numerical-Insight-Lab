#!/usr/bin/env python3
"""Controlled Boundary Condition Extension Study for 1D Heat Equation.

Compares Hard vs. Soft Boundary Condition enforcement in PINNs and evaluates:
1. Baseline Hard BC with standard homogeneous residual integration (upstream reproduction).
2. Soft BC with mathematically consistent inhomogeneous error integration.
3. Soft BC with naive homogeneous error integration (demonstrating breakdown when BC error is ignored).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import torch

# Ensure upstream and local directories are importable
sys.path.insert(0, ".")
sys.path.insert(0, "upstream/pinn-fdm-error-estimation")

import deepxde as dde
dde.config.set_default_float("float64")

from pinn_error.core.fdm import FDMMatrixBuilder, FDMSolverHeatEq
from pinn_error.core.pinn import PINNConfig, PINNTrainer
from pinn_error.problems.heat_1d import Heat1DProblemSineIC
from pinn_error.utils.experiment_factory import ExperimentFactory


class InhomogeneousFDMSolverHeatEq(FDMSolverHeatEq):
    """Crank-Nicolson FDM solver for the error defect equation with inhomogeneous
    initial and boundary conditions:
    
        e_t - alpha * e_xx = -R(x, t)
        e(x, 0) = u_0(x) - phi(x, 0)
        e(0, t) = g_0(t) - phi(0, t)
        e(1, t) = g_1(t) - phi(1, t)
    """

    def __init__(self, nx, nt, problem, domain=None, pinn_model=None, pinn_predict_fn=None, pinn_residual_fn=None):
        if domain is None:
            domain = problem.domain
        super().__init__(nx=nx, nt=nt, problem=problem, domain=domain, pinn_model=pinn_model)
        self.pinn_predict = pinn_predict_fn
        self.pinn_residual = pinn_residual_fn

    def residual_integration_inhomogeneous(self):
        """Integrate defect equation with inhomogeneous BCs and ICs."""
        self._build_matrices()
        start_time = time.time()

        e = np.zeros((self.nt, self.nx))

        # 1. Inhomogeneous Initial Condition at t=0
        pts_t0 = np.column_stack([self.x, np.zeros_like(self.x)])
        phi_t0 = self.pinn_predict(pts_t0).flatten()
        u0 = self.problem.initial_condition(self.x)
        if hasattr(u0, "numpy"):
            u0 = u0.numpy()
        e[0, :] = u0 - phi_t0

        # Residuals at t=0
        R_curr = self.pinn_residual(pts_t0).flatten()
        R_curr[0] = 0.0
        R_curr[-1] = 0.0

        for n in range(self.nt - 1):
            t_next = self.t[n + 1]
            pts_next = np.column_stack([self.x, t_next * np.ones_like(self.x)])
            R_next = self.pinn_residual(pts_next).flatten()
            R_next[0] = 0.0
            R_next[-1] = 0.0

            # Interior Crank-Nicolson RHS
            rhs = self.M_rhs @ e[n]
            residual_source = -0.5 * self.dt * (R_curr + R_next)
            rhs += residual_source

            # Inhomogeneous boundary conditions at t_{n+1}
            # For heat equation with Dirichlet BCs g_0 = 0, g_1 = 0:
            bc_pt_left = np.array([[self.domain.x_min, t_next]])
            bc_pt_right = np.array([[self.domain.x_max, t_next]])
            phi_bc_left = float(self.pinn_predict(bc_pt_left).flatten()[0])
            phi_bc_right = float(self.pinn_predict(bc_pt_right).flatten()[0])

            rhs[0] = 0.0 - phi_bc_left
            rhs[-1] = 0.0 - phi_bc_right

            e[n + 1] = spsolve(self.M_lhs, rhs)
            R_curr = R_next

        self._run_time = time.time() - start_time
        return e


class ModelWrapper:
    """Convenience wrapper for custom predict and residual callables."""
    def __init__(self, pred_fn, res_fn):
        self.predict = pred_fn
        self.residual = res_fn


from scripts.model_cache import train_or_load_soft_pinn_safe


def train_or_load_soft_pinn(problem, pinn_config, cache_dir="results/models"):
    """Train or load cached soft-constrained PINN with cryptographic cache integrity."""
    model, train_time, _ = train_or_load_soft_pinn_safe(
        problem=problem,
        pinn_config=pinn_config,
        cache_dir=cache_dir,
        allow_legacy=True,
        display_every=100
    )
    return model, train_time



def run_boundary_extension_study(config_path="configs/boundary_extension/heat_soft_bc.json"):
    """Execute comparative study of Hard vs Soft BC error estimation."""
    with open(config_path, "r") as f:
        config = json.load(f)

    os.makedirs("results/boundary_extension", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    prob_kw = config["problem_kwargs"]
    problem = Heat1DProblemSineIC(
        x_min=prob_kw["x_min"],
        x_max=prob_kw["x_max"],
        t_max=prob_kw["t_max"],
        diffusivity=prob_kw["diffusivity"],
        frequency=prob_kw["frequency"],
    )

    # 1. Setup Hard BC PINN (Baseline)
    pinn_cfg = PINNConfig(
        n_iterations=config["pinn_config"]["n_iterations"],
        num_domain=config["pinn_config"]["num_domain"],
        seed=config["pinn_config"]["seed"],
        use_cache=True,
    )
    hard_pinn_obj = PINNTrainer(problem=problem, config=pinn_cfg)
    hard_pinn_obj.train()

    # 2. Setup Soft BC PINN
    soft_model, soft_train_time = train_or_load_soft_pinn(problem, config["pinn_config"])

    def soft_predict(xy):
        return soft_model.predict(xy)

    def soft_residual(xy):
        return soft_model.predict(xy, operator=problem.pde)

    soft_wrapper = ModelWrapper(soft_predict, soft_residual)

    # Evaluation resolutions
    grid_resolutions = [16, 32, 64, 128]
    records = []

    print("\nEvaluating estimators across resolutions...")
    for nx in grid_resolutions:
        nt = nx
        dx = 1.0 / (nx - 1)
        dt = 1.0 / (nt - 1)

        x = np.linspace(0.0, 1.0, nx)
        t = np.linspace(0.0, 1.0, nt)
        X, T = np.meshgrid(x, t)
        pts = np.column_stack([X.ravel(), T.ravel()])
        u_exact = problem.exact_solution(X, T)

        # Direct FDM solver (shared reference solution for PDE)
        direct_fdm_solver = FDMSolverHeatEq(nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=None)
        u_fdm = direct_fdm_solver.solve()

        # ---------------------------------------------------------
        # Case A: Hard BC PINN + Standard Homogeneous Solver
        # ---------------------------------------------------------
        u_hard = hard_pinn_obj.predict(pts).reshape((nt, nx))
        e_true_hard = u_exact - u_hard
        pinn_err_hard_l2 = float(np.sqrt(np.mean(e_true_hard**2)))
        pinn_err_hard_max = float(np.max(np.abs(e_true_hard)))

        hard_solver = FDMSolverHeatEq(nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=hard_pinn_obj)
        e_res_hard = hard_solver.residual_integration()
        e_fdm_hard = u_fdm - u_hard

        diff_res_hard = e_true_hard - e_res_hard
        diff_fdm_hard = e_true_hard - e_fdm_hard

        l2_diff_res_hard = float(np.sqrt(np.mean(diff_res_hard**2)))
        max_diff_res_hard = float(np.max(np.abs(diff_res_hard)))
        l2_diff_fdm_hard = float(np.sqrt(np.mean(diff_fdm_hard**2)))
        max_diff_fdm_hard = float(np.max(np.abs(diff_fdm_hard)))
        adv_hard = l2_diff_fdm_hard / l2_diff_res_hard

        records.append({
            "model_type": "hard_bc",
            "solver_type": "standard_homogeneous",
            "nx": nx, "nt": nt, "dx": dx, "dt": dt,
            "pinn_error_l2": pinn_err_hard_l2,
            "pinn_error_max": pinn_err_hard_max,
            "bc_error_max": float(np.max(np.abs(u_hard[:, [0, -1]]))),
            "ic_error_max": float(np.max(np.abs(e_true_hard[0, :]))),
            "est_discrepancy_l2": l2_diff_res_hard,
            "est_discrepancy_max": max_diff_res_hard,
            "fdm_discrepancy_l2": l2_diff_fdm_hard,
            "fdm_discrepancy_max": max_diff_fdm_hard,
            "advantage_factor": adv_hard,
        })

        # ---------------------------------------------------------
        # Case B: Soft BC PINN
        # ---------------------------------------------------------
        u_soft = soft_predict(pts).reshape((nt, nx))
        e_true_soft = u_exact - u_soft
        pinn_err_soft_l2 = float(np.sqrt(np.mean(e_true_soft**2)))
        pinn_err_soft_max = float(np.max(np.abs(e_true_soft)))
        bc_err_soft_max = float(np.max(np.abs(u_soft[:, [0, -1]])))
        ic_err_soft_max = float(np.max(np.abs(e_true_soft[0, :])))

        e_fdm_soft = u_fdm - u_soft
        diff_fdm_soft = e_true_soft - e_fdm_soft
        l2_diff_fdm_soft = float(np.sqrt(np.mean(diff_fdm_soft**2)))
        max_diff_fdm_soft = float(np.max(np.abs(diff_fdm_soft)))

        # Subcase B1: Inhomogeneous Defect Solver (Mathematically Sound)
        inhom_solver = InhomogeneousFDMSolverHeatEq(
            nx=nx, nt=nt, problem=problem, domain=problem.domain,
            pinn_predict_fn=soft_predict,
            pinn_residual_fn=soft_residual
        )
        e_res_soft_inhom = inhom_solver.residual_integration_inhomogeneous()
        diff_res_soft_inhom = e_true_soft - e_res_soft_inhom
        l2_diff_soft_inhom = float(np.sqrt(np.mean(diff_res_soft_inhom**2)))
        max_diff_soft_inhom = float(np.max(np.abs(diff_res_soft_inhom)))
        adv_soft_inhom = l2_diff_fdm_soft / l2_diff_soft_inhom

        records.append({
            "model_type": "soft_bc",
            "solver_type": "inhomogeneous_defect",
            "nx": nx, "nt": nt, "dx": dx, "dt": dt,
            "pinn_error_l2": pinn_err_soft_l2,
            "pinn_error_max": pinn_err_soft_max,
            "bc_error_max": bc_err_soft_max,
            "ic_error_max": ic_err_soft_max,
            "est_discrepancy_l2": l2_diff_soft_inhom,
            "est_discrepancy_max": max_diff_soft_inhom,
            "fdm_discrepancy_l2": l2_diff_fdm_soft,
            "fdm_discrepancy_max": max_diff_fdm_soft,
            "advantage_factor": adv_soft_inhom,
        })

        # Subcase B2: Naive Homogeneous Solver (Upstream Code Unaltered)
        naive_solver = FDMSolverHeatEq(nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=soft_wrapper)
        e_res_soft_naive = naive_solver.residual_integration()
        diff_res_soft_naive = e_true_soft - e_res_soft_naive
        l2_diff_soft_naive = float(np.sqrt(np.mean(diff_res_soft_naive**2)))
        max_diff_soft_naive = float(np.max(np.abs(diff_res_soft_naive)))
        adv_soft_naive = l2_diff_fdm_soft / l2_diff_soft_naive

        records.append({
            "model_type": "soft_bc",
            "solver_type": "naive_homogeneous",
            "nx": nx, "nt": nt, "dx": dx, "dt": dt,
            "pinn_error_l2": pinn_err_soft_l2,
            "pinn_error_max": pinn_err_soft_max,
            "bc_error_max": bc_err_soft_max,
            "ic_error_max": ic_err_soft_max,
            "est_discrepancy_l2": l2_diff_soft_naive,
            "est_discrepancy_max": max_diff_soft_naive,
            "fdm_discrepancy_l2": l2_diff_fdm_soft,
            "fdm_discrepancy_max": max_diff_fdm_soft,
            "advantage_factor": adv_soft_naive,
        })

        print(f"  Nx={nx:3d}: Hard Adv={adv_hard:5.2f}x | Soft Inhom Adv={adv_soft_inhom:5.2f}x | Soft Naive Adv={adv_soft_naive:5.2f}x")

    # Save to CSV
    df = pl.DataFrame(records)
    csv_out = "results/boundary_extension.csv"
    df.write_csv(csv_out)
    print(f"\nSaved boundary extension metrics to {csv_out}")

    # Generate Publication Figure
    plot_boundary_extension_comparison(df, problem, hard_pinn_obj, soft_predict, soft_residual)


def plot_boundary_extension_comparison(df, problem, hard_pinn_obj, soft_predict, soft_residual):
    """Generate 4-panel diagnostic figure analyzing Hard vs Soft BC estimation."""
    nx, nt = 64, 64
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    X, T = np.meshgrid(x, t)
    pts = np.column_stack([X.ravel(), T.ravel()])
    u_exact = problem.exact_solution(X, T)

    u_soft = soft_predict(pts).reshape((nt, nx))
    e_true_soft = u_exact - u_soft

    inhom_solver = InhomogeneousFDMSolverHeatEq(
        nx=nx, nt=nt, problem=problem, domain=problem.domain,
        pinn_predict_fn=soft_predict,
        pinn_residual_fn=soft_residual
    )
    e_res_soft_inhom = inhom_solver.residual_integration_inhomogeneous()

    naive_solver = FDMSolverHeatEq(nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=ModelWrapper(soft_predict, soft_residual))
    e_res_soft_naive = naive_solver.residual_integration()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Error profile slice at t = 0.5
    t_idx = nt // 2
    t_val = t[t_idx]
    ax1 = axes[0, 0]
    ax1.plot(x, e_true_soft[t_idx, :], 'k-', lw=2.0, label=r'True Error $e_{\mathrm{true}}$')
    ax1.plot(x, e_res_soft_inhom[t_idx, :], 'b--', lw=2.0, label=r'Inhomogeneous Est. $e_{\mathrm{res,inhom}}$')
    ax1.plot(x, e_res_soft_naive[t_idx, :], 'r:', lw=2.0, label=r'Naive Homogeneous Est. $e_{\mathrm{res,naive}}$')
    ax1.set_title(f"(a) Soft BC Error Profiles across Space at $t={t_val:.2f}$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Spatial coordinate $x$")
    ax1.set_ylabel("Error magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Boundary Error along x=0 over time t
    ax2 = axes[0, 1]
    ax2.plot(t, e_true_soft[:, 0], 'k-', lw=2.0, label=r'True Boundary Error $e(0, t)$')
    ax2.plot(t, e_res_soft_inhom[:, 0], 'b--', lw=2.0, label=r'Inhomogeneous $e_{\mathrm{res}}(0, t)$')
    ax2.plot(t, e_res_soft_naive[:, 0], 'r:', lw=2.0, label=r'Naive $e_{\mathrm{res}}(0, t) \equiv 0$')
    ax2.set_title("(b) Pointwise Boundary Error along Left Boundary ($x=0$)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time coordinate $t$")
    ax2.set_ylabel(r"Boundary Error $0 - \hat{\phi}(0, t)$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Convergence of Estimator Discrepancy vs Grid Spacing dx
    ax3 = axes[1, 0]
    hard_sub = df.filter(pl.col("solver_type") == "standard_homogeneous").sort("dx")
    soft_inhom_sub = df.filter(pl.col("solver_type") == "inhomogeneous_defect").sort("dx")
    soft_naive_sub = df.filter(pl.col("solver_type") == "naive_homogeneous").sort("dx")

    ax3.loglog(hard_sub["dx"].to_list(), hard_sub["est_discrepancy_l2"].to_list(), 's-', color="#1f77b4", lw=2, label="Hard BC (Standard Homogeneous)")
    ax3.loglog(soft_inhom_sub["dx"].to_list(), soft_inhom_sub["est_discrepancy_l2"].to_list(), 'o-', color="#2ca02c", lw=2, label="Soft BC (Inhomogeneous Defect)")
    ax3.loglog(soft_naive_sub["dx"].to_list(), soft_naive_sub["est_discrepancy_l2"].to_list(), '^--', color="#d62728", lw=2, label="Soft BC (Naive Homogeneous)")
    ax3.loglog(hard_sub["dx"].to_list(), hard_sub["fdm_discrepancy_l2"].to_list(), 'k:', lw=1.5, label=r"Direct FDM $\|u_{\mathrm{FDM}} - u_{\mathrm{true}}\|$")
    ax3.set_title(r"(c) Discrepancy $\|e_{\mathrm{true}} - e_{\mathrm{est}}\|_{L2}$ vs Grid Resolution", fontsize=11, fontweight="bold")
    ax3.set_xlabel(r"Grid spacing $\Delta x = \Delta t$")
    ax3.set_ylabel("Estimator Discrepancy (RMS)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Advantage Factor across Resolutions
    ax4 = axes[1, 1]
    nx_labels = [str(x) for x in hard_sub["nx"].to_list()]
    bar_width = 0.25
    indices = np.arange(len(nx_labels))

    ax4.bar(indices - bar_width, hard_sub["advantage_factor"].to_list(), bar_width, label="Hard BC", color="#1f77b4")
    ax4.bar(indices, soft_inhom_sub["advantage_factor"].to_list(), bar_width, label="Soft BC (Inhomogeneous)", color="#2ca02c")
    ax4.bar(indices + bar_width, soft_naive_sub["advantage_factor"].to_list(), bar_width, label="Soft BC (Naive)", color="#d62728")
    ax4.set_xticks(indices)
    ax4.set_xticklabels([f"Nx={n}" for n in nx_labels])
    ax4.set_title(r"(d) Advantage Factor ($\|\Delta_{\mathrm{FDM}}\| / \|\Delta_{\mathrm{res}}\|$) by Configuration", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Grid Resolution")
    ax4.set_ylabel("Advantage Ratio")
    ax4.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig_path = "figures/boundary_condition_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated comparison figure: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hard vs Soft BC Study")
    parser.add_argument("--config", type=str, default="configs/boundary_extension/heat_soft_bc.json")
    args = parser.parse_args()
    run_boundary_extension_study(args.config)
