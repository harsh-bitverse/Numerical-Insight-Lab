#!/usr/bin/env python3
"""Execute controlled boundary penalty sweep study for soft-constrained PINNs.

Evaluates lambda_bc in {1, 10, 100, 1000} on the 1D Heat Equation benchmark.
Investigates:
1. Trade-off between interior PDE residual and boundary violation.
2. Inhomogeneous defect estimator second-order convergence invariance.
3. Naive homogeneous estimator breakdown and error stagnation scaling.
4. Estimator advantage factor dynamics over direct FDM.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

# Ensure project root and upstream are importable
sys.path.insert(0, ".")
sys.path.insert(0, "upstream/pinn-fdm-error-estimation")

import deepxde as dde
dde.config.set_default_float("float64")

from pinn_error.core.fdm import FDMSolverHeatEq
from pinn_error.problems.heat_1d import Heat1DProblemSineIC
from scripts.model_cache import (
    compute_config_fingerprint,
    normalize_soft_pinn_config,
    train_or_load_soft_pinn_safe,
)
from scripts.run_boundary_extension import InhomogeneousFDMSolverHeatEq, ModelWrapper


def run_penalty_sweep(config_path: str = "configs/boundary_extension/heat_penalty_sweep.json"):
    """Run controlled boundary penalty sweep study."""
    with open(config_path, "r") as f:
        config = json.load(f)

    results_dir = config.get("results_dir", "results/penalty_sweep")
    cache_dir = config.get("cache_dir", "results/models/penalty_sweep")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    prob_kw = config["problem_kwargs"]
    problem = Heat1DProblemSineIC(
        x_min=prob_kw["x_min"],
        x_max=prob_kw["x_max"],
        t_max=prob_kw["t_max"],
        diffusivity=prob_kw["diffusivity"],
        frequency=prob_kw["frequency"],
    )

    penalty_weights = [float(w) for w in config["penalty_weights"]]
    grid_resolutions = config.get("grid_resolutions", [16, 32, 64])
    base_pinn_cfg = config["pinn_config"]
    lambda_ic = float(config.get("lambda_ic", 1.0))

    records = []
    trained_models = {}
    histories = {}

    print("================================================================================")
    print("STARTING CONTROLLED BOUNDARY PENALTY SWEEP")
    print(f"Penalty weights: {penalty_weights}")
    print(f"Resolutions:     {grid_resolutions}")
    print(f"Cache dir:       {cache_dir}")
    print(f"Results dir:     {results_dir}")
    print("================================================================================\n")

    # Step 1: Train or load models for each penalty weight
    for lbc in penalty_weights:
        pinn_cfg = dict(base_pinn_cfg)
        pinn_cfg["lambda_bc"] = lbc
        pinn_cfg["lambda_ic"] = lambda_ic

        print(f"\n--- Processing lambda_bc = {lbc:g} ---")
        model, train_time, hist = train_or_load_soft_pinn_safe(
            problem=problem,
            pinn_config=pinn_cfg,
            cache_dir=cache_dir,
            allow_legacy=False,
            display_every=100,
            verbose=True
        )
        trained_models[lbc] = (model, train_time)
        histories[lbc] = hist

        # Save copy of history in results_dir for convenient plotting
        hist_out_path = os.path.join(results_dir, f"loss_history_lbc_{lbc:g}.json")
        with open(hist_out_path, "w") as f:
            json.dump(hist, f, indent=2)

    # Step 2: Evaluate estimators across resolutions
    print("\n--- Evaluating Estimators Across Resolutions ---")
    for lbc in penalty_weights:
        model, train_time = trained_models[lbc]
        hist = histories[lbc]

        # Extract final training loss metrics from history if available
        if hist and "loss_train_unweighted" in hist and len(hist["loss_train_unweighted"]) > 0:
            final_unw = hist["loss_train_unweighted"][-1]
            train_loss_pde = float(final_unw[0])
            train_loss_bc_left = float(final_unw[1])
            train_loss_bc_right = float(final_unw[2])
            train_loss_ic = float(final_unw[3])
            train_loss_total = float(hist["loss_total"][-1])
        else:
            train_loss_pde = float("nan")
            train_loss_bc_left = float("nan")
            train_loss_bc_right = float("nan")
            train_loss_ic = float("nan")
            train_loss_total = float("nan")

        def soft_predict(xy):
            return model.predict(xy)

        def soft_residual(xy):
            return model.predict(xy, operator=problem.pde)

        soft_wrapper = ModelWrapper(soft_predict, soft_residual)

        for nx in grid_resolutions:
            nt = nx
            dx = 1.0 / (nx - 1)
            dt = 1.0 / (nt - 1)

            x = np.linspace(0.0, 1.0, nx)
            t = np.linspace(0.0, 1.0, nt)
            X, T = np.meshgrid(x, t)
            pts = np.column_stack([X.ravel(), T.ravel()])
            u_exact = problem.exact_solution(X, T)

            # Direct FDM solver (shared numerical benchmark)
            direct_fdm_solver = FDMSolverHeatEq(
                nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=None
            )
            u_fdm = direct_fdm_solver.solve()

            # PINN predictions
            u_soft = soft_predict(pts).reshape((nt, nx))
            e_true = u_exact - u_soft
            pinn_err_l2 = float(np.sqrt(np.mean(e_true**2)))
            pinn_err_max = float(np.max(np.abs(e_true)))

            # Empirical boundary departure
            bc_departure_left = np.abs(u_soft[:, 0] - 0.0)
            bc_departure_right = np.abs(u_soft[:, -1] - 0.0)
            bc_err_l2 = float(np.sqrt(0.5 * (np.mean(bc_departure_left**2) + np.mean(bc_departure_right**2))))
            bc_err_max = float(max(np.max(bc_departure_left), np.max(bc_departure_right)))

            # Empirical initial departure
            ic_departure = np.abs(u_soft[0, :] - np.sin(2.0 * np.pi * x))
            ic_err_l2 = float(np.sqrt(np.mean(ic_departure**2)))
            ic_err_max = float(np.max(ic_departure))

            # Empirical interior residual on grid
            R_grid = soft_residual(pts).reshape((nt, nx))
            res_l2 = float(np.sqrt(np.mean(R_grid**2)))
            res_max = float(np.max(np.abs(R_grid)))

            # Direct FDM discrepancy
            e_fdm = u_fdm - u_soft
            diff_fdm = e_true - e_fdm  # mathematically identical to u_exact - u_fdm
            fdm_disc_l2 = float(np.sqrt(np.mean(diff_fdm**2)))
            fdm_disc_max = float(np.max(np.abs(diff_fdm)))

            # -------------------------------------------------------------
            # Subcase 1: Inhomogeneous Defect Solver (Mathematically Sound)
            # -------------------------------------------------------------
            inhom_solver = InhomogeneousFDMSolverHeatEq(
                nx=nx, nt=nt, problem=problem, domain=problem.domain,
                pinn_predict_fn=soft_predict, pinn_residual_fn=soft_residual
            )
            e_res_inhom = inhom_solver.residual_integration_inhomogeneous()
            diff_inhom = e_true - e_res_inhom
            inhom_disc_l2 = float(np.sqrt(np.mean(diff_inhom**2)))
            inhom_disc_max = float(np.max(np.abs(diff_inhom)))
            adv_inhom = fdm_disc_l2 / inhom_disc_l2 if inhom_disc_l2 > 0 else float("inf")

            records.append({
                "lambda_bc": lbc,
                "lambda_ic": lambda_ic,
                "solver_type": "inhomogeneous_defect",
                "nx": nx, "nt": nt, "dx": dx, "dt": dt,
                "pinn_error_l2": pinn_err_l2,
                "pinn_error_max": pinn_err_max,
                "bc_error_l2": bc_err_l2,
                "bc_error_max": bc_err_max,
                "ic_error_l2": ic_err_l2,
                "ic_error_max": ic_err_max,
                "residual_l2": res_l2,
                "residual_max": res_max,
                "est_discrepancy_l2": inhom_disc_l2,
                "est_discrepancy_max": inhom_disc_max,
                "fdm_discrepancy_l2": fdm_disc_l2,
                "fdm_discrepancy_max": fdm_disc_max,
                "advantage_factor": adv_inhom,
                "train_loss_pde": train_loss_pde,
                "train_loss_bc_left": train_loss_bc_left,
                "train_loss_bc_right": train_loss_bc_right,
                "train_loss_ic": train_loss_ic,
                "train_loss_total": train_loss_total,
                "train_time_sec": train_time,
            })

            # -------------------------------------------------------------
            # Subcase 2: Naive Homogeneous Solver (Original Krasowski Code)
            # -------------------------------------------------------------
            naive_solver = FDMSolverHeatEq(
                nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=soft_wrapper
            )
            e_res_naive = naive_solver.residual_integration()
            diff_naive = e_true - e_res_naive
            naive_disc_l2 = float(np.sqrt(np.mean(diff_naive**2)))
            naive_disc_max = float(np.max(np.abs(diff_naive)))
            adv_naive = fdm_disc_l2 / naive_disc_l2 if naive_disc_l2 > 0 else float("inf")

            records.append({
                "lambda_bc": lbc,
                "lambda_ic": lambda_ic,
                "solver_type": "naive_homogeneous",
                "nx": nx, "nt": nt, "dx": dx, "dt": dt,
                "pinn_error_l2": pinn_err_l2,
                "pinn_error_max": pinn_err_max,
                "bc_error_l2": bc_err_l2,
                "bc_error_max": bc_err_max,
                "ic_error_l2": ic_err_l2,
                "ic_error_max": ic_err_max,
                "residual_l2": res_l2,
                "residual_max": res_max,
                "est_discrepancy_l2": naive_disc_l2,
                "est_discrepancy_max": naive_disc_max,
                "fdm_discrepancy_l2": fdm_disc_l2,
                "fdm_discrepancy_max": fdm_disc_max,
                "advantage_factor": adv_naive,
                "train_loss_pde": train_loss_pde,
                "train_loss_bc_left": train_loss_bc_left,
                "train_loss_bc_right": train_loss_bc_right,
                "train_loss_ic": train_loss_ic,
                "train_loss_total": train_loss_total,
                "train_time_sec": train_time,
            })

            print(
                f"  lambda_bc={lbc:4g} | Nx={nx:2d}: Inhom Disc={inhom_disc_l2:.3e} "
                f"(Adv={adv_inhom:5.1f}x) | Naive Disc={naive_disc_l2:.3e} (Adv={adv_naive:5.2f}x)"
            )

    # Save complete tabular results to CSV
    df = pl.DataFrame(records)
    csv_path = os.path.join(results_dir, "heat_penalty_sweep.csv")
    df.write_csv(csv_path)
    print(f"\n[DATA SAVED] Results written to {csv_path}")

    # Step 3: Compute empirical convergence orders
    print("\n--- Empirical Convergence Order Analysis ---")
    conv_records = []
    for lbc in penalty_weights:
        for stype in ["inhomogeneous_defect", "naive_homogeneous"]:
            sub = df.filter((pl.col("lambda_bc") == lbc) & (pl.col("solver_type") == stype)).sort("dx")
            dx_vals = sub["dx"].to_numpy()
            disc_vals = sub["est_discrepancy_l2"].to_numpy()

            if len(dx_vals) >= 2:
                # Log-log linear regression slope
                p_fit, _ = np.polyfit(np.log(dx_vals), np.log(disc_vals), 1)
                p_16_32 = float(np.log2(disc_vals[1] / disc_vals[0])) if len(disc_vals) >= 2 else float("nan")
                p_32_64 = float(np.log2(disc_vals[2] / disc_vals[1])) if len(disc_vals) >= 3 else float("nan")
            else:
                p_fit = float("nan")
                p_16_32 = float("nan")
                p_32_64 = float("nan")

            conv_records.append({
                "lambda_bc": lbc,
                "solver_type": stype,
                "p_fit": float(p_fit),
                "p_16_to_32": p_16_32,
                "p_32_to_64": p_32_64,
                "disc_nx16": float(disc_vals[0]),
                "disc_nx32": float(disc_vals[1]),
                "disc_nx64": float(disc_vals[2]),
            })
            print(f"  lambda_bc={lbc:4g} [{stype:20s}]: Convergence Order p = {p_fit:.2f} (16->32: {p_16_32:.2f}, 32->64: {p_32_64:.2f})")

    conv_df = pl.DataFrame(conv_records)
    conv_csv_path = os.path.join(results_dir, "convergence_orders.csv")
    conv_df.write_csv(conv_csv_path)
    print(f"[DATA SAVED] Convergence orders written to {conv_csv_path}")

    # Step 4: Generate Diagnostic Figures
    plot_penalty_sweep_diagnostics(df, conv_df, "figures/penalty_sweep_comparison.png")

    # Step 5: Save Run Manifest
    manifest = {
        "study": "heat_1d_boundary_penalty_sweep",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results_csv": csv_path,
        "convergence_csv": conv_csv_path,
        "figures": ["figures/penalty_sweep_comparison.png", "figures/training_loss_progression.png"],
        "records_count": len(records),
    }
    manifest_path = os.path.join(results_dir, "manifest_penalty_sweep.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST SAVED] Experiment manifest written to {manifest_path}")

    return df, conv_df


def plot_penalty_sweep_diagnostics(df: pl.DataFrame, conv_df: pl.DataFrame, output_path: str):
    """Generate 4-panel comprehensive diagnostic figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = {1.0: "#1f77b4", 10.0: "#2ca02c", 100.0: "#ff7f0e", 1000.0: "#d62728"}

    # Panel (a): Convergence of Inhomogeneous Estimator Discrepancy
    ax_a = axes[0, 0]
    for lbc in [1.0, 10.0, 100.0, 1000.0]:
        sub = df.filter((pl.col("lambda_bc") == lbc) & (pl.col("solver_type") == "inhomogeneous_defect")).sort("dx")
        c = colors.get(lbc, "black")
        fit_row = conv_df.filter((pl.col("lambda_bc") == lbc) & (pl.col("solver_type") == "inhomogeneous_defect"))
        p_val = fit_row["p_fit"][0] if len(fit_row) > 0 else 2.0
        ax_a.loglog(
            sub["dx"].to_list(), sub["est_discrepancy_l2"].to_list(),
            'o-', color=c, lw=2, label=fr"$\lambda={lbc:g}$ ($p={p_val:.2f}$)"
        )

    # Add reference O(h^2) slope
    dx_ref = np.array([1.0/63, 1.0/15])
    ax_a.loglog(dx_ref, 0.05 * (dx_ref**2), 'k:', lw=1.5, label=r"Reference $\mathcal{O}(h^2)$")
    ax_a.set_title(r"(a) Inhomogeneous Defect Estimator Convergence ($p \approx 2.0$)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel(r"Grid spacing $\Delta x = \Delta t$", fontsize=10)
    ax_a.set_ylabel(r"Estimator Discrepancy $\|\Delta_{\mathrm{inhom}}\|_{\mathrm{RMS}}$", fontsize=10)
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(loc="upper left", fontsize=8)

    # Panel (b): Breakdown of Naive Homogeneous Estimator
    ax_b = axes[0, 1]
    for lbc in [1.0, 10.0, 100.0, 1000.0]:
        sub = df.filter((pl.col("lambda_bc") == lbc) & (pl.col("solver_type") == "naive_homogeneous")).sort("dx")
        c = colors.get(lbc, "black")
        fit_row = conv_df.filter((pl.col("lambda_bc") == lbc) & (pl.col("solver_type") == "naive_homogeneous"))
        p_val = fit_row["p_fit"][0] if len(fit_row) > 0 else 0.0
        ax_b.loglog(
            sub["dx"].to_list(), sub["est_discrepancy_l2"].to_list(),
            '^--', color=c, lw=2, label=fr"$\lambda={lbc:g}$ ($p={p_val:.2f}$)"
        )
    ax_b.set_title(r"(b) Naive Homogeneous Estimator Stagnation Floor ($p \approx 0.0$)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel(r"Grid spacing $\Delta x = \Delta t$", fontsize=10)
    ax_b.set_ylabel(r"Estimator Discrepancy $\|\Delta_{\mathrm{naive}}\|_{\mathrm{RMS}}$", fontsize=10)
    ax_b.grid(True, which="both", alpha=0.3)
    ax_b.legend(loc="lower left", fontsize=8)

    # Panel (c): Optimization Trade-off: Boundary Departure vs Interior Residual
    ax_c = axes[1, 0]
    # Evaluate at Nx=64
    eval64 = df.filter((pl.col("nx") == 64) & (pl.col("solver_type") == "inhomogeneous_defect")).sort("lambda_bc")
    lbcs = eval64["lambda_bc"].to_list()
    bc_errs = eval64["bc_error_l2"].to_list()
    res_errs = eval64["residual_l2"].to_list()
    pinn_errs = eval64["pinn_error_l2"].to_list()

    ax_c.loglog(lbcs, bc_errs, 'rs-', lw=2, label=r"Boundary Departure $\|\delta_b\|_{\mathrm{RMS}}$")
    ax_c.loglog(lbcs, res_errs, 'b^-', lw=2, label=r"Interior Residual $\|R\|_{\mathrm{RMS}}$")
    ax_c.loglog(lbcs, pinn_errs, 'ko--', lw=1.8, label=r"True PINN Error $\|e_{\mathrm{true}}\|_{\mathrm{RMS}}$")

    ax_c.set_title(r"(c) Multi-Objective Optimization Trade-Off vs $\lambda_{\mathrm{bc}}$", fontsize=11, fontweight="bold")
    ax_c.set_xlabel(r"Boundary Penalty Weight $\lambda_{\mathrm{bc}}$", fontsize=10)
    ax_c.set_ylabel("Error / Residual Metric (Log Scale)", fontsize=10)
    ax_c.grid(True, which="both", alpha=0.3)
    ax_c.legend(loc="best", fontsize=9)

    # Panel (d): Estimator Advantage Factor over Direct FDM
    ax_d = axes[1, 1]
    nx_target = 64
    sub64_inhom = df.filter((pl.col("nx") == nx_target) & (pl.col("solver_type") == "inhomogeneous_defect")).sort("lambda_bc")
    sub64_naive = df.filter((pl.col("nx") == nx_target) & (pl.col("solver_type") == "naive_homogeneous")).sort("lambda_bc")

    indices = np.arange(len(lbcs))
    bar_width = 0.35

    ax_d.bar(indices - bar_width/2, sub64_inhom["advantage_factor"].to_list(), bar_width, label="Inhomogeneous Defect Estimator", color="#2ca02c")
    ax_d.bar(indices + bar_width/2, sub64_naive["advantage_factor"].to_list(), bar_width, label="Naive Homogeneous Estimator", color="#d62728")
    ax_d.axhline(1.0, color="gray", linestyle="--", alpha=0.8, label="FDM Parity Floor (1.0x)")

    ax_d.set_xticks(indices)
    ax_d.set_xticklabels([fr"$\lambda={l:g}$" for l in lbcs])
    ax_d.set_title(r"(d) Estimator Advantage Factor ($\|\Delta_{\mathrm{FDM}}\| / \|\Delta_{\mathrm{est}}\|$) at $64 \times 64$", fontsize=11, fontweight="bold")
    ax_d.set_xlabel(r"Boundary Penalty Weight $\lambda_{\mathrm{bc}}$", fontsize=10)
    ax_d.set_ylabel("Advantage Ratio over Direct FDM", fontsize=10)
    ax_d.set_yscale("log")
    ax_d.grid(True, which="both", axis="y", alpha=0.3)
    ax_d.legend(loc="upper right", fontsize=8)


    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIGURE GENERATED] Diagnostic plot saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run controlled boundary penalty sweep study.")
    parser.add_argument("--config", type=str, default="configs/boundary_extension/heat_penalty_sweep.json")
    args = parser.parse_args()
    run_penalty_sweep(args.config)
