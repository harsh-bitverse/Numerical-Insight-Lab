"""
Controlled Grid-Resolution Study for 1D Heat Equation.
Evaluates the finite-difference residual error estimator across grid resolutions:
Nx = Nt in [8, 16, 32, 64, 128, 256]
for both well-trained and randomly initialized PINNs.
"""

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

# Ensure upstream is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_PATH = PROJECT_ROOT / "upstream" / "pinn-fdm-error-estimation"
if str(UPSTREAM_PATH) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_PATH))

import deepxde as dde
from pinn_error.core.pinn import PINNConfig
from pinn_error.utils.experiment_factory import ExperimentFactory
from pinn_error.utils.setup import set_default_device


def setup_environment(precision: str = "float64", device: str = "cpu"):
    """Configure precision and compute device."""
    dde.config.set_default_float(precision)
    set_default_device(device)


def run_grid_sweep(config_path: Path) -> pl.DataFrame:
    """Execute the full grid sweep defined in config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    problem_name = config["problem"]
    problem_kwargs = config["problem_kwargs"]
    grid_sizes = config["grid_sizes"]
    regimes = config["training_regimes"]
    seed = config.get("seed", 42)

    setup_environment(config.get("precision", "float64"), config.get("device", "cpu"))

    output_dir = PROJECT_ROOT / "results" / "grid_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for regime_name, regime_cfg in regimes.items():
        n_iter = regime_cfg["n_iterations"]
        num_dom = regime_cfg["num_domain"]
        label = regime_cfg["label"]

        print(f"\n{'='*70}")
        print(f"STARTING REGIME: {regime_name.upper()} ({label})")
        print(f"Iterations: {n_iter}, Domain Points: {num_dom}, Seed: {seed}")
        print(f"{'='*70}")

        pinn_config = PINNConfig(
            n_iterations=n_iter,
            num_domain=num_dom,
            seed=seed,
            use_cache=True,
        )

        for nx in grid_sizes:
            nt = nx  # Nx = Nt
            dx = (problem_kwargs["x_max"] - problem_kwargs["x_min"]) / (nx - 1)
            dt = problem_kwargs["t_max"] / (nt - 1)

            print(f"\n--- Running {regime_name} @ Nx=Nt={nx} (dx={dx:.5f}, dt={dt:.5f}) ---")
            t_start = time.time()

            fdm_solver_kwargs = {
                "nx": nx,
                "nt": nt,
                "ny": 16,
            }

            experiment = ExperimentFactory(
                problem_name=problem_name,
                problem_kwargs=problem_kwargs,
                fdm_solver_kwargs=fdm_solver_kwargs,
                pinn_config=pinn_config,
                verbose=False,
            )

            # Execute run
            results = experiment.run_experiment()
            elapsed_wallclock = time.time() - t_start

            # Extract fields
            e_true = np.asarray(results["e_true"]).flatten()
            e_res = np.asarray(results["e_res"]).flatten()
            u_pinn = np.asarray(results["u_pinn"]).flatten()
            u_fdm = np.asarray(results["u_fdm"]).flatten()
            u_true = np.asarray(results["u_true"]).flatten()
            e_fdm_vs_true = np.asarray(results["e_fdm_vs_true"]).flatten()

            # Discrepancies
            diff_res = e_res - e_true
            diff_fdm = e_fdm_vs_true  # u_fdm - u_true

            # Compute scale-invariant RMS L2 norms
            pinn_err_l2 = float(np.sqrt(np.mean(e_true**2)))
            pinn_err_max = float(np.max(np.abs(e_true)))
            u_true_l2 = float(np.sqrt(np.mean(u_true**2)))

            e_res_l2 = float(np.sqrt(np.mean(e_res**2)))
            e_res_vs_true_l2 = float(np.sqrt(np.mean(diff_res**2)))
            e_res_vs_true_max = float(np.max(np.abs(diff_res)))
            e_res_rel_l2 = float(e_res_vs_true_l2 / (pinn_err_l2 + 1e-16))

            e_fdm_vs_true_l2 = float(np.sqrt(np.mean(diff_fdm**2)))
            e_fdm_vs_true_max = float(np.max(np.abs(diff_fdm)))
            e_fdm_rel_l2 = float(e_fdm_vs_true_l2 / (pinn_err_l2 + 1e-16))

            advantage_factor = float(e_fdm_vs_true_l2 / (e_res_vs_true_l2 + 1e-16))

            # Numerical Sanity Checks
            has_nan = any(np.isnan(v) for v in [pinn_err_l2, e_res_vs_true_l2, e_fdm_vs_true_l2])
            has_inf = any(np.isinf(v) for v in [pinn_err_l2, e_res_vs_true_l2, e_fdm_vs_true_l2])
            validation_passed = (not has_nan) and (not has_inf) and (pinn_err_l2 > 0)

            run_record = {
                "problem": problem_name,
                "regime": regime_name,
                "nx": nx,
                "nt": nt,
                "dx": dx,
                "dt": dt,
                "n_iterations": n_iter,
                "num_domain": num_dom,
                "seed": seed,
                "pinn_time": results["pinn_time"],
                "e_res_time": results["e_res_time"],
                "fdm_time": results["e_fdm_time"],
                "wallclock_time": elapsed_wallclock,
                "pinn_error_l2": pinn_err_l2,
                "pinn_error_max": pinn_err_max,
                "e_res_l2": e_res_l2,
                "e_res_vs_true_l2": e_res_vs_true_l2,
                "e_res_vs_true_max": e_res_vs_true_max,
                "e_res_rel_l2": e_res_rel_l2,
                "e_fdm_vs_true_l2": e_fdm_vs_true_l2,
                "e_fdm_vs_true_max": e_fdm_vs_true_max,
                "e_fdm_rel_l2": e_fdm_rel_l2,
                "advantage_factor": advantage_factor,
                "stability_flag": bool(results["stability_flag"]),
                "validation_passed": validation_passed,
            }

            rows.append(run_record)

            # Save individual run JSON
            run_json_path = output_dir / f"{regime_name}_nx{nx}.json"
            with open(run_json_path, "w") as f_json:
                json.dump(run_record, f_json, indent=2)

            print(f"  PINN Err L2:     {pinn_err_l2:.6e}")
            print(f"  e_res Diff L2:   {e_res_vs_true_l2:.6e}")
            print(f"  e_fdm Diff L2:   {e_fdm_vs_true_l2:.6e}")
            print(f"  Advantage:       {advantage_factor:.2f}x")
            print(f"  Validation:      {'PASS' if validation_passed else 'FAIL'}")

    df = pl.DataFrame(rows)
    csv_path = PROJECT_ROOT / "results" / "grid_study.csv"
    df.write_csv(csv_path)
    print(f"\n[OK] Grid sweep complete. Results written to: {csv_path}")

    return df


def plot_grid_study_results(df: pl.DataFrame, author_csv_path: Path, output_plot_path: Path):
    """Generate high-quality comparison plot matching Paper Figure 3(c)."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 1. Overlay Author Reference Data (Seeds 0-9) as faint shaded reference bands
    if author_csv_path.exists():
        try:
            author_df = pl.read_csv(author_csv_path)
            heat_author = author_df.filter(
                (pl.col("problem") == "heat") &
                (pl.col("seed") < 10) &
                (pl.col("num_domain") == 10000)
            )

            # Plot Author e_fdm reference band
            fdm_ref = heat_author.filter(pl.col("n_iterations") == 10000).group_by("nx").agg([
                pl.col("e_fdm_vs_true_l2").mean().alias("mean"),
                pl.col("e_fdm_vs_true_l2").std().alias("std"),
            ]).sort("nx")

            # Plot Author well-trained e_res reference band
            res_good_ref = heat_author.filter(pl.col("n_iterations") == 10000).group_by("nx").agg([
                pl.col("e_res_vs_true_l2").mean().alias("mean"),
                pl.col("e_res_vs_true_l2").std().alias("std"),
            ]).sort("nx")

            # Plot Author random-init e_res reference band
            res_rand_ref = heat_author.filter(pl.col("n_iterations") == 0).group_by("nx").agg([
                pl.col("e_res_vs_true_l2").mean().alias("mean"),
                pl.col("e_res_vs_true_l2").std().alias("std"),
            ]).sort("nx")

            # Shaded bands for author references
            ax.fill_between(
                res_good_ref["nx"].to_numpy(),
                (res_good_ref["mean"] - res_good_ref["std"]).to_numpy(),
                (res_good_ref["mean"] + res_good_ref["std"]).to_numpy(),
                color="#377eb8",
                alpha=0.15,
                label=r"Author Ref (10-seed band, Well-trained)",
            )
            ax.fill_between(
                res_rand_ref["nx"].to_numpy(),
                (res_rand_ref["mean"] - res_rand_ref["std"]).to_numpy(),
                (res_rand_ref["mean"] + res_rand_ref["std"]).to_numpy(),
                color="#984ea3",
                alpha=0.15,
                label=r"Author Ref (10-seed band, Random Init.)",
            )
        except Exception as e:
            print(f"Warning: Could not overlay author references: {e}")

    # 2. Plot Our Measured Curves
    well_trained_df = df.filter(pl.col("regime") == "well_trained").sort("nx")
    random_init_df = df.filter(pl.col("regime") == "random_init").sort("nx")

    # Direct FDM error (e_fdm)
    ax.plot(
        well_trained_df["nx"].to_numpy(),
        well_trained_df["e_fdm_vs_true_l2"].to_numpy(),
        color="#ff7f00",
        linestyle="-",
        marker="o",
        markersize=7,
        linewidth=2.5,
        label=r"$e_{\mathrm{FDM}}$ (Direct FDM Baseline)",
    )

    # Our Measured Well-Trained e_res
    ax.plot(
        well_trained_df["nx"].to_numpy(),
        well_trained_df["e_res_vs_true_l2"].to_numpy(),
        color="#377eb8",
        linestyle="-",
        marker="o",
        markersize=7,
        linewidth=2.5,
        label=r"$e_{\mathrm{res}}$ (Ours, Well-Trained)",
    )

    # Our Measured Random-Init e_res
    ax.plot(
        random_init_df["nx"].to_numpy(),
        random_init_df["e_res_vs_true_l2"].to_numpy(),
        color="#984ea3",
        linestyle="--",
        marker="s",
        markersize=7,
        linewidth=2.5,
        label=r"$e_{\mathrm{res}}$ (Ours, Random Init.)",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(well_trained_df["nx"].to_numpy())
    ax.set_xticklabels([str(x) for x in well_trained_df["nx"].to_numpy()])
    ax.set_xlabel(r"Grid size ($N_x = N_t$)", fontsize=13)
    ax.set_ylabel(r"$L_2$ Error ($\|e_{\mathrm{true}} - e_{\mathrm{method}}\|_{\mathrm{RMS}}$)", fontsize=13)
    ax.set_title("1D Heat Equation: Grid-Resolution Convergence (Fig. 3c Reproduction)", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Plot saved to: {output_plot_path}")


def main():
    config_path = PROJECT_ROOT / "configs" / "grid_study" / "heat_grid_study.json"
    author_csv = UPSTREAM_PATH / "results" / "all_experiments_results_float64.csv"
    plot_path = PROJECT_ROOT / "figures" / "grid_resolution_comparison.png"

    print("Running Controlled Grid-Resolution Study...")
    df = run_grid_sweep(config_path)
    plot_grid_study_results(df, author_csv, plot_path)


if __name__ == "__main__":
    main()
