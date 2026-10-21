#!/usr/bin/env python3
"""Plot actual recorded PINN training loss progression curves.

Requirements satisfied:
1. Plots actual recorded history without interpolation or fabricated data points.
2. Accurately labels steps and individual loss components:
   - Total Loss
   - Interior PDE Residual Loss (Loss 0)
   - Left Boundary Loss (Loss 1)
   - Right Boundary Loss (Loss 2)
   - Initial Condition Loss (Loss 3)
3. Supports single-run diagnostics and multi-weight penalty comparison sweeps.
4. Generates publication-ready figures under figures/.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_single_history(history_data: Dict, output_path: str = "figures/training_loss_progression.png"):
    """Plot detailed loss component progression for a single training run."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    steps = np.array(history_data["steps"])
    total_loss = np.array(history_data["loss_total"])
    unweighted = np.array(history_data["loss_train_unweighted"])  # shape: [N_steps, 4]
    weighted = np.array(history_data["loss_train_weighted"])      # shape: [N_steps, 4]
    weights = history_data.get("loss_weights", [1.0, 1.0, 1.0, 1.0])
    cfg = history_data.get("canonical_config", {})
    lbc = cfg.get("lambda_bc", weights[1] if len(weights) > 1 else 1.0)
    seed = cfg.get("seed", 42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Actual unweighted physical loss components
    ax1 = axes[0]
    ax1.semilogy(steps, total_loss, 'k-', lw=2.2, label="Total Loss (Weighted)")
    ax1.semilogy(steps, unweighted[:, 0], 'b-', lw=1.8, label="PDE Residual Loss")
    ax1.semilogy(steps, unweighted[:, 1], 'r--', lw=1.5, label="Left Boundary Loss ($x=0$)")
    ax1.semilogy(steps, unweighted[:, 2], 'm-.', lw=1.5, label="Right Boundary Loss ($x=1$)")
    ax1.semilogy(steps, unweighted[:, 3], 'g:', lw=1.8, label="Initial Condition Loss ($t=0$)")

    ax1.set_title(fr"Unweighted Loss Components ($\lambda={lbc:g}$, Seed={seed})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Training Step (Adam Iteration)", fontsize=10)
    ax1.set_ylabel("Mean Squared Error (Log Scale)", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Component Breakdown / Proportions
    ax2 = axes[1]
    pde_w = weighted[:, 0]
    bc_w = weighted[:, 1] + weighted[:, 2]
    ic_w = weighted[:, 3]
    tot_w = np.maximum(total_loss, 1e-12)

    ax2.plot(steps, pde_w / tot_w, 'b-', lw=1.8, label="PDE Residual Share")
    ax2.plot(steps, bc_w / tot_w, 'r--', lw=1.8, label="Boundary Condition Share")
    ax2.plot(steps, ic_w / tot_w, 'g:', lw=1.8, label="Initial Condition Share")

    ax2.set_title("Relative Loss Component Shares in Optimization Objective", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Training Step (Adam Iteration)", fontsize=10)
    ax2.set_ylabel("Fraction of Total Objective", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIGURE GENERATED] Saved single-run loss curve: {output_path}")


def plot_penalty_sweep_histories(
    history_files: List[str],
    output_path: str = "figures/training_loss_progression.png"
):
    """Plot multi-curve comparison across penalty weights lambda_bc in {1, 10, 100, 1000}."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    loaded = []
    for fp in history_files:
        with open(fp, "r") as f:
            data = json.load(f)
            cfg = data.get("canonical_config", {})
            lbc = cfg.get("lambda_bc", 1.0)
            loaded.append((lbc, data))

    # Sort by lambda_bc
    loaded.sort(key=lambda x: x[0])

    colors = {1.0: "#1f77b4", 10.0: "#2ca02c", 100.0: "#ff7f0e", 1000.0: "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): Total Loss (weighted)
    ax_a = axes[0, 0]
    # Panel (b): Unweighted PDE Residual Loss
    ax_b = axes[0, 1]
    # Panel (c): Unweighted Boundary Loss (avg of left and right)
    ax_c = axes[1, 0]
    # Panel (d): Unweighted Initial Condition Loss
    ax_d = axes[1, 1]

    for lbc, data in loaded:
        c = colors.get(lbc, "black")
        lbl = fr"$\lambda={lbc:g}$"
        steps = np.array(data["steps"])
        tot = np.array(data["loss_total"])
        unw = np.array(data["loss_train_unweighted"])

        ax_a.semilogy(steps, tot, color=c, lw=1.8, label=lbl)
        ax_b.semilogy(steps, unw[:, 0], color=c, lw=1.8, label=lbl)
        avg_bc = 0.5 * (unw[:, 1] + unw[:, 2])
        ax_c.semilogy(steps, avg_bc, color=c, lw=1.8, label=lbl)
        ax_d.semilogy(steps, unw[:, 3], color=c, lw=1.8, label=lbl)

    ax_a.set_title(r"(a) Total Weighted Training Objective $\mathcal{L}(\theta)$", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Adam Iteration", fontsize=10)
    ax_a.set_ylabel("Weighted Loss", fontsize=10)
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(loc="upper right", fontsize=9)

    ax_b.set_title("(b) Interior PDE Residual Loss (Unweighted)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Adam Iteration", fontsize=10)
    ax_b.set_ylabel("MSE Residual", fontsize=10)
    ax_b.grid(True, which="both", alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=9)

    ax_c.set_title("(c) Boundary Departure Loss (Unweighted)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Adam Iteration", fontsize=10)
    ax_c.set_ylabel("MSE Boundary Violation", fontsize=10)
    ax_c.grid(True, which="both", alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=9)

    ax_d.set_title("(d) Initial Condition Loss (Unweighted)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Adam Iteration", fontsize=10)
    ax_d.set_ylabel("MSE Initial Departure", fontsize=10)
    ax_d.grid(True, which="both", alpha=0.3)
    ax_d.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIGURE GENERATED] Saved multi-penalty loss progression: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot actual recorded PINN loss progression.")
    parser.add_argument("--history", type=str, default=None, help="Path to single history JSON file.")
    parser.add_argument("--sweep_dir", type=str, default=None, help="Directory containing multiple history JSON files.")
    parser.add_argument("--output", type=str, default="figures/training_loss_progression.png", help="Output PNG path.")
    args = parser.parse_args()

    if args.history:
        with open(args.history, "r") as f:
            data = json.load(f)
        plot_single_history(data, args.output)
    elif args.sweep_dir:
        files = glob.glob(os.path.join(args.sweep_dir, "*history*.json"))
        if not files:
            print(f"[ERROR] No *history*.json files found in {args.sweep_dir}")
            sys.exit(1)
        plot_penalty_sweep_histories(files, args.output)
    else:
        # Check default locations
        candidates = (
            glob.glob("results/penalty_sweep/*history*.json") or
            glob.glob("results/models/penalty_sweep/*history*.json") or
            glob.glob("results/models/*history*.json")
        )
        if len(candidates) == 1:
            with open(candidates[0], "r") as f:
                data = json.load(f)
            plot_single_history(data, args.output)
        elif len(candidates) > 1:
            plot_penalty_sweep_histories(candidates, args.output)
        else:
            print("[INFO] No loss history files found to plot yet. Run an experiment first.")



if __name__ == "__main__":
    main()
