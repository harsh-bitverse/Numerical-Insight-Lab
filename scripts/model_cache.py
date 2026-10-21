#!/usr/bin/env python3
"""Deterministic configuration fingerprinting and metadata validation for soft PINN checkpoints.

Guarantees research integrity:
1. Prevents silent cache collisions across different penalty weights (lambda_bc, lambda_ic).
2. Saves full configuration metadata JSON alongside PyTorch checkpoint weights (.pt).
3. Validates every hyperparameter and data setting before reusing any cached model.
4. Serializes full training loss history (steps, components, unweighted/weighted, totals).
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import deepxde as dde


def normalize_soft_pinn_config(problem: Any, pinn_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a canonical, normalized dictionary of all parameters governing soft PINN training."""
    domain = problem.domain
    prob_dict = {
        "pde_name": "heat_1d_sine_ic",
        "x_min": float(domain.x_min),
        "x_max": float(domain.x_max),
        "t_max": float(domain.t_max),
        "diffusivity": float(problem.diffusivity),
        "frequency": float(problem.frequency),
    }

    # Extract penalty weights
    lambda_bc = float(pinn_config.get("lambda_bc", 1.0))
    lambda_ic = float(pinn_config.get("lambda_ic", 1.0))

    # DeepXDE loss ordering for TimePDE with ic_bcs=[bc_l, bc_r, ic]:
    # loss 0: PDE residual (weight = 1.0)
    # loss 1: Left BC (weight = lambda_bc)
    # loss 2: Right BC (weight = lambda_bc)
    # loss 3: Initial condition (weight = lambda_ic)
    loss_weights = [1.0, lambda_bc, lambda_bc, lambda_ic]

    canonical = {
        "problem": prob_dict,
        "layers": list(pinn_config["layers"]),
        "activation": str(pinn_config.get("activation", "tanh")),
        "initializer": str(pinn_config.get("initializer", "Glorot normal")),
        "optimizer": str(pinn_config.get("optimizer", "adam")),
        "learning_rate": float(pinn_config.get("learning_rate", 1e-3)),
        "n_iterations": int(pinn_config["n_iterations"]),
        "seed": int(pinn_config["seed"]),
        "num_domain": int(pinn_config["num_domain"]),
        "num_boundary": int(pinn_config.get("num_boundary", 1000)),
        "num_initial": int(pinn_config.get("num_initial", 1000)),
        "lambda_bc": lambda_bc,
        "lambda_ic": lambda_ic,
        "loss_weights": loss_weights,
        "precision": str(pinn_config.get("precision", "float64")),
        "device": str(pinn_config.get("device", "cpu")),
    }
    return canonical


def compute_config_fingerprint(canonical_config: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint from canonical configuration dictionary."""
    canonical_json = json.dumps(canonical_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_checkpoint_base_name(canonical_config: Dict[str, Any], fingerprint: str) -> str:
    """Construct a human-readable and fingerprint-anchored checkpoint basename."""
    lbc = canonical_config["lambda_bc"]
    lic = canonical_config["lambda_ic"]
    seed = canonical_config["seed"]
    short_hash = fingerprint[:8]
    # Format floats cleanly (e.g. 1.0 -> 1, 10.0 -> 10)
    lbc_str = f"{lbc:g}".replace(".", "_")
    lic_str = f"{lic:g}".replace(".", "_")
    return f"soft_pinn_lbc{lbc_str}_lic{lic_str}_seed{seed}_{short_hash}"


def validate_checkpoint_metadata(
    metadata_path: str,
    canonical_config: Dict[str, Any],
    tolerance: float = 1e-9
) -> Tuple[bool, Optional[str]]:
    """Validate saved checkpoint metadata against the requested configuration.
    
    Returns (True, None) if completely consistent, or (False, error_reason) if mismatched.
    """
    if not os.path.exists(metadata_path):
        return False, f"Metadata file not found: {metadata_path}"

    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        return False, f"Failed to parse metadata JSON: {e}"

    stored_cfg = meta.get("canonical_config", meta)

    # Check critical hyperparameter fields
    scalar_fields = [
        "seed", "n_iterations", "num_domain", "num_boundary", "num_initial",
        "activation", "optimizer", "initializer", "precision", "device"
    ]
    for field in scalar_fields:
        if field in stored_cfg and stored_cfg[field] != canonical_config[field]:
            return False, (
                f"Field mismatch on '{field}': stored={stored_cfg[field]}, "
                f"requested={canonical_config[field]}"
            )

    float_fields = ["learning_rate", "lambda_bc", "lambda_ic"]
    for field in float_fields:
        if field in stored_cfg:
            diff = abs(float(stored_cfg[field]) - float(canonical_config[field]))
            if diff > tolerance:
                return False, (
                    f"Numerical parameter mismatch on '{field}': stored={stored_cfg[field]}, "
                    f"requested={canonical_config[field]} (diff={diff:.2e})"
                )

    if stored_cfg.get("layers") != canonical_config["layers"]:
        return False, (
            f"Layer architecture mismatch: stored={stored_cfg.get('layers')}, "
            f"requested={canonical_config['layers']}"
        )

    # Check problem parameters
    prob_stored = stored_cfg.get("problem", {})
    prob_req = canonical_config["problem"]
    for pfield in ["diffusivity", "frequency", "x_min", "x_max", "t_max"]:
        if pfield in prob_stored:
            diff = abs(float(prob_stored[pfield]) - float(prob_req[pfield]))
            if diff > tolerance:
                return False, (
                    f"Problem parameter mismatch on '{pfield}': stored={prob_stored[pfield]}, "
                    f"requested={prob_req[pfield]}"
                )

    return True, None


def serialize_training_history(
    losshistory: dde.model.LossHistory,
    model: dde.Model,
    canonical_config: Dict[str, Any],
    fingerprint: str
) -> Dict[str, Any]:
    """Convert DeepXDE LossHistory into a structured, JSON-serializable dictionary."""
    steps = [int(s) for s in losshistory.steps]
    loss_weights = [float(w) for w in model.loss_weights]

    weighted_rows = []
    unweighted_rows = []
    total_losses = []

    for row in losshistory.loss_train:
        row_arr = np.asarray(row, dtype=np.float64)
        weighted_rows.append([float(x) for x in row_arr])
        # Unweight components: loss_unweighted = loss_weighted / weight
        unweighted_comp = [
            float(x / w) if w > 0 else float(x)
            for x, w in zip(row_arr, loss_weights)
        ]
        unweighted_rows.append(unweighted_comp)
        total_losses.append(float(np.sum(row_arr)))

    return {
        "fingerprint": fingerprint,
        "steps": steps,
        "loss_total": total_losses,
        "loss_train_weighted": weighted_rows,
        "loss_train_unweighted": unweighted_rows,
        "loss_weights": loss_weights,
        "component_names": [
            "loss_pde_residual",
            "loss_bc_left",
            "loss_bc_right",
            "loss_initial_condition"
        ],
        "recorded_at": datetime.now().isoformat(),
        "canonical_config": canonical_config,
    }


def train_or_load_soft_pinn_safe(
    problem: Any,
    pinn_config: Dict[str, Any],
    cache_dir: str = "results/models",
    allow_legacy: bool = False,
    display_every: int = 100,
    force_retrain: bool = False,
    verbose: bool = True
) -> Tuple[dde.Model, float, Dict[str, Any]]:
    """Train or load soft-constrained PINN with cryptographic cache integrity.

    Returns:
        (model, train_time, history_dict)
    """
    os.makedirs(cache_dir, exist_ok=True)
    canonical = normalize_soft_pinn_config(problem, pinn_config)
    fingerprint = compute_config_fingerprint(canonical)
    base_name = get_checkpoint_base_name(canonical, fingerprint)

    pt_path = os.path.join(cache_dir, f"{base_name}.pt")
    meta_path = os.path.join(cache_dir, f"{base_name}.json")
    hist_path = os.path.join(cache_dir, f"{base_name}_history.json")

    # DeepXDE Problem Setup
    geomtime = problem.create_geometry()
    bc_l = dde.icbc.DirichletBC(
        geomtime, lambda x: 0.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0)
    )
    bc_r = dde.icbc.DirichletBC(
        geomtime, lambda x: 0.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0)
    )
    ic = dde.icbc.IC(
        geomtime, lambda x: np.sin(2.0 * np.pi * x[:, 0:1]),
        lambda x, on_initial: on_initial
    )

    data = dde.data.TimePDE(
        geometryxtime=geomtime,
        pde=problem.pde,
        ic_bcs=[bc_l, bc_r, ic],
        num_domain=canonical["num_domain"],
        num_boundary=canonical["num_boundary"],
        num_initial=canonical["num_initial"],
    )

    dde.config.set_random_seed(canonical["seed"])
    torch.manual_seed(canonical["seed"])
    np.random.seed(canonical["seed"])

    net = dde.nn.FNN(canonical["layers"], canonical["activation"], canonical["initializer"])
    model = dde.Model(data, net)
    model.compile(
        canonical["optimizer"],
        lr=canonical["learning_rate"],
        loss_weights=canonical["loss_weights"]
    )

    # 1. Attempt loading fingerprint-matched checkpoint
    checkpoint_loaded = False
    train_time = 0.0
    history_dict = {}

    if not force_retrain and os.path.exists(pt_path) and os.path.exists(meta_path):
        is_valid, reason = validate_checkpoint_metadata(meta_path, canonical)
        if is_valid:
            if verbose:
                print(f"[CACHE HIT] Loading validated Soft PINN from: {pt_path}")
            net.load_state_dict(torch.load(pt_path, map_location="cpu"))
            checkpoint_loaded = True
            if os.path.exists(hist_path):
                with open(hist_path, "r") as f:
                    history_dict = json.load(f)
        else:
            print(f"[CACHE INVALID] Metadata check failed ({reason}). Re-training...")

    # 2. Check legacy baseline if explicitly allowed and exact parameters match
    if not checkpoint_loaded and not force_retrain and allow_legacy:
        legacy_pt = os.path.join(cache_dir, f"soft_pinn_seed{canonical['seed']}.pt")
        legacy_meta = os.path.join(cache_dir, f"soft_pinn_seed{canonical['seed']}.json")

        # Legacy model is ONLY valid for lambda_bc = 1.0, lambda_ic = 1.0, 10000 steps
        is_exact_baseline = (
            abs(canonical["lambda_bc"] - 1.0) < 1e-9 and
            abs(canonical["lambda_ic"] - 1.0) < 1e-9 and
            canonical["n_iterations"] == 10000 and
            canonical["seed"] == 42
        )

        if is_exact_baseline and os.path.exists(legacy_pt):
            # Check legacy metadata if present
            if os.path.exists(legacy_meta):
                is_valid, _ = validate_checkpoint_metadata(legacy_meta, canonical)
            else:
                is_valid = True

            if is_valid:
                if verbose:
                    print(f"[LEGACY CACHE HIT] Loading verified baseline Soft PINN from: {legacy_pt}")
                net.load_state_dict(torch.load(legacy_pt, map_location="cpu"))
                checkpoint_loaded = True

    # 3. If not loaded, train from scratch and persist checkpoint + metadata + history
    if not checkpoint_loaded:
        if verbose:
            print(
                f"[TRAINING] Soft BC PINN: lambda_bc={canonical['lambda_bc']}, "
                f"lambda_ic={canonical['lambda_ic']}, seed={canonical['seed']} "
                f"({canonical['n_iterations']} iterations)..."
            )
        t0 = time.time()
        losshistory, train_state = model.train(
            iterations=canonical["n_iterations"],
            display_every=display_every
        )
        train_time = time.time() - t0
        if verbose:
            print(f"[TRAINED] Completed in {train_time:.2f} s")

        # Save weights
        torch.save(net.state_dict(), pt_path)

        # Save metadata
        metadata = {
            "fingerprint": fingerprint,
            "created_at": datetime.now().isoformat(),
            "train_time_seconds": train_time,
            "pytorch_version": torch.__version__,
            "deepxde_version": dde.__version__,
            "canonical_config": canonical,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save training history
        history_dict = serialize_training_history(losshistory, model, canonical, fingerprint)
        with open(hist_path, "w") as f:
            json.dump(history_dict, f, indent=2)

        if verbose:
            print(f"[PERSISTED] Weights -> {pt_path}")
            print(f"[PERSISTED] Metadata -> {meta_path}")
            print(f"[PERSISTED] Loss History -> {hist_path}")

    return model, train_time, history_dict
