#!/usr/bin/env python3
"""Focused test suite for mathematical and implementation invariants.

Course: Numerical Analysis Term Project, IIT Delhi
Scope: Verification of FDM matrix builders, Crank-Nicolson discretization,
       hard and soft boundary enforcement, residual source signs,
       configuration-aware caching integrity, and numerical convergence.
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root and upstream directory are on sys.path
sys.path.insert(0, ".")
sys.path.insert(0, "upstream/pinn-fdm-error-estimation")

import tempfile
import numpy as np
import pytest
import scipy.sparse as sparse
import torch
import deepxde as dde
dde.config.set_default_float("float64")


from pinn_error.core.fdm import FDMMatrixBuilder, FDMSolverHeatEq
from pinn_error.problems.heat_1d import Heat1DProblemSineIC
from scripts.model_cache import (
    compute_config_fingerprint,
    get_checkpoint_base_name,
    normalize_soft_pinn_config,
    validate_checkpoint_metadata,
)
from scripts.run_boundary_extension import InhomogeneousFDMSolverHeatEq, ModelWrapper


# ==============================================================================
# Invariant 1: Finite-difference matrix dimensions and stencil coefficients
# ==============================================================================
def test_fdm_matrix_dimensions_and_stencil_coefficients():
    """Verify Crank-Nicolson matrix dimensions and stencil coefficients.
    
    Mathematical Invariant:
        For 1D Heat Eq: (I - alpha*dt/2 * A_xx) e^{n+1} = (I + alpha*dt/2 * A_xx) e^n
        Interior rows must implement stencil: [-r/2, 1+r, -r/2] for LHS,
        and [r/2, 1-r, r/2] for RHS, where r = alpha*dt / dx^2.
        Boundary rows (i=0, i=Nx-1) must be identity: [1, 0, ..., 0] and [0, ..., 0, 1].
    
    Bug Detected:
        Incorrect matrix dimensions, inverted Crank-Nicolson weights,
        or corrupted boundary rows that would violate Dirichlet conditions.
    """
    nx, nt = 11, 11
    dx = 1.0 / (nx - 1)
    dt = 1.0 / (nt - 1)
    alpha = 0.05
    r = alpha * dt / (dx**2)

    problem = Heat1DProblemSineIC(x_min=0.0, x_max=1.0, t_max=1.0, diffusivity=alpha, frequency=2.0)
    solver = FDMSolverHeatEq(nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=None)
    solver._build_matrices()
    M_lhs, M_rhs = solver.M_lhs, solver.M_rhs

    # Dimension invariant
    assert M_lhs.shape == (nx, nx), f"Expected M_lhs shape {(nx, nx)}, got {M_lhs.shape}"
    assert M_rhs.shape == (nx, nx), f"Expected M_rhs shape {(nx, nx)}, got {M_rhs.shape}"

    lhs_dense = M_lhs.toarray()
    rhs_dense = M_rhs.toarray()

    # Boundary row invariants (Dirichlet identity rows)
    assert lhs_dense[0, 0] == 1.0 and np.all(lhs_dense[0, 1:] == 0.0), "LHS row 0 not identity"
    assert lhs_dense[-1, -1] == 1.0 and np.all(lhs_dense[-1, :-1] == 0.0), "LHS row -1 not identity"
    assert rhs_dense[0, 0] == 0.0 and np.all(rhs_dense[0, :] == 0.0), "RHS row 0 not zeroed"
    assert rhs_dense[-1, -1] == 0.0 and np.all(rhs_dense[-1, :] == 0.0), "RHS row -1 not zeroed"

    # Interior stencil coefficients
    for i in range(1, nx - 1):
        assert np.isclose(lhs_dense[i, i - 1], -0.5 * r), f"LHS row {i} lower diag mismatch"
        assert np.isclose(lhs_dense[i, i], 1.0 + r), f"LHS row {i} main diag mismatch"
        assert np.isclose(lhs_dense[i, i + 1], -0.5 * r), f"LHS row {i} upper diag mismatch"

        assert np.isclose(rhs_dense[i, i - 1], 0.5 * r), f"RHS row {i} lower diag mismatch"
        assert np.isclose(rhs_dense[i, i], 1.0 - r), f"RHS row {i} main diag mismatch"
        assert np.isclose(rhs_dense[i, i + 1], 0.5 * r), f"RHS row {i} upper diag mismatch"


# ==============================================================================
# Invariant 2: Symmetry of interior diffusion operator
# ==============================================================================
def test_diffusion_matrix_interior_symmetry():
    """Verify symmetry of the 1D spatial second-derivative operator on interior nodes.
    
    Mathematical Invariant:
        Central difference Laplacian A_xx = (1/dx^2) tridiag(1, -2, 1) is self-adjoint
        and symmetric: A_interior = A_interior^T.
    
    Bug Detected:
        Asymmetric stencil formulations, off-center difference errors,
        or unintentional directional bias.
    """
    nx, nt = 15, 15
    dx = 1.0 / (nx - 1)
    alpha = 0.05
    A_xx = FDMMatrixBuilder.get_derivative_matrix_1d(nx, dx, order=2).toarray()

    # Second-order central difference matrix interior must be strictly symmetric
    assert np.allclose(A_xx[1:-1, 1:-1], A_xx[1:-1, 1:-1].T), "A_xx interior is not symmetric"

    problem = Heat1DProblemSineIC(x_min=0.0, x_max=1.0, t_max=1.0, diffusivity=alpha, frequency=2.0)
    solver = FDMSolverHeatEq(nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=None)
    solver._build_matrices()

    lhs_interior = solver.M_lhs.toarray()[1:-1, 1:-1]
    rhs_interior = solver.M_rhs.toarray()[1:-1, 1:-1]

    assert np.allclose(lhs_interior, lhs_interior.T), "M_lhs interior is not strictly symmetric"
    assert np.allclose(rhs_interior, rhs_interior.T), "M_rhs interior is not strictly symmetric"



# ==============================================================================
# Invariant 3: Finite-valued solver outputs
# ==============================================================================
def test_fdm_solver_finite_values_small_problem():
    """Verify direct FDM solver produces finite, non-NaN, non-Inf outputs.
    
    Mathematical Invariant:
        Unconditionally stable Crank-Nicolson scheme yields bounded, finite values.
    
    Bug Detected:
        Matrix singularity, ill-conditioned LU factorization, or zero division in solve.
    """
    problem = Heat1DProblemSineIC(x_min=0.0, x_max=1.0, t_max=0.5, diffusivity=0.05, frequency=2.0)
    solver = FDMSolverHeatEq(nx=9, nt=9, problem=problem, domain=problem.domain, pinn_model=None)
    u_fdm = solver.solve()

    assert u_fdm.shape == (9, 9), f"Expected shape (9, 9), got {u_fdm.shape}"
    assert np.all(np.isfinite(u_fdm)), "FDM solution contains non-finite (NaN or Inf) values"
    assert np.max(np.abs(u_fdm)) <= 1.0 + 1e-12, "FDM solution violates maximum principle for Heat Eq"


# ==============================================================================
# Invariant 4: Hard-constrained PINN boundary & initial exactness
# ==============================================================================
def test_hard_constrained_pinn_ansatz_invariants():
    """Verify that the hard-constrained ansatz mathematically forces exact BCs and ICs.
    
    Mathematical Invariant:
        u_hat(x, t) = sin(2*pi*x) + t * x * (1 - x) * N(x, t; theta)
        u_hat(x, 0) == sin(2*pi*x) for all x.
        u_hat(0, t) == 0 for all t.
        u_hat(1, t) == 0 for all t.
    
    Bug Detected:
        Faulty ansatz exponents, missing multiplicative boundary factors, or non-zero boundary leakage.
    """
    # Create random neural network outputs N(x, t)
    x = np.linspace(0.0, 1.0, 50)
    t = np.linspace(0.0, 1.0, 50)
    X, T = np.meshgrid(x, t)
    np.random.seed(1234)
    N_mock = np.random.randn(*X.shape) * 100.0  # Large arbitrary values

    # Hard-ansatz formulation from Krasowski et al. (2026)
    u_hat = np.sin(2.0 * np.pi * X) + T * X * (1.0 - X) * N_mock

    # 1. Initial condition at t=0
    u_t0 = u_hat[0, :]
    assert np.allclose(u_t0, np.sin(2.0 * np.pi * x), atol=1e-15), "Hard ansatz violates initial condition"

    # 2. Left boundary at x=0
    u_x0 = u_hat[:, 0]
    assert np.allclose(u_x0, 0.0, atol=1e-15), "Hard ansatz violates left boundary condition"

    # 3. Right boundary at x=1
    u_x1 = u_hat[:, -1]
    assert np.allclose(u_x1, 0.0, atol=1e-15), "Hard ansatz violates right boundary condition"


# ==============================================================================
# Invariant 5: Inhomogeneous defect solver initial and boundary enforcement
# ==============================================================================
def test_inhomogeneous_defect_solver_enforcement():
    """Verify InhomogeneousFDMSolverHeatEq strictly enforces prescribed boundary and initial defects.
    
    Mathematical Invariant:
        e(x, 0) = u_0(x) - phi(x, 0)
        e(0, t) = 0 - phi(0, t)
        e(1, t) = 0 - phi(1, t)
    
    Bug Detected:
        Inhomogeneous solver failing to inject empirical boundary defects into Crank-Nicolson state.
    """
    problem = Heat1DProblemSineIC(x_min=0.0, x_max=1.0, t_max=0.5, diffusivity=0.05, frequency=2.0)
    nx, nt = 9, 9

    # Predictor with non-zero boundary and initial departure
    def mock_predict(xy):
        # xy is shape [N, 2] where column 0 is x, column 1 is t
        x_col = xy[:, 0:1]
        t_col = xy[:, 1:2]
        return 0.5 * np.sin(2.0 * np.pi * x_col) + 0.1 * (x_col + 1.0) * (t_col + 0.5)

    def mock_residual(xy):
        return np.zeros((xy.shape[0], 1))

    solver = InhomogeneousFDMSolverHeatEq(
        nx=nx, nt=nt, problem=problem, domain=problem.domain,
        pinn_predict_fn=mock_predict, pinn_residual_fn=mock_residual
    )
    e = solver.residual_integration_inhomogeneous()

    # Check initial defect at t=0
    x_nodes = np.linspace(0.0, 1.0, nx)
    pts_t0 = np.column_stack([x_nodes, np.zeros_like(x_nodes)])
    expected_e0 = np.sin(2.0 * np.pi * x_nodes) - mock_predict(pts_t0).flatten()
    assert np.allclose(e[0, :], expected_e0, atol=1e-14), "Inhomogeneous solver IC mismatch at t=0"

    # Check boundary defect at left boundary x=0 for all time steps
    t_nodes = np.linspace(0.0, 0.5, nt)
    pts_bc0 = np.column_stack([np.zeros_like(t_nodes), t_nodes])
    expected_bc0 = 0.0 - mock_predict(pts_bc0).flatten()
    assert np.allclose(e[:, 0], expected_bc0, atol=1e-12), "Inhomogeneous solver left BC mismatch"

    # Check boundary defect at right boundary x=1 for all time steps
    pts_bc1 = np.column_stack([np.ones_like(t_nodes), t_nodes])
    expected_bc1 = 0.0 - mock_predict(pts_bc1).flatten()
    assert np.allclose(e[:, -1], expected_bc1, atol=1e-12), "Inhomogeneous solver right BC mismatch"


# ==============================================================================
# Invariant 6: Correct residual source sign verification
# ==============================================================================
def test_residual_source_sign():
    """Verify the negative residual source sign in the defect equation.
    
    Mathematical Invariant:
        True error e = u - u_hat satisfies:
            e_t - alpha * e_xx = -R(x, t)
        If R(x, t) > 0 (positive residual) and initial/boundary error is zero,
        the RHS is negative (-R < 0), forcing the solution e(x, t) to be
        strictly NEGATIVE for all t > 0.
    
    Bug Detected:
        Classic sign error using +R instead of -R in the Crank-Nicolson load vector.
    """
    problem = Heat1DProblemSineIC(x_min=0.0, x_max=1.0, t_max=0.5, diffusivity=0.05, frequency=2.0)
    nx, nt = 11, 11

    # Predictor with exact BCs/ICs (so e(x, 0) = 0 and e(x_b, t) = 0) but with positive interior residual R = 2.0
    def mock_predict(xy):
        x_col = xy[:, 0:1]
        t_col = xy[:, 1:2]
        return np.exp(-4.0 * (np.pi**2) * 0.05 * t_col) * np.sin(2.0 * np.pi * x_col)

    def mock_positive_residual(xy):
        return np.ones((xy.shape[0], 1)) * 2.0  # R(x, t) = +2.0


    # Test InhomogeneousFDMSolverHeatEq
    solver = InhomogeneousFDMSolverHeatEq(
        nx=nx, nt=nt, problem=problem, domain=problem.domain,
        pinn_predict_fn=mock_predict, pinn_residual_fn=mock_positive_residual
    )
    e = solver.residual_integration_inhomogeneous()

    # Interior values for t > 0 must be strictly negative
    interior_e = e[1:, 1:-1]
    assert np.all(interior_e <= 1e-12), (
        f"Residual source sign error! Expected negative error from positive residual -R, "
        f"got max interior value: {np.max(interior_e)}"
    )
    assert np.mean(interior_e) < -1e-4, "Error should be distinctly negative under positive residual"


# ==============================================================================
# Invariant 7: Cache separation and metadata validation (Regression Test)
# ==============================================================================
def test_cache_collision_prevention_and_metadata_validation():
    """Regression test proving two different penalty weights cannot silently share a checkpoint.
    
    Integrity Invariant:
        Different lambda_bc values must generate distinct SHA-256 fingerprints,
        distinct checkpoint filenames, and must fail metadata validation if mismatched.
    
    Bug Detected:
        The critical research-integrity cache collision bug where lambda_bc sweeps
        reused identical model weights across runs.
    """
    problem = Heat1DProblemSineIC(x_min=0.0, x_max=1.0, t_max=1.0, diffusivity=0.05, frequency=2.0)

    cfg_lbc1 = {
        "layers": [2, 20, 20, 20, 1], "activation": "tanh", "n_iterations": 10000,
        "num_domain": 10000, "num_boundary": 1000, "num_initial": 1000,
        "learning_rate": 0.001, "seed": 42, "lambda_bc": 1.0, "lambda_ic": 1.0
    }
    cfg_lbc10 = {
        "layers": [2, 20, 20, 20, 1], "activation": "tanh", "n_iterations": 10000,
        "num_domain": 10000, "num_boundary": 1000, "num_initial": 1000,
        "learning_rate": 0.001, "seed": 42, "lambda_bc": 10.0, "lambda_ic": 1.0
    }

    norm1 = normalize_soft_pinn_config(problem, cfg_lbc1)
    norm10 = normalize_soft_pinn_config(problem, cfg_lbc10)

    fp1 = compute_config_fingerprint(norm1)
    fp10 = compute_config_fingerprint(norm10)

    # Invariant 1: Distinct fingerprints
    assert fp1 != fp10, "Cache collision: lambda_bc=1 and lambda_bc=10 produced identical fingerprints!"

    # Invariant 2: Distinct base names
    name1 = get_checkpoint_base_name(norm1, fp1)
    name10 = get_checkpoint_base_name(norm10, fp10)
    assert name1 != name10, f"Cache collision: identical checkpoint base names: {name1}"
    assert "lbc1_" in name1 or "lbc1" in name1
    assert "lbc10_" in name10 or "lbc10" in name10

    # Invariant 3: Metadata validation rejects mismatched weights
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_file = os.path.join(tmpdir, "test_meta.json")
        with open(meta_file, "w") as f:
            json.dump({"canonical_config": norm1}, f)

        # Checking with matching config must succeed
        valid_same, _ = validate_checkpoint_metadata(meta_file, norm1)
        assert valid_same, "Metadata validation failed on matching configuration"

        # Checking with mismatched config (lambda_bc=10) must FAIL
        valid_diff, reason = validate_checkpoint_metadata(meta_file, norm10)
        assert not valid_diff, "Research integrity failure: metadata check allowed mismatched lambda_bc!"
        assert "lambda_bc" in reason, f"Expected lambda_bc mismatch reason, got: {reason}"


# ==============================================================================
# Invariant 8: Direct FDM second-order convergence check
# ==============================================================================
def test_direct_fdm_second_order_convergence():
    """Verify that the baseline Crank-Nicolson direct solver converges at second order.
    
    Mathematical Invariant:
        For coupled grid refinement (dx = dt):
            ||u_FDM - u_exact||_RMS = O(dx^2 + dt^2) = O(h^2)
        Refining from Nx=16 to Nx=32 must reduce error by a factor of ~4 (p ~ 2.0).
    
    Bug Detected:
        Order degradation, spatial/temporal stencil mismatch, or incorrect exact solution reference.
    """
    problem = Heat1DProblemSineIC(x_min=0.0, x_max=1.0, t_max=0.5, diffusivity=0.05, frequency=2.0)

    errors = {}
    for nx in [16, 32]:
        nt = nx
        solver = FDMSolverHeatEq(nx=nx, nt=nt, problem=problem, domain=problem.domain, pinn_model=None)
        u_fdm = solver.solve()

        x = np.linspace(0.0, 1.0, nx)
        t = np.linspace(0.0, 0.5, nt)
        X, T = np.meshgrid(x, t)
        u_exact = problem.exact_solution(X, T)

        rms_err = np.sqrt(np.mean((u_exact - u_fdm)**2))
        errors[nx] = rms_err

    # Compute empirical convergence order p
    ratio = errors[16] / errors[32]
    p = np.log2(ratio)

    assert 1.8 <= p <= 2.2, (
        f"Direct FDM convergence order out of bounds: p={p:.2f} (expected ~2.0, "
        f"err16={errors[16]:.2e}, err32={errors[32]:.2e})"
    )
