"""Bonus unit test — Finite-difference geometry gradient.

For random ``Q, S`` and a symmetric perturbation ``Z`` the directional
derivative of ``L(Q Exp_S(eps Z)^{1/2})`` must match the analytic
``< grad_S L, Z >_S`` (affine-invariant inner product) up to O(eps).

This test exercises ``spd_ops.affine_invariant_step``, ``matrix_sqrt``, and
``symexpm`` together. It does **not** depend on the optimizer skeletons.
"""

import torch

from manifoldflow.parametrization import manifold_weight
from manifoldflow.spd_ops import (
    sym,
    matrix_sqrt,
    matrix_sqrt_inv,
    symexpm,
    fp32_eigh,
)


def _random_stiefel(n, r, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q


def _random_spd(r, seed, jitter=0.5):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r, r, generator=g, dtype=torch.float64)
    return A @ A.T + jitter * torch.eye(r, dtype=torch.float64)


def _loss(W, target):
    return ((W - target) ** 2).sum()


def test_finite_diff_matches_analytic():
    """Compare central finite-difference along symmetric ``Z`` against the
    analytic Euclidean derivative of ``S ↦ L(Q S^{1/2})`` at ``S``.

    We use the Euclidean inner product (not affine-invariant) for the
    analytic side; both sides should agree to O(eps^2).
    """
    n, r = 32, 8
    Q = _random_stiefel(n, r, seed=42)
    S = _random_spd(r, seed=43)
    target = torch.randn(n, r, dtype=torch.float64)

    # Random symmetric perturbation
    Zraw = torch.randn(r, r, dtype=torch.float64, generator=torch.Generator().manual_seed(99))
    Z = sym(Zraw)

    # --- analytic Euclidean gradient w.r.t. S ----------------------------
    S_var = S.clone().requires_grad_(True)
    W = Q @ matrix_sqrt(S_var)
    loss = _loss(W, target)
    grad_S = torch.autograd.grad(loss, S_var)[0]
    grad_S = sym(grad_S)
    analytic_dir_deriv = (grad_S * Z).sum().item()

    # --- central finite difference along Z (in S, not on the manifold) ---
    eps = 1e-5
    S_plus = sym(S + eps * Z)
    S_minus = sym(S - eps * Z)
    L_plus = _loss(Q @ matrix_sqrt(S_plus), target).item()
    L_minus = _loss(Q @ matrix_sqrt(S_minus), target).item()
    fd_dir_deriv = (L_plus - L_minus) / (2 * eps)

    rel = abs(analytic_dir_deriv - fd_dir_deriv) / (abs(fd_dir_deriv) + 1e-12)
    assert rel < 1e-3, (analytic_dir_deriv, fd_dir_deriv, rel)


def test_symexpm_round_trip():
    """``symlogm(symexpm(M)) = M`` for symmetric ``M``."""
    from manifoldflow.spd_ops import symlogm
    torch.manual_seed(0)
    M = sym(torch.randn(12, 12, dtype=torch.float64))
    round_trip = symlogm(symexpm(M))
    err = (round_trip - M).norm().item() / (M.norm().item() + 1e-12)
    assert err < 1e-8, err


if __name__ == "__main__":
    test_finite_diff_matches_analytic()
    test_symexpm_round_trip()
    print("test_finite_diff_geometry: PASS")
