"""Unit test #1 — Stiefel feasibility.

‖Q^T Q − I‖_F < 1e-5 after random retractions.
"""

import torch

from manifoldflow.retraction import qr_retract, polar_retract


def _random_stiefel(n: int, r: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q


def test_initial_qr_is_stiefel():
    for (n, r) in [(16, 4), (32, 32), (4608, 512)]:
        Q = _random_stiefel(n, r, seed=n * 7 + r)
        err = (Q.T @ Q - torch.eye(r, dtype=Q.dtype)).norm().item()
        assert err < 1e-10, (n, r, err)


def test_qr_retract_preserves_stiefel():
    torch.manual_seed(0)
    Q = _random_stiefel(64, 8)
    for step in range(200):
        V = 0.01 * torch.randn_like(Q)
        Q = qr_retract(Q, V)
        err = (Q.T @ Q - torch.eye(8, dtype=Q.dtype)).norm().item()
        assert err < 1e-5, (step, err)


def test_polar_retract_preserves_stiefel():
    torch.manual_seed(1)
    Q = _random_stiefel(64, 8)
    for step in range(200):
        V = 0.01 * torch.randn_like(Q)
        Q = polar_retract(Q, V)
        err = (Q.T @ Q - torch.eye(8, dtype=Q.dtype)).norm().item()
        assert err < 1e-5, (step, err)


if __name__ == "__main__":
    test_initial_qr_is_stiefel()
    test_qr_retract_preserves_stiefel()
    test_polar_retract_preserves_stiefel()
    print("test_stiefel_feasibility: PASS")
