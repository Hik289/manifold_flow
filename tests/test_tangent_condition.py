"""Unit test #3 — Tangent condition: Q^T G_tan + G_tan^T Q ≈ 0."""

import torch

from manifoldflow.tangent import decompose_tangent_normal


def _random_stiefel(n, r, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q


def test_tangent_skew():
    for trial, (n, r) in enumerate([(16, 4), (64, 8), (1024, 64), (512, 256)]):
        Q = _random_stiefel(n, r, seed=trial * 13 + 1)
        G_bar = torch.randn(n, r, dtype=torch.float64) * 3.0
        split = decompose_tangent_normal(Q, G_bar)
        skew_check = Q.T @ split.G_tan + split.G_tan.T @ Q
        err = skew_check.norm().item() / (split.G_tan.norm().item() + 1e-12)
        assert err < 1e-10, (n, r, err)


def test_consistency_with_normal():
    """G_bar = G_tan + Q P should hold exactly."""
    torch.manual_seed(0)
    Q = _random_stiefel(64, 8, seed=5)
    G_bar = torch.randn(64, 8, dtype=torch.float64)
    split = decompose_tangent_normal(Q, G_bar)
    recon = split.G_tan + Q @ split.P
    err = (recon - G_bar).norm().item()
    assert err < 1e-10, err


if __name__ == "__main__":
    test_tangent_skew()
    test_consistency_with_normal()
    print("test_tangent_condition: PASS")
