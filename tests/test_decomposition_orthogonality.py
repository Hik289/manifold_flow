"""Unit test #4 — Decomposition orthogonality: <G_tan, Q P>_F ≈ 0."""

import torch

from manifoldflow.tangent import decompose_tangent_normal


def _random_stiefel(n, r, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q


def test_orthogonal_decomposition():
    for trial, (n, r) in enumerate([(16, 4), (64, 8), (1024, 64), (256, 256)]):
        Q = _random_stiefel(n, r, seed=trial * 7 + 3)
        G_bar = torch.randn(n, r, dtype=torch.float64) * 5.0
        split = decompose_tangent_normal(Q, G_bar)
        ip = (split.G_tan * (Q @ split.P)).sum().item()
        norms = split.G_tan.norm().item() * (Q @ split.P).norm().item() + 1e-12
        rel = abs(ip) / norms
        assert rel < 1e-10, (n, r, rel, ip)


if __name__ == "__main__":
    test_orthogonal_decomposition()
    print("test_decomposition_orthogonality: PASS")
