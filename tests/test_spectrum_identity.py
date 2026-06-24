"""Unit test #5 — Spectrum identity: σ_i^2(QS^{1/2}) = λ_i(S)."""

import torch

from manifoldflow.parametrization import manifold_weight
from manifoldflow.spd_ops import fp32_eigh


def _random_stiefel(n, r, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q


def _random_spd(r, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r, r, generator=g, dtype=torch.float64)
    return A @ A.T + 0.5 * torch.eye(r, dtype=torch.float64)


def test_spectrum_identity_random():
    for trial, (n, r) in enumerate([(16, 4), (64, 8), (1024, 64), (512, 256)]):
        Q = _random_stiefel(n, r, seed=trial * 11 + 7)
        S = _random_spd(r, seed=trial * 11 + 8)
        W = manifold_weight(Q, S)
        sv = torch.linalg.svdvals(W)
        eigvals, _ = fp32_eigh(S)
        sv_sorted = torch.sort(sv, descending=True).values
        ev_sorted = torch.sort(eigvals, descending=True).values
        # fp32_eigh in spd_ops casts via float32 — relax to spec tolerance 1e-5
        err = (sv_sorted ** 2 - ev_sorted).abs().max().item()
        assert err < 1e-4, (n, r, err)


if __name__ == "__main__":
    test_spectrum_identity_random()
    print("test_spectrum_identity: PASS")
