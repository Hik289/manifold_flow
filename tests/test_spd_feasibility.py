"""Unit test #2 — SPD feasibility (λ_min(S_t) > 0)."""

import torch

from manifoldflow.spd_ops import sym, spectral_clip, affine_invariant_step, fp32_eigh


def test_spectral_clip_floor():
    torch.manual_seed(0)
    A = torch.randn(8, 8, dtype=torch.float64)
    S = sym(A @ A.T) + 1e-4 * torch.eye(8, dtype=torch.float64)
    S_clipped = spectral_clip(S, lambda_min=0.25, lambda_max=4.0)
    eigvals, _ = fp32_eigh(S_clipped)
    assert eigvals.min().item() >= 0.25 - 1e-5
    assert eigvals.max().item() <= 4.0 + 1e-5


def test_random_affine_invariant_walks_stay_spd():
    torch.manual_seed(1)
    S = torch.eye(16, dtype=torch.float64)
    for step in range(50):
        H = sym(torch.randn(16, 16, dtype=torch.float64))
        S_raw = affine_invariant_step(S, H, gamma=0.05)
        S = spectral_clip(S_raw, lambda_min=0.25, lambda_max=4.0)
        eigvals, _ = fp32_eigh(S)
        assert eigvals.min().item() > 0.0, (step, eigvals.min().item())
        assert eigvals.min().item() >= 0.25 - 1e-5, (step, eigvals.min().item())


def test_extreme_H_still_clipped():
    """Even with huge H the post-clip eigenvalues must stay in [0.25, 4]."""
    torch.manual_seed(2)
    S = torch.eye(8, dtype=torch.float64)
    H = sym(10.0 * torch.randn(8, 8, dtype=torch.float64))
    S_raw = affine_invariant_step(S, H, gamma=0.5)
    S_clipped = spectral_clip(S_raw, lambda_min=0.25, lambda_max=4.0)
    eigvals, _ = fp32_eigh(S_clipped)
    assert eigvals.min().item() >= 0.25 - 1e-5
    assert eigvals.max().item() <= 4.0 + 1e-5


if __name__ == "__main__":
    test_spectral_clip_floor()
    test_random_affine_invariant_walks_stay_spd()
    test_extreme_H_still_clipped()
    print("test_spd_feasibility: PASS")
