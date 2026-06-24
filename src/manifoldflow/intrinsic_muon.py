"""Intrinsic Muon: Riemannian Muon on the Stiefel manifold (skeleton).

Reference: arXiv:2605.09238 (2026). The tangent gradient is orthogonalised
through Newton–Schulz iterations *before* retraction. No ``S_t`` is learned.
Used as the head-to-head competitor for H0.lit_1 / H0.method_5.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


def newton_schulz_5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """5-iteration Newton–Schulz orthogonalisation of an n×r matrix.

    Returns an approximation to the polar factor ``U`` from ``G = U Σ V^T``
    (i.e., a matrix with all singular values ≈ 1). Pure function, used by
    both Intrinsic Muon and ManifoldFlow-Muon.

    Coefficients follow Keller Jordan's standard implementation.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    if X.size(-2) > X.size(-1):
        X = X.transpose(-1, -2)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


class IntrinsicMuonOptimizer(Optimizer):
    """Stiefel-constrained Riemannian Muon (skeleton).

    Step: tangent grad → NS5 orthogonalisation → tangent projection →
    QR retraction. No ``S_t`` learning.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-2,
        momentum: float = 0.95,
        ns_steps: int = 5,
        log_pressure: bool = False,
    ) -> None:
        raise NotImplementedError(
            "IntrinsicMuonOptimizer is implemented in the RUNNING phase."
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        raise NotImplementedError
