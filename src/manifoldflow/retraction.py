"""Retractions on the Stiefel manifold and basis-rotation alignment."""

from __future__ import annotations

import torch
from torch import Tensor


def qr_retract(Q: Tensor, V: Tensor) -> Tensor:
    """QR-based retraction: R(Q, V) = qr(Q + V).

    Both Fixed-Stiefel and ManifoldFlow share this retraction so that
    ``test_frozen_equivalence`` can compare trajectories at machine
    precision when ``gamma=0``.
    """
    Q_new, R = torch.linalg.qr(Q + V, mode="reduced")
    # Make the sign of Q_new diagonal of R positive for uniqueness
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q_new * sign.unsqueeze(-2)


def polar_retract(Q: Tensor, V: Tensor) -> Tensor:
    """Polar retraction via SVD: R(Q, V) = (Q+V) (I + V^T V)^{-1/2}.

    Implemented through SVD for numerical stability; used as a backup in
    unit tests where QR sign conventions matter.
    """
    M = Q + V
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def procrustes_align(A: Tensor) -> Tensor:
    """Closest orthogonal matrix to ``A`` via SVD: O = U V^T.

    Used to transport pressure EMA state when the basis ``Q`` changes
    between iterations (ManifoldFlow.md §2.3).
    """
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh
