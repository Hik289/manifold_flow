"""Tangent / normal decomposition on the Stiefel manifold.

For ``Q ∈ St(n, r)`` and any ``G_bar ∈ R^{n×r}`` the orthogonal
decomposition (ManifoldFlow.md §2.2) is

    G_bar = G_tan + Q P     with    P = sym(Q^T G_bar)
    Q^T G_tan + G_tan^T Q = 0
    < G_tan, Q P >_F = 0

These identities are what unit tests #3 and #4 check.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .spd_ops import sym


class TangentNormalSplit(NamedTuple):
    G_tan: Tensor
    P: Tensor
    G_nor: Tensor   # = Q P


def decompose_tangent_normal(Q: Tensor, G_bar: Tensor) -> TangentNormalSplit:
    """Split a Euclidean gradient ``G_bar`` into its tangent + normal parts.

    Args:
        Q:     ``[..., n, r]`` Stiefel matrix (``Q^T Q = I``).
        G_bar: ``[..., n, r]`` Euclidean gradient already multiplied by R = S^{1/2}.

    Returns:
        TangentNormalSplit with ``G_tan``, ``P = sym(Q^T G_bar)``, and ``G_nor = Q P``.
    """
    P = sym(Q.transpose(-1, -2) @ G_bar)
    G_nor = Q @ P
    G_tan = G_bar - G_nor
    return TangentNormalSplit(G_tan=G_tan, P=P, G_nor=G_nor)


def project_tangent(Q: Tensor, V: Tensor) -> Tensor:
    """Re-project a vector ``V`` onto the tangent space at ``Q``.

    Used after the base optimizer's update direction (e.g. Adam moment) to
    guard against numerical drift.
    """
    return V - Q @ sym(Q.transpose(-1, -2) @ V)
