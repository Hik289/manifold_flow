"""Weight parametrizations for Stiefel and SPD-relaxed Stiefel layers."""

from __future__ import annotations

import torch
from torch import Tensor

from .spd_ops import matrix_sqrt, sym


def manifold_weight(Q: Tensor, S: Tensor) -> Tensor:
    """Return ``W = Q @ S^{1/2}`` with ``S^{1/2}`` computed via eigh.

    Caller is responsible for ensuring ``Q^T Q = I`` and ``S`` is SPD.
    """
    return Q @ matrix_sqrt(sym(S))


def manifold_weight_cotiefel(Q: Tensor, S: Tensor) -> Tensor:
    """Co-Stiefel variant for *wide* layers (``n < r``): ``W^T = Q @ S^{1/2}``
    with ``Q ∈ R^{r × n}, Q Q^T = I``, ``S ∈ R^{n × n}``.

    The returned tensor has the original layer shape ``(n, r)``.
    """
    Wt = Q @ matrix_sqrt(sym(S))
    return Wt.transpose(-1, -2)
