from __future__ import annotations

from torch import Tensor

from .spd_ops import matrix_sqrt, sym


def manifold_weight(Q: Tensor, S: Tensor) -> Tensor:

    return Q @ matrix_sqrt(sym(S))


def manifold_weight_cotiefel(Q: Tensor, S: Tensor) -> Tensor:

    Wt = Q @ matrix_sqrt(sym(S))
    return Wt.transpose(-1, -2)
