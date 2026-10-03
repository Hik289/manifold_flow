from __future__ import annotations

from typing import NamedTuple

from torch import Tensor

from .spd_ops import sym


class TangentNormalSplit(NamedTuple):
    G_tan: Tensor
    P: Tensor
    G_nor: Tensor


def decompose_tangent_normal(Q: Tensor, G_bar: Tensor) -> TangentNormalSplit:

    P = sym(Q.transpose(-1, -2) @ G_bar)
    G_nor = Q @ P
    G_tan = G_bar - G_nor
    return TangentNormalSplit(G_tan=G_tan, P=P, G_nor=G_nor)


def project_tangent(Q: Tensor, V: Tensor) -> Tensor:

    return V - Q @ sym(Q.transpose(-1, -2) @ V)
