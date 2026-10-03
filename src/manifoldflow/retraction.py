from __future__ import annotations

import torch
from torch import Tensor


def qr_retract(Q: Tensor, V: Tensor) -> Tensor:

    Q_new, R = torch.linalg.qr(Q + V, mode="reduced")
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q_new * sign.unsqueeze(-2)


def polar_retract(Q: Tensor, V: Tensor) -> Tensor:

    M = Q + V
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def procrustes_align(A: Tensor) -> Tensor:

    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh
