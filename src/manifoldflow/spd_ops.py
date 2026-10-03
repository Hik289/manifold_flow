from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def sym(A: Tensor) -> Tensor:

    return 0.5 * (A + A.transpose(-1, -2))


def fp32_eigh(S: Tensor) -> Tuple[Tensor, Tensor]:

    orig_dtype = S.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        S_calc = sym(S.to(torch.float32))
    else:
        S_calc = sym(S)
    eigvals, eigvecs = torch.linalg.eigh(S_calc)
    return eigvals.to(orig_dtype), eigvecs.to(orig_dtype)


def spectral_clip(
    S: Tensor,
    lambda_min: float = 0.25,
    lambda_max: float = 4.0,
    floor: float = 1e-6,
) -> Tensor:

    eigvals, eigvecs = fp32_eigh(S)
    eigvals = torch.clamp(eigvals, min=floor)
    eigvals = torch.clamp(eigvals, min=lambda_min, max=lambda_max)
    S_new = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
    return sym(S_new)


def _spectral_fn(S: Tensor, fn) -> Tensor:

    eigvals, eigvecs = fp32_eigh(S)
    eigvals = torch.clamp(eigvals, min=1e-12)
    new_eigvals = fn(eigvals)
    return sym(eigvecs @ torch.diag_embed(new_eigvals) @ eigvecs.transpose(-1, -2))


def matrix_sqrt(S: Tensor) -> Tensor:

    return _spectral_fn(S, torch.sqrt)


def matrix_sqrt_inv(S: Tensor) -> Tensor:

    return _spectral_fn(S, lambda lam: torch.rsqrt(lam))


def symlogm(S: Tensor) -> Tensor:

    return _spectral_fn(S, torch.log)


def symexpm(M: Tensor) -> Tensor:

    M = sym(M)
    eigvals, eigvecs = fp32_eigh(M)
    new_eigvals = torch.exp(eigvals)
    return sym(eigvecs @ torch.diag_embed(new_eigvals) @ eigvecs.transpose(-1, -2))


def affine_invariant_step(S: Tensor, H: Tensor, gamma: float) -> Tensor:

    R = matrix_sqrt(S)
    R_inv = matrix_sqrt_inv(S)
    M = -gamma * sym(R_inv @ sym(H) @ R_inv)
    return sym(R @ symexpm(M) @ R)
