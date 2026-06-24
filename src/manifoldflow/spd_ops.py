"""Symmetric / SPD numerical primitives.

All eigendecompositions are forced to fp32 even when the rest of the network
runs in mixed precision (see exp_design.md §3.4). After any update to an
SPD candidate we symmetrise and clamp eigenvalues to ``lambda_min``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Basic primitives
# ---------------------------------------------------------------------------

def sym(A: Tensor) -> Tensor:
    """Symmetric part: (A + A^T) / 2 (last two dims)."""
    return 0.5 * (A + A.transpose(-1, -2))


def fp32_eigh(S: Tensor) -> Tuple[Tensor, Tensor]:
    """Symmetric eigendecomposition, forced to at-least float32 precision.

    Behavior:
      * If ``S`` is float16 / bfloat16: cast to float32, run eigh, cast back.
      * If ``S`` is float32: run eigh in float32.
      * If ``S`` is float64: run eigh in float64 (no down-cast — we keep the
        higher precision for unit tests / theoretical checks).

    Returns ``(eigvals, eigvecs)`` in the original dtype/device.
    """
    orig_dtype = S.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        S_calc = sym(S.to(torch.float32))
    else:
        S_calc = sym(S)
    eigvals, eigvecs = torch.linalg.eigh(S_calc)
    return eigvals.to(orig_dtype), eigvecs.to(orig_dtype)


# ---------------------------------------------------------------------------
# Spectral clipping (essential for SPD invariants)
# ---------------------------------------------------------------------------

def spectral_clip(
    S: Tensor,
    lambda_min: float = 0.25,
    lambda_max: float = 4.0,
    floor: float = 1e-6,
) -> Tensor:
    """Clip eigenvalues to ``[lambda_min, lambda_max]``.

    First we floor at ``floor`` to guarantee strict positive-definiteness
    even if numerical noise drove some eigenvalue below zero, then we apply
    the user range.
    """
    eigvals, eigvecs = fp32_eigh(S)
    eigvals = torch.clamp(eigvals, min=floor)
    eigvals = torch.clamp(eigvals, min=lambda_min, max=lambda_max)
    S_new = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
    return sym(S_new)


# ---------------------------------------------------------------------------
# Functional calculus on SPD
# ---------------------------------------------------------------------------

def _spectral_fn(S: Tensor, fn) -> Tensor:
    """Apply a scalar function ``fn`` to the eigenvalues of an SPD matrix."""
    eigvals, eigvecs = fp32_eigh(S)
    eigvals = torch.clamp(eigvals, min=1e-12)
    new_eigvals = fn(eigvals)
    return sym(eigvecs @ torch.diag_embed(new_eigvals) @ eigvecs.transpose(-1, -2))


def matrix_sqrt(S: Tensor) -> Tensor:
    """Principal square root of an SPD matrix via eigendecomposition."""
    return _spectral_fn(S, torch.sqrt)


def matrix_sqrt_inv(S: Tensor) -> Tensor:
    """Inverse principal square root."""
    return _spectral_fn(S, lambda lam: torch.rsqrt(lam))


def symlogm(S: Tensor) -> Tensor:
    """Symmetric log of an SPD matrix."""
    return _spectral_fn(S, torch.log)


def symexpm(M: Tensor) -> Tensor:
    """Symmetric matrix exponential of a symmetric ``M``."""
    M = sym(M)
    eigvals, eigvecs = fp32_eigh(M)
    new_eigvals = torch.exp(eigvals)
    return sym(eigvecs @ torch.diag_embed(new_eigvals) @ eigvecs.transpose(-1, -2))


def affine_invariant_step(S: Tensor, H: Tensor, gamma: float) -> Tensor:
    """One affine-invariant SPD step:  S_{t+1} = S^{1/2} exp(-gamma S^{-1/2} H S^{-1/2}) S^{1/2}.

    Implements eq. (S-update) from ManifoldFlow.md §2.5.
    """
    R = matrix_sqrt(S)
    R_inv = matrix_sqrt_inv(S)
    M = -gamma * sym(R_inv @ sym(H) @ R_inv)
    return sym(R @ symexpm(M) @ R)
