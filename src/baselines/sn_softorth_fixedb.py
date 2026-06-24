"""Spectral Norm / Soft Orthogonality / Fixed-B Stiefel baselines (skeleton).

* Spectral Norm: forward pre-hook power iteration (PyTorch builtin reused).
* Soft Orthogonality: adds ``lambda_reg * ||W^T W - I||_F^2`` to the loss.
* Fixed-B Stiefel: generalized Stiefel ``W^T B W = I`` with **fixed** SPD
  ``B`` (Zhu & Sra 2024). Riemannian SGD on this constraint manifold.

Implementations land during the adjacent-baselines run.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import Tensor


def spectral_norm_wrap(module: torch.nn.Module, n_power_iter: int = 1) -> torch.nn.Module:
    """Wrap a module so that its weight matrices are forced to ``sigma_max ≤ 1``."""
    raise NotImplementedError("Adjacent baselines implemented in RUNNING phase.")


def soft_orthogonality_penalty(weights: Iterable[Tensor], lambda_reg: float = 1e-3) -> Tensor:
    """Sum of ``lambda_reg * ||W^T W - I||_F^2`` over the supplied weights."""
    raise NotImplementedError("Adjacent baselines implemented in RUNNING phase.")


class FixedBStiefelOptimizer(torch.optim.Optimizer):
    """Riemannian SGD on the fixed-B generalized Stiefel manifold."""

    def __init__(
        self,
        params: Iterable,
        B: Tensor,
        lr: float = 1e-2,
        momentum: float = 0.9,
    ) -> None:
        raise NotImplementedError(
            "FixedBStiefelOptimizer is implemented in the RUNNING phase."
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        raise NotImplementedError
