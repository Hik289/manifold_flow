from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import Tensor


def spectral_norm_wrap(module: torch.nn.Module, n_power_iter: int = 1) -> torch.nn.Module:

    raise NotImplementedError("Adjacent baselines implemented in RUNNING phase.")


def soft_orthogonality_penalty(weights: Iterable[Tensor], lambda_reg: float = 1e-3) -> Tensor:

    raise NotImplementedError("Adjacent baselines implemented in RUNNING phase.")


class FixedBStiefelOptimizer(torch.optim.Optimizer):


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
