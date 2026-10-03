from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


def newton_schulz_5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    if X.size(-2) > X.size(-1):
        X = X.transpose(-1, -2)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


class IntrinsicMuonOptimizer(Optimizer):


    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-2,
        momentum: float = 0.95,
        ns_steps: int = 5,
        log_pressure: bool = False,
    ) -> None:
        raise NotImplementedError(
            "IntrinsicMuonOptimizer is implemented in the RUNNING phase."
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        raise NotImplementedError
