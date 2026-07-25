"""Fixed-Stiefel paired baseline.

S_t == I enforced. Shares _stiefel_sgd_step with ManifoldFlowOptimizer to
guarantee numerical equivalence under rho_geo=0 (unit test #6).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.optim import Optimizer

from .tangent import decompose_tangent_normal
from .manifoldflow_optimizer import _stiefel_adam_step, _stiefel_sgd_step

BaseOptim = Literal["sgd", "adam"]


class FixedStiefelOptimizer(Optimizer):
    """Fixed-Stiefel SGD (S=I). Shares tangent step with ManifoldFlowOptimizer."""

    def __init__(
        self,
        params,
        base_optim="sgd",
        lr=1e-2,
        momentum=0.0,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        log_pressure=False,
    ):
        if base_optim not in {"sgd", "adam"}:
            raise ValueError(f"unsupported base optimizer: {base_optim}")
        defaults = dict(lr=lr, momentum=momentum, betas=betas,
                        weight_decay=weight_decay, base_optim=base_optim)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            base_optim = group["base_optim"]

            for Q in group["params"]:
                if Q.grad is None:
                    continue

                G_bar = Q.grad.to(Q.dtype)
                state = self.state[Q]
                if len(state) == 0:
                    state["step"] = 0

                split = decompose_tangent_normal(Q, G_bar)
                G_tan = split.G_tan

                if base_optim == "adam":
                    Q_new = _stiefel_adam_step(
                        Q,
                        G_tan,
                        state,
                        lr,
                        group["betas"],
                    )
                else:
                    Q_new = _stiefel_sgd_step(Q, G_tan, state, lr, momentum)

                Q.data.copy_(Q_new)
                state["step"] += 1

        return loss
