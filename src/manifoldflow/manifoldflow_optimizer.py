"""ManifoldFlow optimizer.

The optimizer shares the Stiefel tangent step with the Fixed-Stiefel baseline
and adds an affine-invariant SPD update for the learnable spectrum. Setting
``rho_geo=0`` recovers the frozen-spectrum Fixed-Stiefel trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from .spd_ops import sym, symlogm, affine_invariant_step, spectral_clip, fp32_eigh
from .retraction import qr_retract, procrustes_align
from .tangent import decompose_tangent_normal, project_tangent

BaseOptim = Literal["sgd", "adam", "shampoo", "muon"]


@dataclass
class ManifoldFlowConfig:
    """Hyper-parameters for the ManifoldFlow geometry mechanism."""
    rho_geo: float = 1e-2
    beta_P: float = 0.95
    lambda_S: float = 1e-3
    K_geo: int = 10
    tau_c: float = 0.1
    tau_r: float = 0.0
    alpha_c: float = 5.0
    alpha_r: float = 2.0
    lambda_min: float = 0.25
    lambda_max: float = 4.0
    warmup_frac: float = 0.05


def _stiefel_sgd_step(Q, G_tan, state, lr, momentum):
    """Shared tangent-SGD retraction step. Used by both optimizers."""
    dev = Q.device
    if momentum > 0.0:
        V = state.get("V")
        if V is None:
            V = torch.zeros_like(G_tan)
        else:
            V = V.to(dev)  # DEVICE FIX: ensure V is on correct device
        D = momentum * V + G_tan
        D = project_tangent(Q, D)
        state["V"] = D
    else:
        D = G_tan
    Q_new = qr_retract(Q, -lr * D)
    if momentum > 0.0 and "V" in state:
        state["V"] = project_tangent(Q_new, state["V"])
    return Q_new


class ManifoldFlowOptimizer(Optimizer):
    """ManifoldFlow optimizer with online SPD geometry learning.

    Each parameter must be a Q tensor (Stiefel factor). S is in optimizer state.
    When rho_geo=0, trajectory is identical to FixedStiefelOptimizer (test #6).
    """

    def __init__(
        self,
        params,
        base_optim="sgd",
        lr=1e-2,
        momentum=0.0,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        mf_config=None,
        total_steps=None,
        log_pressure=True,
    ):
        if mf_config is None:
            mf_config = ManifoldFlowConfig()
        defaults = dict(lr=lr, momentum=momentum, betas=betas,
                        weight_decay=weight_decay, base_optim=base_optim)
        super().__init__(params, defaults)
        self.mf_config = mf_config
        self.total_steps = total_steps
        self.log_pressure = log_pressure
        self._pressure_log = {}

    def _warmup_steps(self):
        if self.total_steps is None:
            return 0
        return int(math.ceil(self.mf_config.warmup_frac * self.total_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        cfg = self.mf_config
        eps = 1e-8

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            gamma_t = cfg.rho_geo * lr

            for Q in group["params"]:
                if Q.grad is None:
                    continue

                dev = Q.device  # canonical device for this step
                G_bar = Q.grad.to(Q.dtype)
                state = self.state[Q]

                if len(state) == 0:
                    state["step"] = 0
                    r = Q.shape[-1]
                    state["S"] = torch.eye(r, dtype=Q.dtype, device=dev)
                    state["M_P"] = torch.zeros(r, r, dtype=Q.dtype, device=dev)
                    state["Q_prev"] = Q.clone()

                t = state["step"]
                # DEVICE FIX: force state tensors to Q.device at each step
                S = state["S"].to(dev)
                M_P = state["M_P"].to(dev)
                Q_prev = state["Q_prev"].to(dev)

                split = decompose_tangent_normal(Q, G_bar)
                G_tan = split.G_tan
                P_t = split.P

                Q_new = _stiefel_sgd_step(Q, G_tan, state, lr, momentum)

                if t > 0:
                    A = Q.T @ Q_prev
                    O_t = procrustes_align(A)
                    M_P_aligned = O_t @ M_P @ O_t.T
                else:
                    M_P_aligned = M_P

                G_bar_norm = G_bar.norm() + eps
                P_normalized = P_t / G_bar_norm
                M_P_prev = M_P_aligned.clone()
                M_P_new = cfg.beta_P * M_P_aligned + (1.0 - cfg.beta_P) * P_normalized
                state["M_P"] = M_P_new

                warmup_done = t >= self._warmup_steps()
                do_geo_update = (gamma_t > 0.0) and warmup_done and ((t % cfg.K_geo) == 0)

                if do_geo_update:
                    P_norm = P_t.norm() + eps
                    M_prev_norm = M_P_prev.norm() + eps
                    c_t = (P_t * M_P_prev).sum() / (P_norm * M_prev_norm)
                    G_nor_norm = (Q @ P_t).norm() + eps
                    G_tan_norm = G_tan.norm() + eps
                    r_t = G_nor_norm / G_tan_norm
                    log_r_t = torch.log(r_t)
                    a_t_c = torch.sigmoid(torch.tensor(
                        cfg.alpha_c * (c_t.item() - cfg.tau_c), dtype=Q.dtype, device=dev))
                    a_t_r = torch.sigmoid(cfg.alpha_r * (log_r_t - cfg.tau_r))
                    a_t = (a_t_c * a_t_r).item()
                    H_t = sym(M_P_new) + cfg.lambda_S * symlogm(S)
                    S_raw = affine_invariant_step(S, H_t, gamma_t * a_t)
                    state["S"] = spectral_clip(S_raw, cfg.lambda_min, cfg.lambda_max)

                Q.data.copy_(Q_new)
                state["Q_prev"] = Q_new.clone()
                state["step"] = t + 1

                if self.log_pressure:
                    eigvals, _ = fp32_eigh(state["S"])
                    self._pressure_log[id(Q)] = {
                        "P_norm": P_t.norm().item(),
                        "grad_tan_norm": G_tan.norm().item(),
                        "grad_nor_norm": (Q @ P_t).norm().item(),
                        "lambda_min": eigvals.min().item(),
                        "lambda_max": eigvals.max().item(),
                        "step": t,
                    }

        return loss

    def get_pressure_log(self):
        return self._pressure_log

    def get_S(self, Q_param):
        return self.state[Q_param]["S"]
