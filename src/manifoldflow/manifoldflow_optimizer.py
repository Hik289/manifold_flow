from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch.optim import Optimizer

from .spd_ops import sym, symlogm, matrix_sqrt, affine_invariant_step, spectral_clip, fp32_eigh
from .retraction import qr_retract, procrustes_align
from .tangent import decompose_tangent_normal, project_tangent

BaseOptim = Literal["sgd", "adam"]


@dataclass
class ManifoldFlowConfig:

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
    use_gate: bool = True
    pressure_mode: str = "gradient"


def _stiefel_sgd_step(Q, G_tan, state, lr, momentum):

    dev = Q.device
    if momentum > 0.0:
        V = state.get("V")
        if V is None:
            V = torch.zeros_like(G_tan)
        else:
            V = V.to(dev)
        D = momentum * V + G_tan
        D = project_tangent(Q, D)
        state["V"] = D
    else:
        D = G_tan
    Q_new = qr_retract(Q, -lr * D)
    if momentum > 0.0 and "V" in state:
        state["V"] = project_tangent(Q_new, state["V"])
    return Q_new


def _stiefel_adam_step(Q, G_tan, state, lr, betas, eps=1e-8):

    beta1, beta2 = betas
    first_moment = state.get("adam_m")
    second_moment = state.get("adam_v")
    adam_step = state.get("adam_step", 0) + 1
    if first_moment is None:
        first_moment = torch.zeros_like(G_tan)
        second_moment = torch.zeros_like(G_tan)
    else:
        first_moment = first_moment.to(Q.device)
        second_moment = second_moment.to(Q.device)

    first_moment = beta1 * first_moment + (1.0 - beta1) * G_tan
    second_moment = beta2 * second_moment + (1.0 - beta2) * G_tan.square()
    first_unbiased = first_moment / (1.0 - beta1**adam_step)
    second_unbiased = second_moment / (1.0 - beta2**adam_step)
    direction = project_tangent(
        Q,
        first_unbiased / (second_unbiased.clamp_min(0.0).sqrt() + eps),
    )
    Q_new = qr_retract(Q, -lr * direction)

    state["adam_m"] = project_tangent(Q_new, first_moment)
    state["adam_v"] = second_moment
    state["adam_step"] = adam_step
    return Q_new


class ManifoldFlowOptimizer(Optimizer):


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
        if mf_config.K_geo < 1:
            raise ValueError("K_geo must be positive")
        if not 0.0 < mf_config.lambda_min <= 1.0 <= mf_config.lambda_max:
            raise ValueError("spectral bounds must contain the identity")
        if not 0.0 <= mf_config.beta_P < 1.0:
            raise ValueError("beta_P must be in [0, 1)")
        if not 0.0 <= mf_config.warmup_frac <= 1.0:
            raise ValueError("warmup_frac must be in [0, 1]")
        if mf_config.rho_geo < 0.0 or mf_config.lambda_S < 0.0:
            raise ValueError("geometry rate and regularization must be nonnegative")
        if mf_config.pressure_mode not in {"gradient", "random"}:
            raise ValueError("unsupported pressure mode")
        if base_optim not in {"sgd", "adam"}:
            raise ValueError(f"unsupported base optimizer: {base_optim}")
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
            base_optim = group["base_optim"]
            gamma_t = cfg.rho_geo * lr

            for Q in group["params"]:
                if Q.grad is None:
                    continue

                dev = Q.device
                G_bar = Q.grad.to(Q.dtype)
                state = self.state[Q]

                if len(state) == 0:
                    state["step"] = 0
                    r = Q.shape[-1]
                    state["S"] = torch.eye(r, dtype=Q.dtype, device=dev)
                    state["M_P"] = torch.zeros(r, r, dtype=Q.dtype, device=dev)
                    state["Q_prev"] = Q.clone()

                t = state["step"]
                S = state["S"].to(dev)
                M_P = state["M_P"].to(dev)
                Q_prev = state["Q_prev"].to(dev)

                split = decompose_tangent_normal(Q, G_bar)
                G_tan = split.G_tan
                P_t = split.P
                if cfg.pressure_mode == "random":
                    random_pressure = sym(torch.randn_like(P_t))
                    P_t = random_pressure * (P_t.norm() / random_pressure.norm().clamp_min(eps))

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
                    P_norm = P_t.norm()
                    M_prev_norm = M_P_prev.norm()
                    c_t = (P_t * M_P_prev).sum() / (P_norm * M_prev_norm + eps)
                    G_nor_norm = (Q @ P_t).norm()
                    G_tan_norm = G_tan.norm() + eps
                    r_t = G_nor_norm / G_tan_norm
                    if cfg.use_gate:
                        log_r_t = torch.log(r_t.clamp_min(eps))
                        a_t_c = torch.sigmoid(cfg.alpha_c * (c_t - cfg.tau_c))
                        a_t_r = torch.sigmoid(cfg.alpha_r * (log_r_t - cfg.tau_r))
                        eigenvalues, _ = fp32_eigh(S)
                        spectral_damping = (eigenvalues.min() / eigenvalues.max()).clamp(max=1.0)
                        a_t = (a_t_c * a_t_r * spectral_damping).item()
                    else:
                        a_t = 1.0
                    R = matrix_sqrt(S)
                    H_t = sym(M_P_new) + cfg.lambda_S * (R @ symlogm(S) @ R)
                    S_raw = affine_invariant_step(S, H_t, gamma_t * a_t)
                    state["S"] = spectral_clip(S_raw, cfg.lambda_min, cfg.lambda_max)

                state["Q_prev"] = Q.clone()
                Q.data.copy_(Q_new)
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
