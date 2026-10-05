from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from manifoldflow.fixed_stiefel import FixedStiefelOptimizer
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig, ManifoldFlowOptimizer
from manifoldflow.retraction import procrustes_align
from manifoldflow.spd_ops import matrix_sqrt, sym
from manifoldflow.tangent import decompose_tangent_normal


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return device


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False, default=_json_default)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class PlateauStopper:
    def __init__(self, min_epochs, patience, min_delta=0.0, mode="min"):
        if min_epochs < 1 or patience < 1 or min_delta < 0.0 or mode not in {"min", "max"}:
            raise ValueError("invalid stopping configuration")
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0
        self._plateau_value = self.best_value
        self._plateau_epoch = 0

    def update(self, value, epoch):
        if not math.isfinite(value):
            raise FloatingPointError("validation metric is nonfinite")
        if epoch < 1:
            raise ValueError("epochs are one-based")
        better = value < self.best_value if self.mode == "min" else value > self.best_value
        if better:
            self.best_value = value
            self.best_epoch = epoch
        gain = self._plateau_value - value if self.mode == "min" else value - self._plateau_value
        if gain > self.min_delta:
            self._plateau_value = value
            self._plateau_epoch = epoch
        return epoch >= self.min_epochs and epoch - self._plateau_epoch >= self.patience


class SpectrumLinear(nn.Module):
    def __init__(self, in_features, out_features, mode, alpha=0.25, beta=4.0, bias=True):
        super().__init__()
        if mode not in {"dense", "fs", "scalar", "diagonal", "mf"}:
            raise ValueError(f"unsupported spectrum mode: {mode}")
        if min(in_features, out_features) < 1 or not 0.0 < alpha <= 1.0 <= beta:
            raise ValueError("dimensions must be positive and spectral bounds must contain one")
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.transpose = out_features < in_features
        self.p = max(in_features, out_features)
        self.r = min(in_features, out_features)
        basis, triangular = torch.linalg.qr(torch.randn(self.p, self.r), mode="reduced")
        signs = torch.diagonal(triangular).sign()
        basis = basis * torch.where(signs == 0.0, torch.ones_like(signs), signs)
        if mode == "dense":
            self.register_parameter("Q", None)
            self.weight = nn.Parameter(basis.T.contiguous() if self.transpose else basis)
        else:
            self.Q = nn.Parameter(basis)
            self.register_parameter("weight", None)
        if mode in {"scalar", "diagonal"}:
            self.scale = nn.Parameter(torch.ones(1 if mode == "scalar" else self.r))
        else:
            self.register_parameter("scale", None)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("gram_S", torch.eye(self.r) if mode == "mf" else None)
        self.register_buffer("sqrt_S", torch.eye(self.r) if mode == "mf" else None)

    def effective_weight(self):
        if self.mode == "dense":
            return self.weight
        if self.mode == "mf":
            matrix = self.Q @ self.sqrt_S
        elif self.scale is not None:
            matrix = self.Q * self.scale
        else:
            matrix = self.Q
        return matrix.T if self.transpose else matrix

    def forward(self, inputs):
        return F.linear(inputs, self.effective_weight(), self.bias)

    @torch.no_grad()
    def set_gram(self, gram):
        if self.mode != "mf":
            raise ValueError("only MF has a full SPD Gram factor")
        self.gram_S = sym(gram).detach().clone()
        self.sqrt_S = matrix_sqrt(self.gram_S).detach()

    @torch.no_grad()
    def project_scale(self):
        if self.scale is not None:
            self.scale.clamp_(math.sqrt(self.alpha), math.sqrt(self.beta))


class OptimizerBundle:
    def __init__(self, model, ordinary, constrained, layers, collect_diagnostics):
        self.model = model
        self.ordinary = ordinary
        self.constrained = constrained
        self.layers = layers
        self.collect_diagnostics = collect_diagnostics
        self._previous_pressure = {}
        self._last_metrics = {}

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        if self.collect_diagnostics:
            for name, layer in self.layers:
                if layer.Q is None or layer.Q.grad is None:
                    continue
                split = decompose_tangent_normal(layer.Q, layer.Q.grad)
                pressure = split.P
                cosine = None
                previous = self._previous_pressure.get(name)
                if previous is not None:
                    basis, old_pressure = previous
                    rotation = procrustes_align(layer.Q.T @ basis)
                    aligned = rotation @ old_pressure @ rotation.T
                    cosine = float((pressure * aligned).sum() / (pressure.norm() * aligned.norm() + 1e-8))
                self._last_metrics[name] = {
                    "pressure_norm": float(pressure.norm()),
                    "tangent_gradient_norm": float(split.G_tan.norm()),
                    "normal_gradient_norm": float((layer.Q @ pressure).norm()),
                    "aligned_pressure_cosine": cosine,
                }
                self._previous_pressure[name] = (layer.Q.detach().clone(), pressure.detach().clone())
        for optimizer in self.constrained:
            optimizer.step()
        if self.ordinary is not None:
            self.ordinary.step()
        for _, layer in self.layers:
            layer.project_scale()
            if layer.mode == "mf":
                optimizer = next((opt for opt in self.constrained if layer.Q in opt.state), None)
                if optimizer is None:
                    continue
                gram = optimizer.state[layer.Q].get("S")
                if gram is not None and gram is not getattr(layer, "_optimizer_gram", None):
                    layer.set_gram(gram)
                    layer._optimizer_gram = gram

    @torch.no_grad()
    def diagnostics(self, include_gradient_metrics=True):
        result = {}
        for name, layer in self.layers:
            if layer.mode == "mf":
                eigenvalues = torch.linalg.eigvalsh(sym(layer.gram_S))
            elif layer.mode == "dense":
                eigenvalues = torch.linalg.svdvals(layer.weight).square().sort().values
            elif layer.scale is not None:
                eigenvalues = layer.scale.square().expand(layer.r).sort().values
            else:
                eigenvalues = torch.ones(layer.r, device=layer.Q.device)
            minimum = float(eigenvalues.min())
            maximum = float(eigenvalues.max())
            metrics = dict(self._last_metrics.get(name, {})) if include_gradient_metrics else {}
            metrics.update({
                "mode": layer.mode,
                "gram_orientation": "WWt" if layer.transpose else "WtW",
                "gram_eigenvalues": eigenvalues.cpu().tolist(),
                "lambda_min": minimum,
                "lambda_max": maximum,
                "lambda_ratio": maximum / max(minimum, 1e-12),
            })
            result[name] = metrics
        return result


def make_optimizers(model, optimizer_name, lr, total_steps, mf_config=None,
                    momentum=0.9, weight_decay=0.0, collect_diagnostics=True):
    if optimizer_name not in {"adam", "sgd"} or lr <= 0.0:
        raise ValueError("invalid base optimizer or learning rate")
    config = mf_config if mf_config is not None else ManifoldFlowConfig()
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, SpectrumLinear)]
    fixed = [layer.Q for _, layer in layers if layer.mode in {"fs", "scalar", "diagonal"}]
    learned = [layer.Q for _, layer in layers if layer.mode == "mf"]
    for _, layer in layers:
        if layer.mode == "mf" and (layer.alpha != config.lambda_min or layer.beta != config.lambda_max):
            raise ValueError("layer and optimizer spectral bounds differ")
    excluded = {id(parameter) for parameter in fixed + learned}
    scales = [layer.scale for _, layer in layers if layer.scale is not None]
    excluded.update(id(parameter) for parameter in scales)
    ordinary_parameters = [parameter for parameter in model.parameters()
                           if parameter.requires_grad and id(parameter) not in excluded]
    groups = []
    if ordinary_parameters:
        groups.append({"params": ordinary_parameters, "weight_decay": weight_decay})
    if scales:
        groups.append({"params": scales, "weight_decay": 0.0})
    ordinary = None
    if groups:
        ordinary = torch.optim.Adam(groups, lr=lr) if optimizer_name == "adam" else torch.optim.SGD(groups, lr=lr, momentum=momentum)
    constrained = []
    if fixed:
        constrained.append(FixedStiefelOptimizer(fixed, base_optim=optimizer_name, lr=lr, momentum=momentum))
    if learned:
        constrained.append(ManifoldFlowOptimizer(learned, base_optim=optimizer_name, lr=lr,
                                                momentum=momentum, mf_config=config,
                                                total_steps=total_steps, log_pressure=False))
    return OptimizerBundle(model, ordinary, constrained, layers, collect_diagnostics)
