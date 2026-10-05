import pytest
import torch

from manifoldflow.fixed_stiefel import FixedStiefelOptimizer
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig, ManifoldFlowOptimizer


def _orthogonal_parameter(seed=0):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return torch.nn.Parameter(q)


def test_adam_preserves_stiefel_constraint():
    parameter = _orthogonal_parameter()
    optimizer = ManifoldFlowOptimizer(
        [parameter],
        base_optim="adam",
        lr=1e-2,
        mf_config=ManifoldFlowConfig(rho_geo=0.0),
    )
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    identity = torch.eye(parameter.shape[1], dtype=parameter.dtype)
    assert torch.allclose(parameter.T @ parameter, identity, atol=1e-10)
    assert torch.all(optimizer.state[parameter]["adam_v"] >= 0)


def test_frozen_adam_matches_fixed_stiefel():
    manifold_parameter = _orthogonal_parameter()
    fixed_parameter = torch.nn.Parameter(manifold_parameter.detach().clone())
    manifold = ManifoldFlowOptimizer(
        [manifold_parameter],
        base_optim="adam",
        lr=1e-2,
        mf_config=ManifoldFlowConfig(rho_geo=0.0),
    )
    fixed = FixedStiefelOptimizer([fixed_parameter], base_optim="adam", lr=1e-2)
    generator = torch.Generator().manual_seed(11)
    for _ in range(4):
        gradient = torch.randn(8, 4, generator=generator, dtype=torch.float64)
        manifold_parameter.grad = gradient.clone()
        fixed_parameter.grad = gradient.clone()
        manifold.step()
        fixed.step()
    assert torch.equal(manifold_parameter, fixed_parameter)


def test_unknown_base_optimizer_is_rejected():
    with pytest.raises(ValueError, match="unsupported base optimizer"):
        ManifoldFlowOptimizer([_orthogonal_parameter()], base_optim="muon")

import math

import torch

from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig, ManifoldFlowOptimizer


def test_pressure_history_follows_a_changed_basis():
    parameter = torch.nn.Parameter(torch.eye(3, dtype=torch.float64)[:, :2])
    optimizer = ManifoldFlowOptimizer(
        [parameter], lr=0.01, log_pressure=False,
        mf_config=ManifoldFlowConfig(rho_geo=0.0, beta_P=0.5),
    )
    original = parameter.detach().clone()
    parameter.grad = original @ torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64))
    parameter.grad[2] = torch.tensor([0.5, -0.25], dtype=torch.float64)
    optimizer.step()
    assert not torch.equal(parameter.detach(), original)
    assert torch.equal(optimizer.state[parameter]["Q_prev"], original)
    history = optimizer.state[parameter]["M_P"].clone()
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    with torch.no_grad():
        parameter.copy_(original @ rotation)
    basis_before_step = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    expected = 0.5 * rotation.T @ history @ rotation
    assert torch.allclose(optimizer.state[parameter]["M_P"], expected, atol=1e-12)
    assert torch.equal(optimizer.state[parameter]["Q_prev"], basis_before_step)


def test_isotropic_regularizer_contracts_log_eigenvalues():
    parameter = torch.nn.Parameter(torch.eye(3, dtype=torch.float64)[:, :2])
    config = ManifoldFlowConfig(
        rho_geo=0.1, lambda_S=0.3, beta_P=0.0, K_geo=1,
        warmup_frac=0.0, use_gate=False,
    )
    optimizer = ManifoldFlowOptimizer([parameter], lr=0.2, mf_config=config, log_pressure=False)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["S"] = 2.0 * torch.eye(2, dtype=torch.float64)
    optimizer.step()
    expected = 2.0 ** (1.0 - 0.2 * 0.1 * 0.3)
    assert torch.allclose(optimizer.state[parameter]["S"], expected * torch.eye(2, dtype=torch.float64), atol=1e-12)


def test_anisotropy_reduces_the_geometry_step():
    parameter = torch.nn.Parameter(torch.eye(3, dtype=torch.float64)[:, :2])
    config = ManifoldFlowConfig(
        rho_geo=0.1, lambda_S=0.0, beta_P=0.0, K_geo=1,
        warmup_frac=0.0, alpha_c=0.0, alpha_r=0.0,
    )
    optimizer = ManifoldFlowOptimizer([parameter], lr=0.2, mf_config=config, log_pressure=False)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    spectrum = torch.tensor([0.5, 2.0], dtype=torch.float64)
    optimizer.state[parameter]["S"] = torch.diag(spectrum)
    parameter.grad = parameter.detach().clone()
    optimizer.step()
    gate = 0.5 * 0.5 * (0.5 / 2.0)
    pressure = 1.0 / (math.sqrt(2.0) + 1e-8)
    expected = spectrum * torch.exp(-0.2 * 0.1 * gate * pressure / spectrum)
    assert torch.allclose(torch.diagonal(optimizer.state[parameter]["S"]), expected, atol=1e-12)
