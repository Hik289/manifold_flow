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
