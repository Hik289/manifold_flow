"""Unit test #6 - Frozen geometry equivalence.

When rho_geo=0, ManifoldFlow-SGD trajectory must match Fixed-Stiefel-SGD
step-for-step within <= 1e-6.
"""

import torch


def _make_stiefel_q(n, r, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)
    return Q.clone().requires_grad_(True)


def _quadratic_loss(Q, target):
    return 0.5 * ((Q - target) ** 2).sum()


def test_frozen_equivalence_sgd():
    """ManifoldFlowOptimizer(rho_geo=0) == FixedStiefelOptimizer step-for-step."""
    from manifoldflow.manifoldflow_optimizer import ManifoldFlowOptimizer, ManifoldFlowConfig
    from manifoldflow.fixed_stiefel import FixedStiefelOptimizer

    torch.manual_seed(42)
    n, r = 32, 8
    seed = 7
    lr = 0.01
    n_steps = 20

    target = torch.randn(n, r, dtype=torch.float32)

    # Run A: Fixed-Stiefel-SGD
    Q_fs = _make_stiefel_q(n, r, seed)
    opt_fs = FixedStiefelOptimizer([Q_fs], lr=lr, momentum=0.0)
    Q_fs_history = []
    for _ in range(n_steps):
        opt_fs.zero_grad()
        loss = _quadratic_loss(Q_fs, target)
        loss.backward()
        opt_fs.step()
        Q_fs_history.append(Q_fs.data.clone())

    # Run B: ManifoldFlow with rho_geo=0
    Q_mf = _make_stiefel_q(n, r, seed)
    cfg = ManifoldFlowConfig(rho_geo=0.0, warmup_frac=0.0)
    opt_mf = ManifoldFlowOptimizer([Q_mf], lr=lr, momentum=0.0, mf_config=cfg, total_steps=n_steps)
    Q_mf_history = []
    for _ in range(n_steps):
        opt_mf.zero_grad()
        loss = _quadratic_loss(Q_mf, target)
        loss.backward()
        opt_mf.step()
        Q_mf_history.append(Q_mf.data.clone())

    max_dev = 0.0
    for step_idx, (Q_a, Q_b) in enumerate(zip(Q_fs_history, Q_mf_history)):
        dev = (Q_a - Q_b).abs().max().item()
        max_dev = max(max_dev, dev)
        assert dev < 1e-6, f"Step {step_idx}: deviation {dev:.3e}"

    print(f"test_frozen_equivalence_sgd PASS -- max deviation = {max_dev:.3e}")


def test_frozen_equivalence_with_momentum():
    """ManifoldFlowOptimizer(rho_geo=0, momentum=0.9) == FixedStiefelOptimizer(momentum=0.9)."""
    from manifoldflow.manifoldflow_optimizer import ManifoldFlowOptimizer, ManifoldFlowConfig
    from manifoldflow.fixed_stiefel import FixedStiefelOptimizer

    torch.manual_seed(123)
    n, r = 16, 4
    seed = 99
    lr = 0.005
    momentum = 0.9
    n_steps = 30

    target = torch.randn(n, r, dtype=torch.float32)

    Q_fs = _make_stiefel_q(n, r, seed)
    opt_fs = FixedStiefelOptimizer([Q_fs], lr=lr, momentum=momentum)
    Q_fs_history = []
    for _ in range(n_steps):
        opt_fs.zero_grad()
        loss = _quadratic_loss(Q_fs, target)
        loss.backward()
        opt_fs.step()
        Q_fs_history.append(Q_fs.data.clone())

    Q_mf = _make_stiefel_q(n, r, seed)
    cfg = ManifoldFlowConfig(rho_geo=0.0, warmup_frac=0.0)
    opt_mf = ManifoldFlowOptimizer([Q_mf], lr=lr, momentum=momentum, mf_config=cfg, total_steps=n_steps)
    Q_mf_history = []
    for _ in range(n_steps):
        opt_mf.zero_grad()
        loss = _quadratic_loss(Q_mf, target)
        loss.backward()
        opt_mf.step()
        Q_mf_history.append(Q_mf.data.clone())

    max_dev = 0.0
    for step_idx, (Q_a, Q_b) in enumerate(zip(Q_fs_history, Q_mf_history)):
        dev = (Q_a - Q_b).abs().max().item()
        max_dev = max(max_dev, dev)
        assert dev < 1e-6, f"Step {step_idx}: dev {dev:.3e} with momentum"

    print(f"test_frozen_equivalence_with_momentum PASS -- max deviation = {max_dev:.3e}")


if __name__ == "__main__":
    test_frozen_equivalence_sgd()
    test_frozen_equivalence_with_momentum()
    print("test_frozen_equivalence: PASS")
