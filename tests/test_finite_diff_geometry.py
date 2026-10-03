import torch

from manifoldflow.spd_ops import (
    sym,
    matrix_sqrt,
    symexpm,
)


def _random_stiefel(n, r, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q


def _random_spd(r, seed, jitter=0.5):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r, r, generator=g, dtype=torch.float64)
    return A @ A.T + jitter * torch.eye(r, dtype=torch.float64)


def _loss(W, target):
    return ((W - target) ** 2).sum()


def test_finite_diff_matches_analytic():

    n, r = 32, 8
    Q = _random_stiefel(n, r, seed=42)
    S = _random_spd(r, seed=43)
    target = torch.randn(n, r, dtype=torch.float64)

    Zraw = torch.randn(r, r, dtype=torch.float64, generator=torch.Generator().manual_seed(99))
    Z = sym(Zraw)

    S_var = S.clone().requires_grad_(True)
    W = Q @ matrix_sqrt(S_var)
    loss = _loss(W, target)
    grad_S = torch.autograd.grad(loss, S_var)[0]
    grad_S = sym(grad_S)
    analytic_dir_deriv = (grad_S * Z).sum().item()

    eps = 1e-5
    S_plus = sym(S + eps * Z)
    S_minus = sym(S - eps * Z)
    L_plus = _loss(Q @ matrix_sqrt(S_plus), target).item()
    L_minus = _loss(Q @ matrix_sqrt(S_minus), target).item()
    fd_dir_deriv = (L_plus - L_minus) / (2 * eps)

    rel = abs(analytic_dir_deriv - fd_dir_deriv) / (abs(fd_dir_deriv) + 1e-12)
    assert rel < 1e-3, (analytic_dir_deriv, fd_dir_deriv, rel)


def test_symexpm_round_trip():

    from manifoldflow.spd_ops import symlogm
    torch.manual_seed(0)
    M = sym(torch.randn(12, 12, dtype=torch.float64))
    round_trip = symlogm(symexpm(M))
    err = (round_trip - M).norm().item() / (M.norm().item() + 1e-12)
    assert err < 1e-8, err


if __name__ == "__main__":
    test_finite_diff_matches_analytic()
    test_symexpm_round_trip()
    print("test_finite_diff_geometry: PASS")
