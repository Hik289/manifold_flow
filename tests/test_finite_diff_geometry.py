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

import math

import torch

from experiments.b12_lstm_5seeds import TransformerLanguageModel, current_evaluate as sequence_evaluate, current_load_corpus as load_corpus, current_parser as sequence_parser


def test_validation_and_test_words_do_not_enter_the_training_vocabulary(tmp_path):
    (tmp_path / "train.txt").write_text("alpha beta alpha\n", encoding="utf-8")
    (tmp_path / "valid.txt").write_text("unseen alpha\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("other beta\n", encoding="utf-8")
    args = sequence_parser().parse_args(["--data-dir", str(tmp_path), "--max-vocab-size", "5"])
    corpus = load_corpus("wikitext2", args)
    vocabulary = corpus["metadata"]["vocabulary"]["tokens"]
    assert "unseen" not in vocabulary
    assert "other" not in vocabulary
    assert corpus["encoded"]["validation"][0].item() == 0
    assert corpus["encoded"]["test"][0].item() == 0


def test_transformer_predictions_cannot_depend_on_future_tokens():
    torch.manual_seed(13)
    args = sequence_parser().parse_args(["--architectures", "transformer"])
    model = TransformerLanguageModel(11, 128, "fs", args).eval()
    inputs = torch.randint(11, (6, 2))
    changed = inputs.clone()
    changed[3:] = (changed[3:] + 1) % 11
    with torch.no_grad():
        original, _ = model(inputs)
        alternative, _ = model(changed)
    assert torch.allclose(original[:6], alternative[:6], atol=1e-6)


def test_perplexity_counts_a_partial_final_segment_correctly():
    class UniformModel(torch.nn.Module):
        def forward(self, inputs, hidden=None):
            return torch.zeros(inputs.numel(), 5), None

    source = torch.arange(14).view(7, 2) % 5
    metrics = sequence_evaluate(UniformModel(), source, 4, torch.device("cpu"))
    assert metrics["tokens"] == 12
    assert math.isclose(metrics["ppl"], 5.0, rel_tol=1e-6)

import numpy as np
import pytest
import torch

from experiments.mlp_batch9 import MLP, current_parser as mlp_parser, current_preprocess_tabular as preprocess_tabular


def test_tabular_normalization_and_category_encoding_use_training_rows_only():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    train = pd.DataFrame({"number": [0.0, 2.0], "category": ["a", "b"]})
    held_out = pd.DataFrame({"number": [1000.0], "category": ["unseen"]})
    arrays, metadata = preprocess_tabular(train, held_out, held_out)
    assert metadata["fit_split"] == "train"
    assert np.allclose(arrays[0][:, 0], [-1.0, 1.0])
    assert np.allclose(arrays[1][0], [999.0, -3.0])
    assert np.isfinite(arrays[2]).all()


def test_mlp_baselines_share_the_initial_feature_maps_and_dense_classifier():
    args = mlp_parser().parse_args([])
    features = torch.randn(2, 6)
    outputs = []
    for method in ("dense", "fs", "scalar", "diagonal", "mf"):
        model = MLP(6, 3, method, 42, args).eval()
        with torch.no_grad():
            outputs.append(model(features))
    assert all(torch.allclose(outputs[0], output, atol=1e-6, rtol=1e-6) for output in outputs[1:])

if __name__ == "__main__":
    test_finite_diff_matches_analytic()
    test_symexpm_round_trip()
    print("test_finite_diff_geometry: PASS")
