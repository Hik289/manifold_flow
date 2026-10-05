import numpy as np
import pytest
import torch

from experiments.mlp_convergence import MLP, parser, preprocess_tabular


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
    args = parser().parse_args([])
    features = torch.randn(2, 6)
    outputs = []
    for method in ("dense", "fs", "scalar", "diagonal", "mf"):
        model = MLP(6, 3, method, 42, args).eval()
        with torch.no_grad():
            outputs.append(model(features))
    assert all(torch.allclose(outputs[0], output, atol=1e-6, rtol=1e-6) for output in outputs[1:])
