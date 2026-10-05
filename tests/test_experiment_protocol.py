import pytest
import torch

from experiments.common import PlateauStopper, SpectrumLinear, make_optimizers, set_seed
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig


@pytest.mark.parametrize("dimensions", [(5, 3), (3, 5), (4, 4)])
def test_spectral_baselines_start_from_the_same_realized_weight(dimensions):
    weights = []
    for method in ("dense", "fs", "scalar", "diagonal", "mf"):
        set_seed(19)
        layer = SpectrumLinear(*dimensions, mode=method)
        weights.append(layer.effective_weight().detach())
    assert all(torch.equal(weights[0], weight) for weight in weights[1:])


def test_selected_model_restores_its_spd_spectrum():
    set_seed(2)
    model = SpectrumLinear(4, 6, mode="mf")
    model.set_gram(torch.diag(torch.tensor([0.5, 1.0, 2.0, 3.0])))
    checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = torch.randn(2, 4)
    selected_output = model(inputs).detach().clone()
    model.set_gram(torch.eye(4))
    model.load_state_dict(checkpoint)
    assert torch.equal(model(inputs), selected_output)
    assert torch.allclose(model.effective_weight().T @ model.effective_weight(), model.gram_S, atol=1e-6)


def test_projected_scale_updates_preserve_the_two_sided_bound():
    for method in ("scalar", "diagonal"):
        set_seed(4)
        model = SpectrumLinear(3, 5, mode=method)
        bundle = make_optimizers(model, "sgd", lr=100.0, total_steps=1, collect_diagnostics=False)
        model.Q.grad = torch.zeros_like(model.Q)
        model.scale.grad = torch.ones_like(model.scale)
        bundle.step()
        values = torch.linalg.svdvals(model.effective_weight())
        assert bool((values >= 0.5 - 1e-6).all())
        assert bool((values <= 2.0 + 1e-6).all())


def test_stopping_never_uses_test_metrics_or_stops_before_the_minimum():
    stopper = PlateauStopper(min_epochs=5, patience=2, mode="max")
    for epoch, accuracy in enumerate([50.0, 60.0, 59.0, 58.0], 1):
        assert not stopper.update(accuracy, epoch)
    assert stopper.update(57.0, 5)
    assert stopper.best_epoch == 2
    assert stopper.best_value == 60.0


def test_stopping_tracks_the_actual_best_even_with_a_plateau_tolerance():
    stopper = PlateauStopper(min_epochs=3, patience=2, min_delta=0.1, mode="min")
    assert not stopper.update(2.0, 1)
    assert not stopper.update(1.98, 2)
    assert stopper.update(1.97, 3)
    assert stopper.best_epoch == 3
    assert stopper.best_value == 1.97


def test_optimizer_rejects_mismatched_model_and_gram_bounds():
    layer = SpectrumLinear(4, 6, mode="mf", alpha=0.5, beta=2.0)
    with pytest.raises(ValueError, match="spectral bounds differ"):
        make_optimizers(layer, "adam", 0.01, 2, mf_config=ManifoldFlowConfig())
