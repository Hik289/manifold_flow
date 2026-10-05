import math

import torch

from experiments.sequence_models import TransformerLanguageModel, evaluate, load_corpus, parser


def test_validation_and_test_words_do_not_enter_the_training_vocabulary(tmp_path):
    (tmp_path / "train.txt").write_text("alpha beta alpha\n", encoding="utf-8")
    (tmp_path / "valid.txt").write_text("unseen alpha\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("other beta\n", encoding="utf-8")
    args = parser().parse_args(["--data-dir", str(tmp_path), "--max-vocab-size", "5"])
    corpus = load_corpus("wikitext2", args)
    vocabulary = corpus["metadata"]["vocabulary"]["tokens"]
    assert "unseen" not in vocabulary
    assert "other" not in vocabulary
    assert corpus["encoded"]["validation"][0].item() == 0
    assert corpus["encoded"]["test"][0].item() == 0


def test_transformer_predictions_cannot_depend_on_future_tokens():
    torch.manual_seed(13)
    args = parser().parse_args(["--architectures", "transformer"])
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
    metrics = evaluate(UniformModel(), source, 4, torch.device("cpu"))
    assert metrics["tokens"] == 12
    assert math.isclose(metrics["ppl"], 5.0, rel_tol=1e-6)
