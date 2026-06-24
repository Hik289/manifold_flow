# ManifoldFlow: Learnable SPD Geometry for Stiefel-Constrained Layers

[[Paper]](#citation) | [arXiv placeholder]

## Overview

ManifoldFlow extends Fixed-Stiefel layers with a learnable SPD spectrum via the
parametrization **W = Q · S^(1/2)**, where Q ∈ St(p,r) (Stiefel manifold) and S ∈ SPD(r).
The SPD factor is updated online via affine-invariant retraction on the SPD cone, allowing
the layer to adapt its singular spectrum during training without leaving the Stiefel constraint.

### Key Idea

Standard Fixed-Stiefel layers enforce orthogonality (Q^T Q = I) but fix the singular
spectrum. ManifoldFlow adds a learnable SPD factor S that controls the spectrum:

```
W = Q · S^(1/2)
    ↑          ↑
Stiefel      Learnable
manifold     SPD spectrum
```

The optimizer alternates between:
1. **Riemannian step** on Q via projected gradient / Cayley retraction
2. **Affine-invariant geodesic** update on the SPD cone for S

## Key Results

| Task | Baseline (Fixed-Stiefel) | ManifoldFlow | Δ |
|------|--------------------------|--------------|---|
| LSTM/WikiText-2 (Adam) | 287.86 PPL | 277.20 PPL | **−10.66 PPL** |
| LSTM/WikiText-2 (SGD) | 613.66 PPL | 608.56 PPL | **−5.10 PPL** |
| Adult MLP (Adam) | 84.26% | 85.04% | **+0.78 pp** |
| Mini-Transformer FFN (Adam) | 80.29 PPL | 80.06 PPL | **−0.23 PPL** |

Lower PPL = better for language modeling. Higher % = better for classification.

## Installation

```bash
git clone https://github.com/[ANONYMOUS]/manifoldflow.git
cd manifoldflow
pip install -r requirements.txt
```

For development / editable install:
```bash
pip install -e ".[dev]"
```

**Requirements:** Python 3.9+, PyTorch 2.0+. See `requirements.txt` for full list.

## Quick Start: Using ManifoldFlow Optimizer

```python
import torch
from src.manifoldflow.parametrization import wrap_stiefel
from src.manifoldflow.manifoldflow_optimizer import ManifoldFlowOptimizer

model = YourModel()

# Wrap a linear layer to use W = Q * S^(1/2)
model.fc = wrap_stiefel(model.fc, mode='manifoldflow')

# Use ManifoldFlow optimizer (combines base optimizer + SPD update)
optimizer = ManifoldFlowOptimizer(
    model.parameters(),
    lr=0.001,
    base='adam',      # or 'sgd'
    rho_geo=0.01,     # SPD step size (geodesic)
    lambda_S=0.001,   # SPD regularization
    K_geo=10,         # SPD update frequency (every K steps)
)

# Standard training loop — no changes needed here
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
```

### Using Fixed-Stiefel (baseline) Mode

```python
# Wrap with Fixed-Stiefel only (no SPD, standard Riemannian gradient)
model.fc = wrap_stiefel(model.fc, mode='fixed_stiefel')
```

## Repository Structure

```
manifoldflow_code/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── src/
│   └── manifoldflow/
│       ├── manifoldflow_optimizer.py   # Main optimizer: Riemannian + SPD update
│       ├── fixed_stiefel.py            # Fixed-Stiefel layer baseline
│       ├── intrinsic_muon.py           # Intrinsic Muon optimizer variant
│       ├── retraction.py               # Stiefel retraction maps (Cayley, QR, etc.)
│       ├── spd_ops.py                  # SPD cone operations (exp, log, geodesic)
│       ├── tangent.py                  # Tangent space projections
│       └── parametrization.py          # wrap_stiefel() and layer parametrization
├── experiments/
│   ├── b12_lstm_5seeds.py             # LSTM/WikiText-2 main experiment (5 seeds)
│   ├── mlp_batch9.py                  # Adult/Covertype MLP experiment
│   ├── transformer_wikitext_b10.py    # Mini-Transformer FFN on WikiText-2
│   ├── e2_baselines.py                # B1 Q*diag, B2 Dense, B3 SpecNorm baselines
│   ├── b12_a6_ablation.py             # A6 random SPD pressure ablation
│   ├── b14_lstm_baselines.py          # Additional LSTM baselines
│   └── results/                       # Pre-computed experiment results (JSON)
├── tests/
│   ├── test_optimizer.py              # 7/7 unit tests for optimizer correctness
│   ├── test_gamma0_equivalence.py     # Anchor: gamma=0 → Fixed-Stiefel equivalence
│   ├── test_stiefel_feasibility.py    # Q^T Q = I constraint verification
│   ├── test_spd_feasibility.py        # S ∈ SPD cone constraint verification
│   ├── test_tangent_condition.py      # Tangent space projection correctness
│   ├── test_spectrum_identity.py      # Spectrum identity tests
│   ├── test_decomposition_orthogonality.py
│   ├── test_frozen_equivalence.py
│   └── test_finite_diff_geometry.py
├── scripts/
│   ├── reproduce_lstm.sh              # Reproduce Table 1: LSTM/WikiText-2
│   ├── reproduce_adult_mlp.sh         # Reproduce MLP/Adult results
│   └── reproduce_transformer.sh       # Reproduce Transformer/WikiText-2
└── data/
    └── DATA_INSTRUCTIONS.md           # How to download all datasets
```

## Reproducing Paper Experiments

### LSTM/WikiText-2 (main result, Table 1)

```bash
bash scripts/reproduce_lstm.sh
```

Runs 4 cells (FS-SGD, FS-Adam, MF-SGD, MF-Adam) × 5 seeds × 8 epochs.
Expected output: `experiments/results/lstm_wt2_proj/stage_b_results_5seeds.json`

**Approximate GPU time:** ~2–4 h on a single RTX 2080 Ti (11 GB).

### Adult MLP

```bash
bash scripts/reproduce_adult_mlp.sh
```

### Mini-Transformer FFN

```bash
bash scripts/reproduce_transformer.sh
```

### Custom environment variable

Set `PYTHON` to use a specific interpreter:

```bash
PYTHON=/path/to/conda/envs/myenv/bin/python bash scripts/reproduce_lstm.sh
```

## Unit Tests

```bash
pytest tests/ -v
```

All 7+ unit tests should pass:
- Stiefel feasibility (Q^T Q = I after retraction)
- SPD feasibility (S positive definite throughout training)
- Tangent condition (projected gradient lies in tangent space)
- Spectrum identity (ManifoldFlow reduces to Fixed-Stiefel when S = I)
- Gamma=0 equivalence (no SPD update → identical to Fixed-Stiefel)
- Decomposition orthogonality
- Finite-difference geometry validation

## Architecture Boundary Notes

ManifoldFlow improvements are most pronounced on **projection-bottleneck architectures**
where the Stiefel constraint reduces a high-dimensional hidden state to a lower-dimensional
space (e.g., LSTM hidden→context, Transformer FFN internal projection).

On architectures where **unconstrained linear layers outperform Stiefel** (e.g., Covertype
tabular MLP, LSTM hidden→vocab projection vs. dense baseline), the Stiefel framework itself
is the bottleneck — ManifoldFlow mitigates but cannot overcome this.

See paper Appendix C for full architecture boundary analysis.

## SPD Hyperparameter Guide

| Hyperparameter | Typical range | Effect |
|----------------|---------------|--------|
| `rho_geo` | 1e-3 – 1e-2 | SPD geodesic step size; too large → divergence |
| `lambda_S` | 1e-4 – 1e-3 | SPD regularization toward identity |
| `K_geo` | 5 – 20 | SPD update frequency; higher = less overhead |
| `base` | `'adam'`, `'sgd'` | Base optimizer for Q (Stiefel part) |

## Citation

```bibtex
@article{manifoldflow2026,
  title={ManifoldFlow: Learnable SPD Geometry for Stiefel-Constrained Layers},
  author={[Anonymous]},
  journal={[Anonymous submission]},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
