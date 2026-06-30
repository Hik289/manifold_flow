# ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](requirements.txt)

Official code for **ManifoldFlow**, a spectrum-learnable relaxation of
Fixed-Stiefel neural layers.

> Fixed Stiefel learns an orthonormal basis but fixes every represented
> singular value at one. ManifoldFlow keeps the basis and learns bounded
> singular values through an SPD factor.

<p align="center">
  <img src="fig_method_update_detailed.png" width="96%" alt="Detailed ManifoldFlow update schematic">
</p>

## Method

A Fixed-Stiefel layer represents a weight by a semi-orthogonal factor
`Q in St(p, r)`, so `Q.T @ Q = I` and every represented singular value is
fixed at one. ManifoldFlow introduces a minimal SPD relaxation:

```text
W = Q S^{1/2},        Q in St(p, r),        S in SPD(r).
```

The identity `W.T @ W = S` makes the parameterization directly interpretable:
the eigenvalues of `S` are exactly the squared singular values of the realized
weight `W`. ManifoldFlow therefore keeps the Stiefel basis while learning a
bounded singular spectrum by clipping the eigenvalues of `S`.

The optimizer uses the same tangent Stiefel step as the Fixed-Stiefel baseline
for `Q`, and updates `S` with an affine-invariant SPD retraction. The default
SPD direction is built from the normal-gradient pressure discarded by the
Stiefel tangent projection.

## What Is Included

- `src/manifoldflow/`: core geometry, Stiefel retractions, SPD operations,
  Fixed-Stiefel optimizer, ManifoldFlow optimizer, and spectral utilities.
- `experiments/`: paired sequence, tabular, image, and baseline scripts used
  for the paper experiments.
- `tests/`: geometry and optimizer invariance tests for Stiefel feasibility,
  SPD feasibility, tangent projection, spectrum identity, and frozen-spectrum
  equivalence.
- `scripts/`: reproducibility entry points for the main experiment families.
- `experiments/results/`: lightweight JSON records for archived runs.

Raw datasets, model checkpoints, caches, notebooks, and local analysis scratch
files are intentionally not bundled.

## Installation

```bash
git clone https://github.com/Hik289/manifold_flow.git
cd manifold_flow

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

For a lighter install that only uses the core optimizer and tests, `pip install
-e .` is sufficient. The full `requirements.txt` adds experiment dependencies
such as `datasets`, `torchvision`, `transformers`, and `scikit-learn`.

## Quick Start

```python
import torch

from manifoldflow.manifoldflow_optimizer import (
    ManifoldFlowConfig,
    ManifoldFlowOptimizer,
)
from manifoldflow.parametrization import manifold_weight

n, r = 128, 64
Q0, _ = torch.linalg.qr(torch.randn(n, r), mode="reduced")
Q = torch.nn.Parameter(Q0)

cfg = ManifoldFlowConfig(
    rho_geo=1e-2,
    lambda_S=1e-3,
    K_geo=10,
    lambda_min=0.25,
    lambda_max=4.0,
)
optimizer = ManifoldFlowOptimizer(
    [Q],
    lr=1e-2,
    momentum=0.9,
    mf_config=cfg,
    total_steps=1000,
)

def current_weight():
    state = optimizer.state.get(Q, {})
    S = state.get("S", torch.eye(r, device=Q.device, dtype=Q.dtype))
    return manifold_weight(Q, S)

for x, y in dataloader:
    optimizer.zero_grad()
    W = current_weight()
    logits = x @ W
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
```

For the paired Fixed-Stiefel baseline, either use
`manifoldflow.fixed_stiefel.FixedStiefelOptimizer` or set
`ManifoldFlowConfig(rho_geo=0.0)`, which recovers the frozen-spectrum update
path tested in `tests/test_frozen_equivalence.py`.

## Reproducing Experiments

All scripts assume they are launched from the repository root. The shell
wrappers set `PYTHONPATH=src` automatically.

```bash
# LSTM / WikiText-2 hidden-to-vocabulary projection
bash scripts/reproduce_lstm.sh

# Adult Census MLP
bash scripts/reproduce_adult_mlp.sh

# Mini-Transformer feed-forward layers on WikiText-2
bash scripts/reproduce_transformer.sh
```

Use a specific Python interpreter with:

```bash
PYTHON=/path/to/python bash scripts/reproduce_lstm.sh
```

Dataset download instructions are in [`data/DATA_INSTRUCTIONS.md`](data/DATA_INSTRUCTIONS.md).

## Tests

```bash
python -m pytest tests -q
```

The test suite checks:

- `Q.T @ Q = I` after Stiefel retraction.
- `S` stays positive definite under SPD updates.
- tangent projections satisfy the Stiefel tangent condition.
- `W = Q S^{1/2}` satisfies `sigma_i(W)^2 = lambda_i(S)`.
- `rho_geo = 0` matches the Fixed-Stiefel trajectory.
- finite-difference geometry sanity checks.

## Repository Layout

```text
.
├── data/
│   └── DATA_INSTRUCTIONS.md
├── experiments/
│   ├── b12_lstm_5seeds.py
│   ├── mlp_batch9.py
│   ├── transformer_wikitext_b10.py
│   ├── e2_baselines.py
│   └── results/
├── scripts/
│   ├── reproduce_adult_mlp.sh
│   ├── reproduce_lstm.sh
│   └── reproduce_transformer.sh
├── src/
│   ├── baselines/
│   └── manifoldflow/
│       ├── fixed_stiefel.py
│       ├── manifoldflow_optimizer.py
│       ├── parametrization.py
│       ├── retraction.py
│       ├── spd_ops.py
│       └── tangent.py
├── tests/
├── fig_method_update_detailed.png
├── pyproject.toml
└── requirements.txt
```

## Citation

```bibtex
@misc{manifoldflow2026,
  title        = {ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum},
  author       = {Anonymous},
  year         = {2026},
  howpublished = {\url{https://github.com/Hik289/manifold_flow}},
}
```

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
