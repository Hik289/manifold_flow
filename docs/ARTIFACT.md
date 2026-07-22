# Artifact Guide

Operational notes for reproducing `ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum` from the public `manifold_flow` repository.

## Review Path

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `experiments/`: Experiment drivers, ablations, and benchmark-specific runners.
- `tests/`: Local tests or smoke checks for fresh checkouts.
- `data/`: Small fixtures, schemas, manifests, or data-layout notes; large data should stay outside git.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `pyproject.toml`: Package metadata and optional extras when available.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
python -m pytest tests -q
```

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `bash scripts/reproduce_adult_mlp.sh`
- `bash scripts/reproduce_lstm.sh`
- `bash scripts/reproduce_transformer.sh`
- `python experiments/mlp_batch9.py`

## Data Layout Notes

- `data/DATA_INSTRUCTIONS.md`

## Figure Assets

- `fig_method_update_detailed.png`

## Data And Outputs

- Keep local dataset paths, downloaded corpora, checkpoints, and generated run artifacts outside git unless the README identifies them as small checked-in fixtures.
- Record dataset version, preprocessing command, seed, and hardware/runtime notes for every reproduced table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
