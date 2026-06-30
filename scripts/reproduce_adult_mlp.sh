#!/bin/bash
# Reproduce Adult Census MLP results
# Expected output in experiments/ (printed to stdout)

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[reproduce_adult_mlp] Starting Adult MLP experiment..."
$PYTHON experiments/mlp_batch9.py --dataset adult

echo "[reproduce_adult_mlp] Done."
