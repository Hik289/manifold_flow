#!/bin/bash

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

if [[ "$1" == "--tasks" ]]; then
    "$PYTHON" experiments/e2_baselines.py "$@"
else
    "$PYTHON" experiments/b12_lstm_5seeds.py "$@"
fi
