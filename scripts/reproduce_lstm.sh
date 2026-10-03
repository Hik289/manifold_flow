#!/bin/bash

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[reproduce_lstm] Starting LSTM/WikiText-2 experiment..."
$PYTHON experiments/b12_lstm_5seeds.py

echo "[reproduce_lstm] Done. Results in experiments/results/lstm_wt2_proj/"
