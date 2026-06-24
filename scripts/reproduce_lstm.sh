#!/bin/bash
# Reproduce LSTM/WikiText-2 results (Table 1 main result)
# Runs 4 cells: FS-SGD, FS-Adam, MF-SGD, MF-Adam x 5 seeds x 8 epochs
# Expected output: experiments/results/lstm_wt2_proj/stage_b_results_5seeds.json

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}

echo "[reproduce_lstm] Starting LSTM/WikiText-2 experiment..."
$PYTHON experiments/b12_lstm_5seeds.py

echo "[reproduce_lstm] Done. Results in experiments/results/lstm_wt2_proj/"
