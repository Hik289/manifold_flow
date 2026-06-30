#!/bin/bash
# Reproduce Mini-Transformer FFN results on WikiText-2
# Expected output: experiments/results/transformer_wikitext/

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[reproduce_transformer] Starting Transformer/WikiText-2 experiment..."
$PYTHON experiments/transformer_wikitext_b10.py

echo "[reproduce_transformer] Done."
