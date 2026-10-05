#!/bin/bash

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

"$PYTHON" experiments/transformer_wikitext_b10.py "$@"
