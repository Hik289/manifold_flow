#!/bin/bash

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

if [[ "$1" == "--legacy" ]]; then
    shift
    "$PYTHON" experiments/mlp_batch9.py --legacy --task adult "$@"
else
    "$PYTHON" experiments/mlp_batch9.py --datasets adult "$@"
fi
