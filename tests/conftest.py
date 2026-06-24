"""Pytest configuration: make ``src`` importable as a top-level package."""

import sys
from pathlib import Path

# repo/experiments/src
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
