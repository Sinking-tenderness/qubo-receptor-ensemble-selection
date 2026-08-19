"""Run or plan the canonical experiment pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qubo_receptor_ensemble.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
