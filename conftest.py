"""Pytest root configuration.

Adds the ``src`` layout package to ``sys.path`` so tests can import
``qubo_receptor_ensemble`` without requiring an editable install.
Equivalent to ``python -m pip install -e .`` for import resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
