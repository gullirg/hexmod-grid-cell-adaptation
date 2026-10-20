"""Shared test configuration.

Marker (registered in ``pyproject.toml``):

``results``
    check the results folder of a complete run (``tests/test_plotted_vs_stats.py``); the folder
    is given in ``HEXMOD_RESULTS``, and the tests are skipped when it holds no ``figures/`` and
    ``stats/``.

Tests must not write into the repository; use pytest's ``tmp_path`` and set ``HEXMOD_RESULTS``
to a temporary folder when running steps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLBACKEND", "Agg")
