"""Shared test configuration.

Markers (registered in ``pyproject.toml``):

``smoke``
    quick end-to-end runs on synthetic data; always run.
``data``
    need the raw datasets; skipped unless the data folder (``HEXMOD_DATA`` or ``<repo>/data``)
    holds at least one dataset folder.
``frozen``
    compare with a complete results folder of an earlier run, laid out like ``results/``
    (no such folder is distributed); skipped unless ``HEXMOD_FROZEN`` or ``HEXMOD_FROZEN_RESULTS``
    points to an existing folder. The ``frozen_dir`` fixture returns ``HEXMOD_FROZEN`` (for
    example ``<HEXMOD_FROZEN>/rat/cohort/cells.csv``) and skips tests that need it when only
    ``HEXMOD_FROZEN_RESULTS`` is set; ``tests/test_plotted_vs_stats.py`` takes either (a folder with
    ``figures/`` and ``stats/``, such as ``results/`` after a run).
``slow``
    long runs; skipped unless ``--runslow`` is given or ``HEXMOD_RUN_SLOW=1``.

Tests must not write into the repository; use pytest's ``tmp_path`` and set ``HEXMOD_RESULTS``
to a temporary folder when running steps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked slow")


def _data_available() -> bool:
    from hexmod import config, datasets

    base = config.data_dir()
    if not base.is_dir():
        return False
    for key in datasets.CANONICAL_ORDER:
        first = datasets.DATASETS[key]["raw_subpaths"][0]
        if (base / first).exists():
            return True
    return False


def _frozen_available() -> bool:
    for name in ("HEXMOD_FROZEN", "HEXMOD_FROZEN_RESULTS"):
        value = os.environ.get(name)
        if value and Path(value).expanduser().is_dir():
            return True
    return False


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow") or os.environ.get("HEXMOD_RUN_SLOW") == "1"
    has_data = _data_available()
    has_frozen = _frozen_available()
    skip_slow = pytest.mark.skip(reason="slow test: use --runslow or HEXMOD_RUN_SLOW=1")
    skip_data = pytest.mark.skip(reason="raw datasets not found (set HEXMOD_DATA)")
    skip_frozen = pytest.mark.skip(reason="reference results not found (set HEXMOD_FROZEN or HEXMOD_FROZEN_RESULTS)")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "data" in item.keywords and not has_data:
            item.add_marker(skip_data)
        if "frozen" in item.keywords and not has_frozen:
            item.add_marker(skip_frozen)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root."""
    return REPO


@pytest.fixture(scope="session")
def frozen_dir() -> Path:
    """Reference results folder (``HEXMOD_FROZEN``); skips the test when it is absent."""
    value = os.environ.get("HEXMOD_FROZEN")
    if not value or not Path(value).expanduser().is_dir():
        pytest.skip("reference results not found (set HEXMOD_FROZEN)")
    return Path(value).expanduser()


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Raw-data folder; skips the test when no dataset is present."""
    from hexmod import config

    if not _data_available():
        pytest.skip("raw datasets not found (set HEXMOD_DATA)")
    return config.data_dir()


def frozen_file(frozen: Path, relpath: str) -> Path:
    """Return ``frozen / relpath`` or skip the test when the file is missing."""
    path = frozen / relpath
    if not path.exists():
        pytest.skip(f"reference file missing: {relpath}")
    return path


@pytest.fixture
def frozen_path(frozen_dir):
    """Function ``f(relpath)`` returning a reference file path, skipping when it is missing."""
    return lambda relpath: frozen_file(frozen_dir, relpath)
