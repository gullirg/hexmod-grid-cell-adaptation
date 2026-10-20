"""Paths, results folder and worker count.

Configuration comes from command-line flags (in the scripts) and these environment variables:

``HEXMOD_DATA``
    Folder with the raw datasets (default ``<repo>/data``).
``HEXMOD_RESULTS``
    Folder for all outputs (default ``<repo>/results``).
``HEXMOD_FONT_DIR``
    Optional folder with Arial TTF files for the figures.
``HEXMOD_WORKERS``
    Number of worker processes (default 4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


#: Default number of worker processes.
DEFAULT_WORKERS = 4


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "hexmod").is_dir():
            return parent
    return Path.cwd()


#: Root of the repository (the folder that holds ``pyproject.toml``).
REPO_ROOT: Path = _find_repo_root()


def repo_root() -> Path:
    """Return the repository root."""
    return REPO_ROOT


def data_dir() -> Path:
    """Return the raw-data folder (``HEXMOD_DATA`` or ``<repo>/data``)."""
    return Path(os.environ.get("HEXMOD_DATA") or (REPO_ROOT / "data")).expanduser()


def results_dir() -> Path:
    """Return the results folder (``HEXMOD_RESULTS`` or ``<repo>/results``)."""
    return Path(os.environ.get("HEXMOD_RESULTS") or (REPO_ROOT / "results")).expanduser()


def precomputed_dir() -> Path:
    """Return the folder of shipped simulation results (``<repo>/results_precomputed``)."""
    return REPO_ROOT / "results_precomputed"


def inputs_dir() -> Path:
    """Return the folder of small shipped inputs (``<repo>/inputs``)."""
    return REPO_ROOT / "inputs"


def font_dir() -> Optional[Path]:
    """Return ``HEXMOD_FONT_DIR`` as a path, or None when it is not set."""
    value = os.environ.get("HEXMOD_FONT_DIR")
    return Path(value).expanduser() if value else None


def workers(value: Optional[int] = None) -> int:
    """Return the number of worker processes.

    Parameters
    ----------
    value : int, optional
        Explicit value (for example from ``--workers``); overrides ``HEXMOD_WORKERS``.

    Returns
    -------
    int
        At least 1.
    """
    if value is None:
        env = os.environ.get("HEXMOD_WORKERS")
        value = int(env) if env else DEFAULT_WORKERS
    value = int(value)
    if value < 1:
        raise ValueError(f"the number of workers must be at least 1, got {value}")
    return value


def thread_env() -> Dict[str, str]:
    """Return environment settings that keep numerical libraries single-threaded per worker."""
    return {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def limit_threads() -> None:
    """Set single-thread defaults for numerical libraries unless the user has set them.

    Call this before numpy is imported (the step scripts do so first thing). The numerical
    libraries read these variables once, when they are loaded; worker processes started by
    ``hexmod.parallel.map_ordered`` inherit them, so code that uses the library from an
    interactive session should call this function before importing numpy, or each of N workers
    may start N threads.
    """
    for name, value in thread_env().items():
        os.environ.setdefault(name, value)
