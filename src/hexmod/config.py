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
``HEXMOD_FROZEN``
    Optional folder with reference results in the layout of ``results/``; used only by tests
    marked ``frozen`` and by ``step40_check_statistics.py --compare-frozen``.

A smoke run (``--smoke``) writes synthetic results. Its results folder is marked with the file
``SMOKE_MARKER``, and a step run without ``--smoke`` refuses a marked folder, so that synthetic
results never mix with results of the recorded data (``guard_smoke_results``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional


class ResultsFolderError(SystemExit, RuntimeError):
    """The results folder cannot be used by this run (it holds results of the other kind of run).

    It derives from ``SystemExit`` so that a step started from the command line stops with this
    one-line message (exit status 1) instead of a traceback, and from ``RuntimeError`` so that
    library code and tests can catch it as an ordinary error.
    """


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


def frozen_dir() -> Optional[Path]:
    """Return ``HEXMOD_FROZEN`` as a path when it is set and exists, else None."""
    value = os.environ.get("HEXMOD_FROZEN")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_dir() else None


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


#: File that marks a results folder written by a smoke run.
SMOKE_MARKER = ".smoke_run"


def _non_smoke_record(res: Path) -> Optional[str]:
    """Name of the first run record in ``res/logs`` written without ``--smoke``, or None."""
    for path in sorted((res / "logs").glob("*.json")):
        try:
            argv = json.loads(path.read_text(encoding="utf-8")).get("argv") or []
        except (OSError, ValueError, AttributeError):  # unreadable record: treat as not a smoke run
            argv = []
        if "--smoke" not in argv:
            return f"logs/{path.name}"
    return None


def guard_smoke_results(smoke: bool) -> None:
    """Keep the results of smoke runs and of runs on the recorded data apart.

    A smoke step marks the results folder with ``SMOKE_MARKER`` and refuses a folder that holds run
    records of steps run without ``--smoke``; any other step refuses a marked folder.

    Raises
    ------
    ResultsFolderError
        With a one-line message, when the folder belongs to the other kind of run.
    """
    res = results_dir()
    marker = res / SMOKE_MARKER
    if smoke:
        if marker.is_file():
            return
        found = _non_smoke_record(res)
        if found is not None:
            raise ResultsFolderError(f"error: {res} holds results of a run that was not a smoke run ({found}); a "
                                     "smoke run would overwrite them. Point HEXMOD_RESULTS to another folder.")
        res.mkdir(parents=True, exist_ok=True)
        marker.write_text("results of a smoke run on synthetic inputs\n", encoding="utf-8")
    elif marker.is_file():
        raise ResultsFolderError(f"error: {res} holds results of a smoke run on synthetic inputs ({SMOKE_MARKER}); "
                                 "run without --smoke in another results folder (HEXMOD_RESULTS) or delete this one.")


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
