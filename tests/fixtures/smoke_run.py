"""One smoke run of the pipeline per test session, shared by the smoke tests.

``tests/test_pipeline_smoke.py``, ``tests/test_figures_smoke.py`` and ``tests/test_plotted_vs_stats.py`` load
this file by path under the name ``MODULE_NAME`` (so that they share its state) and call :func:`smoke_run`,
which runs ``scripts/run_pipeline.py --smoke`` once, in a separate process, with ``HEXMOD_RESULTS`` set to a
temporary folder, and returns where the results are. Set ``HEXMOD_SMOKE_RESULTS`` to the results folder of an
earlier smoke run to check those results without running the pipeline again.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE_NAME = "hexmod_tests_smoke_run"
_STATE: dict = {}


def smoke_run(tmp_path_factory) -> dict:
    """Run the smoke pipeline once and return its outcome.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Source of the temporary folder for the results.

    Returns
    -------
    dict
        ``results`` (the results folder), ``returncode``, ``runtime_s``, ``log`` (the output of
        the run) and ``reused`` (True when ``HEXMOD_SMOKE_RESULTS`` was used instead of a run).
    """
    if "run" in _STATE:
        return _STATE["run"]
    reuse = os.environ.get("HEXMOD_SMOKE_RESULTS")
    if reuse:
        _STATE["run"] = {"results": Path(reuse).expanduser().resolve(), "returncode": 0, "runtime_s": float("nan"),
                         "log": "", "reused": True}
        return _STATE["run"]
    base = tmp_path_factory.mktemp("smoke_run")
    env = dict(os.environ)
    env.update(HEXMOD_RESULTS=str(base / "results"), MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    cmd = [sys.executable, str(REPO / "scripts" / "run_pipeline.py"), "--smoke"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(base), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    runtime = time.perf_counter() - t0
    (base / "smoke_run.log").write_text(proc.stdout, encoding="utf-8")
    _STATE["run"] = {"results": base / "results", "returncode": proc.returncode, "runtime_s": runtime,
                     "log": proc.stdout, "reused": False}
    return _STATE["run"]


def log_tail(run: dict, n: int = 60) -> str:
    """Last ``n`` lines of the output of the smoke run (for assertion messages)."""
    return "\n".join(run["log"].splitlines()[-n:])
