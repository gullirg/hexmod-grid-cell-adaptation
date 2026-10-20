"""Every figure module builds from the smoke results and the shipped simulation results.

The smoke run (``run_pipeline.py --smoke``, shared with ``test_pipeline_smoke.py``) draws all twelve figures
with ``step30_figures.py --smoke --strict`` from its synthetic results, falling back on
``results_precomputed/`` for the simulation results it does not compute. The first test checks the output of
each figure. The second draws the simulation figures (Figs 6, 7 and 8) from ``results_precomputed/`` alone,
into an empty results folder, as a reader without the recorded data would.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hexmod.figures import FIGURES

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[1]
SIMULATION_FIGURES = ("Fig6", "Fig7", "Fig8")


def _helper():
    name = "hexmod_tests_smoke_run"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / "fixtures" / "smoke_run.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _step30():
    name = "hexmod_tests_step30_figures"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "step30_figures.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(scope="session")
def res(tmp_path_factory) -> Path:
    run = _helper().smoke_run(tmp_path_factory)
    if run["returncode"] != 0:
        pytest.fail(f"the smoke run failed (exit status {run['returncode']}):\n{_helper().log_tail(run)}")
    return run["results"]


def check_figure_folder(folder: Path) -> None:
    """Assert that a figure folder holds a complete, readable build."""
    from hexmod import io

    assert folder.is_dir(), f"{folder.name} was not drawn"
    problems = _step30().output_problems(folder)
    assert not problems, f"{folder.name}: {problems}"
    for pdf in folder.glob("*.pdf"):
        assert pdf.read_bytes()[:5] == b"%PDF-"
        assert (folder / (pdf.stem + ".png")).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert b"<svg" in (folder / (pdf.stem + ".svg")).read_bytes()[:2000]
    positions = io.read_csv(folder / "positions.csv")
    assert (positions["element"] == "axes").any(), f"{folder.name}: no axes in positions.csv"
    tables = sorted((folder / "source_data").glob("*.csv"))
    assert tables, f"{folder.name}: no source data"
    rows = 0
    for path in tables:
        assert path.name.startswith(folder.name), f"source data {path.name} not named after {folder.name}"
        rows += len(io.read_csv(path))
    assert rows > 0, f"{folder.name}: all source data tables are empty"


@pytest.mark.parametrize("figure", FIGURES)
def test_figure_builds_from_smoke_results(res, figure):
    from hexmod import io

    record = io.read_json(res / "logs" / "step30_figures.json")
    status = record["notes"]["figures"][figure]
    assert status["status"] == "ok", f"{figure}: {status}"
    check_figure_folder(res / "figures" / figure)


def test_simulation_figures_build_from_precomputed_results(tmp_path):
    env = dict(os.environ, HEXMOD_RESULTS=str(tmp_path), MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    cmd = [sys.executable, str(REPO / "scripts" / "step30_figures.py"), "--precomputed", "--strict", "--figure",
           ",".join(SIMULATION_FIGURES)]
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    assert proc.returncode == 0, proc.stdout[-4000:]
    for figure in SIMULATION_FIGURES:
        found = [p for p in tmp_path.rglob(f"figures/{figure}") if p.is_dir()]
        assert len(found) == 1, f"{figure}: figure folders {found}"
        check_figure_folder(found[0])
    assert not list(tmp_path.rglob("rat")), "a figure step wrote results of the rat analysis"
