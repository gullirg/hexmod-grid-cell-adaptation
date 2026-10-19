"""Run the analysis pipeline in order.

Each step is a script ``scripts/stepNN_<name>.py`` (or a data script) with a ``main(argv)``
function; see docs/PIPELINE.md for what every step reads and writes. Examples::

    python scripts/run_pipeline.py --workers 4                  # D2 to 40
    python scripts/run_pipeline.py --from 02 --to 12
    python scripts/run_pipeline.py --full-sim --workers 10       # includes the Fig 7 run (about 6 h)
    python scripts/run_pipeline.py --quick --workers 10          # Fig 7A to 7E from participants 0 to 29
    python scripts/run_pipeline.py --list

By default the data download (D1) and the full simulation of Fig 7 (step 24) are skipped; step
25 then uses the shipped results in ``results_precomputed/``. ``--quick`` simulates participants 0
to 29 of both modules (the participants of Fig 7A to 7E) and step 25 takes those panels from them;
Fig 7F, which needs participants 30 to 1,499, then comes from the shipped results. All steps write
to the results folder ``HEXMOD_RESULTS`` (default ``<repo>/results``).

``--smoke`` runs the pipeline on small synthetic inputs, in a temporary results folder unless
``HEXMOD_RESULTS`` is set (about a minute on one core, no data needed; the folder is
printed on the first line and is not removed afterwards):

- D1 and D2 are skipped;
- step 01 is replaced by synthetic recordings made with the simulator (``hexmod.synthetic``: two
  sessions, 22 grid cells with known adaptation);
- step 09 is skipped, because it reads the raw files of Vollan et al. (2025); every synthetic cell
  is taken as a unit of its own (``hexmod.synthetic.write_separate_units``);
- the other steps run with their ``--smoke`` sizes (a 500-cell pool, 2 participants of 200 cells,
  a 3-speed sweep), and step 30 draws every figure from the smoke results and
  ``results_precomputed/`` (``--strict``: a figure that cannot be drawn fails the run);
- step 40 writes the statistics report and does not fail on differences from the published values,
  which synthetic results cannot match.

Workers default to 1 in a smoke run. The run stops before any work when the results folder already
holds results of a run that was not a smoke run (preprocessed recordings that are not synthetic, or a run
record in ``logs/`` written without ``--smoke``). The smoke results folder is marked
(``hexmod.config.SMOKE_MARKER``), and steps run later without ``--smoke`` refuse it.

Step arguments: the steps that use parallel workers receive ``--workers N``; ``--quick``, ``--smoke``
and ``--compare-frozen DIR`` are passed on to the steps that accept them (see ``STEPS`` below).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))  # step modules importable by name in worker processes

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

# id, script, takes --workers, takes --quick, takes --smoke, short description
STEPS = [
    ("D1", "data_download.py", False, False, False, "download the six datasets"),
    ("D2", "data_check.py", False, False, False, "verify the data"),
    ("01", "step01_preprocess.py", True, False, True, "preprocess raw recordings"),
    ("02", "step02_cohort.py", True, False, True, "grid cell cohort and characterisation"),
    ("03", "step03_adaptation_parametric.py", True, False, True, "parametric adaptation and HexMod fits"),
    ("04", "step04_adaptation_nonparametric.py", True, False, True, "firing rate by time since last visit"),
    ("05", "step05_prospective_coding.py", True, False, True, "prospective coding control"),
    ("06", "step06_visit_sequences.py", True, False, True, "field visit sequences (Fig 3F)"),
    ("07", "step07_adaptation_timescales.py", True, False, True, "adaptation timescales (Supp Fig 4)"),
    ("08", "step08_methods_numbers.py", True, False, True, "numbers quoted in the Methods"),
    ("09", "step09_same_cell_links.py", True, False, True, "repeated recordings of the same cell"),
    ("10", "step10_conjunctive_cells.py", True, False, True, "conjunctive cells (Fig 2B to 2E)"),
    ("11", "step11_example_cells.py", False, False, True, "example cells"),
    ("12", "step12_rat_statistics.py", False, False, True, "rat statistics and figure tables"),
    ("21", "step21_simulation_pools.py", False, False, True, "simulated grid cell pools"),
    ("22", "step22_simulation_maps.py", False, False, True, "simulated maps and trajectory"),
    ("23", "step23_simulation_proximity.py", False, False, True, "proximity of simulated cells (Fig 6D)"),
    ("24", "step24_simulation_participants.py", True, True, True, "simulated participants (Fig 7)"),
    ("25", "step25_simulation_collect.py", False, True, True, "Fig 7 values"),
    ("26", "step26_speed_sweep.py", True, False, True, "running-speed sweep (Fig 8D)"),
    ("27", "step27_simulation_statistics.py", False, False, True, "simulation statistics"),
    ("30", "step30_figures.py", False, False, True, "figures"),
    ("40", "step40_check_statistics.py", False, False, True, "check every statistic"),
]
IDS = [s[0] for s in STEPS]

#: Steps that the smoke run replaces, and why.
SMOKE_SUBSTITUTES = {
    "01": "replaced by synthetic recordings made with the simulator (hexmod.synthetic)",
    "09": "skipped in the smoke run: it reads the raw files of Vollan et al. (2025); every synthetic cell is "
          "taken as a unit of its own (rat/same_cell/entries.csv from hexmod.synthetic)",
}
#: Run records of the smoke substitutes (logs/<name>.json).
SMOKE_RECORDS = {"01": "smoke_recordings", "09": "smoke_separate_units"}


def _index(step_id: str) -> int:
    key = step_id.upper().zfill(2) if step_id[:1].isdigit() else step_id.upper()
    if key not in IDS:
        raise SystemExit(f"unknown step {step_id!r}; steps are {', '.join(IDS)}")
    return IDS.index(key)


def load_main(script: Path):
    """Import a step script and return its ``main`` function."""
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[script.stem] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError(f"{script.name} has no main(argv) function")
    return module.main


def select_steps(args) -> List[tuple]:
    first = _index(args.from_step) if args.from_step else _index("D2")
    last = _index(args.to_step) if args.to_step else len(STEPS) - 1
    chosen = []
    for i, step in enumerate(STEPS):
        sid = step[0]
        if i < first or i > last:
            continue
        if sid == "D1" and not args.download and not (args.from_step or "").upper() == "D1":
            continue
        if sid in ("D1", "D2") and args.smoke:  # the smoke run uses synthetic inputs, no data
            continue
        if sid == "24" and not (args.full_sim or args.quick or args.smoke):
            continue
        chosen.append(step)
    return chosen


def n_workers(args) -> int:
    """Worker processes of the run: ``--workers``, else 1 in a smoke run, else ``HEXMOD_WORKERS`` or 4."""
    if args.smoke and args.workers is None:
        return 1
    return config.workers(args.workers)


def run_smoke_substitute(sid: str) -> None:
    """Run the smoke-run replacement of step ``sid`` (see ``SMOKE_SUBSTITUTES``)."""
    from hexmod import synthetic
    from hexmod.runlog import RunRecord

    results = config.results_dir()
    with RunRecord(SMOKE_RECORDS[sid], workers=1, argv=["--smoke"]) as rec:
        if sid == "01":
            paths = synthetic.write_recordings(results, record=rec)
        else:
            paths = synthetic.write_separate_units(results, record=rec)
    for p in paths:
        print(f"     wrote {p}", flush=True)


def smoke_guard() -> Optional[str]:
    """Return a message when the smoke run would overwrite results of a run that was not a smoke run."""
    res = config.results_dir()
    found = None
    if (res / "preprocessed" / "trajectories.pkl").exists() and not (res / "logs" / "smoke_recordings.json").exists():
        found = "preprocessed/trajectories.pkl"
    else:
        for path in sorted((res / "logs").glob("*.json")):
            try:
                argv = json.loads(path.read_text(encoding="utf-8")).get("argv") or []
            except (OSError, ValueError, AttributeError):  # unreadable record: treat as not a smoke run
                argv = []
            if "--smoke" not in argv:
                found = f"logs/{path.name}"
                break
    if found is None:
        return None
    return (f"{res} holds results of a run that was not a smoke run ({found}); the smoke run would overwrite "
            "them. Unset HEXMOD_RESULTS or point it to another folder.")


def step_argv(step: tuple, args) -> List[str]:
    sid, _, takes_workers, takes_quick, takes_smoke, _ = step
    argv = []
    if takes_workers:
        argv += ["--workers", str(n_workers(args))]
    if args.quick and takes_quick:
        argv.append("--quick")
    if args.smoke and takes_smoke:
        argv.append("--smoke")
    if args.smoke and sid == "30":
        argv.append("--strict")  # every figure must be drawn from the smoke results
    if sid == "40" and args.compare_frozen:
        argv += ["--compare-frozen", str(args.compare_frozen)]
    if sid == "D1":
        argv = ["--datasets", "all", "--dest", str(config.data_dir())]
    if sid == "D2":
        argv = []
    return argv


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    p.add_argument("--from", dest="from_step", default=None, help="first step id (for example 02)")
    p.add_argument("--to", dest="to_step", default=None, help="last step id (for example 12)")
    p.add_argument("--quick", action="store_true",
                   help="simulate participants 0 to 29 (Fig 7A to 7E, about 1.2 worker hours); Fig 7F from the "
                        "shipped results")
    p.add_argument("--smoke", action="store_true",
                   help="synthetic inputs, temporary results folder unless HEXMOD_RESULTS is set "
                        "(about a minute on one core)")
    p.add_argument("--full-sim", action="store_true",
                   help="run the full Fig 7 simulation (about 6 h with 10 workers, 62 worker hours)")
    p.add_argument("--download", action="store_true", help="also run D1 (download the six datasets)")
    p.add_argument("--compare-frozen", default=None, metavar="DIR",
                   help="reference results folder for step 40 (layout of results/)")
    p.add_argument("--keep-going", action="store_true", help="continue after a failing step")
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.list:
        for sid, script, *_rest, desc in STEPS:
            present = "" if (HERE / script).is_file() else "   (not yet available)"
            print(f"{sid:>3}  {script:<38} {desc}{present}")
        return 0
    if args.smoke and not os.environ.get("HEXMOD_RESULTS"):
        import tempfile

        os.environ["HEXMOD_RESULTS"] = tempfile.mkdtemp(prefix="hexmod_smoke_")
        print(f"smoke run: results in {os.environ['HEXMOD_RESULTS']}")
    if args.smoke:
        problem = smoke_guard()
        if problem:
            print(problem, file=sys.stderr)
            return 2
    try:
        config.guard_smoke_results(smoke=args.smoke)  # a smoke run marks the folder as holding smoke results
    except config.ResultsFolderError as exc:      # one-line message, no traceback
        print(exc.code, file=sys.stderr)
        return 2
    print(f"results: {config.results_dir()}; workers: {n_workers(args)}")
    failures, missing = [], []
    t_all = time.perf_counter()
    for step in select_steps(args):
        sid, script = step[0], step[1]
        path = HERE / script
        if args.smoke and sid in SMOKE_SUBSTITUTES:
            print(f"[{sid}] {script}: {SMOKE_SUBSTITUTES[sid]}", flush=True)
            t0 = time.perf_counter()
            try:
                run_smoke_substitute(sid)
            except Exception as exc:
                print(f"[{sid}] failed: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                failures.append(sid)
                if not args.keep_going:
                    break
                continue
            print(f"[{sid}] done in {time.perf_counter() - t0:.1f} s", flush=True)
            continue
        if not path.is_file():
            print(f"[{sid}] {script}: not available, skipped")
            missing.append(sid)
            continue
        argv_step = step_argv(step, args)
        print(f"[{sid}] {script} {' '.join(argv_step)}", flush=True)
        t0 = time.perf_counter()
        try:
            rc = load_main(path)(argv_step)
        except SystemExit as exc:          # argparse errors, results-folder errors and explicit exits
            if exc.code is not None and not isinstance(exc.code, int):
                print(f"[{sid}] {exc.code}", flush=True)
            rc = exc.code if isinstance(exc.code, int) or exc.code is None else 1
        except Exception as exc:           # report and stop (or continue with --keep-going)
            print(f"[{sid}] failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            rc = 1
        dt = time.perf_counter() - t0
        if rc not in (0, None):
            failures.append(sid)
            print(f"[{sid}] exit code {rc} after {dt:.1f} s", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"[{sid}] done in {dt:.1f} s", flush=True)
    print(f"total {time.perf_counter() - t_all:.1f} s; failed: {failures or 'none'}; "
          f"not available: {missing or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
