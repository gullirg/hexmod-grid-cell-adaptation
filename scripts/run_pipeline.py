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
to the results folder ``HEXMOD_RESULTS`` (default ``<repo>/results``). The run stops at the first
step that fails.

Step arguments: the steps that use parallel workers receive ``--workers N``; ``--quick`` is passed
on to the steps that accept it (see ``STEPS`` below).
"""

from __future__ import annotations

import argparse
import importlib.util
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

# id, script, takes --workers, takes --quick, short description
STEPS = [
    ("D1", "data_download.py", False, False, "download the six datasets"),
    ("D2", "data_check.py", False, False, "verify the data"),
    ("01", "step01_preprocess.py", True, False, "preprocess raw recordings"),
    ("02", "step02_cohort.py", True, False, "grid cell cohort and characterisation"),
    ("03", "step03_adaptation_parametric.py", True, False, "parametric adaptation and HexMod fits"),
    ("04", "step04_adaptation_nonparametric.py", True, False, "firing rate by time since last visit"),
    ("05", "step05_prospective_coding.py", True, False, "prospective coding control"),
    ("06", "step06_visit_sequences.py", True, False, "field visit sequences (Fig 3F)"),
    ("07", "step07_adaptation_timescales.py", True, False, "adaptation timescales (Supp Fig 4)"),
    ("08", "step08_methods_numbers.py", True, False, "numbers quoted in the Methods"),
    ("09", "step09_same_cell_links.py", True, False, "repeated recordings of the same cell"),
    ("10", "step10_conjunctive_cells.py", False, False, "conjunctive cells (Fig 2B to 2E)"),
    ("11", "step11_example_cells.py", False, False, "example cells"),
    ("12", "step12_rat_statistics.py", False, False, "rat statistics and figure tables"),
    ("21", "step21_simulation_pools.py", False, False, "simulated grid cell pools"),
    ("22", "step22_simulation_maps.py", False, False, "simulated maps and trajectory"),
    ("23", "step23_simulation_proximity.py", False, False, "proximity of simulated cells (Fig 6D)"),
    ("24", "step24_simulation_participants.py", True, True, "simulated participants (Fig 7)"),
    ("25", "step25_simulation_collect.py", False, True, "Fig 7 values"),
    ("26", "step26_speed_sweep.py", True, False, "running-speed sweep (Fig 8D)"),
    ("27", "step27_simulation_statistics.py", False, False, "simulation statistics"),
    ("30", "step30_figures.py", False, False, "figures"),
    ("40", "step40_check_statistics.py", False, False, "check every statistic"),
]
IDS = [s[0] for s in STEPS]


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
        if sid == "24" and not (args.full_sim or args.quick):
            continue
        chosen.append(step)
    return chosen


def step_argv(step: tuple, args) -> List[str]:
    sid, _, takes_workers, takes_quick, _ = step
    argv = []
    if takes_workers:
        argv += ["--workers", str(config.workers(args.workers))]
    if args.quick and takes_quick:
        argv.append("--quick")
    if sid == "D1":
        argv = ["--datasets", "all", "--data", str(config.data_dir())]
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
    p.add_argument("--full-sim", action="store_true",
                   help="run the full Fig 7 simulation (about 6 h with 10 workers, 62 worker hours)")
    p.add_argument("--download", action="store_true", help="also run D1 (download the six datasets)")
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.list:
        for sid, script, *_rest, desc in STEPS:
            print(f"{sid:>3}  {script:<38} {desc}")
        return 0
    print(f"results: {config.results_dir()}; workers: {config.workers(args.workers)}")
    failures = []
    t_all = time.perf_counter()
    for step in select_steps(args):
        sid, script = step[0], step[1]
        path = HERE / script
        argv_step = step_argv(step, args)
        print(f"[{sid}] {script} {' '.join(argv_step)}", flush=True)
        t0 = time.perf_counter()
        try:
            rc = load_main(path)(argv_step)
        except SystemExit as exc:          # argparse errors and explicit exits
            if exc.code is not None and not isinstance(exc.code, int):
                print(f"[{sid}] {exc.code}", flush=True)
            rc = exc.code if isinstance(exc.code, int) or exc.code is None else 1
        except Exception as exc:           # report and stop
            print(f"[{sid}] failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            rc = 1
        dt = time.perf_counter() - t0
        if rc not in (0, None):
            failures.append(sid)
            print(f"[{sid}] exit code {rc} after {dt:.1f} s", flush=True)
            break
        print(f"[{sid}] done in {dt:.1f} s", flush=True)
    print(f"total {time.perf_counter() - t_all:.1f} s; failed: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
