"""Step 26: firing rate and proximity against running speed (Fig 8D).

Example::

    python scripts/step26_speed_sweep.py --workers 10

Three modules of 1,024 perfect grid cells (grid scales 30, 50 and 70 vm;
``hexmod.sim.pool.set_gridcells_pool`` with ``gridtype="perfectGrid"``, base seed 0) are simulated with
adaptation 0.1 (time constant 2 s) along straight wall-to-wall runs (``hexmod.sim.trajectory_v2``,
``moveConstraint="wall2wall"``, runs of 70 vm, no pauses, four blocks of 600 s; worker function
``hexmod.sim.speed_sweep.sweep_run``) at 37 running speeds
(5 to 149 vm/s in steps of 4; the trajectory of speed number i has seed i). Samples faster than
1 vm/s are analysed. For each of the 111 runs:

- ``FRby12dir_0`` to ``FRby12dir_330``: the population rate (mean over cells) in 12 bins of running
  direction relative to the grid (``hexmod.sim.simulate.calc_hm``, all samples);
- ``FR_align``, ``FR_misalign``: the mean of the even (within 15 deg of a grid axis) and the odd
  bins; ``FR_mis_minus_align_pct``: misaligned minus aligned rate in percent of their mean;
- ``proximity_align``, ``proximity_misalign``: the in-field proximity ``exp(-t_prev / 2 s)`` for
  aligned and misaligned running, averaged over cells; ``proximity_align_minus_mis``;
- ``n_time``: samples of the trajectory; ``plotted``: the run is drawn in Fig 8D (every second speed).

Writes ``sim/fig8/fig8d_sweep.csv`` (111 rows, scales in the order 30, 50, 70 vm, speeds ascending).
About 35 worker minutes; runs are independent, so the result does not depend on the number of
workers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hexmod import io as hio  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402
from hexmod.sim.speed_sweep import sweep_run  # noqa: E402  (worker function, importable by spawned workers)

STEP = "step26_speed_sweep"
#: Grid scale (vm) of each module, in the order of the table.
SCALES = {"small": 30, "medium": 50, "large": 70}
N_CELLS = 1024
SPEEDS = np.arange(5, 150, 4)
ALPHA = 0.1
SPEED_THRESHOLD = 1.0


def sweep_inputs() -> list:
    """One work item per run: (run description, cells of the module, parameters of the module)."""
    from hexmod.sim.pool import set_gridcells_pool, set_param

    items = []
    for scale_level, (name, scale) in enumerate(SCALES.items()):
        par = set_param(dict(gridScale=scale, gridtype="perfectGrid"))
        cells = set_gridcells_pool(par, n_cells=N_CELLS, base_seed=0)
        par["speedThres"] = SPEED_THRESHOLD
        par["adaptEff_mean"] = ALPHA
        for speed_level in range(len(SPEEDS)):
            run = dict(scale=scale, scaleLevel=scale_level, speedLevel=int(speed_level),
                       moveSpeed=float(SPEEDS[speed_level]))
            items.append((run, cells, par))
    return items


def sweep_table(results: list) -> pd.DataFrame:
    """The Fig 8D table from the run results (in run order)."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "FRby12dir"} for r in results])
    rates = np.array([r["FRby12dir"] for r in results])
    for j in range(12):
        df[f"FRby12dir_{30 * j}"] = rates[:, j]
    df["FR_mis_minus_align_pct"] = (-(df["FR_align"] - df["FR_misalign"])
                                    / df[["FR_align", "FR_misalign"]].mean(axis=1) * 100)
    df["proximity_align_minus_mis"] = df["proximity_align"] - df["proximity_misalign"]
    df["plotted"] = df["speedLevel"] % 2 == 0  # Fig 8D draws every second speed
    return df


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workers = config.workers(args.workers)
    from hexmod.parallel import map_ordered

    out = config.results_dir() / "sim" / "fig8" / "fig8d_sweep.csv"
    with RunRecord(STEP, workers=workers, argv=argv) as rec:
        items = sweep_inputs()
        results = map_ordered(sweep_run, items, workers=workers, desc="runs")
        table = sweep_table(results)
        hio.write_csv(table, out)
        rec.add_output(out)
        rec.note("n_runs", int(len(table)))
    print(f"speed sweep written to {out} ({len(table)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
