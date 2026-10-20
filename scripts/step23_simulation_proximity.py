"""Step 23: proximity of the previous field visit for runs aligned and misaligned with the grid (Fig 6D).

Example::

    python scripts/step23_simulation_proximity.py

For each module the 10,000 cells of participant 0 (``default_rng(0).choice(100000, 10000)``; pools of
step 21) are simulated along the trajectory of Fig 6A (``hexmod.sim.trajectory_v2``, seed 0) with
adaptation 0.1, time constant 2 s and no speed gain (``hexmod.sim.simulate.simulate_grid_cells``, in
chunks of 1,000 cells; no random draw). For every cell the proximity ``exp(-t_prev / 2 s)`` inside
its fields is averaged over samples moving within 15 deg of a grid axis (``aligned``,
cos 6 theta > 0) and further from the axes (``misaligned``).

Writes ``sim/fig6/fig6d_small.csv`` and ``fig6d_large.csv``: ``cell_index`` (pool row), ``aligned``,
``misaligned``; 10,000 rows in draw order. About 45 s per module on one core.
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

STEP = "step23_simulation_proximity"
MODULES = ("small", "large")
N_CELLS = 10000
CHUNK = 1000
PARTICIPANT = 0
ALPHA = 0.1


def load_pool(module: str, rec: Optional[RunRecord] = None):
    """Pool of a module written by step 21 (``hexmod.sim.io_pool.load_module_pool``)."""
    from hexmod.sim import io_pool

    table, path = io_pool.load_module_pool(config.results_dir(), module)
    if rec is not None:
        rec.add_input(path)
    return table


def proximity_table(pool_table, module: str, chunk: int = CHUNK) -> pd.DataFrame:
    """Per-cell aligned and misaligned proximity of the cells of participant 0 of a module."""
    from hexmod.sim import pool, simulate, trajectory_v2

    par_small = pool.module_param("small", adaptEff_mean=ALPHA, speedGamma_mean=0)
    par = pool.module_param(module, adaptEff_mean=ALPHA, speedGamma_mean=0)
    traj = trajectory_v2.make_trajectory(par_small, base_seed=PARTICIPANT)
    n_draw = min(N_CELLS, len(pool_table))
    cells = np.random.default_rng(PARTICIPANT).choice(len(pool_table), n_draw, replace=False)
    g = pool_table.loc[cells]
    res = []
    for c0 in range(0, len(g), chunk):
        o = simulate.simulate_grid_cells({"gcTable": g.iloc[c0:c0 + chunk], "traj": traj, "par": par})
        res.append(np.asarray(o["tprevbyAlignMis"], float))
        del o
    x = np.vstack(res)
    return pd.DataFrame({"cell_index": cells, "aligned": x[:, 0], "misaligned": x[:, 1]})


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    parse_args(argv)
    out = config.results_dir() / "sim" / "fig6"
    with RunRecord(STEP, argv=argv) as rec:
        for m in MODULES:
            table = proximity_table(load_pool(m, rec), m)
            hio.write_csv(table, out / f"fig6d_{m}.csv")
            rec.add_output(out / f"fig6d_{m}.csv")
            rec.note(f"{m}_mean_aligned", float(table["aligned"].to_numpy().mean()))
            rec.note(f"{m}_mean_misaligned", float(table["misaligned"].to_numpy().mean()))
    print(f"proximity tables written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
