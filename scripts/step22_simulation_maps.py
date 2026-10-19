"""Step 22: simulated trajectory and rate maps of Fig 6A to 6C, Supp Fig 3E and Fig 8A.

Example::

    python scripts/step22_simulation_maps.py

Reads the pools of step 21 (``sim/pools/``; the large pool falls back to the shipped
``inputs/simulation/large_module_pool_70cm.npz``) and writes:

- ``sim/fig6/trajectory_seed0.csv`` and ``trajectory_seed0_objects.json`` (Fig 6A): the trajectory
  of participant 0 (``hexmod.sim.trajectory_v2``, small-module parameters, seed 0; 23,990 samples
  at 10 Hz) and the object locations of its four blocks;
- ``sim/fig6/maps.npz`` (Fig 6B, 6C, Supp Fig 3E):
  - six example rate maps per module, cells ``default_rng(1).choice(100000, 10000)[:6]``;
  - the summed rate maps of the 10,000 cells of participants 0 and 1
    (``default_rng(seed).choice(100000, 10000)``), in percent of their mean;
  - the Supp Fig 3E maps: per module one cell per panel drawn with
    ``DataFrame.sample(n=1, random_state=6 to 11)`` from the cells with gridness in (0.5, 0.7]
    (panels 1, 2), (0.7, 0.9] (panels 3, 4) and (0.9, 1.5] (panels 5, 6), with their pool index,
    gridness and peak rate;
- ``sim/fig8/fig8a_ideal_grids.npz`` (Fig 8A): field centres and delineated fields of perfect grids
  of 30, 50 and 70 vm (``set_gridcells_pool`` with ``gridtype="perfectGrid"``, 2 vm bins, cell 0).

Rate maps are indexed [x, y] (first axis x), 34 x 34 bins of 3 vm over the 100 x 100 vm
environment (``extent`` = [0, 100, 0, 100]). The titles of the Supp Fig 3E maps (gridness with one
decimal and the peak rate, ``supp3e_title``) are listed in the run record. Nothing here is random
beyond the seeds above. ``--smoke`` uses the smoke pools of step 21 as they are.
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

from hexmod import io as hio  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step22_simulation_maps"
MODULES = ("small", "large")
N_EXAMPLES = 6
N_PARTICIPANT_CELLS = 10000
SUMMED_SEEDS = (0, 1)
EXAMPLE_SEED = 1
#: Gridness classes of the six Supp Fig 3E panels and the gridness interval of each class.
SUPP3E_CLASSES = ("low", "low", "medium", "medium", "high", "high")
SUPP3E_INTERVALS = {"low": (0.5, 0.7), "medium": (0.7, 0.9), "high": (0.9, 1.5)}
#: Panel p (1 to 6) draws its cell with ``random_state = SUPP3E_RANDOM_STATE + p``.
SUPP3E_RANDOM_STATE = 5
#: Grid scales (vm) of the perfect grids of Fig 8A.
FIG8A_SCALES = {"small": 30, "medium": 50, "large": 70}
TRAJECTORY_COLUMNS = ["time", "x", "y", "moveSpeed", "moveDir", "trialId", "expPhase"]


def load_pool(module: str, rec: Optional[RunRecord] = None, smoke: bool = False):
    """Pool of a module written by step 21 (``hexmod.sim.io_pool.load_module_pool``)."""
    from hexmod.sim import io_pool

    table, path = io_pool.load_module_pool(config.results_dir(), module, smoke=smoke)
    if rec is not None:
        rec.add_input(path)
    return table


def supp3e_title(gridness: float, peak_rate_hz: float) -> str:
    """Title of a Supp Fig 3E map: gridness with one decimal and the peak rate in Hz.

    The peak rate is a whole number, or one decimal when it is below 1 Hz (so that no map reads
    "0 Hz"); for example ``"0.6, 3 Hz"`` and ``"0.7, 0.5 Hz"``.
    """
    if peak_rate_hz >= 1:
        return f"{gridness:.1f}, {peak_rate_hz:.0f} Hz"
    return f"{gridness:.1f}, {peak_rate_hz:.1f} Hz"


def example_maps(pool_table, par, cells):
    """Rate maps (len(cells), x bins, y bins) of the given pool rows."""
    from hexmod.sim.pool import ratemap_from_centers

    maps = []
    for c in cells:
        cell_data = pool_table.loc[c]
        maps.append(ratemap_from_centers(cell_data["centers"], cell_data["peakFR_jitter"], par))
    return np.array(maps)


def summed_map_pct(pool_table, par, cells):
    """Sum of the rate maps of ``cells`` in percent of its mean (100 = mean)."""
    from hexmod.sim.pool import ratemap_from_centers

    g = pool_table.loc[cells]
    centers = g["centers"].to_numpy()
    peaks = g["peakFR_jitter"].to_numpy()
    ratemap_sum = np.zeros_like(ratemap_from_centers(centers[0], peaks[0], par))
    for i in range(len(centers)):
        ratemap_sum += ratemap_from_centers(centers[i], peaks[i], par)
    return ratemap_sum / np.mean(ratemap_sum) * 100


def supp3e_cells(pool_table, par):
    """The six Supp Fig 3E cells of a pool: pool indices, gridness, peak rates and maps."""
    from hexmod.sim.pool import ratemap_from_centers

    cells, gridness, peaks, maps = [], [], [], []
    for panel, cls in enumerate(SUPP3E_CLASSES, start=1):
        lo, hi = SUPP3E_INTERVALS[cls]
        in_class = (pool_table["gridness"] > lo) & (pool_table["gridness"] <= hi)
        cell_data = pool_table[in_class].reset_index().sample(n=1, random_state=SUPP3E_RANDOM_STATE + panel).iloc[0]
        rate_map = ratemap_from_centers(cell_data["centers"], cell_data["peakFR_jitter"], par)
        cells.append(int(cell_data["index"]))
        gridness.append(float(cell_data["gridness"]))
        peaks.append(float(rate_map.max()))
        maps.append(rate_map)
    return (np.array(cells, dtype=np.int64), np.array(gridness, dtype=np.float64), np.array(peaks, dtype=np.float64),
            np.stack(maps))


def ideal_grids():
    """Field centres and delineated fields (x, y, radius) of the perfect grids of Fig 8A."""
    from hexmod.sim.pool import set_gridcells_pool, set_param

    out = {}
    for name, scale in FIG8A_SCALES.items():
        par = set_param(dict(gridScale=scale, gridtype="perfectGrid", binSize_ratemap=2))
        cell = set_gridcells_pool(par, n_cells=4, base_seed=0).loc[0]
        out[f"{name}__field_locations"] = np.asarray(np.c_[cell["peaks_x"], cell["peaks_y"], cell["radius"]], float)
        out[f"{name}__centers"] = np.asarray(cell["centers"], float)
        out[f"{name}__grid_scale_vm"] = np.array(scale, dtype=np.int64)
    return out


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="use the smoke pools of step 21")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config.guard_smoke_results(args.smoke)
    from hexmod.sim import pool, trajectory_v2

    res = config.results_dir()
    out6, out8 = res / "sim" / "fig6", res / "sim" / "fig8"
    with RunRecord(STEP, argv=argv) as rec:
        pools = {m: load_pool(m, rec, smoke=args.smoke) for m in MODULES}
        par = {m: pool.module_param(m) for m in MODULES}
        n_pool = len(pools["small"])
        n_draw = min(N_PARTICIPANT_CELLS, n_pool)

        # Fig 6A: trajectory of participant 0 and the object locations of its blocks
        traj = trajectory_v2.make_trajectory(par["small"], base_seed=0)
        hio.write_csv(traj[TRAJECTORY_COLUMNS], out6 / "trajectory_seed0.csv")
        rec.add_output(out6 / "trajectory_seed0.csv")
        objs = [np.asarray(o, float).tolist() for o in par["small"]["objLocs"]]
        hio.write_json({"objLocs": objs[0], "objLocs_by_block": objs,
                        "block_duration_s": float(par["small"]["expDurBlock"]), "n_blocks": int(par["small"]["nBlock"]),
                        "env_size_vm": [float(par["small"]["envSize_x"]), float(par["small"]["envSize_y"])]},
                       out6 / "trajectory_seed0_objects.json")
        rec.add_output(out6 / "trajectory_seed0_objects.json")
        rec.note("fig6a_n_samples", int(len(traj)))

        # Fig 6B, 6C: example maps and summed maps; Supp Fig 3E maps
        maps = {"extent": np.array([0, par["small"]["envSize_x"], 0, par["small"]["envSize_y"]], float)}
        examples = np.random.default_rng(EXAMPLE_SEED).choice(n_pool, n_draw, replace=False)[:N_EXAMPLES]
        for m in MODULES:
            maps[f"{m}__example_maps"] = example_maps(pools[m], par[m], examples)
            maps[f"{m}__example_cells"] = examples
        for seed in SUMMED_SEEDS:
            cells = np.random.default_rng(seed).choice(n_pool, n_draw, replace=False)
            for m in MODULES:
                maps[f"seed{seed}__{m}"] = summed_map_pct(pools[m], par[m], cells)
        titles = {}
        for m in MODULES:
            cells, gridness, peaks, smaps = supp3e_cells(pools[m], par[m])
            maps[f"supp3e__{m}__maps"] = smaps
            maps[f"supp3e__{m}__cells"] = cells
            maps[f"supp3e__{m}__gridness"] = gridness
            maps[f"supp3e__{m}__peak_rate_hz"] = peaks
            titles[m] = [supp3e_title(g, r) for g, r in zip(gridness, peaks)]
        maps["supp3e__gridness_class"] = np.array(SUPP3E_CLASSES, dtype=str)
        hio.save_npz(out6 / "maps.npz", arrays=maps)
        rec.add_output(out6 / "maps.npz")
        rec.note("example_cells", examples.tolist())
        rec.note("supp3e_cells", {m: maps[f"supp3e__{m}__cells"].tolist() for m in MODULES})
        rec.note("supp3e_titles", titles)
        nm = maps["seed0__large"]
        rec.note("fig6c_large_participant0_pct", {"max": float(nm.max() - 100), "min": float(nm.min() - 100),
                                                  "sd": float(nm.std())})

        # Fig 8A: perfect grids
        hio.save_npz(out8 / "fig8a_ideal_grids.npz", arrays=ideal_grids())
        rec.add_output(out8 / "fig8a_ideal_grids.npz")
        rec.note("smoke", bool(args.smoke))
    print(f"maps written to {out6} and {out8}; Supp Fig 3E titles: {titles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
