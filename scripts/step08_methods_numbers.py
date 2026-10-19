"""Step 08: numbers quoted in the Methods.

Example::

    python scripts/step08_methods_numbers.py --workers 8

- Interval between consecutive field visits by third of grid scale: the t_prev of every field entry
  (except the first visit of each cell) of every cohort cell, with the fields of step 02. The
  Methods quote the pooled median of the small-scale third (intervals up to 60 s).
- Running speed in the 100 cm arenas: for each cohort session whose rate maps end between 85 and
  115 cm, the mean speed of the samples between 5 and 150 cm/s; averaged over sessions.
- Ratio of the firing rate in the 0 to 1 s t_prev bin to that in the bin above 3 s (step 04),
  averaged over cells; it compares the measured adaptation with the time course
  ``1 - alpha exp(-t_prev / tau)`` used to choose alpha in the simulations.
- The time constant of the adaptation regressor and the range of spike shifts of the prospective
  coding control, as used by steps 03 and 05.

Reads ``preprocessed/trajectories.pkl``, ``rat/cohort/cells.csv``, ``rat/cohort/field_locations.npz``,
``rat/cohort/cells_full.npz`` (rate-map edges) and ``rat/adaptation/fr_percent_change.npy``. Writes
``rat/methods/methods_numbers.json`` and ``stats/step08_methods_numbers.json``.
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
from hexmod.parallel import map_ordered  # noqa: E402
from hexmod.rat import timescales as ts  # noqa: E402
from hexmod.rat.adaptation import ADAPTATION_TAU_S  # noqa: E402  (tau of the parametric fits, step 03)
from hexmod.rat.gridcell import make_cell_data  # noqa: E402
from hexmod.rat.prospective import DEFAULT_SHIFTS_MS  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step08_methods_numbers"


#: Keys of each scale third in methods_numbers.json.
THIRD_KEYS = ("n_cells", "scale_range_cm", "n_events", "pooled_median_s", "pooled_iqr_s", "median_of_cell_medians_s")


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    p.add_argument("--smoke", action="store_true", help="smoke run on synthetic results")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config.guard_smoke_results(args.smoke)
    workers = config.workers(args.workers)
    res = config.results_dir()
    with RunRecord(STEP, workers=workers, argv=argv) as rec:
        traj_path = res / "preprocessed" / "trajectories.pkl"
        cells_path = res / "rat" / "cohort" / "cells.csv"
        fl_path = res / "rat" / "cohort" / "field_locations.npz"
        full_path = res / "rat" / "cohort" / "cells_full.npz"
        pct_path = res / "rat" / "adaptation" / "fr_percent_change.npy"
        for p in (traj_path, cells_path, fl_path, full_path, pct_path):
            rec.add_input(p)
        traj = hio.load_cache(traj_path)
        cells = hio.read_csv(cells_path)
        fl = hio.load_npz(fl_path)
        if not np.array_equal(fl["row"], cells["row"].to_numpy()):
            raise ValueError("field_locations.npz is not row-aligned with cells.csv")
        with np.load(full_path, allow_pickle=False) as z:
            if not np.array_equal(z["row"], cells["row"].to_numpy()):
                raise ValueError("cells_full.npz is not row-aligned with cells.csv")
            x_edges = hio.ragged_from_npz(z, "x_edges")

        # intervals between field visits
        items = []
        for i, row in enumerate(cells.itertuples(index=False)):
            cd = make_cell_data(traj[row.animal_id][row.session_id], int(row.cell_id), row.animal_id, row.session_id)
            items.append(ts.interval_inputs(cd, fl["field_locations"][i]))
        intervals = map_ordered(ts.field_visit_intervals, items, workers, desc="cells")
        del items
        iv = ts.interval_statistics(intervals, cells["scale_cm"].to_numpy(float))

        # running speed in the 100 cm arenas
        sessions = list(dict.fromkeys(zip(cells["animal_id"], cells["session_id"])))
        speed = {s: ts.session_speed(traj[a][s]["position"]) for a, s in sessions}
        arena = ts.arena_size_by_session(cells["session_id"].tolist(), x_edges)
        running = ts.running_speed_summary([s for _, s in sessions], speed, arena)

        ratio = ts.first_to_last_bin_ratio(hio.load_npy(pct_path))

        numbers = {"intervals_by_scale_third": {name: {k: v[k] for k in THIRD_KEYS if k in v}
                                                for name, v in iv["intervals_by_scale_third"].items()},
                   "scale_third_edges_cm": iv["scale_third_edges_cm"],
                   "running_speed_100cm_arenas": running}
        p = hio.write_json(numbers, res / "rat" / "methods" / "methods_numbers.json")
        rec.add_output(p)

        stats_out = {"methods.running_speed.n_sessions": running["n_sessions"],
                     "methods.ratio_0_1s_to_above_3s": ratio,
                     "methods.tau_s": ADAPTATION_TAU_S,
                     "methods.shift_min_ms": int(DEFAULT_SHIFTS_MS.min()),
                     "methods.shift_max_ms": int(DEFAULT_SHIFTS_MS.max())}
        small = iv["intervals_by_scale_third"]["small"]
        if "pooled_median_s" in small:
            stats_out["methods.interfield_median_s.small_third"] = small["pooled_median_s"]
        if "mean_speed_cm_s" in running:
            stats_out["methods.running_speed_cm_s"] = running["mean_speed_cm_s"]
        if not np.isfinite(ratio):
            del stats_out["methods.ratio_0_1s_to_above_3s"]
        p = hio.write_json(stats_out, res / "stats" / f"{STEP}.json")
        rec.add_output(p)
        for name, v in iv["intervals_by_scale_third"].items():
            for key in ("pooled_share_under_1s", "pooled_share_under_2s"):
                if key in v:
                    rec.note(f"{name}.{key}", v[key])
        rec.note("n_cohort_sessions", len(sessions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
