"""Step 04: firing rate by time since the last field visit (non-parametric fits, Fig 3C).

Example::

    python scripts/step04_adaptation_nonparametric.py --workers 8

Reads ``results/preprocessed/trajectories.pkl`` and, from step 02, ``rat/cohort/cells.csv``
and ``field_locations.npz``. For every cell it fits
:func:`hexmod.rat.adaptation.adaptation_nonparametric`: a multiplicative Poisson model of the
firing rate with 5 cm location bins, 8 speed bins of equal sample count and t_prev bins 0 to 1,
1 to 2, 2 to 3 and above 3 s (masks ``all`` and ``infield``). The rate in each t_prev bin is
expressed as percent change from the cell's mean rate in the mask. The model uses neither the
grid orientation nor the field rates.

Writes ``rat/adaptation/fr_percent_change.npy`` (mask ``all``, the values of Fig 3C) and
``rat/adaptation/fr_percent_change_infield.npy``, both (n_cells, 4) and row-aligned with
``cells.csv``.
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

from hexmod.io import load_cache, load_npz, read_csv, save_npy  # noqa: E402
from hexmod.parallel import map_ordered  # noqa: E402
from hexmod.rat.adaptation import adaptation_nonparametric  # noqa: E402
from hexmod.rat.gridcell import make_cell_data  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step04_adaptation_nonparametric"

TPREV_BINS = np.array([0, 1, 2, 3, np.inf])  # s
N_BINS_SPEED = 8
SPATIAL_RESOLUTION = 5  # cm
MASKS = ["all", "infield"]
OUTPUTS = {"all": "fr_percent_change.npy", "infield": "fr_percent_change_infield.npy"}


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    p.add_argument("--smoke", action="store_true", help="smoke run on synthetic data")
    return p.parse_args(argv)


def percent_change(results, mask):
    """Rate per t_prev bin as percent change from the mean rate of the mask, one row per cell."""
    mean_fr = np.array([r[f"meanFR_{mask}"] for r in results], float)
    if not results:
        return np.zeros((0, len(TPREV_BINS) - 1))
    return np.row_stack([r[f"tprev_nonparam_{mask}"] for r in results]) / mean_fr[:, None] * 100 - 100


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config.guard_smoke_results(args.smoke)
    workers = config.workers(args.workers)
    res = config.results_dir()
    coh = res / "rat" / "cohort"
    out_dir = res / "rat" / "adaptation"
    traj_path = res / "preprocessed" / "trajectories.pkl"

    with RunRecord(STEP, workers=workers, argv=argv if argv is not None else sys.argv[1:]) as rec:
        for p in (traj_path, coh / "cells.csv", coh / "field_locations.npz"):
            rec.add_input(p)
        cells = read_csv(coh / "cells.csv")
        fl = load_npz(coh / "field_locations.npz")
        if not np.array_equal(fl["row"], cells["row"].to_numpy()):
            raise RuntimeError("cohort files hold different rows")
        data = load_cache(traj_path)
        inputs = []
        for i, (animal_id, session_id, cell_id) in enumerate(zip(cells["animal_id"], cells["session_id"],
                                                                 cells["cell_id"])):
            cd = make_cell_data(data[animal_id][session_id], int(cell_id), animal_id, session_id)
            inputs.append({"cell_data": cd, "fieldLoc": fl["field_locations"][i], "tprev_bins": TPREV_BINS,
                           "n_bins_speed": N_BINS_SPEED, "spatialResolution": SPATIAL_RESOLUTION,
                           "mask_type": MASKS})
        results = map_ordered(adaptation_nonparametric, inputs, workers, progress=True, desc="non-parametric fits")
        for r, s, c in zip(results, cells["session_id"], cells["cell_id"]):
            if r["session_id"] != s or int(r["cell_id"]) != int(c):
                raise RuntimeError(f"fit result of {r['session_id']} cell {r['cell_id']} is out of order")
        for mask in MASKS:
            X = percent_change(results, mask)
            save_npy(out_dir / OUTPUTS[mask], X)
            rec.add_output(out_dir / OUTPUTS[mask])
            rec.note(f"n_cells_with_nan_{mask}", int(np.isnan(X).any(axis=1).sum()))
        rec.note("n_cells", len(results))
    print(f"percent change of {len(results)} cells written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
