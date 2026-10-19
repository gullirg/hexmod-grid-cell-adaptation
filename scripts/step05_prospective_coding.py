"""Step 05: prospective coding control (Supp Fig 2).

Example::

    python scripts/step05_prospective_coding.py --workers 8

Reads ``results/preprocessed/trajectories.pkl``, ``rat/cohort/cells.csv`` (step 02) and
``rat/adaptation/param_cells.csv`` (step 03).

1. For every cell, the spike train is shifted against the positions by -400 to +380 ms in steps
   of 20 ms (:func:`hexmod.rat.prospective.check_prospective_coding`); the best shift is the one
   with the highest spatial information of the rate map.
2. Every cell is characterised again with its spikes shifted by its best shift (as in step 02);
   cells that remain grid cells (gridness at least 0.3) are kept.
3. The kept cells are refitted with their shifted spikes and the new fields: the non-parametric
   model (as in step 04) and the parametric model with the ``tprevXloc`` regressor, masks
   ``all``, ``infield`` and ``outfield`` (as in step 03). Orientation and field rates of the
   new characterisation enter the fits unchanged.

``alpha_original`` in ``shift_cells.csv`` is ``alpha_tprevXloc_all`` of step 03 after one CSV text
round trip (docs/PIPELINE.md, section 2.7), the values of the Supp Fig 2 comparison.

Writes ``rat/prospective/shift_ms.npy``, ``si_shift.npy``, ``peak_shift.csv``,
``shift_cells.csv``, ``fr_percent_change_shifted.npy`` and ``shift_field_locations.npz``.
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

from hexmod.io import (csv_float_roundtrip, load_cache, read_csv, save_npy, save_npz,  # noqa: E402
                       write_csv)
from hexmod.parallel import map_ordered  # noqa: E402
from hexmod.rat.adaptation import ADAPTATION_TAU_S, adaptation_nonparametric, adaptation_parametric  # noqa: E402
from hexmod.rat.characterise import characterise_gridcell  # noqa: E402
from hexmod.rat.gridcell import make_cell_data  # noqa: E402
from hexmod.rat.prospective import DEFAULT_SHIFTS_MS, check_prospective_coding  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step05_prospective_coding"

SHIFTS_MS = np.asarray(DEFAULT_SHIFTS_MS)  # -400 to +380 ms in steps of 20 ms
MIN_DIST_BTW_PEAKS_REL = 2 / 3
PROMINENCE = 0.4
TPREV_BINS = np.array([0, 1, 2, 3, np.inf])  # s
N_BINS_SPEED = 8
SPATIAL_RESOLUTION = 5  # cm
TAU_ADAPT = ADAPTATION_TAU_S  # s, hexmod.rat.adaptation
ADAPT_TYPES = ["tprevXloc"]
MASKS = ["all", "infield", "outfield"]
ID_COLUMNS = ["row", "animal_id", "session_id", "cell_id"]


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    p.add_argument("--smoke", action="store_true", help="smoke run on synthetic data")
    return p.parse_args(argv)


def _cell_data(data, animal_id, session_id, cell_id, shift_ms=None):
    shift = None if shift_ms is None else shift_ms / 1000
    return make_cell_data(data[animal_id][session_id], int(cell_id), animal_id, session_id, spike_shift_s=shift)


def _check_keys(results, session_ids, cell_ids):
    for r, s, c in zip(results, session_ids, cell_ids):
        if r["session_id"] != s or int(r["cell_id"]) != int(c):
            raise RuntimeError(f"result of {r['session_id']} cell {r['cell_id']} is out of order")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config.guard_smoke_results(args.smoke)
    workers = config.workers(args.workers)
    res = config.results_dir()
    out_dir = res / "rat" / "prospective"
    traj_path = res / "preprocessed" / "trajectories.pkl"
    cells_path = res / "rat" / "cohort" / "cells.csv"
    param_path = res / "rat" / "adaptation" / "param_cells.csv"

    with RunRecord(STEP, workers=workers, argv=argv if argv is not None else sys.argv[1:]) as rec:
        for p in (traj_path, cells_path, param_path):
            rec.add_input(p)
        cells = read_csv(cells_path)
        param = read_csv(param_path)
        if not np.array_equal(param["row"].to_numpy(), cells["row"].to_numpy()):
            raise RuntimeError("param_cells.csv and cells.csv hold different rows")
        data = load_cache(traj_path)
        ids = list(zip(cells["animal_id"], cells["session_id"], cells["cell_id"]))

        # 1. spatial information for every shift; best shift
        inputs = [{"cell_data": _cell_data(data, a, s, c), "shift_time_list": SHIFTS_MS, "isSplitHalfTest": False}
                  for a, s, c in ids]
        pr = map_ordered(check_prospective_coding, inputs, workers, progress=True, desc="time shifts")
        si = np.row_stack([r["SI"] for r in pr]) if pr else np.zeros((0, len(SHIFTS_MS)))
        peak = SHIFTS_MS[np.argmax(si, axis=1)] if len(si) else np.zeros(0, dtype=SHIFTS_MS.dtype)
        save_npy(out_dir / "shift_ms.npy", SHIFTS_MS.astype(np.int64))
        save_npy(out_dir / "si_shift.npy", si)
        write_csv(pd.DataFrame({**{c: cells[c].to_numpy() for c in ID_COLUMNS},
                                "peak_shift_ms": peak.astype(np.int64)}), out_dir / "peak_shift.csv")

        # 2. characterisation with shifted spikes
        inputs = [{"cell_data": _cell_data(data, a, s, c, peak[i]), "min_dist_btw_peaks_rel": MIN_DIST_BTW_PEAKS_REL,
                   "prominence": PROMINENCE} for i, (a, s, c) in enumerate(ids)]
        rs = map_ordered(characterise_gridcell, inputs, workers, progress=True, desc="characterise shifted")
        kept = [i for i, r in enumerate(rs) if r is not None and "fieldLoc" in r]
        n_no_field = sum(1 for r in rs if r is not None and "fieldLoc" not in r)
        shifted = [rs[i] for i in kept]

        # 3. fits with shifted spikes and the new fields (exact orientation and field rates)
        inputs = []
        for i, r in zip(kept, shifted):
            a, s, c = ids[i]
            inputs.append({"cell_data": _cell_data(data, a, s, c, peak[i]), "fieldLoc": r["fieldLoc"],
                           "tprev_bins": TPREV_BINS, "n_bins_speed": N_BINS_SPEED,
                           "gridOri": float(r["orientation_deg"]), "peakFR": r["FR_inhomogeneity_regionFR"],
                           "spatialResolution": SPATIAL_RESOLUTION, "adapt_type": ADAPT_TYPES, "tau_adapt": TAU_ADAPT,
                           "mask_type": MASKS})
        npr = map_ordered(adaptation_nonparametric, inputs, workers, progress=True, desc="non-parametric fits")
        ppr = map_ordered(adaptation_parametric, inputs, workers, progress=True, desc="parametric fits")
        sub = cells.iloc[kept]
        for results in (shifted, npr, ppr):
            _check_keys(results, sub["session_id"], sub["cell_id"])

        alpha_original = csv_float_roundtrip(param["alpha_tprevXloc_all"].to_numpy(float), 1)[kept]
        fit_names = [f"{q}_{t}_{m}" for m in MASKS for t in ADAPT_TYPES
                     for q in ("speed", "alpha", "converge", "align", "misalign")]
        table = {**{c: sub[c].to_numpy() for c in ID_COLUMNS},
                 "peakShift_SI": peak[kept].astype(np.int64),
                 "gridness": np.array([r["gridness"] for r in shifted], float),
                 "scale_cm": np.array([r["scale_cm"] for r in shifted], float)}
        for m in MASKS:
            table[f"meanFR_{m}"] = np.array([r[f"meanFR_{m}"] for r in npr], float)
        for name in fit_names:
            values = [r[name] for r in ppr]
            is_flag = name.startswith("converge")
            table[name] = np.array([bool(v) for v in values]) if is_flag else np.array(values, float)
        table["alpha_original"] = alpha_original
        write_csv(pd.DataFrame(table), out_dir / "shift_cells.csv")
        if npr:
            mean_fr = np.array([r["meanFR_all"] for r in npr], float)
            frpc = np.row_stack([r["tprev_nonparam_all"] for r in npr]) / mean_fr[:, None] * 100 - 100
        else:
            frpc = np.zeros((0, len(TPREV_BINS) - 1))
        save_npy(out_dir / "fr_percent_change_shifted.npy", frpc)
        save_npz(out_dir / "shift_field_locations.npz", {"row": sub["row"].to_numpy()},
                 ragged={"field_locations": [r["fieldLoc"] for r in shifted]})

        for name in ["shift_ms.npy", "si_shift.npy", "peak_shift.csv", "shift_cells.csv",
                     "fr_percent_change_shifted.npy", "shift_field_locations.npz"]:
            rec.add_output(out_dir / name)
        rec.note("n_cells", len(ids))
        rec.note("n_cells_kept", len(kept))
        rec.note("n_kept_grid_cells_without_fields_excluded", n_no_field)
        rec.note("median_best_shift_ms", float(np.median(peak)) if len(peak) else None)
    print(f"best shifts of {len(ids)} cells; {len(kept)} remain grid cells with shifted spikes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
