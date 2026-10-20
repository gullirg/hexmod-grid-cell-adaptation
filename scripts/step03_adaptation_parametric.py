"""Step 03: parametric adaptation fits and hexadirectional modulation of every grid cell.

Example::

    python scripts/step03_adaptation_parametric.py --workers 8

Reads ``results/preprocessed/trajectories.pkl`` and, from step 02, ``rat/cohort/cells.csv``,
``field_locations.npz`` and ``cells_full.npz`` (field rates). For every cell it fits
:func:`hexmod.rat.adaptation.adaptation_parametric` (adaptation regressors ``tprev`` and
``tprevXloc``, time constant 2 s, 5 cm location bins; masks
``all``, ``oddMinute``, ``evenMinute``, ``infield``) and
:func:`hexmod.rat.hexmod_fit.measure_hexmod` (mask ``all``). The grid orientation and the field
rates enter the fits after the text round trips of docs/PIPELINE.md (Rounded per-cell inputs).

It also computes, for each cell, the limits within which the fit keeps alpha
(:func:`hexmod.rat.paper_stats.alpha_bounds`) and counts the cells whose alpha equals a limit;
``--no-alpha-bounds`` skips this.

Writes ``rat/adaptation/param_cells.csv``, ``rat/adaptation/alpha_bounds.csv`` and
``stats/step03_adaptation_parametric.json`` (the limit counts, a diagnostic).
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

from hexmod.io import (array_text_roundtrip, csv_float_roundtrip, load_cache, load_npz,  # noqa: E402
                       read_csv, write_csv, write_json)
from hexmod.parallel import map_ordered  # noqa: E402
from hexmod.rat import paper_stats  # noqa: E402
from hexmod.rat.adaptation import ADAPTATION_TAU_S, adaptation_parametric  # noqa: E402
from hexmod.rat.gridcell import make_cell_data  # noqa: E402
from hexmod.rat.hexmod_fit import measure_hexmod  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step03_adaptation_parametric"

SPATIAL_RESOLUTION = 5  # cm
TAU_ADAPT = ADAPTATION_TAU_S  # s, hexmod.rat.adaptation
ADAPT_TYPES = ["tprev", "tprevXloc"]
PARAM_MASKS = ["all", "oddMinute", "evenMinute", "infield"]
HEXMOD_MASKS = ["all"]
ID_COLUMNS = ["row", "animal_id", "session_id", "cell_id"]


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    p.add_argument("--no-alpha-bounds", action="store_true", help="skip the count of alpha values on a fit limit")
    return p.parse_args(argv)


def fit_inputs(cells, field_locations, field_rates, data):
    """Per-cell inputs shared by the parametric and the HexMod fits.

    The orientation is passed after one CSV text round trip read with pandas' default parser and
    the field rates after their 8-decimal text form (docs/PIPELINE.md, Rounded per-cell inputs); with
    these inputs the fits give the reference results exactly.
    """
    ori = csv_float_roundtrip(cells["orientation_deg"].to_numpy(float), 1)
    out = []
    for i, (animal_id, session_id, cell_id) in enumerate(zip(cells["animal_id"], cells["session_id"],
                                                             cells["cell_id"])):
        cd = make_cell_data(data[animal_id][session_id], int(cell_id), animal_id, session_id)
        out.append({"cell_data": cd, "fieldLoc": field_locations[i], "gridOri": float(ori[i]),
                    "peakFR": array_text_roundtrip(field_rates[i]), "spatialResolution": SPATIAL_RESOLUTION,
                    "adapt_type": ADAPT_TYPES, "tau_adapt": TAU_ADAPT})
    return out


def _check_keys(results, cells):
    for r, s, c in zip(results, cells["session_id"], cells["cell_id"]):
        if r["session_id"] != s or int(r["cell_id"]) != int(c):
            raise RuntimeError(f"fit result of {r['session_id']} cell {r['cell_id']} is out of order")


def _columns(results, names):
    out = {}
    for name in names:
        values = [r[name] for r in results]
        out[name] = np.array([bool(v) for v in values]) if name.startswith("converge") else np.array(values, float)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workers = config.workers(args.workers)
    res = config.results_dir()
    coh = res / "rat" / "cohort"
    out_dir = res / "rat" / "adaptation"
    traj_path = res / "preprocessed" / "trajectories.pkl"

    with RunRecord(STEP, workers=workers, argv=argv if argv is not None else sys.argv[1:]) as rec:
        for p in (traj_path, coh / "cells.csv", coh / "field_locations.npz", coh / "cells_full.npz"):
            rec.add_input(p)
        cells = read_csv(coh / "cells.csv")
        fl = load_npz(coh / "field_locations.npz")
        full = load_npz(coh / "cells_full.npz")
        for z in (fl, full):
            if not np.array_equal(z["row"], cells["row"].to_numpy()):
                raise RuntimeError("cohort files hold different rows")
        data = load_cache(traj_path)
        base = fit_inputs(cells, fl["field_locations"], full["field_rate_hz"], data)

        param_in = [{**b, "mask_type": PARAM_MASKS} for b in base]
        rp = map_ordered(adaptation_parametric, param_in, workers, progress=True, desc="parametric fits")
        _check_keys(rp, cells)
        rh = map_ordered(measure_hexmod, [{**b, "mask_type": HEXMOD_MASKS} for b in base], workers,
                         progress=True, desc="hexmod fits")
        _check_keys(rh, cells)

        param_names = [f"{q}_{t}_{m}" for m in PARAM_MASKS for t in ADAPT_TYPES
                       for q in ("speed", "alpha", "converge", "align", "misalign")]
        hex_names = [f"{q}_{m}" for m in HEXMOD_MASKS for q in ("speed", "beta", "converge")]
        param = pd.DataFrame({**{c: cells[c].to_numpy() for c in ID_COLUMNS}, **_columns(rp, param_names),
                              **_columns(rh, hex_names)}, columns=ID_COLUMNS + param_names + hex_names)
        write_csv(param, out_dir / "param_cells.csv")
        rec.add_output(out_dir / "param_cells.csv")
        n_conv = {c: int(param[c].sum()) for c in param.columns if c.startswith("converge")}
        rec.note("n_cells", int(len(param)))
        rec.note("n_converged", n_conv)

        stats = {}
        if not args.no_alpha_bounds:
            rb = map_ordered(paper_stats.alpha_bounds, param_in, workers, progress=True, desc="alpha limits")
            _check_keys(rb, cells)
            combos = [f"{t}_{m}" for m in PARAM_MASKS for t in ADAPT_TYPES]
            bnames = [f"alpha_{side}_{c}" for c in combos for side in ("lower", "upper")]
            bounds = pd.DataFrame({**{c: cells[c].to_numpy() for c in ID_COLUMNS},
                                   **{n: np.array([r[n] for r in rb], float) for n in bnames}},
                                  columns=ID_COLUMNS + bnames)
            for c in combos:
                a = param[f"alpha_{c}"].to_numpy(float)
                bounds[f"at_bound_{c}"] = (a == bounds[f"alpha_lower_{c}"].to_numpy()) | \
                                          (a == bounds[f"alpha_upper_{c}"].to_numpy())
            write_csv(bounds, out_dir / "alpha_bounds.csv")
            rec.add_output(out_dir / "alpha_bounds.csv")
            stats = paper_stats.alpha_bound_counts(param, bounds, combos)
            rec.note("alpha_at_bound", {c: stats[f"diagnostic.alpha_at_bound.{c}.n_total"] for c in combos})
        write_json(stats, res / "stats" / f"{STEP}.json")
        rec.add_output(res / "stats" / f"{STEP}.json")
    print(f"fits of {len(param)} cells written to {out_dir / 'param_cells.csv'}")
    if stats:
        k = "diagnostic.alpha_at_bound.tprevXloc_all"
        print(f"alpha_tprevXloc_all on a fit limit: {stats[k + '.n_lower']} lower, {stats[k + '.n_upper']} upper "
              f"of {stats[k + '.n_cells']} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
