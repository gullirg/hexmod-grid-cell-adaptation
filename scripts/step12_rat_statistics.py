"""Step 12: rat figure tables and the printed statistics of Fig 3C, 3D, 4C, 4D, 5 and Supp Fig 1, 2, 3, 5.

    python scripts/step12_rat_statistics.py

Reads the per-cell outputs of steps 02 to 05 (``cells.csv``, ``cells_full.npz``, ``param_cells.csv``,
``fr_percent_change.npy``, ``peak_shift.csv``, ``shift_cells.csv``,
``fr_percent_change_shifted.npy``). Writes ``results/rat/figure_tables/`` and
``results/stats/step12_rat_statistics.json`` (see docs/PIPELINE.md, step 12 and section 4):

- ``cells_thirds.csv`` and ``thirds_edges.csv``: thirds of the adaptation coefficient, of the
  proximity difference and of grid scale (Fig 4D, Fig 5);
- ``supp3bcd_arrays.npz`` and ``supp3bcd_fits.csv``: field deviations and field rates of small-
  and large-scale cells with their gamma fits (location 0);
- ``supp5_widths.csv``: circular SD of the direction tuning of the conjunctive cells;
- the statistics file: every key listed for this step in docs/PIPELINE.md section 4 (Fig 3C to
  Supp Fig 2 from ``hexmod.rat.paper_stats``; Supp Fig 3 and 5 from
  ``hexmod.rat.figure_tables``), plus the pairwise comparisons between the groups of Fig 3C, 4D,
  5B and Supp Fig 2, Holm-corrected within each panel.
"""

from __future__ import annotations

import argparse
import math
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
from hexmod.rat import figure_tables as ft  # noqa: E402
from hexmod.rat import paper_stats  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step12_rat_statistics"


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="synthetic smoke run (statistics that cannot be computed "
                   "on the small synthetic cohort are left out)")
    return p.parse_args(argv)


def finite_only(values: dict):
    """Split a statistics dict into the finite numbers and the keys whose value is NaN or infinite."""
    kept, dropped = {}, []
    for k, v in values.items():
        if isinstance(v, (bool, np.bool_)):
            kept[k] = int(v)
        elif isinstance(v, (int, np.integer)):
            kept[k] = int(v)
        elif math.isfinite(float(v)):
            kept[k] = float(v)
        else:
            dropped.append(k)
    return kept, dropped


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config.guard_smoke_results(args.smoke)
    res = config.results_dir()
    out = res / "rat" / "figure_tables"
    stats_path = res / "stats" / f"{STEP}.json"
    paths = {
        "cells": res / "rat" / "cohort" / "cells.csv",
        "cells_full": res / "rat" / "cohort" / "cells_full.npz",
        "param": res / "rat" / "adaptation" / "param_cells.csv",
        "fr": res / "rat" / "adaptation" / "fr_percent_change.npy",
        "peak_shift": res / "rat" / "prospective" / "peak_shift.csv",
        "shift_cells": res / "rat" / "prospective" / "shift_cells.csv",
        "fr_shifted": res / "rat" / "prospective" / "fr_percent_change_shifted.npy",
    }
    with RunRecord(STEP, workers=1, argv=argv) as rec:
        for p in paths.values():
            rec.add_input(p)
        cells = hio.read_csv(paths["cells"])
        param = hio.read_csv(paths["param"])
        full = hio.load_npz(paths["cells_full"])
        fr = hio.load_npy(paths["fr"])
        peak_shift = hio.read_csv(paths["peak_shift"])
        shift_cells = hio.read_csv(paths["shift_cells"])
        fr_shifted = hio.load_npy(paths["fr_shifted"])
        n = len(cells)
        if not np.array_equal(cells["row"].to_numpy(), np.arange(n)):
            raise ValueError("cells.csv rows are not 0 to n - 1")
        if not (len(param) == len(full["rate_map"]) == len(fr) == len(peak_shift) == n):
            raise ValueError("per-cell inputs have different numbers of cells")
        if not np.array_equal(peak_shift["row"].to_numpy(), cells["row"].to_numpy()):
            raise ValueError("peak_shift.csv is not row-aligned with cells.csv")
        if len(fr_shifted) != len(shift_cells):
            raise ValueError("fr_percent_change_shifted.npy is not row-aligned with shift_cells.csv")

        # figure tables
        out.mkdir(parents=True, exist_ok=True)
        thirds, edges = ft.cells_thirds(cells, param, full["x_edges"], full["rate_map"])
        hio.write_csv(thirds, out / "cells_thirds.csv")
        hio.write_csv(edges, out / "thirds_edges.csv")
        supp3 = ft.supp3bcd_arrays(cells, full["x_edges"], full["field_deviation"], full["field_rate_hz"])
        fits = ft.gamma_fits(supp3)
        hio.save_npz(out / "supp3bcd_arrays.npz", arrays=supp3)
        hio.write_csv(fits, out / "supp3bcd_fits.csv")
        widths = ft.supp5_widths(cells)
        hio.write_csv(widths, out / "supp5_widths.csv")
        for name in ("cells_thirds.csv", "thirds_edges.csv", "supp3bcd_arrays.npz", "supp3bcd_fits.csv",
                     "supp5_widths.csv"):
            rec.add_output(out / name)

        # statistics: those of hexmod.rat.paper_stats, the pairwise comparisons, Supp Fig 3 and 5
        values = paper_stats.core_statistics(cells, param, fr, peak_shift["peak_shift_ms"].to_numpy(), shift_cells,
                                             fr_shifted)
        values.update(ft.posthoc_statistics(fr, thirds["proximity_diff"], cells["scale_cm"],
                                            param["alpha_tprevXloc_all"], param["beta_all"], fr_shifted))
        values.update(ft.supp3_statistics(supp3, fits))
        values.update(ft.supp5_statistics(widths))
        kept, dropped = finite_only(values)
        if dropped and not args.smoke:
            raise ValueError(f"statistics without a finite value: {dropped}")
        hio.write_json(kept, stats_path)
        rec.add_output(stats_path)
        rec.note("n_statistics", len(kept))
        rec.note("statistics_not_finite", dropped)
        rec.note("n_cells", n)
        print(f"{STEP}: {len(kept)} statistics; tables in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
