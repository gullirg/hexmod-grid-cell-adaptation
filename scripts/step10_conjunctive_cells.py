"""Step 10: preferred direction of conjunctive grid cells relative to the grid axes (Fig 2B to 2E).

Example::

    python scripts/step10_conjunctive_cells.py

Reads ``rat/cohort/cells.csv``, ``rat/cohort/dir_maps.npy``, ``rat/same_cell/entries.csv`` (step 09) and the
shipped module labels (``inputs/rat/module_labels.csv``). Writes the tables of ``rat/conjunctive/`` and the Fig 2B
to 2E statistics to ``stats/step10_conjunctive_cells.json``. The analyses are described in
``hexmod.rat.conjunctive``; ``hexmod.rat.conjunctive.fig2_results`` computes everything written here.
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

STEP = "step10_conjunctive_cells"


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    parse_args(argv)
    from hexmod import io
    from hexmod.preprocessing import labels
    from hexmod.rat import conjunctive
    from hexmod.runlog import RunRecord

    res = config.results_dir()
    out = res / "rat" / "conjunctive"
    cells_path = res / "rat" / "cohort" / "cells.csv"
    maps_path = res / "rat" / "cohort" / "dir_maps.npy"
    entries_path = res / "rat" / "same_cell" / "entries.csv"
    modules_path = config.inputs_dir() / "rat" / "module_labels.csv"
    print(f"results: {res}", flush=True)

    with RunRecord(STEP, workers=1, argv=argv if argv is not None else sys.argv[1:]) as rec:
        for p in (cells_path, maps_path, entries_path, modules_path):
            rec.add_input(p)
        cells = io.read_csv(cells_path)
        dir_maps = io.load_npy(maps_path)
        entries = io.read_csv(entries_path)
        if len(dir_maps) != len(cells) or not np.array_equal(entries["row"].to_numpy(), cells["row"].to_numpy()):
            raise RuntimeError("cells.csv, dir_maps.npy and entries.csv are not row-aligned; rerun steps 02 and 09")
        tables, S, fl = conjunctive.fig2_results(
            cells, dir_maps, entries["kept_earliest"].to_numpy(bool), labels.read_module_bins(modules_path)
        )
        dropped = fl.pop("statistics_not_finite", [])
        rec.note("statistics_not_finite", dropped)
        if dropped:
            raise ValueError(f"statistics without a finite value: {dropped}")
        for name, df in tables.items():
            io.write_csv(df, out / name)
            rec.add_output(out / name)
        stats_path = res / "stats" / f"{STEP}.json"
        io.write_json(S, stats_path)
        rec.add_output(stats_path)
        rec.note("fig2d_test", fl)
        rec.note("counts", {name: int(len(df)) for name, df in tables.items()})
        nan = float("nan")
        print(
            f"Fig 2B: {S['fig2b.n_aligned']} of {S['fig2b.n_cells']} aligned, "
            f"p = {S.get('fig2b.binomial.p', nan):.4f}; "
            f"Fig 2C: {S['fig2c.n_animals']} animals, {S['fig2c.n_cells']} cells, median width "
            f"{S.get('fig2c.circular_sd.median_deg', nan):.2f} deg",
            flush=True,
        )
        print(
            f"Fig 2D: {S['fig2d.n_cells']} cells, {S['fig2d.n_animals']} animals, "
            f"r = {S.get('fig2d.fisher_lee.r', nan):.5f}, "
            f"p = {S.get('fig2d.permutation.p', nan):.4f}",
            flush=True,
        )
        print(
            f"Fig 2E: {S['fig2e.n_animals']} animals, {S['fig2e.n_cells']} cells, "
            f"p < 0.05 in {S['fig2e.n_p_below_0_05']}, "
            f"lowest Holm p {S.get('fig2e.min_p_holm', nan):.3f}, binomial p {S.get('fig2e.binomial.p', nan):.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
