"""Step 11: arrays of the example cells drawn in Fig 2A, 3A, 3B, 4A, 4B, Supp Fig 2A and 3A.

    python scripts/step11_example_cells.py

Reads the preprocessed trajectories (step 01), the cohort tables of step 02 (``cells.csv``,
``cells_full.npz``, ``field_locations.npz``, ``dir_maps.npy``), the spatial information per
time shift of step 05 (``shift_ms.npy``, ``si_shift.npy``, ``peak_shift.csv``) and the list of
example cells ``inputs/rat/example_cells.csv``. Writes ``results/rat/examples/`` (see
docs/PIPELINE.md, step 11):

- Fig 2A: the direction tuning curves of five conjunctive cells;
- Fig 3A and 3B (one cell) and Fig 4A and 4B (a small-scale and a large-scale cell): every
  speed-filtered sample of the session with spikes, t_prev, location input and adaptation
  variable, the rate map, the fields numbered by their first visit in the drawn time window, and
  the field visits; for Fig 4B also the grid alignment of the running direction;
- Supp Fig 2A: spike positions and fields of the Fig 3 cell with its spikes shifted by its best
  time shift, by 0 ms and by minus its best shift, and its spatial information at every shift;
- Supp Fig 3A: rate maps, fields and fitted lattice origins of twelve cells.

The run record notes whether the listed cells follow the selection rules stated in
``example_cells.csv``.
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
from hexmod.preprocessing import labels  # noqa: E402
from hexmod.rat import examples as ex  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step11_example_cells"

#: Example trajectories: output name, panel and position in example_cells.csv, the panel that
#: shows the same cell and window, and whether the grid alignment of the running direction is stored.
TRAJECTORIES = (("fig3ab", "Fig3A", 1, "Fig3B", False),
                ("fig4ab_small", "Fig4A", 1, "Fig4B", True),
                ("fig4ab_large", "Fig4A", 2, "Fig4B", True))

#: Panels drawn from the trajectories (the others need only the cohort tables).
TRAJECTORY_PANELS = ("Fig3A", "Fig3B", "SuppFig2A", "Fig4A", "Fig4B")

#: Module (``inputs/rat/module_labels.csv``) of the Fig 4A cells by position, for the selection check.
FIG4_MODULES = {1: 1, 2: 3}


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="synthetic smoke run: example cells missing from the cohort are replaced by cohort cells")
    return p.parse_args(argv)


def one_example(examples, panel, position=None):
    """The single row of ``examples`` for a panel (and position)."""
    sel = examples[examples["panel"] == panel]
    if position is not None:
        sel = sel[sel["position"] == position]
    if len(sel) != 1:
        raise ValueError(f"expected one example cell for {panel} position {position}, found {len(sel)}")
    return sel.iloc[0]


def smoke_examples(examples, cells, trajectories):
    """Replace example cells that are not in the cohort by cohort cells (synthetic smoke run only).

    Cells are taken in order of decreasing gridness, first for the panels drawn from the
    trajectories; panels that show the same cell (Fig 3A, 3B and Supp Fig 2A; Fig 4A and 4B)
    keep sharing one cell, and time windows become 10 to 22 s after the first position sample of
    the session.
    """
    present = set(zip(cells["session_id"].astype(str), cells["cell_id"].astype(int)))
    pool = cells.sort_values("gridness", ascending=False, kind="mergesort")
    pool = list(zip(pool["animal_id"], pool["session_id"], pool["cell_id"]))
    if not pool:
        raise ValueError("the cohort is empty")
    out = examples.copy().reset_index(drop=True)
    shared = {"Fig3B": "Fig3A", "SuppFig2A": "Fig3A", "Fig4B": "Fig4A"}
    first = [i for i in out.index if out.loc[i, "panel"] in TRAJECTORY_PANELS]
    chosen = {}
    for i in first + [i for i in out.index if i not in first]:
        r = out.loc[i]
        if (str(r["session_id"]), int(r["cell_id"])) in present:
            continue
        key = (shared.get(r["panel"], r["panel"]), 1 if r["panel"] == "SuppFig2A" else int(r["position"]))
        if key not in chosen:
            chosen[key] = pool[len(chosen) % len(pool)]
        a, s, c = chosen[key]
        out.loc[i, ["animal_id", "session_id"]] = [a, s]
        out.loc[i, "cell_id"] = int(c)
        if np.isfinite(r["t_start_s"]):
            t0 = float(trajectories[a][s]["position"]["time"][0])
            out.loc[i, ["t_start_s", "t_end_s"]] = [t0 + 10.0, t0 + 22.0]
    out["cell_id"] = out["cell_id"].astype(np.int64)
    return out


def selection_checks(cells, peak_shift_ms, rows):
    """Check that the listed example cells follow the selection rules of ``example_cells.csv``."""
    checks = {}
    fig2a = rows["Fig2A"]
    session = cells.loc[fig2a[0], "session_id"]
    conj = cells[(cells["session_id"] == session) & (cells["Rayleigh_mag"] > 0.3)]["row"].tolist()
    checks["fig2a_last_five_conjunctive_cells_of_session"] = conj[-5:] == list(fig2a)
    best = np.asarray(peak_shift_ms)
    first = np.flatnonzero((best > 100) & (np.abs(best) < 150))
    checks["supp2a_first_cell_with_best_shift_100_to_150_ms"] = bool(len(first)) and int(first[0]) == rows["SuppFig2A"]
    modules = labels.assign_modules(cells["animal_id"].tolist(), cells["scale_cm"].to_numpy())
    for pos, module in FIG4_MODULES.items():
        row = rows[f"Fig4A_{pos}"]
        cand = cells[(cells["session_id"] == cells.loc[row, "session_id"]) & (modules == module)
                     & (cells["gridness"] > 1.0)]["row"].tolist()
        key = f"fig4a_position{pos}_second_cell_with_gridness_above_1_in_module_{module}"
        checks[key] = len(cand) > 1 and cand[1] == row
    return checks


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config.guard_smoke_results(args.smoke)
    res = config.results_dir()
    out = res / "rat" / "examples"
    paths = {
        "trajectories": res / "preprocessed" / "trajectories.pkl",
        "cells": res / "rat" / "cohort" / "cells.csv",
        "cells_full": res / "rat" / "cohort" / "cells_full.npz",
        "field_locations": res / "rat" / "cohort" / "field_locations.npz",
        "dir_maps": res / "rat" / "cohort" / "dir_maps.npy",
        "shift_ms": res / "rat" / "prospective" / "shift_ms.npy",
        "si_shift": res / "rat" / "prospective" / "si_shift.npy",
        "peak_shift": res / "rat" / "prospective" / "peak_shift.csv",
        "example_cells": config.inputs_dir() / "rat" / "example_cells.csv",
    }
    with RunRecord(STEP, workers=1, argv=argv) as rec:
        for p in paths.values():
            rec.add_input(p)
        cells = hio.read_csv(paths["cells"])
        if not np.array_equal(cells["row"].to_numpy(), np.arange(len(cells))):
            raise ValueError("cells.csv rows are not 0 to n - 1")
        full = hio.load_npz(paths["cells_full"])
        field_locations = hio.load_npz(paths["field_locations"])["field_locations"]
        dir_maps = hio.load_npy(paths["dir_maps"])
        shift_ms = hio.load_npy(paths["shift_ms"])
        si_shift = hio.load_npy(paths["si_shift"])
        peak_shift = hio.read_csv(paths["peak_shift"])
        if not (len(full["rate_map"]) == len(field_locations) == len(dir_maps) == len(si_shift) == len(peak_shift)
                == len(cells)):
            raise ValueError("the cohort tables have different numbers of cells")
        trajectories = hio.load_cache(paths["trajectories"])
        examples = labels.read_example_cells()
        if args.smoke:
            examples = smoke_examples(examples, cells, trajectories)
        out.mkdir(parents=True, exist_ok=True)
        rows = {}

        def written(name):
            rec.add_output(out / name)

        # Fig 2A: direction tuning curves of five conjunctive cells
        rows["Fig2A"] = ex.example_rows(cells, examples[examples["panel"] == "Fig2A"].sort_values("position"))
        table, arrays = ex.fig2a_outputs(cells, dir_maps, rows["Fig2A"])
        hio.write_csv(table, out / "fig2a_cells.csv")
        hio.save_npz(out / "fig2a_cells.npz", arrays=arrays)
        written("fig2a_cells.csv")
        written("fig2a_cells.npz")

        # Fig 3A, 3B, 4A, 4B: example trajectories
        for name, panel, position, twin, with_alignment in TRAJECTORIES:
            e = one_example(examples, panel, position)
            e2 = one_example(examples, twin, position)
            if (e2["session_id"], int(e2["cell_id"]), e2["t_start_s"], e2["t_end_s"]) != \
                    (e["session_id"], int(e["cell_id"]), e["t_start_s"], e["t_end_s"]):
                raise ValueError(f"{panel} and {twin} position {position} must show the same cell and window")
            row = ex.example_rows(cells, e.to_frame().T)[0]
            rows[f"{panel}_{position}"] = row
            session = trajectories[cells.loc[row, "animal_id"]][cells.loc[row, "session_id"]]
            arrays, visits, sequence = ex.trajectory_outputs(session, cells, row, field_locations, full,
                                                             [e["t_start_s"], e["t_end_s"]], with_alignment)
            hio.save_npz(out / f"{name}.npz", arrays=arrays)
            hio.write_csv(visits, out / f"{name}_field_visits.csv")
            written(f"{name}.npz")
            written(f"{name}_field_visits.csv")
            rec.note(f"{name}.field_sequence_in_window", sequence)

        # Supp Fig 2A: maps with spikes shifted by the best shift, 0 ms and minus the best shift
        row = ex.example_rows(cells, one_example(examples, "SuppFig2A").to_frame().T)[0]
        rows["SuppFig2A"] = row
        session = trajectories[cells.loc[row, "animal_id"]][cells.loc[row, "session_id"]]
        try:
            arrays, best = ex.supp2a_outputs(session, cells, row, shift_ms, si_shift[row])
        except ValueError:
            if not args.smoke:
                raise
            arrays, best = ex.supp2a_outputs(session, cells, row, shift_ms, si_shift[row], shifts_ms=[0])
        if int(peak_shift["peak_shift_ms"].iloc[row]) != best:
            raise ValueError("peak_shift.csv disagrees with si_shift.npy for the Supp Fig 2A cell")
        hio.save_npz(out / "supp2a_shifted_maps.npz", arrays=arrays)
        written("supp2a_shifted_maps.npz")
        rec.note("supp2a.best_shift_ms", best)

        # Supp Fig 3A: rate maps, fields and lattice origins of twelve cells
        s3 = examples[examples["panel"] == "SuppFig3A"].sort_values("position")
        rows["SuppFig3A"] = ex.example_rows(cells, s3)
        table, arrays = ex.supp3a_outputs(cells, field_locations, full, rows["SuppFig3A"], s3["position"].tolist())
        hio.write_csv(table, out / "supp3a_cells.csv")
        hio.save_npz(out / "supp3a_cells.npz", arrays=arrays)
        written("supp3a_cells.csv")
        written("supp3a_cells.npz")

        rec.note("rows", {k: (list(map(int, v)) if isinstance(v, list) else int(v)) for k, v in rows.items()})
        if not args.smoke:
            checks = selection_checks(cells, peak_shift["peak_shift_ms"].to_numpy(), rows)
            rec.note("selection_checks", checks)
            for key, ok in checks.items():
                if not ok:
                    print(f"warning: an example cell does not follow its selection rule: {key}", file=sys.stderr)
        print(f"{STEP}: wrote {len(rec.outputs)} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
