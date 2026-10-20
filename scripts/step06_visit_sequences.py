"""Step 06: firing rate on successive field visits (Fig 3F).

Examples::

    python scripts/step06_visit_sequences.py --workers 4
    python scripts/step06_visit_sequences.py --rule library        # settings of the other rat analyses

Reads ``results/preprocessed/trajectories.pkl`` (step 01) and
``results/rat/cohort/cells.csv`` (step 02). For every cohort cell the fields are
delineated again and the field visits are extracted (``hexmod.rat.visit_sequences``). A cell
whose extraction raises an error stops the step (with the number of such cells and the first
error).
Sequences of three consecutive visits followed by a fourth are then compared between the group in
which the third visit follows the second sooner than the second followed the first, and the
opposite group.

Outputs (docs/PIPELINE.md, step 06) under ``results/rat/visits/``: ``paired_cells.csv``,
``summary.csv``, ``nesting.csv``, ``diagnostics.csv``, and the Fig 3F statistics in
``stats/step06_visit_sequences.json``.

``--rule printed`` (default) uses the extraction that produced the published Fig 3F: fields from a
rate map of spike counts and the gap-aware t_prev rule. ``--rule library`` uses the binary rate
map (the cohort fields) and the standard t_prev rule of the other rat analyses; its tables go to
``rat/visits/library_rule/``, its run record to ``logs/step06_visit_sequences_library_rule.json``,
and it writes no statistics file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hexmod.io import load_cache, read_csv, write_csv, write_json  # noqa: E402
from hexmod.parallel import map_ordered  # noqa: E402
from hexmod.rat import visit_sequences as vs  # noqa: E402
from hexmod.rat.gridcell import make_cell_data  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step06_visit_sequences"
ID_COLUMNS = ["row", "animal_id", "session_id", "cell_id", "unique_cell_id", "dataset"]
PAIRED_COLUMNS = ID_COLUMNS + ["frr_T23_lt_T12", "frr_T23_gt_T12", "n_events_T23_lt_T12", "n_events_T23_gt_T12",
                               "paired_difference", "antisymmetry_offset"]


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    p.add_argument("--rule", choices=sorted(vs.RULES), default="printed",
                   help="visit extraction: 'printed' (Fig 3F as published, default) or 'library' (binary rate "
                        "map and standard t_prev rule of the other rat analyses)")
    return p.parse_args(argv)


def extract_visits(cells, trajectories: Path, settings: dict, workers: int) -> dict:
    """Field visits of every cohort cell, in cohort order."""
    data = load_cache(trajectories)
    items = []
    for rec in cells[["row", "animal_id", "session_id", "cell_id"]].itertuples(index=False):
        session = data[rec.animal_id][rec.session_id]
        cell_data = make_cell_data(session, int(rec.cell_id), rec.animal_id, rec.session_id)
        items.append({"row": int(rec.row), "cell_data": cell_data, **settings})
    results = map_ordered(vs.visit_events_for_cell, items, workers, progress=False, desc="cells")
    del data, items

    columns = {col: [] for col in vs.EVENT_COLUMNS}
    event_rows = []
    per_cell = {k: [] for k in ("row", "n_events", "gridness", "scale_cm", "orientation_deg", "n_fields")}
    reasons = []
    errors = []
    for res in results:
        ev = res["events"]
        n = 0 if ev is None else len(ev["event_idx"])
        if ev is not None:
            for col in vs.EVENT_COLUMNS:
                columns[col].append(np.asarray(ev[col]))
            event_rows.append(np.full(n, res["row"], dtype=np.int64))
        per_cell["row"].append(res["row"])
        per_cell["n_events"].append(n)
        for k in ("gridness", "scale_cm", "orientation_deg", "n_fields"):
            per_cell[k].append(res[k])
        reasons.append("" if res["reason"] is None else str(res["reason"]))
        if res.get("error") is not None:
            errors.append((int(res["row"]), str(res["error"])))
    events = {"row": np.concatenate(event_rows) if event_rows else np.zeros(0, np.int64)}
    for col in vs.EVENT_COLUMNS:
        events[col] = np.concatenate(columns[col]) if columns[col] else np.zeros(0)
    cell_info = {k: np.asarray(v) for k, v in per_cell.items()}
    cell_info["reason"] = np.array(reasons, dtype=str)
    return {"events": events, "cells": cell_info, "errors": errors}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workers = config.workers(args.workers)
    res = config.results_dir()
    rule = args.rule
    settings = vs.extraction_settings(rule)
    record_name = STEP if rule == "printed" else STEP + "_library_rule"
    out_dir = res / "rat" / "visits" if rule == "printed" else res / "rat" / "visits" / "library_rule"
    trajectories = res / "preprocessed" / "trajectories.pkl"
    cells_path = res / "rat" / "cohort" / "cells.csv"

    with RunRecord(record_name, workers=workers, argv=argv) as rec:
        for path, step in ((trajectories, "01"), (cells_path, "02")):
            if not path.is_file():
                raise FileNotFoundError(f"{path} is missing; run step {step} first")
        rec.add_input(trajectories)
        rec.add_input(cells_path)
        cells = read_csv(cells_path)
        cells = cells[ID_COLUMNS].copy()
        rec.note("rule", rule)
        rec.note("settings", settings)

        # field visits
        t0 = time.perf_counter()
        print(f"extracting field visits of {len(cells)} cells with {workers} workers ({rule} rule)", flush=True)
        visits = extract_visits(cells, trajectories, settings, workers)
        errors = visits.pop("errors")
        if errors:
            row, first = errors[0]
            rec.note("cells_with_errors", [{"row": r, "error": e} for r, e in errors])
            raise RuntimeError(f"the visit extraction failed for {len(errors)} cells (first: cohort row "
                               f"{row}: {first})")
        rec.note("visit_extraction_s", round(time.perf_counter() - t0, 1))
        info = visits["cells"]
        reasons = Counter(r for r in info["reason"].tolist() if r)
        rec.note("n_cells_cohort", int(len(cells)))
        rec.note("n_cells_with_visits", int(np.sum(info["n_events"] > 0)))
        rec.note("n_cells_without_visits", dict(reasons))
        rec.note("n_visits", int(len(visits["events"]["row"])))

        # sequences, cell values and tests
        t1 = time.perf_counter()
        events = vs.events_frame(visits["events"], cells)
        all_results = vs.build_triplet_table(events, vs.FIG3F_SPECS)
        final = all_results[all_results["analysis"] == vs.FIG3F_ANALYSIS].copy()
        paired = vs.paired_cell_values(final)
        nesting = vs.nesting_sensitivity(paired)
        summary = vs.visit_sequence_summary(paired)
        diagnostics = [vs.sampling_diagnostic(all_results[all_results["analysis"] == s.name].copy(), s.name)
                       for s in vs.FIG3F_SPECS]
        rec.note("sequence_analysis_s", round(time.perf_counter() - t1, 1))

        paired_out = paired.sort_values("row", kind="mergesort")[PAIRED_COLUMNS].reset_index(drop=True)
        paired_out["row"] = paired_out["row"].astype(np.int64)
        paired_out["cell_id"] = paired_out["cell_id"].astype(np.int64)
        outputs = {
            "paired_cells.csv": paired_out,
            "summary.csv": pd.DataFrame([summary]),
            "nesting.csv": nesting,
            "diagnostics.csv": pd.DataFrame(diagnostics),
        }
        for name, table in outputs.items():
            write_csv(table, out_dir / name)
            rec.add_output(out_dir / name)
        if rule == "printed":
            stats_path = res / "stats" / f"{STEP}.json"
            write_json(vs.fig3f_statistics(summary), stats_path)
            rec.add_output(stats_path)

        rec.note("n_paired_cells", summary["n_paired_cells"])
        rec.note("t_shorter", summary["t_lt"])
        rec.note("t_longer", summary["t_gt"])
        rec.note("t_paired", summary["paired_t"])
        for level in ("session", "animal"):
            rec.note(f"paired_{level}_level", {k: vs.nesting_value(nesting, "all", level, k)
                                               for k in ("n", "t", "p_two_sided")})
        print(f"Fig 3F ({rule} rule): n = {summary['n_paired_cells']}; T23<T12 mean {summary['mean_lt']:.4f}, "
              f"t = {summary['t_lt']:.3f}, p = {summary['p_lt']:.4g}; T23>T12 mean {summary['mean_gt']:.4f}, "
              f"t = {summary['t_gt']:.3f}, p = {summary['p_gt']:.4g}; paired difference {summary['paired_diff']:.4f}, "
              f"t = {summary['paired_t']:.3f}, p = {summary['paired_p']:.4g}", flush=True)
        for level in ("session", "animal"):
            n, t, p = (vs.nesting_value(nesting, "all", level, k) for k in ("n", "t", "p_two_sided"))
            print(f"paired difference over {level} means: n = {n}, t = {t:.3f}, p = {p:.4g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
