"""Step 02: grid cell cohort and characterisation.

Example::

    python scripts/step02_cohort.py --workers 8

Reads ``results/preprocessed/trajectories.pkl`` (step 01). A session enters the cohort
when it has at least 30,000 position samples (10 min at 50 Hz) with running speed above 5 cm/s.
Every unit of these sessions is characterised with
:func:`hexmod.rat.characterise.characterise_gridcell` (rate map, autocorrelogram, gridness,
scale, orientation, fields with a minimum peak distance of 2/3 of the scale and a rate threshold
at the 40th percentile, movement-direction tuning) and kept as a grid cell when its gridness is
at least 0.3. Cells are in canonical order (the order of the preprocessed data).

Writes under ``results/``: ``rat/cohort/cells.csv``, ``dir_maps.npy``,
``field_locations.npz``, ``cells_full.npz``, ``sessions.csv``, ``table1.csv`` and
``stats/step02_cohort.json`` (its printed keys are listed in docs/STATISTICS.md).
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

from hexmod import datasets as ds  # noqa: E402
from hexmod.io import load_cache, save_npy, save_npz, write_csv, write_json  # noqa: E402
from hexmod.parallel import map_ordered  # noqa: E402
from hexmod.preprocessing import labels  # noqa: E402
from hexmod.rat import paper_stats  # noqa: E402
from hexmod.rat.characterise import characterise_gridcell  # noqa: E402
from hexmod.rat.gridcell import make_cell_data  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step02_cohort"

#: Minimum number of position samples with speed above ``MIN_SPEED_CM_S`` for a session to enter.
SAMPLE_RATE_HZ = 50
MIN_FAST_SAMPLES = SAMPLE_RATE_HZ * 60 * 10
MIN_SPEED_CM_S = 5
#: Field delineation settings of the characterisation.
MIN_DIST_BTW_PEAKS_REL = 2 / 3
PROMINENCE = 0.4

CELL_COLUMNS = ["row", "animal_id", "session_id", "cell_id", "unique_cell_id", "dataset", "gridness", "scale_cm",
                "orientation_deg", "Rayleigh_mag", "Rayleigh_angle", "meanFR_all", "meanFR_infield", "meanFR_outfield",
                "n_fields"]


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    return p.parse_args(argv)


def cohort_rule_statistics() -> dict:
    """The session rule as statistics: samples above the speed threshold, and the running time they make at 50 Hz."""
    return {"cohort.rule.min_samples_above_speed": int(MIN_FAST_SAMPLES),
            "cohort.rule.speed_threshold_cm_s": int(MIN_SPEED_CM_S),
            "cohort.rule.min_running_time_min": MIN_FAST_SAMPLES / SAMPLE_RATE_HZ / 60.0}


def cohort_sessions(data, dataset_keys):
    """Return ``(animal_id, session_id, session)`` of the sessions that pass the sample rule."""
    out = []
    for animal_id, sessions in data.items():
        if ds.key_from_label(animal_id) not in dataset_keys:
            continue
        for session_id, session in sessions.items():
            if (session["position"]["speed"] > MIN_SPEED_CM_S).sum() >= MIN_FAST_SAMPLES:
                out.append((animal_id, session_id, session))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workers = config.workers(args.workers)
    res = config.results_dir()
    keys = list(ds.CANONICAL_ORDER)
    traj_path = res / "preprocessed" / "trajectories.pkl"
    out_dir = res / "rat" / "cohort"
    text_path = config.inputs_dir() / "rat" / "table1_text.csv"

    with RunRecord(STEP, workers=workers, argv=argv if argv is not None else sys.argv[1:]) as rec:
        rec.add_input(traj_path)
        rec.add_input(text_path)
        data = load_cache(traj_path)

        sessions = cohort_sessions(data, keys)
        tasks = [make_cell_data(session, cell_id, animal_id, session_id)
                 for animal_id, session_id, session in sessions for cell_id in session.get("spike_times", {})]
        print(f"{len(tasks)} units in {len(sessions)} sessions with at least {MIN_FAST_SAMPLES} samples above "
              f"{MIN_SPEED_CM_S} cm/s", flush=True)
        inputs = [{"cell_data": cd, "min_dist_btw_peaks_rel": MIN_DIST_BTW_PEAKS_REL, "prominence": PROMINENCE}
                  for cd in tasks]
        results = map_ordered(characterise_gridcell, inputs, workers, progress=True, desc="characterise")
        grid = [r for r in results if r is not None]
        # a grid cell without any delineated field cannot enter the field-based analyses
        no_field = [r for r in grid if "fieldLoc" not in r]
        grid = [r for r in grid if "fieldLoc" in r]

        cells = pd.DataFrame({
            "row": np.arange(len(grid), dtype=np.int64),
            "animal_id": [r["animal_id"] for r in grid],
            "session_id": [r["session_id"] for r in grid],
            "cell_id": np.array([r["cell_id"] for r in grid], dtype=np.int64),
            "unique_cell_id": [r["unique_cell_id"] for r in grid],
            "dataset": [ds.key_from_label(r["animal_id"]) for r in grid],
            **{c: np.array([r[c] for r in grid], dtype=float)
               for c in ["gridness", "scale_cm", "orientation_deg", "Rayleigh_mag", "Rayleigh_angle", "meanFR_all",
                         "meanFR_infield", "meanFR_outfield"]},
            "n_fields": np.array([r["fieldLoc"].shape[0] for r in grid], dtype=np.int64),
        }, columns=CELL_COLUMNS)
        if len(cells) and not labels.is_canonical(cells):
            raise labels.ManifestError("the cohort cells are not in the canonical order of the unit manifest")

        rows = cells["row"].to_numpy()
        write_csv(cells, out_dir / "cells.csv")
        save_npy(out_dir / "dir_maps.npy", np.row_stack([r["dir_map"] for r in grid]) if grid
                 else np.zeros((0, 72)))
        save_npz(out_dir / "field_locations.npz", {"row": rows},
                 ragged={"field_locations": [r["fieldLoc"] for r in grid]})
        save_npz(out_dir / "cells_full.npz", {"row": rows},
                 ragged={"rate_map": [r["rate_map"] for r in grid], "x_edges": [r["x_edges"] for r in grid],
                         "y_edges": [r["y_edges"] for r in grid],
                         "field_rate_hz": [r["FR_inhomogeneity_regionFR"] for r in grid],
                         "field_deviation": [r["distFromPerfectGrid"] for r in grid]})

        conj = paper_stats.conjunctive_mask(cells["Rayleigh_mag"])
        srows = []
        for animal_id, session_id, session in sessions:
            m = (cells["session_id"] == session_id).to_numpy()
            if not m.any():
                continue
            pos = session["position"]
            srows.append({"dataset": ds.key_from_label(animal_id), "animal_id": animal_id, "session_id": session_id,
                          "n_units": len(session.get("spike_times", {})), "n_cells": int(m.sum()),
                          "n_conjunctive": int((m & conj).sum()), "n_samples": int(len(pos["time"])),
                          "n_samples_speed_above_5": int((pos["speed"] > MIN_SPEED_CM_S).sum()),
                          "duration_s": float(pos["time"][-1] - pos["time"][0])})
        write_csv(pd.DataFrame(srows, columns=["dataset", "animal_id", "session_id", "n_units", "n_cells",
                                               "n_conjunctive", "n_samples", "n_samples_speed_above_5",
                                               "duration_s"]), out_dir / "sessions.csv")
        text = pd.read_csv(text_path, dtype=str, keep_default_na=False)
        write_csv(paper_stats.table1(cells, text, keys), out_dir / "table1.csv")
        stats = paper_stats.cohort_statistics(cells, keys)
        stats.update(cohort_rule_statistics())
        write_json(stats, res / "stats" / f"{STEP}.json")

        for name in ["cells.csv", "dir_maps.npy", "field_locations.npz", "cells_full.npz", "sessions.csv",
                     "table1.csv"]:
            rec.add_output(out_dir / name)
        rec.add_output(res / "stats" / f"{STEP}.json")
        rec.note("n_sessions_passing_sample_rule", len(sessions))
        rec.note("n_units_characterised", len(tasks))
        rec.note("n_grid_cells", int(len(cells)))
        rec.note("n_grid_cells_without_fields_excluded", len(no_field))
        rec.note("n_animals", stats["cohort.n_animals"])
        rec.note("n_sessions", stats["cohort.n_sessions"])
        rec.note("n_conjunctive", stats["cohort.conjunctive.n_cells"])
    print(f"cohort: {stats['cohort.n_cells']} grid cells, {stats['cohort.n_animals']} animals, "
          f"{stats['cohort.n_sessions']} sessions; conjunctive {stats['cohort.conjunctive.n_cells']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
