"""Step 09: link repeated recordings of the same unit and keep the earliest recording of each unit.

Example::

    python scripts/step09_same_cell_links.py --workers 4

Reads ``rat/cohort/cells.csv`` and ``preprocessed/trajectories.pkl``, the shipped manifests and the
Vollan et al. (2025) session files of the animals recorded in more than one cohort session. Writes
``rat/same_cell/entries.csv`` (one row per cohort cell: its unit and whether it is the unit's earliest recording),
``units.csv`` (one row per unit) and ``links.csv`` (one row per accepted link). The link rules are described in
``hexmod.rat.same_cell``.
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

STEP = "step09_same_cell_links"
VOLLAN = "vollan2025"


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--workers", type=int, default=None, help="processes reading the Vollan files (default HEXMOD_WORKERS or 4)"
    )
    return p.parse_args(argv)


def _clean(obj):
    """Replace NaN by None in nested dicts and lists (run-record notes are plain JSON)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def vollan_groups(cells: pd.DataFrame):
    """Vollan animals with cohort cells in more than one session, their files and the file of each session.

    Returns
    -------
    groups : list of list of str
        Files (names without extension) of each animal in increasing session number; animals in canonical order.
    session_files : dict
        ``{session_id: file}`` for these files.
    paths : dict
        ``{file: raw path relative to the dataset folder}``.
    """
    from hexmod.preprocessing import labels
    from hexmod.rat.same_cell import session_number

    raw = {r["session_id"]: r["raw_path"] for r in labels.sessions_of(VOLLAN)}
    v = cells[cells["dataset"] == VOLLAN]
    groups, session_files, paths = [], {}, {}
    for animal in pd.unique(v["animal_id"]):
        sessions = sorted(pd.unique(v.loc[v["animal_id"] == animal, "session_id"]), key=session_number)
        if len(sessions) < 2:
            continue
        files = []
        for s in sessions:
            f = Path(raw[s]).stem
            files.append(f)
            session_files[s] = f
            paths[f] = raw[s]
        groups.append(files)
    return groups, session_files, paths


def tetrode_keys(cells: pd.DataFrame, trajectories: dict):
    """Rate maps and comparison features of the recordings of the tetrode datasets."""
    from hexmod import io
    from hexmod.preprocessing import labels
    from hexmod.rat.gridcell import make_cell_data
    from hexmod.rat.same_cell import hafting_label, tetrode_entry_maps, tetrode_rate_map

    t = cells[cells["dataset"] != VOLLAN]
    # grid orientation and scale enter the comparison after two CSV round trips
    # (docs/PIPELINE.md, Rounded per-cell inputs); this matters only for pairs exactly at a threshold
    ori = io.csv_float_roundtrip(t["orientation_deg"].to_numpy(float), 2)
    scale = io.csv_float_roundtrip(t["scale_cm"].to_numpy(float), 2)
    hafting = {}
    if (t["dataset"] == "hafting2005").any():
        for sid, rows in labels.units_of("hafting2005").items():
            for u in rows:
                hafting[(sid, int(u["cell_id"]))] = hafting_label(u["raw_unit_key"])
    centres, rows, maps = {}, [], []
    for k, r in enumerate(t.itertuples(index=False)):
        sd = trajectories[r.animal_id][r.session_id]
        cd = make_cell_data(sd, int(r.cell_id), r.animal_id, r.session_id)
        spk, dwell, centre, rate = tetrode_entry_maps(cd, centres.get(r.session_id))
        centres.setdefault(r.session_id, centre)
        maps.append(tetrode_rate_map(spk, dwell))
        rows.append(
            {
                "row": int(r.row),
                "animal_id": r.animal_id,
                "session_id": r.session_id,
                "cell_id": int(r.cell_id),
                "dataset": r.dataset,
                "orientation_deg": float(ori[k]),
                "scale_cm": float(scale[k]),
                "Ex_cm": float(centre[2]),
                "Ey_cm": float(centre[3]),
                "rate_hz": float(rate),
                "label": hafting.get((r.session_id, int(r.cell_id))) if r.dataset == "hafting2005" else None,
            }
        )
    keys = pd.DataFrame(
        rows,
        columns=[
            "row",
            "animal_id",
            "session_id",
            "cell_id",
            "dataset",
            "orientation_deg",
            "scale_cm",
            "Ex_cm",
            "Ey_cm",
            "rate_hz",
            "label",
        ],
    )
    return keys, (np.stack(maps) if maps else np.zeros((0, 50, 50)))


def link_table(vlinks: pd.DataFrame, file_sessions: dict, row_of: dict, tpairs: pd.DataFrame) -> pd.DataFrame:
    """One row per accepted link with the rows of both recordings (empty for units outside the cohort)."""
    out = []
    for r in vlinks.itertuples(index=False):
        sa, sb = file_sessions[r.file_a], file_sessions[r.file_b]
        ca, cb = int(r.idx_a), int(r.idx_b)
        if r.rule == "same_recording":
            score = float(r.f_coinc)
            ev = (
                f"coincident spike fraction {r.f_coinc:.4f}, chance {r.chance:.4g}, depth difference {r.dpos_um:.1f} um"
            )
        elif r.rule == "waveform_and_map":
            score = float(r.rm + r.w)
            ev = f"rate-map r {r.rm:.4f}, waveform r {r.w:.4f}, depth difference {r.dpos_um:.1f} um"
        else:
            score = float(r.rm)
            ev = f"rate-map r {r.rm:.4f}, depth difference {r.dpos_um:.1f} um"
        out.append(
            {
                "row_a": row_of.get((sa, ca)),
                "row_b": row_of.get((sb, cb)),
                "session_a": sa,
                "cell_a": ca,
                "session_b": sb,
                "cell_b": cb,
                "rule": r.rule,
                "evidence": ev,
                "score": score,
            }
        )
    for r in tpairs[tpairs["link"]].itertuples(index=False):
        if r.map_link:
            rule, score = "tetrode_map", float(r.rm)
            ev = (
                f"rate-map r {r.rm:.4f}, orientation difference {r.dori:.2f} deg, relative scale difference "
                f"{r.dscale_rel:.3f}"
            )
        else:
            rule, score = "tetrode_label", np.nan
            ev = f"same tetrode and cell label {r.label_a}"
        out.append(
            {
                "row_a": int(r.row_a),
                "row_b": int(r.row_b),
                "session_a": r.session_a,
                "cell_a": int(r.cell_a),
                "session_b": r.session_b,
                "cell_b": int(r.cell_b),
                "rule": rule,
                "evidence": ev,
                "score": score,
            }
        )
    cols = ["row_a", "row_b", "session_a", "cell_a", "session_b", "cell_b", "rule", "evidence", "score"]
    df = pd.DataFrame(out, columns=cols)
    for c in ("row_a", "row_b"):
        df[c] = df[c].astype("Int64")
    return df


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    from hexmod import io
    from hexmod.parallel import map_ordered
    from hexmod.preprocessing import labels
    from hexmod.runlog import RunRecord
    from hexmod.rat import same_cell

    workers = config.workers(args.workers)
    res = config.results_dir()
    out_dir = res / "rat" / "same_cell"
    cells_path = res / "rat" / "cohort" / "cells.csv"
    traj_path = res / "preprocessed" / "trajectories.pkl"
    print(f"results: {res}; workers: {workers}", flush=True)

    with RunRecord(STEP, workers=workers, argv=argv if argv is not None else sys.argv[1:]) as rec:
        rec.add_input(cells_path)
        cells = io.read_csv(cells_path)
        rec.add_input(labels.sessions_path())
        if (cells["dataset"] == "hafting2005").any():
            rec.add_input(labels.units_path())
        entries = list(zip(cells["session_id"], cells["cell_id"].astype(int)))
        row_of = {e: int(r) for e, r in zip(entries, cells["row"])}

        # ---------------------------------------------------------------- Vollan files of animals with several sessions
        groups, session_files, rel_paths = vollan_groups(cells)
        base = config.data_dir() / VOLLAN
        missing = [rel_paths[f] for fs in groups for f in fs if not (base / rel_paths[f]).is_file()]
        if missing:
            print(
                f"error: {len(missing)} Vollan et al. (2025) session files are missing under {base} "
                f"(first: {missing[0]}); run scripts/data_download.py",
                file=sys.stderr,
            )
            return 1
        jobs = [(str(base / rel_paths[f]), f) for fs in groups for f in fs]
        for job in jobs:
            rec.add_input(Path(job[0]), hash_it=False)
        recs = dict(
            zip(
                [j[1] for j in jobs],
                map_ordered(same_cell.extract_vollan_units_job, jobs, workers, desc="Vollan files"),
            )
        )
        print(
            f"Vollan: {len(groups)} animals, {len(jobs)} files, "
            f"{sum(len(r['units']) for r in recs.values())} grid units read",
            flush=True,
        )
        vlinks, vdetails = same_cell.link_vollan(recs, groups)
        print(f"Vollan links: {len(vlinks)} ({vlinks['rule'].value_counts().to_dict()})", flush=True)

        # ---------------------------------------------------------------- tetrode datasets
        tetrode_cells = cells[cells["dataset"] != VOLLAN]
        if len(tetrode_cells):
            rec.add_input(traj_path, hash_it=False)
            trajectories = io.load_cache(traj_path)
            keys, maps = tetrode_keys(cells, trajectories)
            del trajectories
        else:
            keys, maps = tetrode_keys(cells.iloc[:0], {})
        tpairs, tthr = same_cell.link_tetrode(keys, maps)
        print(
            f"tetrode recordings: {len(keys)}; pairs compared: {len(tpairs)}; links: {int(tpairs['link'].sum())}",
            flush=True,
        )

        # ---------------------------------------------------------------- units and kept recordings
        entry_links = [
            ((r.session_a, int(r.cell_a)), (r.session_b, int(r.cell_b)))
            for r in tpairs[tpairs["link"]].itertuples(index=False)
        ]
        units = same_cell.unit_components(entries, vlinks, session_files, entry_links)
        snum = [same_cell.session_number(s) for s in cells["session_id"]]
        kept = same_cell.kept_earliest(units, snum)
        n_entries = np.bincount(units)[units]
        ent = pd.DataFrame(
            {
                "row": cells["row"].astype(np.int64),
                "animal_id": cells["animal_id"],
                "session_id": cells["session_id"],
                "cell_id": cells["cell_id"].astype(np.int64),
                "unit": units,
                "n_entries": n_entries.astype(np.int64),
                "kept_earliest": kept,
            }
        )
        rows_by_unit = {}
        for u, r in zip(units.tolist(), ent["row"].tolist()):
            rows_by_unit.setdefault(u, []).append(r)
        kept_row = dict(zip(units[kept].tolist(), ent["row"][kept].tolist()))
        first = {}
        for k, u in enumerate(units.tolist()):
            first.setdefault(u, k)
        uni = pd.DataFrame(
            {
                "unit": np.arange(len(rows_by_unit), dtype=np.int64),
                "animal_id": [cells["animal_id"].iloc[first[u]] for u in range(len(rows_by_unit))],
                "dataset": [cells["dataset"].iloc[first[u]] for u in range(len(rows_by_unit))],
                "kept_row": [kept_row[u] for u in range(len(rows_by_unit))],
                "n_entries": [len(rows_by_unit[u]) for u in range(len(rows_by_unit))],
                "rows": [";".join(str(r) for r in rows_by_unit[u]) for u in range(len(rows_by_unit))],
            }
        )
        file_sessions = {f: s for s, f in session_files.items()}
        links = link_table(vlinks, file_sessions, row_of, tpairs)

        for name, df in (("entries.csv", ent), ("units.csv", uni), ("links.csv", links)):
            io.write_csv(df, out_dir / name)
            rec.add_output(out_dir / name)
        counts = {
            "n_cells": int(len(ent)),
            "n_units": int(len(uni)),
            "n_cells_in_units_with_several_recordings": int((n_entries > 1).sum()),
            "n_links": int(len(links)),
            "links_by_rule": {k: int(v) for k, v in links["rule"].value_counts().items()},
            "units_by_dataset": {k: int(v) for k, v in uni["dataset"].value_counts().items()},
            "conjunctive_cells_kept": int((kept & (cells["Rayleigh_mag"].to_numpy(float) > 0.3)).sum()),
        }
        rec.note("counts", counts)
        rec.note("vollan_thresholds", _clean(vdetails["thresholds"]))
        rec.note("vollan_file_pairs", _clean(vdetails["pairs"]))
        rec.note("tetrode_thresholds", _clean(tthr))
        print(
            f"units: {counts['n_units']} from {counts['n_cells']} cells; links {counts['n_links']}; "
            f"conjunctive cells kept {counts['conjunctive_cells_kept']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
