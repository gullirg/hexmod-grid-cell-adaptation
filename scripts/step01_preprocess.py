"""Step 01: preprocess the raw recordings into the trajectories used by every analysis.

Examples::

    python scripts/step01_preprocess.py --workers 4
    python scripts/step01_preprocess.py --smoke          # synthetic recordings, no data needed

Reads the raw files of the six datasets (``HEXMOD_DATA`` or ``--data``) and the manifests in
``inputs/manifests/``. Writes to ``results/preprocessed/``:

- ``<dataset>.pkl``: the recordings of one dataset before smoothing (``hexmod.preprocessing.<dataset>``);
- ``trajectories.pkl``: the six datasets merged in canonical order, with smoothed positions,
  running speed and movement direction (``hexmod.preprocessing.trajectories``);
- ``sessions.csv``: one row per session.

The layout of these files is given in docs/PIPELINE.md (step 01); load the pickles with
``hexmod.io.load_cache``. Only the open-field files of Vollan et al. (2025) are read in parallel
(``--workers``); each worker holds one file of up to a few GB in memory. Before any work the step checks
that every raw file is present with the size given in its manifest; ``scripts/data_check.py`` also
compares the checksums.

``--smoke`` replaces the raw data by synthetic recordings (grid cells along a random walk, under the
labels of the three sessions of Hafting et al. (2005)), so that the pipeline can be tested without data.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

import numpy as np  # noqa: E402

from hexmod import data_sources, datasets, io  # noqa: E402
from hexmod.preprocessing import check_file_sizes, dataset_module, labels, manifest_files, trajectories  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step01_preprocess"
SMOKE_DATASET = "hafting2005"


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None,
                   help="files of Vollan et al. read in parallel (default HEXMOD_WORKERS or 4)")
    p.add_argument("--data", default=None, metavar="DIR", help="raw-data folder (default HEXMOD_DATA or <repo>/data)")
    p.add_argument("--smoke", action="store_true", help="synthetic recordings instead of the raw data")
    return p.parse_args(argv)


def peak_memory_mb() -> Dict[str, float]:
    """Peak resident memory of this process and of its finished worker processes (MB)."""
    import resource

    scale = 1 / (1024 * 1024) if sys.platform == "darwin" else 1 / 1024  # bytes on macOS, kB on Linux
    return {"main": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale, 1),
            "workers": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale, 1)}


def session_table(key: str, data: dict):
    """Return the rows of ``preprocessed/sessions.csv`` for one dataset."""
    source = {s["session_id"]: s["raw_path"] for s in labels.sessions_of(key)}
    rows = []
    for animal_id, sessions in data.items():
        for session_id, session in sessions.items():
            t = session["position"]["time"]
            rows.append({
                "dataset": key, "animal_id": animal_id, "session_id": session_id,
                "source": source.get(session_id, ""),
                "n_units": len(session["spike_times"]),
                "n_samples": int(len(t)),
                "sample_rate_hz": float(session["position"]["sample_rate"]),
                "duration_s": float(t[-1] - t[0]),
            })
    return rows


def smoke_recordings(seed: int = 0) -> dict:
    """Synthetic grid-cell recordings under the labels of the Hafting et al. (2005) sessions.

    Each session is 12 min of a random walk at 50 Hz in a 100 cm box; each unit fires as a grid cell
    (random spacing, orientation and phase; peak rate 12 Hz) with Bernoulli spikes per sample.
    """
    rng = np.random.default_rng(seed)
    units = labels.units_of(SMOKE_DATASET)
    data: Dict[str, dict] = {}
    n, dt, box = 36000, 0.02, 100.0
    t = np.arange(n) * dt
    for s in labels.sessions_of(SMOKE_DATASET):
        xy = np.empty((n, 2))
        pos, heading = np.array([box / 2, box / 2]), rng.uniform(0, 2 * np.pi)
        speeds = np.clip(rng.normal(20, 6, n), 2, 50)
        turns = rng.normal(0, 0.25, n)
        for i in range(n):
            heading += turns[i]
            step = speeds[i] * dt * np.array([np.cos(heading), np.sin(heading)])
            nxt = pos + step
            if not (0 <= nxt[0] <= box and 0 <= nxt[1] <= box):
                heading += np.pi
                nxt = np.clip(pos - step, 0, box)
            pos = nxt
            xy[i] = pos
        session = {"position": {"xy": xy, "time": t.copy(), "sample_rate": 50}, "spike_times": {}, "mod_id": {},
                   "unique_cell_ids": {}, "tetrode_ids": {}, "animal_id": s["animal_id"],
                   "session_id": s["session_id"]}
        for u in units[s["session_id"]]:
            spacing, ori = rng.uniform(35, 60), rng.uniform(0, np.pi / 3)
            phase = rng.uniform(0, spacing, 2)
            k = 4 * np.pi / (np.sqrt(3) * spacing)
            g = sum(np.cos(k * ((xy - phase) @ np.array([np.cos(ori + j * np.pi / 3), np.sin(ori + j * np.pi / 3)])))
                    for j in range(3))
            rate = 12.0 * ((g + 1.5) / 4.5) ** 3
            fired = rng.random(n) < rate * dt
            spikes = np.clip(t[fired] - rng.uniform(0, dt, int(fired.sum())), 0, None)
            c = u["cell_id"]
            session["spike_times"][c] = spikes
            session["mod_id"][c] = 1
            session["unique_cell_ids"][c] = u["unique_cell_id"]
            session["tetrode_ids"][c] = -1
        data.setdefault(s["animal_id"], {})[s["session_id"]] = session
    return labels.canonical_order(data, SMOKE_DATASET)


def check_raw_files(keys: List[str], base: Path) -> None:
    """Stop before any work when a raw file of the given datasets is missing or has the wrong size."""
    problems = []
    for key in keys:
        try:
            check_file_sizes(manifest_files(key), base)
        except FileNotFoundError as exc:
            problems.append(f"{key}: {exc}")
    if problems:
        raise FileNotFoundError("; ".join(problems) + f" (data folder {base}; see scripts/data_download.py)")


def run(base: Path, workers: int, smoke: bool, rec: RunRecord) -> None:
    out = config.results_dir() / "preprocessed"
    out.mkdir(parents=True, exist_ok=True)
    keys = [SMOKE_DATASET] if smoke else list(datasets.CANONICAL_ORDER)
    if not smoke:
        check_raw_files(keys, base)
    for name in ("sessions.csv", "units.csv"):
        rec.add_input(config.inputs_dir() / "manifests" / name)
    timings: Dict[str, float] = {}
    counts: Dict[str, Dict[str, int]] = {}
    table = []
    for key in keys:
        t0 = time.perf_counter()
        if smoke:
            data = smoke_recordings()
        else:
            rec.add_input(data_sources.manifest_path(key))
            for row in data_sources.final_files(data_sources.read_manifest(key)):
                rec.add_input(base / row["path"], hash_it=False)
            print(f"{key}: reading raw files", flush=True)
            data = dataset_module(key).preprocess(base, workers=workers)
        path = io.save_cache(data, out / f"{key}.pkl")
        rec.add_output(path)
        table += session_table(key, data)
        counts[key] = {"animals": len(data), "sessions": sum(len(v) for v in data.values()),
                       "units": sum(len(s["spike_times"]) for v in data.values() for s in v.values())}
        timings[key] = round(time.perf_counter() - t0, 1)
        print(f"{key}: {counts[key]['animals']} animals, {counts[key]['sessions']} sessions, "
              f"{counts[key]['units']} units ({timings[key]} s)", flush=True)
        del data
        gc.collect()

    t0 = time.perf_counter()
    per_dataset = {key: io.load_cache(out / f"{key}.pkl") for key in keys}
    merged = trajectories.merge_and_smooth(per_dataset, strict=True)
    path = io.save_cache(merged, out / "trajectories.pkl")
    rec.add_output(path)
    timings["trajectories"] = round(time.perf_counter() - t0, 1)
    import pandas as pd

    path = io.write_csv(pd.DataFrame(table, columns=["dataset", "animal_id", "session_id", "source", "n_units",
                                                     "n_samples", "sample_rate_hz", "duration_s"]),
                        out / "sessions.csv")
    rec.add_output(path)
    rec.note("datasets", counts)
    rec.note("runtime_s_by_part", timings)
    rec.note("smoke", bool(smoke))
    rec.note("peak_memory_mb", peak_memory_mb())
    total = {k: sum(c[k] for c in counts.values()) for k in ("animals", "sessions", "units")}
    print(f"preprocessed {total['animals']} animals, {total['sessions']} sessions, {total['units']} units "
          f"into {out}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base = Path(args.data).expanduser() if args.data else config.data_dir()
    if args.data:
        os.environ["HEXMOD_DATA"] = str(base)
    config.guard_smoke_results(args.smoke)
    workers = config.workers(args.workers)
    try:
        with RunRecord(STEP, workers=workers, argv=argv if argv is not None else sys.argv[1:],
                       hash_inputs=True) as rec:
            run(base, workers, args.smoke, rec)
    except (FileNotFoundError, labels.ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
