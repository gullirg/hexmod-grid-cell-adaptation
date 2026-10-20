"""Step 24: simulated participants of Fig 7 (one file per participant).

Examples::

    python scripts/step24_simulation_participants.py --module small --seeds 0:1500 --workers 12
    python scripts/step24_simulation_participants.py --module large --seeds 0:1500 --workers 12
    python scripts/step24_simulation_participants.py --quick          # participants 0 to 29 of both modules

Participant s of a module (pools of step 21) has the 10,000 cells
``default_rng(s).choice(100000, 10000, replace=False)`` and the trajectory of
``hexmod.sim.trajectory_v1.make_trajectory(par, base_seed=s)`` (object-location task, four blocks
of 600 s); there is no other random draw. Rates are simulated with adaptation 0.1, time constant 2 s
and no speed gain, and the hexadirectional modulation of the population rate is estimated for
symmetries 1 to 7 with the simple and the location-controlled estimator
(``hexmod.sim.simulate.simulate_participant``, cells simulated in chunks of 1,000). One participant
takes about 85 s (small module) or 65 s (large module) on one core and 1 GB of memory; all 1,500
participants of both modules take about 62 worker hours.

Writes ``sim/fig7/participants/<module>/p0000.npz`` (``p`` and the participant number in four
digits). Each npz holds every entry of the participant result as plain arrays:

- ``seed``, ``module`` (0-d), ``alpha``, ``tau``, ``n_samples``, ``n_moving``,
  ``FR_adapt_mean_move``, ``FR_noadapt_mean_move``, ``tprev_HMness``, ``gridness_mean``;
- ``FR_adapt``, ``FR_noadapt`` (moving samples) [Hz], ``tprevby12dir`` (12,),
  ``tprevbyAlignMis_nanmean`` (2,), ``ratemap_sum`` (34, 34), ``foldSymList`` (1 to 7);
- ``meta_n_cells`` (cells simulated) and ``meta_pool_fingerprint`` (``hexmod.sim.io_pool.pool_fingerprint``
  of the pool the cells were drawn from);
- the two fit tables column by column: ``crossval__<column>`` (112 rows: 2 models x 2 estimators x
  4 left-out blocks x 7 symmetries) and ``noncross__<column>`` (4 rows), with the column order in
  ``crossval__columns`` and ``noncross__columns``. Columns holding one array per row are stored as a
  2-d array when every row has one (``noncross__FRby12dir``), otherwise as ``<key>__data`` and
  ``<key>__shapes`` (``hexmod.io.pack_ragged``) with an empty array for a row without one
  (``crossval__FRby12dir_test`` exists only for six-fold rows; ``noncross__speedby12dir`` only for
  the simple estimator).

``load_participant`` in ``step25_simulation_collect.py`` rebuilds the result (with the two tables as
pandas DataFrames). A rerun skips participants whose file is present, readable and was computed
with the same number of cells and the same pool, so an interrupted run can be resumed with the same
command.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hexmod import io as hio  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402
from hexmod.sim.io_pool import MODULES  # noqa: E402

STEP = "step24_simulation_participants"
N_PARTICIPANTS = 1500
N_CELLS = 10000
ALPHA = 0.1
CHUNK = 1000
QUICK_SEEDS = (0, 30)
TABLES = {"crossvalResult": "crossval", "noncrossResult": "noncross"}
#: Entries that describe how a participant file was computed (checked before a file is reused).
META_KEYS = ("meta_n_cells", "meta_pool_fingerprint")


def participants_dir(module: str) -> Path:
    """``results/sim/fig7/participants/<module>``."""
    return config.results_dir() / "sim" / "fig7" / "participants" / module


def participant_file(seed: int) -> str:
    """File name of participant ``seed`` (``p0000.npz``)."""
    return f"p{int(seed):04d}.npz"


def parse_seeds(text: str) -> range:
    """``"a:b"`` to ``range(a, b)``."""
    try:
        a, b = (int(v) for v in text.split(":"))
    except ValueError:
        raise SystemExit(f"error: --seeds takes A:B (for example 0:1500), got {text!r}")
    if not 0 <= a < b:
        raise SystemExit(f"error: --seeds needs 0 <= A < B, got {text!r}")
    return range(a, b)


def load_pool(module: str, rec: Optional[RunRecord] = None):
    """Pool of a module written by step 21 (``hexmod.sim.io_pool.load_module_pool``)."""
    from hexmod.sim import io_pool

    table, path = io_pool.load_module_pool(config.results_dir(), module)
    if rec is not None:
        rec.add_input(path)
    return table


def participant_metadata(n_cells: int, pool_fingerprint: str) -> Dict[str, object]:
    """The ``meta_*`` entries of a participant file computed now from a pool with this fingerprint."""
    return {"meta_n_cells": int(n_cells), "meta_pool_fingerprint": str(pool_fingerprint)}


def read_metadata(path: Path) -> Dict[str, object]:
    """The ``meta_*`` entries of a participant file (missing entries are left out)."""
    out: Dict[str, object] = {}
    with np.load(path, allow_pickle=False) as z:
        for key in META_KEYS:
            if key in z.files:
                value = z[key].item()
                out[key] = int(value) if key == "meta_n_cells" else str(value)
    return out


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def table_to_arrays(table: pd.DataFrame, prefix: str):
    """Arrays (and ragged lists) holding a fit table column by column (see the module docstring)."""
    arrays: Dict[str, np.ndarray] = {f"{prefix}__columns": np.array([str(c) for c in table.columns], dtype=str)}
    ragged: Dict[str, List[np.ndarray]] = {}
    for col in table.columns:
        key = f"{prefix}__{col}"
        values = table[col].to_numpy()
        if values.dtype != object:
            arrays[key] = values
        elif all(isinstance(v, str) for v in values):
            arrays[key] = values.astype(str)
        else:
            items = []
            for v in values:
                if isinstance(v, np.ndarray):
                    items.append(v)
                elif _is_missing(v):
                    items.append(None)
                else:
                    raise TypeError(f"column {col!r} holds a {type(v).__name__}")
            shapes = {v.shape for v in items if v is not None}
            if all(v is not None for v in items) and len(shapes) == 1:
                arrays[key] = np.stack(items)
            else:
                ragged[key] = [v if v is not None else np.zeros(0) for v in items]
    return arrays, ragged


def record_to_arrays(record: dict, module: str):
    """Arrays and ragged lists of one participant result (``simulate.simulate_and_measure``)."""
    arrays: Dict[str, np.ndarray] = {"module": np.array(module)}
    ragged: Dict[str, List[np.ndarray]] = {}
    for key, value in record.items():
        if key in TABLES:
            a, r = table_to_arrays(value, TABLES[key])
            arrays.update(a)
            ragged.update(r)
        elif key == "foldSymList":
            arrays[key] = np.asarray(value, dtype=np.int64)
        else:
            value = np.asarray(value)
            if value.dtype == object:
                raise TypeError(f"entry {key!r} is not numeric")
            arrays[key] = value
    return arrays, ragged


def save_participant(record: dict, module: str, path: Path, metadata: Optional[Dict[str, object]] = None) -> Path:
    """Write one participant result (and its ``meta_*`` entries) as an npz file."""
    arrays, ragged = record_to_arrays(record, module)
    for key, value in (metadata or {}).items():
        arrays[key] = np.array(value)
    return hio.save_npz(path, arrays=arrays, ragged=ragged)


def is_complete(path: Path, seed: int, module: str, metadata: Optional[Dict[str, object]] = None) -> bool:
    """True when ``path`` is a readable participant file of ``seed`` and ``module``.

    With ``metadata`` (``participant_metadata``), the file must also hold the same ``meta_*`` entries:
    the same number of cells and pool.
    """
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            ok = int(z["seed"]) == int(seed) and str(z["module"]) == module and "crossval__beta" in z.files
        return ok and (metadata is None or read_metadata(path) == dict(metadata))
    except Exception:  # an unreadable or truncated file is recomputed
        return False


def run_module(module: str, seeds: Sequence[int], workers: int, rec: RunRecord) -> dict:
    """Simulate the participants ``seeds`` of one module that are not yet on disk."""
    from hexmod.parallel import map_ordered
    from hexmod.sim import io_pool, pool, simulate

    folder = participants_dir(module)
    folder.mkdir(parents=True, exist_ok=True)
    pool_table = load_pool(module, rec)
    metadata = participant_metadata(N_CELLS, io_pool.pool_fingerprint(pool_table))
    todo = [s for s in seeds if not is_complete(folder / participant_file(s), s, module, metadata)]
    info = {"requested": len(seeds), "already_present": len(seeds) - len(todo), "computed": 0, "seconds": 0.0,
            "pool_fingerprint": metadata["meta_pool_fingerprint"]}
    if not todo:
        return info
    par = pool.module_param(module, adaptEff_mean=ALPHA, speedGamma_mean=0, nGridCells=N_CELLS)
    batch = 3 * workers  # participants handed to the workers per round
    t0 = time.perf_counter()
    for b0 in range(0, len(todo), batch):
        part = todo[b0:b0 + batch]
        items = [(par, simulate.participant_cells(pool_table, par, s), s, CHUNK, tuple(simulate.FOLD_SYM_LIST))
                 for s in part]
        results = map_ordered(simulate.simulate_participant, items, workers=workers, desc=f"{module} participants")
        for s, record in zip(part, results):
            if int(record["seed"]) != int(s):
                raise RuntimeError(f"participant {s}: result of participant {record['seed']}")
            save_participant(record, module, folder / participant_file(s), metadata)
        del items, results
        info["computed"] += len(part)
        elapsed = time.perf_counter() - t0
        print(f"{module}: {info['computed']} of {len(todo)} participants done ({elapsed / 60:.1f} min)", flush=True)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    return info


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--module", choices=["small", "large", "both"], default="both", help="module(s) to simulate")
    p.add_argument("--seeds", default=f"0:{N_PARTICIPANTS}", metavar="A:B",
                   help=f"participants A to B - 1 (default 0:{N_PARTICIPANTS})")
    p.add_argument("--quick", action="store_true", help="participants 0 to 29 (the first experiment) only")
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workers = config.workers(args.workers)
    seeds = range(*QUICK_SEEDS) if args.quick else parse_seeds(args.seeds)
    if seeds.stop > N_PARTICIPANTS:
        print(f"note: participants above {N_PARTICIPANTS - 1} are not part of the published results", file=sys.stderr)
    modules = MODULES if args.module == "both" else (args.module,)
    with RunRecord(STEP, workers=workers, argv=argv) as rec:
        for m in modules:
            info = run_module(m, list(seeds), workers, rec)
            rec.note(m, {"seeds": [seeds.start, seeds.stop], **info})
            rec.add_output(participants_dir(m))
        rec.note("cells_per_participant", N_CELLS)
        rec.note("symmetries", list(range(1, 8)))
    print(f"participant files in {participants_dir(modules[0]).parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
