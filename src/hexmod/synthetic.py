"""Synthetic recordings made with the grid-cell simulator, for the smoke run.

``scripts/run_pipeline.py --smoke`` runs the rat analyses on these recordings instead of the raw data, so
that the whole pipeline can be exercised without any recorded data. Nothing here is derived from a
recording.

Model. Each session is a random walk of 12 min sampled at 50 Hz in a 100 cm square box. Each unit is a
simulated grid cell of :mod:`hexmod.sim`: a cell of a pool built by
:func:`hexmod.sim.pool.set_gridcells_pool` (Gaussian fields on a jittered triangular lattice with
field-to-field rate variability; one grid scale and orientation per session), whose rate along the walk
is computed by :func:`hexmod.sim.simulate.simulate_grid_cells` with firing-rate adaptation::

    rate = RATE_GAIN * map(position) * (1 - ADAPTATION * exp(-t_prev / ADAPTATION_TAU_S)),

where ``t_prev`` is the time since the walk left the previously visited field of the cell. The
conjunctive cells of a session are in addition tuned to movement direction by a von Mises factor
``exp(kappa cos(theta - preferred)) / I0(kappa)``. Spikes are Bernoulli events per position sample.
The true parameters of every unit are returned by :func:`ground_truth` and stored in the run record of
the smoke run (``logs/smoke_recordings.json``, note ``units``).

Labels. The recordings carry the labels of real sessions and units of the shipped manifests
(``SMOKE_SESSIONS``), so that every step can look up animals, sessions and units as usual; their
content is entirely simulated.

Files. :func:`write_recordings` writes the files of step 01 (docs/PIPELINE.md):
``preprocessed/<dataset>.pkl``, ``preprocessed/trajectories.pkl`` and ``preprocessed/sessions.csv``.
:func:`write_separate_units` writes ``rat/same_cell/entries.csv`` and ``units.csv`` with every cohort
cell as a unit of its own (true by construction), which the smoke run uses in place of step 09.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

__all__ = ["SMOKE_SESSIONS", "SAMPLE_RATE_HZ", "DURATION_S", "BOX_CM", "ADAPTATION", "ADAPTATION_TAU_S",
           "RATE_GAIN", "DIRECTION_KAPPA", "random_walk", "session_units", "synthetic_session",
           "synthetic_recordings", "ground_truth", "write_recordings", "write_separate_units"]

#: Position sampling rate (Hz) and session duration (s).
SAMPLE_RATE_HZ = 50
DURATION_S = 720.0
#: Side of the square box (cm).
BOX_CM = 100.0
#: Strength and time constant (s) of the simulated adaptation.
ADAPTATION = 0.3
ADAPTATION_TAU_S = 2.0
#: Factor applied to the simulated rates (pool cells have peak rates of a few Hz).
RATE_GAIN = 3.0
#: Concentration of the movement-direction tuning of the conjunctive cells.
DIRECTION_KAPPA = 2.0

#: Sessions of the smoke run: manifest labels, number of units (the first units of the session in
#: manifest order), grid scale (cm), grid orientation (deg), number of conjunctive units and seed.
SMOKE_SESSIONS = (
    {"dataset": "vollan2025", "session_id": "V11_5", "n_cells": 12, "grid_scale_cm": 40.0,
     "orientation_deg": 7.0, "n_conjunctive": 7, "seed": 11},
    {"dataset": "hafting2005", "session_id": "H1_1", "n_cells": 10, "grid_scale_cm": 62.0,
     "orientation_deg": 22.0, "n_conjunctive": 7, "seed": 12},
)


def random_walk(n: int, dt: float, rng: np.random.Generator, box: float = BOX_CM, speed_mean: float = 20.0,
                speed_sd: float = 5.0, turn_sd: float = 0.12, margin: float = 1.0):
    """Random walk in a square box with reflection at the walls.

    Parameters
    ----------
    n : int
        Number of samples.
    dt : float
        Sample interval (s).
    rng : numpy.random.Generator
        Random numbers.
    box, margin : float
        Side of the box and distance from the walls at which the walk is reflected (cm).
    speed_mean, speed_sd : float
        Running speed (cm/s), drawn per sample and clipped to 3 to 45 cm/s.
    turn_sd : float
        Standard deviation of the change of heading per sample (rad).

    Returns
    -------
    xy : ndarray, shape (n, 2)
        Positions (cm).
    speed : ndarray, shape (n,)
        Running speed (cm/s).
    heading : ndarray, shape (n,)
        Movement direction in [0, 2 pi) (rad).
    """
    speed = np.clip(rng.normal(speed_mean, speed_sd, n), 3.0, 45.0)
    turns = rng.normal(0.0, turn_sd, n)
    xy = np.empty((n, 2))
    heading = np.empty(n)
    pos = np.array([box / 2, box / 2])
    h = rng.uniform(0, 2 * np.pi)
    lo, hi = margin, box - margin
    for i in range(n):
        h += turns[i]
        nxt = pos + speed[i] * dt * np.array([np.cos(h), np.sin(h)])
        if not lo <= nxt[0] <= hi:
            h = np.pi - h
        if not lo <= nxt[1] <= hi:
            h = -h
        nxt = np.clip(pos + speed[i] * dt * np.array([np.cos(h), np.sin(h)]), lo, hi)
        pos = nxt
        xy[i] = pos
        heading[i] = h % (2 * np.pi)
    return xy, speed, heading


def session_units(spec: dict) -> List[dict]:
    """Manifest rows (``cell_id``, ``unique_cell_id``, ...) of the units used for a smoke session."""
    from hexmod.preprocessing import labels

    units = labels.units_of(spec["dataset"])[spec["session_id"]]
    if len(units) < spec["n_cells"]:
        raise ValueError(f"{spec['session_id']} has {len(units)} units in the manifest, {spec['n_cells']} asked")
    return list(units[: spec["n_cells"]])


def _animal_of(spec: dict) -> str:
    from hexmod.preprocessing import labels

    for s in labels.sessions_of(spec["dataset"]):
        if s["session_id"] == spec["session_id"]:
            return s["animal_id"]
    raise ValueError(f"session {spec['session_id']} is not in the manifest of {spec['dataset']}")


def synthetic_session(spec: dict, adaptation: float = ADAPTATION, tau_s: float = ADAPTATION_TAU_S):
    """Simulate one session.

    Parameters
    ----------
    spec : dict
        One entry of ``SMOKE_SESSIONS``.
    adaptation, tau_s : float
        Strength and time constant (s) of the adaptation.

    Returns
    -------
    session : dict
        Session in the format of ``preprocessed/<dataset>.pkl`` (positions before smoothing).
    truth : list of dict
        True parameters of every unit (see :func:`ground_truth`).
    """
    import pandas as pd
    from scipy.special import i0

    from hexmod.sim import pool, simulate

    rng = np.random.default_rng(spec["seed"])
    dt = 1.0 / SAMPLE_RATE_HZ
    n = int(round(DURATION_S * SAMPLE_RATE_HZ))
    t = np.arange(n) * dt
    xy, speed, heading = random_walk(n, dt, rng)
    units = session_units(spec)
    par = pool.module_param("small", gridScale=spec["grid_scale_cm"], gridOri=spec["orientation_deg"],
                            envSize_x=BOX_CM, envSize_y=BOX_CM, adaptEff_mean=adaptation, adaptTau=tau_s,
                            speedThres=0.0, nGridCells=len(units))
    cells = pool.set_gridcells_pool(par, len(units), base_seed=spec["seed"])
    traj = pd.DataFrame({"time": t, "x": xy[:, 0], "y": xy[:, 1], "moveSpeed": speed,
                         "moveDir": np.degrees(heading)})
    rates = simulate.simulate_grid_cells(cells, traj, par)["FR_adapt"] * RATE_GAIN

    animal_id = _animal_of(spec)
    vollan = spec["dataset"] == "vollan2025"
    position = {"xy": xy.astype(np.float32) if vollan else xy.copy(), "time": t.copy(),
                "sample_rate": SAMPLE_RATE_HZ}
    # optional entries as in the preprocessed recordings of each dataset (docs/PIPELINE.md, step 01)
    session = {"animal_id": animal_id, "session_id": spec["session_id"], "position": position,
               "spike_times": {}, "unique_cell_ids": {}, "mod_id": {}}
    if vollan:
        session["time"] = position["time"]
    else:
        session["tetrode_ids"] = {}
    truth = []
    preferred = rng.uniform(0, 2 * np.pi, len(units))
    for k, u in enumerate(units):
        conjunctive = k < spec["n_conjunctive"]
        rate = rates[k]
        if conjunctive:
            rate = rate * np.exp(DIRECTION_KAPPA * np.cos(heading - preferred[k])) / i0(DIRECTION_KAPPA)
        fired = rng.random(n) < np.minimum(rate * dt, 1.0)
        # a spike in (t[i - 1], t[i]] is counted at sample i by the analyses
        spikes = np.maximum(t[fired] - rng.random(int(fired.sum())) * dt, t[0])
        c = int(u["cell_id"])
        session["spike_times"][c] = spikes
        session["unique_cell_ids"][c] = u["unique_cell_id"]
        session["mod_id"][c] = 1
        if not vollan:
            session["tetrode_ids"][c] = -1
        truth.append({"dataset": spec["dataset"], "animal_id": animal_id, "session_id": spec["session_id"],
                      "cell_id": c, "unique_cell_id": u["unique_cell_id"],
                      "grid_scale_cm": float(spec["grid_scale_cm"]), "orientation_deg": float(spec["orientation_deg"]),
                      "conjunctive": bool(conjunctive),
                      "preferred_direction_deg": float(np.degrees(preferred[k])) if conjunctive else None,
                      "adaptation": float(adaptation), "adaptation_tau_s": float(tau_s),
                      "n_spikes": int(len(spikes))})
    return session, truth


def synthetic_recordings(sessions=SMOKE_SESSIONS, adaptation: float = ADAPTATION,
                         tau_s: float = ADAPTATION_TAU_S):
    """Simulate the smoke sessions.

    Returns
    -------
    per_dataset : dict
        ``{dataset: {animal_id: {session_id: session}}}`` in canonical order, the content of the
        files ``preprocessed/<dataset>.pkl``.
    truth : list of dict
        True parameters of every unit, in the order of the recordings.
    """
    from hexmod.preprocessing import labels

    per_dataset: Dict[str, dict] = {}
    truth: List[dict] = []
    for spec in sessions:
        session, rows = synthetic_session(spec, adaptation, tau_s)
        per_dataset.setdefault(spec["dataset"], {}).setdefault(session["animal_id"], {})[spec["session_id"]] = session
        truth += rows
    ordered = {key: labels.canonical_order(data, key, strict=False) for key, data in per_dataset.items()}
    return ordered, truth


def ground_truth(sessions=SMOKE_SESSIONS, adaptation: float = ADAPTATION, tau_s: float = ADAPTATION_TAU_S):
    """True parameters of the synthetic units, one row per unit in canonical order.

    Returns
    -------
    pandas.DataFrame
        ``dataset``, ``animal_id``, ``session_id``, ``cell_id``, ``unique_cell_id``, ``grid_scale_cm``,
        ``orientation_deg``, ``conjunctive``, ``preferred_direction_deg`` (NaN for cells without
        direction tuning), ``adaptation``, ``adaptation_tau_s``, ``n_spikes``.
    """
    import pandas as pd

    from hexmod.datasets import canonical_rank

    _, truth = synthetic_recordings(sessions, adaptation, tau_s)
    df = pd.DataFrame(truth)
    rank = df["dataset"].map(canonical_rank)
    return df.assign(_rank=rank).sort_values("_rank", kind="stable").drop(columns="_rank").reset_index(drop=True)


def _session_rows(key: str, data: dict) -> List[dict]:
    from hexmod.preprocessing import labels

    source = {s["session_id"]: s["raw_path"] for s in labels.sessions_of(key)}
    rows = []
    for animal_id, sessions in data.items():
        for session_id, session in sessions.items():
            t = session["position"]["time"]
            rows.append({"dataset": key, "animal_id": animal_id, "session_id": session_id,
                         "source": source.get(session_id, ""), "n_units": len(session["spike_times"]),
                         "n_samples": int(len(t)), "sample_rate_hz": float(session["position"]["sample_rate"]),
                         "duration_s": float(t[-1] - t[0])})
    return rows


def write_recordings(results_dir, sessions=SMOKE_SESSIONS, adaptation: float = ADAPTATION,
                     tau_s: float = ADAPTATION_TAU_S, record=None) -> List[Path]:
    """Write the synthetic recordings in the layout of step 01.

    Parameters
    ----------
    results_dir : str or Path
        ``results/``; the files go to its ``preprocessed/`` folder.
    sessions, adaptation, tau_s
        As for :func:`synthetic_recordings`.
    record : hexmod.runlog.RunRecord, optional
        Run record that receives the outputs and the true parameters as notes.

    Returns
    -------
    list of Path
        The files written.
    """
    import pandas as pd

    from hexmod import config, io
    from hexmod.datasets import canonical_rank
    from hexmod.preprocessing import trajectories

    out = Path(results_dir) / "preprocessed"
    out.mkdir(parents=True, exist_ok=True)
    per_dataset, truth = synthetic_recordings(sessions, adaptation, tau_s)
    keys = sorted(per_dataset, key=canonical_rank)
    paths = [io.save_cache(per_dataset[key], out / f"{key}.pkl") for key in keys]
    table = [row for key in keys for row in _session_rows(key, per_dataset[key])]
    # smoothing writes into the position arrays: merge copies read back from the files just written
    merged = trajectories.merge_and_smooth({key: io.load_cache(out / f"{key}.pkl") for key in keys}, strict=False)
    paths.append(io.save_cache(merged, out / "trajectories.pkl"))
    columns = ["dataset", "animal_id", "session_id", "source", "n_units", "n_samples", "sample_rate_hz",
               "duration_s"]
    paths.append(io.write_csv(pd.DataFrame(table, columns=columns), out / "sessions.csv"))
    if record is not None:
        for name in ("sessions.csv", "units.csv"):
            record.add_input(config.inputs_dir() / "manifests" / name)
        for p in paths:
            record.add_output(p)
        record.note("model", {"adaptation": adaptation, "adaptation_tau_s": tau_s, "rate_gain": RATE_GAIN,
                              "direction_kappa": DIRECTION_KAPPA, "duration_s": DURATION_S,
                              "sample_rate_hz": SAMPLE_RATE_HZ, "box_cm": BOX_CM})
        record.note("sessions", [dict(s) for s in sessions])
        record.note("units", truth)
    return paths


def write_separate_units(results_dir, record=None) -> List[Path]:
    """Write ``rat/same_cell/entries.csv`` and ``units.csv`` with every cohort cell as its own unit.

    Every synthetic cell is simulated independently, so no two recordings belong to the same unit. The
    files have the columns of step 09 (docs/PIPELINE.md) and are row-aligned with ``rat/cohort/cells.csv``.

    Parameters
    ----------
    results_dir : str or Path
        ``results/``.
    record : hexmod.runlog.RunRecord, optional
        Run record that receives the inputs and outputs.

    Returns
    -------
    list of Path
        The files written.
    """
    import pandas as pd

    from hexmod import io

    results_dir = Path(results_dir)
    cells_path = results_dir / "rat" / "cohort" / "cells.csv"
    cells = io.read_csv(cells_path)
    out = results_dir / "rat" / "same_cell"
    n = len(cells)
    unit = np.arange(n, dtype=np.int64)
    entries = pd.DataFrame({"row": cells["row"].to_numpy(np.int64), "animal_id": cells["animal_id"],
                            "session_id": cells["session_id"], "cell_id": cells["cell_id"].to_numpy(np.int64),
                            "unit": unit, "n_entries": np.ones(n, dtype=np.int64),
                            "kept_earliest": np.ones(n, dtype=bool)})
    units = pd.DataFrame({"unit": unit, "animal_id": cells["animal_id"], "dataset": cells["dataset"],
                          "kept_row": cells["row"].to_numpy(np.int64), "n_entries": np.ones(n, dtype=np.int64),
                          "rows": cells["row"].astype(str)})
    paths = [io.write_csv(entries, out / "entries.csv"), io.write_csv(units, out / "units.csv")]
    if record is not None:
        record.add_input(cells_path)
        for p in paths:
            record.add_output(p)
        record.note("n_units", int(n))
    return paths
