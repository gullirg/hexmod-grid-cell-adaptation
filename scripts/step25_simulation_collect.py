"""Step 25: the values of Fig 7 from the simulated participants.

Examples::

    python scripts/step25_simulation_collect.py                    # files of step 24, else the shipped tables
    python scripts/step25_simulation_collect.py --experiments 50   # Fig 7F over participants 0 to 1,499
    python scripts/step25_simulation_collect.py --quick            # Fig 7A to 7E from participants 0 to 29

Sources: when all 1,500 participant files of both modules are present, the step reads the
participant files of step 24 (``sim/fig7/participants/<module>/p0000.npz`` to ``p1499.npz``;
source label ``run``); otherwise it reads the shipped participant tables
``results_precomputed/sim/fig7/participants_<module>.csv.gz`` (source label ``precomputed``). The
files of a run are used only when each was computed with 10,000 cells and from the pool that step
24 uses now (``sim/pools/pool_checksums.json`` of step 21, or the shipped large pool); step 24
records both in the ``meta_*`` entries of each file. Otherwise the step stops with a message.
``--quick`` takes participants 0 to 29 (Fig 7A to 7E) from the files of ``step24 --quick`` and
participants 30 to 1,499 (Fig 7F) from the shipped tables (source label ``quick``).

Per participant and estimator (simple; location-controlled) the participant table holds:

- ``rate_dir_0_hz`` to ``rate_dir_330_hz``: the population rate of the adapting cells in 12 bins of
  running direction relative to the grid (0 = along a grid axis), from all moving samples;
- ``dir_0`` to ``dir_330``: the same in percent change from the participant's mean over the bins;
- ``fold_1`` to ``fold_7``: the cross-validated modulation coefficient for 1- to 7-fold symmetry,
  the mean over the four left-out blocks, in percent of the mean moving rate;
- ``orientation_deg_mod60``: the six-fold orientation, the circular mean (period 60 deg) of the
  orientations estimated in the four cross-validation runs;
- ``mean_rate_moving_hz``: the mean moving rate.

From these (rows converted to C order before row-wise means):

- Fig 7A, 7D: paired t-test of the mean of the even (aligned) against the odd (misaligned) bins of
  ``rate_dir_*_hz`` over participants 0 to 29, simple and location-controlled estimator;
- Fig 7B, 7E: two-sided one-sample t-test of each ``fold_k`` against 0, participants 0 to 29;
- Fig 7C: participants 0 to 29 per 30 deg bin of six times the orientation;
- Fig 7F: experiments of 30 consecutive participants; per experiment and symmetry the one-sided
  (greater) one-sample t-test of ``fold_k``; count and share of experiments with p < 0.05. By
  default over participants 30 to 1,499 (49 experiments, source label ``rest_49_experiments``), as
  drawn in the paper; ``--experiments 50`` uses participants 0 to 1,499 (``all_50_experiments``)
  for the ``fig7f.*`` values. ``fig7f_detection.csv``, ``fig7f_p_per_experiment.csv`` and the
  ``fig7f_50.*`` values always cover both.

Writes ``sim/fig7/participants_small.csv.gz``, ``participants_large.csv.gz`` (3,000 rows each,
``simple`` rows for participants 0 to 1,499, then ``location-controlled``),
``fig7ad_direction_bins.csv``, ``fig7be_beta.csv``, ``fig7c_orientation.csv``,
``fig7f_detection.csv``, ``fig7f_p_per_experiment.csv`` and ``fig7_values.json`` (keys
``fig7.*``, ``fig7a.*`` to ``fig7f.*`` and ``fig7f_50.*``, each listed with its reference value in
``reference/expected_statistics.json``). From the shipped tables every output is identical to the
shipped ``results_precomputed/sim/fig7`` files.
"""

from __future__ import annotations

import argparse
import sys
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
import scipy.stats  # noqa: E402

from hexmod import io as hio  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402
from hexmod.sim.io_pool import MODULES  # noqa: E402

STEP = "step25_simulation_collect"
#: Cells of each simulated participant.
N_CELLS = 10000
#: Entries of ``sim/pools/pool_checksums.json`` (step 21).
POOL_SUMMARY_KEYS = {"small": "small_module_pool_40cm", "large": "large_module_pool_70cm"}
N_PARTICIPANTS = 1500
N_PER_EXPERIMENT = 30
ALPHA_THRESHOLD = 0.05
DIRS = [30 * j for j in range(12)]
FOLDS = list(range(1, 8))
#: Estimators as named in the simulation results and in the tables.
ESTIMATORS = ("simple", "multiplicative")
METHOD = {"simple": "simple", "multiplicative": "location-controlled"}
KEY_WORD = {"simple": "simple", "location-controlled": "location_controlled"}
#: Participants of the two Fig 7F variants.
F_SOURCES = {"rest_49_experiments": range(30, N_PARTICIPANTS), "all_50_experiments": range(N_PARTICIPANTS)}
TABLE_PREFIXES = {"crossval": "crossvalResult", "noncross": "noncrossResult"}
N_BINS_C = 12
SIX = 6


# ------------------------------------------------------------------------------- participant files
def _object_array(items) -> np.ndarray:
    out = np.empty(len(items), dtype=object)
    for i, v in enumerate(items):
        out[i] = v
    return out


def arrays_to_table(raw: Dict[str, np.ndarray], prefix: str) -> pd.DataFrame:
    """Rebuild a fit table stored by step 24 (``<prefix>__<column>``) as a DataFrame."""
    columns = [str(c) for c in raw[f"{prefix}__columns"].tolist()]
    data = {}
    for col in columns:
        key = f"{prefix}__{col}"
        if key in raw:
            values = raw[key]
            if values.ndim == 2:  # one array per row
                data[col] = _object_array([np.array(v) for v in values])
            elif values.dtype.kind == "U":
                data[col] = values.astype(object)
            else:
                data[col] = values
        else:  # ragged: an empty array stands for a row without an array
            items = hio.unpack_ragged(raw[key + hio.RAGGED_DATA], raw[key + hio.RAGGED_SHAPES])
            data[col] = _object_array([v if v.size else None for v in items])
    return pd.DataFrame(data, columns=columns)


def load_participant(path: Path, tables_only: bool = False) -> dict:
    """Read a participant file of step 24 into the participant result.

    Parameters
    ----------
    path : Path
        ``p0000.npz`` file.
    tables_only : bool
        Return only ``seed``, ``module``, ``FR_adapt_mean_move`` and the two fit tables.

    Returns
    -------
    dict
        Entries as returned by ``hexmod.sim.simulate.simulate_and_measure`` (plus ``module``); the
        fit tables ``crossvalResult`` and ``noncrossResult`` as DataFrames, 0-d entries as Python
        numbers.
    """
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    out = {}
    for prefix, name in TABLE_PREFIXES.items():
        out[name] = arrays_to_table(raw, prefix)
    keep = {"seed", "module", "FR_adapt_mean_move"} if tables_only else None
    for key, value in raw.items():
        if key.split("__")[0] in TABLE_PREFIXES or (keep is not None and key not in keep):
            continue
        if key == "foldSymList":
            out[key] = [int(v) for v in value]
        elif value.ndim == 0:
            out[key] = value.item()
        else:
            out[key] = value
    return out


def participant_rows(records: Sequence[dict], module: str) -> pd.DataFrame:
    """Participant table rows (method-major) from participant results.

    Rates by direction are stacked row by row, the cross-validated coefficients are averaged per
    participant and symmetry with pandas, and the orientations are averaged with
    ``scipy.stats.circmean`` (period 60 deg).
    """
    records = sorted(records, key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in records]
    noncross = pd.concat([r["noncrossResult"] for r in records])
    crossval = pd.concat([r["crossvalResult"] for r in records])
    mean_rate_signed = np.array([r["FR_adapt_mean_move"] for r in records])
    mean_rate = np.abs([r["FR_adapt_mean_move"] for r in records])
    rows = []
    for est in ESTIMATORS:
        mask = (noncross["estType"] == est) & (noncross["model"] == "adapt")
        if noncross.loc[mask, "sub"].to_numpy().tolist() != seeds:
            raise ValueError(f"{module} {est}: direction profiles do not match the participants")
        rate_by_dir = np.row_stack(noncross.loc[mask, "FRby12dir"])
        rate_by_dir_pct = rate_by_dir / rate_by_dir.mean(axis=1, keepdims=True) * 100 - 100
        cmask = (crossval["model"] == "adapt") & (crossval["estType"] == est)
        beta = crossval[cmask].groupby(["sub", "fold"])["beta"].mean().reset_index().pivot(index="sub", columns="fold",
                                                                                          values="beta")
        if beta.index.tolist() != seeds:
            raise ValueError(f"{module} {est}: coefficients do not match the participants")
        beta = beta[[f for f in beta.columns if 1 <= f <= 7]]
        if list(beta.columns) != FOLDS:
            raise ValueError(f"{module} {est}: symmetries {list(beta.columns)}, expected 1 to 7")
        beta_pct = beta.to_numpy() * 100 / mean_rate[:, np.newaxis]
        omask = (crossval["fold"] == SIX) & (crossval["model"] == "adapt") & (crossval["estType"] == est)
        ori = crossval[omask].groupby(["sub"])["ori"].apply(lambda x: scipy.stats.circmean(x, high=360 / SIX, low=0))
        if ori.index.tolist() != seeds:
            raise ValueError(f"{module} {est}: orientations do not match the participants")
        ori = ori.to_numpy()
        for i, s in enumerate(seeds):
            row = {"module": module, "participant": s, "method": METHOD[est]}
            row.update({f"dir_{d}": float(rate_by_dir_pct[i, j]) for j, d in enumerate(DIRS)})
            row.update({f"fold_{k}": float(beta_pct[i, k - 1]) for k in FOLDS})
            row["orientation_deg_mod60"] = float(ori[i])
            row.update({f"rate_dir_{d}_hz": float(rate_by_dir[i, j]) for j, d in enumerate(DIRS)})
            row["mean_rate_moving_hz"] = float(mean_rate_signed[i])
            rows.append(row)
    return pd.DataFrame(rows)


def participant_folder(module: str) -> Path:
    return config.results_dir() / "sim" / "fig7" / "participants" / module


def complete_run(participants: Sequence[int] = range(N_PARTICIPANTS)) -> bool:
    """True when the participant files of both modules are present (0 to 1,499 by default)."""
    return all((participant_folder(m) / f"p{s:04d}.npz").is_file() for m in MODULES for s in participants)


def expected_pool_fingerprint(module: str) -> str:
    """Fingerprint of the pool that step 24 uses for ``module`` now (``hexmod.sim.io_pool.pool_fingerprint``)."""
    from hexmod.sim import io_pool

    pools = config.results_dir() / "sim" / "pools"
    if (pools / io_pool.POOL_FILES[module]).is_file():
        summary_path = pools / "pool_checksums.json"
    elif module == "large":   # step 24 then reads the shipped pool
        shipped = config.inputs_dir() / "simulation" / io_pool.SHIPPED_LARGE_POOL
        return io_pool.pool_fingerprint(io_pool.load_pool(shipped))
    else:
        raise SystemExit(f"error: {pools / io_pool.POOL_FILES[module]} not found; the participant files cannot be "
                         "checked against their pool (run step 21 first)")
    summary = hio.read_json(summary_path).get(POOL_SUMMARY_KEYS[module]) if summary_path.is_file() else None
    fingerprint = io_pool.fingerprint_from_summary(summary) if summary else None
    if fingerprint is None:
        raise SystemExit(f"error: {summary_path} does not describe the {module} pool; run step 21 again")
    return fingerprint


def check_metadata(path: Path, seed: int, module: str, fingerprint: str) -> Optional[str]:
    """Why a participant file of step 24 cannot be used (None when it can)."""
    with np.load(path, allow_pickle=False) as z:
        meta = {k: z[k].item() for k in ("seed", "module", "meta_n_cells", "meta_pool_fingerprint") if k in z.files}
    if int(meta.get("seed", -1)) != int(seed) or str(meta.get("module")) != module:
        return "seed or module differ from the file name"
    if "meta_n_cells" not in meta:
        return "the file does not record how it was computed (run step 24 again)"
    if int(meta["meta_n_cells"]) != N_CELLS:
        return f"{int(meta['meta_n_cells'])} cells instead of {N_CELLS}"
    if str(meta.get("meta_pool_fingerprint")) != fingerprint:
        return "computed from another pool than the one of sim/pools"
    return None


def tables_from_run(participants: Sequence[int] = range(N_PARTICIPANTS)) -> Dict[str, pd.DataFrame]:
    """Participant tables of both modules from the participant files of step 24 (checked with ``check_metadata``)."""
    out = {}
    for m in MODULES:
        folder = participant_folder(m)
        missing = [s for s in participants if not (folder / f"p{s:04d}.npz").is_file()]
        if missing:
            raise SystemExit(f"error: {len(missing)} participant files of the {m} module are missing in {folder} "
                             f"(first: participant {missing[0]}); run step 24")
        fingerprint = expected_pool_fingerprint(m)
        bad = [(s, why) for s in participants
               if (why := check_metadata(folder / f"p{s:04d}.npz", s, m, fingerprint)) is not None]
        if bad:
            s, why = bad[0]
            raise SystemExit(f"error: {len(bad)} participant files of the {m} module in {folder} cannot be used "
                             f"(participant {s}: {why}); run step 24 again, which recomputes them")
        records = [load_participant(folder / f"p{s:04d}.npz", tables_only=True) for s in participants]
        out[m] = participant_rows(records, m)
    return out


def tables_quick(shipped: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Participants 0 to 29 from the files of ``step24 --quick``, 30 to 1,499 from the shipped tables."""
    run = tables_from_run(range(N_PER_EXPERIMENT))
    rank = {name: i for i, name in enumerate(METHOD[e] for e in ESTIMATORS)}
    out = {}
    for m in MODULES:
        rest = shipped[m][shipped[m]["participant"] >= N_PER_EXPERIMENT]
        merged = pd.concat([run[m][list(shipped[m].columns)], rest], ignore_index=True)
        order = np.lexsort((merged["participant"].to_numpy(), merged["method"].map(rank).to_numpy()))
        out[m] = merged.iloc[order].reset_index(drop=True)
    return out


# ------------------------------------------------------------------------------------- Fig 7 values
def _select(table: pd.DataFrame, method: str, participants: Sequence[int]) -> pd.DataFrame:
    t = table[(table["method"] == method) & (table["participant"].isin(participants))]
    if t["participant"].tolist() != list(participants):
        raise ValueError(f"participant table lacks participants of {method}")
    return t


def _block(table: pd.DataFrame, columns: Sequence[str], order: str = "C") -> np.ndarray:
    """Columns of a table as a float array in a given memory order.

    The order of the additions in a mean along one axis depends on the memory layout, which can
    change the last digit. Row-wise means are taken of row-stacked (C-ordered) arrays and
    column-wise means of column-major (Fortran-ordered) arrays; ``order`` sets the layout.
    """
    values = table[list(columns)].to_numpy(dtype=float)
    return np.ascontiguousarray(values) if order == "C" else np.asfortranarray(values)


def fig7_values(tables: Dict[str, pd.DataFrame], experiments: int = 49):
    """Fig 7 tables and statistics from the participant tables of both modules.

    Returns
    -------
    values : dict
        Flat ``{key: number}``, the contents of ``fig7_values.json`` (see the module docstring).
    out : dict of DataFrame
        ``fig7ad_direction_bins``, ``fig7be_beta``, ``fig7c_orientation``, ``fig7f_detection``,
        ``fig7f_p_per_experiment``.
    """
    if experiments not in (49, 50):
        raise ValueError("experiments must be 49 or 50")
    first = range(N_PER_EXPERIMENT)
    values: Dict[str, float] = {}
    rows_ad, rows_be, rows_c, rows_f, rows_p = [], [], [], [], []
    for est in ESTIMATORS:
        method = METHOD[est]
        panel_ad = "fig7a" if est == "simple" else "fig7d"
        panel_be = "fig7b" if est == "simple" else "fig7e"
        for m in MODULES:
            t = _select(tables[m], method, first)
            # Fig 7A, 7D: aligned (even bins) against misaligned (odd bins)
            rate = _block(t, [f"rate_dir_{d}_hz" for d in DIRS])
            tt = scipy.stats.ttest_rel(np.mean(rate[:, 0::2], axis=1), np.mean(rate[:, 1::2], axis=1))
            values[f"{panel_ad}.{m}.n"] = len(t)
            values[f"{panel_ad}.{m}.t"] = float(tt.statistic)
            values[f"{panel_ad}.{m}.df"] = len(t) - 1
            values[f"{panel_ad}.{m}.p"] = float(tt.pvalue)
            for _, r in t.iterrows():
                rows_ad.append(dict(module=m, method=method, participant=int(r["participant"]),
                                    **{f"dir_{d}": float(r[f"dir_{d}"]) for d in DIRS}))
            # Fig 7B, 7E: coefficient of each symmetry against 0
            beta = _block(t, [f"fold_{k}" for k in FOLDS])
            for k in FOLDS:
                r_ = scipy.stats.ttest_1samp(beta[:, k - 1], 0)
                values[f"{panel_be}.{m}.fold_{k}.t"] = float(r_.statistic)
                values[f"{panel_be}.{m}.fold_{k}.df"] = len(t) - 1
                values[f"{panel_be}.{m}.fold_{k}.p"] = float(r_.pvalue)
                values[f"{panel_be}.{m}.fold_{k}.significant"] = int(float(r_.pvalue) < ALPHA_THRESHOLD)
            for _, r in t.iterrows():
                rows_be.append(dict(module=m, method=method, participant=int(r["participant"]),
                                    **{f"fold_{k}": float(r[f"fold_{k}"]) for k in FOLDS}))
            # Fig 7C: six-fold orientations in 30 deg bins
            x = t["orientation_deg_mod60"].to_numpy() * SIX
            idx = np.digitize(x, np.arange(-360 / N_BINS_C, 361, 360 / N_BINS_C) + 360 / N_BINS_C / 2) - 1
            idx[idx == N_BINS_C] = 0
            counts = np.bincount(idx, minlength=N_BINS_C)
            for j in range(N_BINS_C):
                values[f"fig7c.{m}.{KEY_WORD[method]}.count_{30 * j}"] = int(counts[j])
            for _, r in t.iterrows():
                rows_c.append(dict(module=m, method=method, participant=int(r["participant"]),
                                   orientation_deg_mod60=float(r["orientation_deg_mod60"])))
    # Fig 7F: detection in experiments of 30 participants
    chosen = "rest_49_experiments" if experiments == 49 else "all_50_experiments"
    for m in MODULES:
        for label, parts in F_SOURCES.items():
            for est in ESTIMATORS:
                method = METHOD[est]
                beta = _block(_select(tables[m], method, parts), [f"fold_{k}" for k in FOLDS], order="F")
                n_exp = int(beta.shape[0] / N_PER_EXPERIMENT)
                p_all = np.zeros((n_exp, len(FOLDS)))
                for e in range(n_exp):
                    sl = np.arange(N_PER_EXPERIMENT) + e * N_PER_EXPERIMENT
                    _, p = scipy.stats.ttest_1samp(beta[sl, :], 0, axis=0, alternative="greater")
                    p_all[e, :] = p
                    rows_p.append(dict(module=m, method=method, source=label, experiment=e,
                                       **{f"p_fold_{k}": float(p[k - 1]) for k in FOLDS}))
                rate = np.mean(p_all < ALPHA_THRESHOLD, axis=0)
                count = (p_all < ALPHA_THRESHOLD).sum(axis=0)
                rows_f.append(dict(module=m, method=method, source=label, experiments=n_exp,
                                   participants=int(beta.shape[0]),
                                   **{f"rate_fold_{k}": float(rate[k - 1]) for k in FOLDS},
                                   **{f"count_fold_{k}": int(count[k - 1]) for k in FOLDS}))
                prefixes = ((["fig7f"] if label == chosen else [])
                            + (["fig7f_50"] if label == "all_50_experiments" else []))
                for prefix in prefixes:
                    values[f"{prefix}.n_experiments"] = n_exp
                    values[f"{prefix}.first_participant"] = int(parts[0])
                    values[f"{prefix}.last_participant"] = int(parts[-1])
                    for k in FOLDS:
                        values[f"{prefix}.{m}.{KEY_WORD[method]}.fold_{k}.count"] = int(count[k - 1])
                        values[f"{prefix}.{m}.{KEY_WORD[method]}.fold_{k}.rate"] = float(rate[k - 1])
    for m in MODULES:
        values[f"fig7.{m}.n_participants"] = int(tables[m]["participant"].nunique())
    values["fig7.participants_per_experiment"] = N_PER_EXPERIMENT
    values["fig7.alpha_threshold"] = ALPHA_THRESHOLD
    out = {"fig7ad_direction_bins": pd.DataFrame(rows_ad), "fig7be_beta": pd.DataFrame(rows_be),
           "fig7c_orientation": pd.DataFrame(rows_c), "fig7f_detection": pd.DataFrame(rows_f),
           "fig7f_p_per_experiment": pd.DataFrame(rows_p)}
    return values, out


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", type=int, choices=[49, 50], default=49,
                   help="Fig 7F over participants 30 to 1,499 (49, as published) or 0 to 1,499 (50)")
    p.add_argument("--quick", action="store_true",
                   help="participants 0 to 29 (Fig 7A to 7E) from step24 --quick, the others (Fig 7F) from the "
                        "shipped tables")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = config.results_dir() / "sim" / "fig7"
    pre = config.precomputed_dir() / "sim" / "fig7"
    with RunRecord(STEP, argv=argv) as rec:
        if args.quick:
            source = "quick"
        else:
            source = "run" if complete_run() else "precomputed"
        if source == "run":
            tables = tables_from_run()
        else:
            tables = {}
            for m in MODULES:
                path = pre / f"participants_{m}.csv.gz"
                rec.add_input(path)
                tables[m] = hio.read_csv(path)
            if source == "quick":
                tables = tables_quick(tables)
        if source in ("run", "quick"):
            for m in MODULES:
                rec.add_input(participant_folder(m))
        for m in MODULES:
            hio.write_csv(tables[m], out / f"participants_{m}.csv.gz")
            rec.add_output(out / f"participants_{m}.csv.gz")
        values, frames = fig7_values(tables, args.experiments)
        for name, frame in frames.items():
            hio.write_csv(frame, out / f"{name}.csv")
            rec.add_output(out / f"{name}.csv")
        hio.write_json(values, out / "fig7_values.json")
        rec.add_output(out / "fig7_values.json")
        rec.note("source", source)
        if source == "quick":
            rec.note("source_detail", "participants 0 to 29 from step 24, 30 to 1,499 from results_precomputed")
        rec.note("experiments_fig7f", args.experiments)
        printed = {k: values[k] for k in ("fig7a.small.t", "fig7a.large.t", "fig7d.small.t", "fig7d.large.t")}
        rec.note("printed_t", printed)
    print(f"Fig 7 values written to {out} (source: {source}); 7A t: small {values['fig7a.small.t']:.4f}, "
          f"large {values['fig7a.large.t']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
