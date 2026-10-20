"""Repeated recordings of the same unit, and the earliest recording of each unit.

Some units were recorded more than once: the Vollan et al. (2025) data hold two exports of one
recording (same time base) and several sessions of one animal, and the tetrode datasets hold
several sessions of one animal on the same tetrodes. Counting such a unit once per recording would
give it extra weight in tests that treat cells as independent (Fig 2D and 2E). This module links
the recordings of one unit and keeps the earliest recording of each unit.

Links (the link set used in the paper):

- **Same recording** (Vollan files with an identical time base): two units on the same probe and
  shank are linked when at least half of the spikes of the shorter train have a spike of the other
  train within 1 ms, and this fraction is at least ten times the chance level; one-to-one
  assignment (``linear_sum_assignment``) between the two files.
- **Different recordings of one animal** (Vollan): units on the same probe and shank whose depths
  along the shank differ by at most 60 um are compared by the correlation of their rate maps
  (``units.rmf.pos.z`` of the data files, bins valid in both, only when the position grids are
  identical) and, when the probe geometry is identical, by the correlation of their mean
  waveforms (on the union of each unit's 8 largest-amplitude sites, sites matched by position).
  Thresholds are the 99th percentiles of the same measures over pairs of distinct units recorded
  together in one file that meet the same probe, shank and depth condition (rate-map threshold at
  least 0.5). With both measures a pair passes when both reach their thresholds; with the rate map
  alone the threshold is raised so that the expected share of false passes is the same. One-to-one
  assignment per pair of files.
- **Tetrode datasets**: rate maps of each recording on a common 5 cm grid centred on the arena
  (speed 5 to 150 cm/s; Gaussian smoothing of one bin on spike and dwell maps; bins with at
  least 0.2 s of smoothed dwell), compared by the largest Pearson correlation over shifts of up to
  one bin. A pair of recordings in different sessions of one animal is linked when the correlation
  reaches the 99th percentile of pairs of distinct cells recorded together in the same dataset (at
  least 0.5), the grid orientations differ by at most 6 degrees (modulo 60), the grid scales by at
  most 15 percent, and both arenas have the same extent within 10 percent; one-to-one per pair of
  sessions. In the Hafting et al. (2005) data the two trials of one animal hold the same tetrode
  and cell labels, which link the recordings directly.

Units are the connected components of the links (recordings of Vollan units outside the cohort
can connect two cohort recordings). The kept recording of a unit is its earliest one: the lowest
session number (the number after the underscore of the session label), and the first in the
canonical order among recordings of the same session number. The preferred direction, which is
the quantity tested in Fig 2D and 2E, is never used to link recordings.
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

__all__ = [
    "extract_vollan_units",
    "spike_coincidence",
    "ratemap_similarity",
    "waveform_similarity",
    "probe_geometry_equal",
    "same_recording",
    "link_vollan",
    "tetrode_entry_maps",
    "tetrode_rate_map",
    "map_similarity",
    "pair_features",
    "link_tetrode",
    "hafting_label",
    "UnionFind",
    "unit_components",
    "kept_earliest",
    "session_number",
    "extract_vollan_units_job",
    "COINCIDENCE_WINDOW_S",
    "MAX_DEPTH_DIFFERENCE_UM",
    "TETRODE_GRID_EDGES_CM",
]

#: Window for coincident spikes of two exports of one recording (s).
COINCIDENCE_WINDOW_S = 0.001
#: Largest difference of depth along the shank of two recordings of one unit (um).
MAX_DEPTH_DIFFERENCE_UM = 60
#: Bin edges of the arena-centred grid of the tetrode rate maps (cm).
TETRODE_GRID_EDGES_CM = np.arange(-125, 125 + 5, 5.0)
#: Smoothed dwell required in a bin of a tetrode rate map (s).
TETRODE_MIN_DWELL_S = 0.2
#: Pass rules of the tetrode map links.
TETRODE_MAX_ORIENTATION_DIFF_DEG = 6
TETRODE_MAX_RELATIVE_SCALE_DIFF = 0.15
TETRODE_MAX_RELATIVE_ARENA_DIFF = 0.10
#: Lowest rate-map correlation threshold of any link rule.
MIN_MAP_THRESHOLD = 0.5


# ------------------------------------------------------------------------------------- Vollan files
def _as_array(value, dtype=None):
    """``np.asarray(value)``, cast to ``dtype``; None when the value cannot be cast (a MATLAB object)."""
    a = np.asarray(value)
    if dtype is None:
        return a
    try:
        return a.astype(dtype)
    except (TypeError, ValueError):
        return None


def extract_vollan_units(path, file_key: Optional[str] = None) -> dict:
    """Read the unit features of one Vollan et al. (2025) session file.

    Only units with ``isGrid == 1`` are kept. For each: ``idx`` (position in ``units.mec``,
    counted from 1, equal to the ``cell_id`` of the preprocessed data), ``probeId``, ``shank``,
    ``shankPos`` (um), sorted spike times (s), the mean waveform ``wv`` (samples x sites, float32,
    None when the file holds no waveform), the rate map ``rm`` (``rmf.pos.z``, float32) and its
    valid bins ``rm_valid``.

    Parameters
    ----------
    path : str or Path
        Session file (``<data>/vollan2025/sharing_v4/navigation/of/<file>.mat``).
    file_key : str, optional
        Name of the file without extension (default: taken from ``path``).

    Returns
    -------
    dict
        ``file``, ``t0``, ``t1`` (first and last time, s), ``nt`` (samples), ``gv_pos`` (position
        grid of the rate maps), ``pcm`` (list per probe of ``xcoords``, ``ycoords``, ``shankInd``,
        ``connected``), ``units`` (list of dicts as above) and ``n_mec`` (MEC units in the file).
    """
    import scipy.io as sio
    from pathlib import Path

    path = Path(path)
    m = sio.loadmat(str(path), variable_names=["Dsession"], squeeze_me=True, struct_as_record=False)
    D = m["Dsession"]
    del m
    t = _as_array(D.t, float)
    rec = {
        "file": file_key or path.stem,
        "t0": float(t[0]),
        "t1": float(t[-1]),
        "nt": int(len(t)),
        "gv_pos": _as_array(D.gv.pos, float),
    }
    pcm = []
    for p in np.atleast_1d(D.probeChannelMaps):
        pcm.append(
            {k: _as_array(getattr(p, k)) for k in ("xcoords", "ycoords", "shankInd", "connected") if hasattr(p, k)}
        )
    rec["pcm"] = pcm
    U = np.atleast_1d(D.units.mec)
    units = []
    for i, u in enumerate(U, start=1):
        if int(u.isGrid) != 1:
            continue
        rp = u.rmf.pos
        units.append(
            {
                "idx": i,
                "probeId": int(u.probeId),
                "shank": int(u.shank),
                "shankPos": float(u.shankPos),
                "spikeTimes": np.sort(np.atleast_1d(np.asarray(_as_array(u.spikeTimes, float), float))),
                "wv": _as_array(u.wv, np.float32),
                "rm": _as_array(rp.z, np.float32),
                "rm_valid": _as_array(rp.validBin, bool),
            }
        )
    rec["units"] = units
    rec["n_mec"] = int(len(U))
    return rec


def extract_vollan_units_job(item) -> dict:
    """Worker function: ``(path, file_key)`` to the record of :func:`extract_vollan_units`."""
    path, file_key = item
    return extract_vollan_units(path, file_key)


def site_coordinates(rec: dict, probe: int) -> np.ndarray:
    """Coordinates (x, y, shank) of the connected recording sites of a probe (counted from 1)."""
    p = rec["pcm"][probe - 1]
    con = np.asarray(p["connected"]).astype(bool)
    return np.stack([np.asarray(p["xcoords"])[con], np.asarray(p["ycoords"])[con], np.asarray(p["shankInd"])[con]], 1)


def top_sites(unit: dict, k: int = 8) -> np.ndarray:
    """Indices of the k sites with the largest peak-to-peak amplitude of the mean waveform."""
    w = unit["wv"]
    ptp = w.max(0) - w.min(0)
    return np.argsort(ptp)[::-1][:k]


def waveform_similarity(ua: dict, ra: dict, ub: dict, rb: dict) -> float:
    """Correlation of two mean waveforms on the union of each unit's 8 largest sites.

    Sites are matched by their (x, y, shank) coordinates; NaN when a waveform is missing, the
    site count does not match the probe map, or fewer than 4 sites are shared.
    """
    if ua["wv"] is None or ub["wv"] is None:
        return np.nan
    ca = site_coordinates(ra, ua["probeId"])
    cb = site_coordinates(rb, ub["probeId"])
    if ca.shape[0] != ua["wv"].shape[1] or cb.shape[0] != ub["wv"].shape[1]:
        return np.nan
    key_b = {tuple(c): j for j, c in enumerate(cb)}
    sites = set(tuple(ca[j]) for j in top_sites(ua)) | set(tuple(cb[j]) for j in top_sites(ub))
    ja, jb = [], []
    for s in sites:
        ia = np.where((ca == np.array(s)).all(1))[0]
        if len(ia) and s in key_b:
            ja.append(ia[0])
            jb.append(key_b[s])
    if len(ja) < 4:
        return np.nan
    xa = ua["wv"][:, ja].ravel().astype(float)
    xb = ub["wv"][:, jb].ravel().astype(float)
    return float(np.corrcoef(xa, xb)[0, 1])


def ratemap_similarity(ua: dict, ub: dict) -> float:
    """Pearson correlation of two rate maps over bins valid in both (NaN below 50 bins)."""
    a = ua["rm"]
    b = ub["rm"]
    m = ua["rm_valid"] & ub["rm_valid"] & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def probe_geometry_equal(ra: dict, rb: dict) -> bool:
    """True when two files have the same probes with identical site coordinates and shanks."""
    if len(ra["pcm"]) != len(rb["pcm"]):
        return False
    return all(
        all(np.array_equal(np.asarray(pa[k]), np.asarray(pb[k])) for k in ("xcoords", "ycoords", "shankInd"))
        for pa, pb in zip(ra["pcm"], rb["pcm"])
    )


def spike_coincidence(sa: np.ndarray, sb: np.ndarray, window: float = COINCIDENCE_WINDOW_S) -> Tuple[float, int]:
    """Fraction of spikes of the shorter train with a spike of the other train within ``window``.

    Parameters
    ----------
    sa, sb : ndarray
        Sorted spike times (s).
    window : float
        Coincidence window (s).

    Returns
    -------
    fraction : float
    n : int
        Number of spikes of the shorter train.
    """
    if len(sa) > len(sb):
        sa, sb = sb, sa
    j = np.searchsorted(sb, sa)
    lo = np.clip(j - 1, 0, len(sb) - 1)
    hi = np.clip(j, 0, len(sb) - 1)
    d = np.minimum(np.abs(sa - sb[lo]), np.abs(sa - sb[hi]))
    return float(np.mean(d <= window)), len(sa)


def same_recording(ra: dict, rb: dict) -> bool:
    """True when two files share the time base (first and last time within 0.5 s, same sample count)."""
    return abs(ra["t0"] - rb["t0"]) <= 0.5 and abs(ra["t1"] - rb["t1"]) <= 0.5 and ra["nt"] == rb["nt"]


def _same_shank_near(ua: dict, ub: dict) -> bool:
    return (
        ua["probeId"] == ub["probeId"]
        and ua["shank"] == ub["shank"]
        and abs(ua["shankPos"] - ub["shankPos"]) <= MAX_DEPTH_DIFFERENCE_UM
    )


def link_vollan(recs: Dict[str, dict], groups: Sequence[Sequence[str]]):
    """Link the units of the Vollan files of each animal.

    Parameters
    ----------
    recs : dict
        ``{file: record}`` from :func:`extract_vollan_units`.
    groups : sequence of sequences of str
        Files of each animal, in increasing session number.

    Returns
    -------
    links : pandas.DataFrame
        One row per link: ``rule`` (``"same_recording"``, ``"waveform_and_map"`` or
        ``"map_only"``), ``file_a``, ``idx_a``, ``file_b``, ``idx_b``, ``f_coinc``, ``chance``,
        ``dpos_um``, ``rm``, ``w``.
    details : dict
        ``null`` (pairs of co-recorded units: ``file``, ``idx_a``, ``idx_b``, ``dpos_um``, ``rm``,
        ``w``), ``candidates`` (compared pairs of different recordings, with ``pass``),
        ``thresholds`` and per-pair statistics ``pairs``.
    """
    import pandas as pd

    links: List[dict] = []
    pair_rows: List[dict] = []
    null_rows: List[dict] = []
    pair_stats: Dict[str, dict] = {}
    pairs_all = [(a, b) for fs in groups for a, b in itertools.combinations(fs, 2)]
    same_rec = [(a, b) for a, b in pairs_all if same_recording(recs[a], recs[b])]
    # ------------------------------------------------ exports of one recording: coincident spikes
    for a, b in same_rec:
        ra, rb = recs[a], recs[b]
        dur = ra["t1"] - ra["t0"]
        ua_l, ub_l = ra["units"], rb["units"]
        fmat = np.full((len(ua_l), len(ub_l)), np.nan)
        chance = np.full_like(fmat, np.nan)
        same = np.zeros_like(fmat, bool)
        for i, ua in enumerate(ua_l):
            for j, ub in enumerate(ub_l):
                f, n = spike_coincidence(ua["spikeTimes"], ub["spikeTimes"])
                other = ub if len(ua["spikeTimes"]) <= len(ub["spikeTimes"]) else ua
                fmat[i, j] = f
                chance[i, j] = min(1.0, 0.002 * len(other["spikeTimes"]) / dur)
                same[i, j] = (ua["probeId"] == ub["probeId"]) and (ua["shank"] == ub["shank"])
        elig = same & (fmat >= 0.5) & (fmat >= 10 * chance)
        cost = np.where(elig, -fmat, 1e3)
        ri, ci = linear_sum_assignment(cost)
        nl = 0
        for i, j in zip(ri, ci):
            if elig[i, j]:
                nl += 1
                ua, ub = ua_l[i], ub_l[j]
                links.append(
                    {
                        "rule": "same_recording",
                        "file_a": a,
                        "idx_a": ua["idx"],
                        "file_b": b,
                        "idx_b": ub["idx"],
                        "f_coinc": fmat[i, j],
                        "chance": chance[i, j],
                        "dpos_um": abs(ua["shankPos"] - ub["shankPos"]),
                        "rm": ratemap_similarity(ua, ub),
                        "w": waveform_similarity(ua, ra, ub, rb),
                    }
                )
        pair_stats[f"{a}|{b}"] = {"rule": "same_recording", "units_a": len(ua_l), "units_b": len(ub_l), "links": nl}
    # ------------------------------------------------ different recordings: rate maps and waveforms
    diff_rec = [p for p in pairs_all if p not in same_rec]
    grid_ok = {p: bool(np.allclose(recs[p[0]]["gv_pos"], recs[p[1]]["gv_pos"])) for p in diff_rec}
    geo_ok = {p: probe_geometry_equal(recs[p[0]], recs[p[1]]) for p in diff_rec}
    null_files = sorted(set(b for a, b in diff_rec))
    for fb in null_files:
        rb = recs[fb]
        U = rb["units"]
        for i, j in itertools.combinations(range(len(U)), 2):
            ua, ub = U[i], U[j]
            if _same_shank_near(ua, ub):
                null_rows.append(
                    {
                        "file": fb,
                        "idx_a": ua["idx"],
                        "idx_b": ub["idx"],
                        "dpos_um": abs(ua["shankPos"] - ub["shankPos"]),
                        "rm": ratemap_similarity(ua, ub),
                        "w": waveform_similarity(ua, rb, ub, rb),
                    }
                )
    N = pd.DataFrame(null_rows, columns=["file", "idx_a", "idx_b", "dpos_um", "rm", "w"])
    thresholds: dict = {"n_null_pairs": int(len(N)), "null_files": null_files}
    if len(N):
        rm_star = max(MIN_MAP_THRESHOLD, float(np.nanpercentile(N.rm, 99)))
        Nw = N[N.w.notna()]
        w_star = float(np.nanpercentile(Nw.w, 99))
        q_joint = float(np.mean((Nw.rm >= rm_star) & (Nw.w >= w_star)))
        rm_alone = float(np.nanpercentile(N.rm, 100 * (1 - q_joint))) if q_joint > 0 else float(np.nanmax(N.rm))
        rm_alone = max(rm_alone, rm_star)
        q_rm_alone = float(np.mean(N.rm >= rm_alone))
        thresholds.update(
            {
                "n_null_pairs_with_waveform": int(len(Nw)),
                "rm_threshold": rm_star,
                "w_threshold": w_star,
                "joint_null_pass_rate": q_joint,
                "rm_alone_threshold": rm_alone,
                "rm_alone_null_pass_rate": q_rm_alone,
            }
        )
    for a, b in diff_rec:
        ra, rb = recs[a], recs[b]
        UA, UB = ra["units"], rb["units"]
        use_w = geo_ok[(a, b)] and all(u["wv"] is not None for u in UA[:1] + UB[:1])
        cand = []
        for i, ua in enumerate(UA):
            for j, ub in enumerate(UB):
                if _same_shank_near(ua, ub):
                    rmv = ratemap_similarity(ua, ub) if grid_ok[(a, b)] else np.nan
                    wv = waveform_similarity(ua, ra, ub, rb) if use_w else np.nan
                    cand.append((i, j, rmv, wv, abs(ua["shankPos"] - ub["shankPos"])))
        C = pd.DataFrame(cand, columns=["i", "j", "rm", "w", "dpos_um"])
        if use_w:
            C["pass"] = (C.rm >= thresholds["rm_threshold"]) & (C.w >= thresholds["w_threshold"])
            C["score"] = C.rm + C.w
            q = thresholds["joint_null_pass_rate"]
            rule = "waveform_and_map"
        else:
            C["pass"] = C.rm >= thresholds["rm_alone_threshold"]
            C["score"] = C.rm
            q = thresholds["rm_alone_null_pass_rate"]
            rule = "map_only"
        cost = np.full((len(UA), len(UB)), 1e3)
        for r_ in C[C["pass"]].itertuples():
            cost[r_.i, r_.j] = -r_.score
        ri, ci = linear_sum_assignment(cost)
        nl = 0
        for i, j in zip(ri, ci):
            if cost[i, j] < 1e3:
                nl += 1
                ua, ub = UA[i], UB[j]
                rr = C[(C.i == i) & (C.j == j)].iloc[0]
                links.append(
                    {
                        "rule": rule,
                        "file_a": a,
                        "idx_a": ua["idx"],
                        "file_b": b,
                        "idx_b": ub["idx"],
                        "f_coinc": np.nan,
                        "chance": np.nan,
                        "dpos_um": rr.dpos_um,
                        "rm": rr.rm,
                        "w": rr.w,
                    }
                )
        for r_ in C.itertuples():
            pair_rows.append(
                {
                    "file_a": a,
                    "idx_a": UA[r_.i]["idx"],
                    "file_b": b,
                    "idx_b": UB[r_.j]["idx"],
                    "rm": r_.rm,
                    "w": r_.w,
                    "dpos_um": r_.dpos_um,
                    "pass": bool(r_._6),
                }
            )
        exp_false = q * len(C)
        pair_stats[f"{a}|{b}"] = {
            "rule": rule,
            "units_a": len(UA),
            "units_b": len(UB),
            "candidate_pairs": int(len(C)),
            "pairs_passing": int(C["pass"].sum()),
            "links": nl,
            "expected_false_links": exp_false,
            "fdr_estimate": (exp_false / nl) if nl else None,
        }
    cols = ["rule", "file_a", "idx_a", "file_b", "idx_b", "f_coinc", "chance", "dpos_um", "rm", "w"]
    L = pd.DataFrame(links, columns=cols)
    details = {
        "null": N,
        "candidates": pd.DataFrame(
            pair_rows, columns=["file_a", "idx_a", "file_b", "idx_b", "rm", "w", "dpos_um", "pass"]
        ),
        "thresholds": thresholds,
        "pairs": pair_stats,
    }
    return L, details


# --------------------------------------------------------------------------------- tetrode datasets
def tetrode_entry_maps(cell_data: dict, centre: Optional[Tuple[float, float, float, float]] = None):
    """Raw spike and dwell maps of one recording on the arena-centred 5 cm grid.

    Parameters
    ----------
    cell_data : dict
        Per-cell input of :func:`hexmod.rat.gridcell.make_cell_data`.
    centre : tuple, optional
        ``(cx, cy, ex, ey)`` of the session: arena centre (midpoint of the 0.5 and 99.5
        percentiles of the speed-filtered positions) and extents (99.5 minus 0.5 percentile), cm.
        Computed from this recording when not given (it depends only on the session).

    Returns
    -------
    spikes : ndarray, float32 (50, 50)
        Spikes per bin (first index along x).
    dwell : ndarray, float32 (50, 50)
        Time per bin (s).
    centre : tuple
        ``(cx, cy, ex, ey)``.
    rate_hz : float
        Mean rate over the speed-filtered samples.
    """
    from .gridcell import GridCellAnalysis

    g = GridCellAnalysis(cell_data)
    x, y = np.asarray(g.x, float), np.asarray(g.y, float)
    t, spk = np.asarray(g.time, float), np.asarray(g.spike_array, float)
    dtv = float(g.dt)
    n_spikes = float(spk.sum())
    rate_hz = n_spikes / (len(t) * dtv) if len(t) else np.nan
    if centre is None:
        centre = (
            (np.percentile(x, 0.5) + np.percentile(x, 99.5)) / 2,
            (np.percentile(y, 0.5) + np.percentile(y, 99.5)) / 2,
            np.percentile(x, 99.5) - np.percentile(x, 0.5),
            np.percentile(y, 99.5) - np.percentile(y, 0.5),
        )
    cx, cy, ex, ey = centre
    GE = TETRODE_GRID_EDGES_CM
    S, _, _ = np.histogram2d(x - cx, y - cy, bins=[GE, GE], weights=spk)
    T, _, _ = np.histogram2d(x - cx, y - cy, bins=[GE, GE])
    T = T * dtv
    return S.astype(np.float32), T.astype(np.float32), centre, rate_hz


def tetrode_rate_map(spikes: np.ndarray, dwell: np.ndarray) -> np.ndarray:
    """Rate map from raw spike and dwell maps: Gaussian smoothing of one bin on both, NaN where the
    smoothed dwell is below 0.2 s."""
    S = gaussian_filter(np.asarray(spikes).astype(float), 1, mode="constant")
    T = gaussian_filter(np.asarray(dwell).astype(float), 1, mode="constant")
    r = np.full(S.shape, np.nan)
    ok = T >= TETRODE_MIN_DWELL_S
    r[ok] = S[ok] / T[ok]
    return r


def map_similarity(a: np.ndarray, b: np.ndarray, max_shift: int = 1) -> float:
    """Largest Pearson correlation of two rate maps over shifts of up to ``max_shift`` bins.

    Only shifts with at least 30 bins valid in both maps count; NaN when there is none.
    """
    best = -np.inf
    n = a.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for dx in range(-max_shift, max_shift + 1):
            for dy in range(-max_shift, max_shift + 1):
                xa = slice(max(0, dx), n + min(0, dx))
                xb = slice(max(0, -dx), n + min(0, -dx))
                ya = slice(max(0, dy), n + min(0, dy))
                yb = slice(max(0, -dy), n + min(0, -dy))
                p = a[xa, ya]
                q = b[xb, yb]
                m = np.isfinite(p) & np.isfinite(q)
                if m.sum() < 30:
                    continue
                c = np.corrcoef(p[m], q[m])[0, 1]
                if np.isfinite(c) and c > best:
                    best = c
    return float(best) if np.isfinite(best) else np.nan


def pair_features(ka: dict, kb: dict, map_a: np.ndarray, map_b: np.ndarray) -> dict:
    """Comparison of two tetrode recordings.

    Parameters
    ----------
    ka, kb : dict
        ``orientation_deg``, ``scale_cm``, ``Ex_cm``, ``Ey_cm`` (arena extents) and ``rate_hz``.
    map_a, map_b : ndarray
        Rate maps from :func:`tetrode_rate_map`.

    Returns
    -------
    dict
        ``rm`` (map correlation), ``dori`` (orientation difference modulo 60, deg), ``dscale_rel``
        (scale difference over the mean scale), ``same_size`` (bool) and ``log_rate_ratio``.
    """
    dori = abs((ka["orientation_deg"] - kb["orientation_deg"] + 30) % 60 - 30)
    dsc = abs(ka["scale_cm"] - kb["scale_cm"]) / ((ka["scale_cm"] + kb["scale_cm"]) / 2)
    tol = TETRODE_MAX_RELATIVE_ARENA_DIFF
    same_size = (
        abs(ka["Ex_cm"] - kb["Ex_cm"]) / ((ka["Ex_cm"] + kb["Ex_cm"]) / 2) <= tol
        and abs(ka["Ey_cm"] - kb["Ey_cm"]) / ((ka["Ey_cm"] + kb["Ey_cm"]) / 2) <= tol
    )
    lr = abs(np.log(max(ka["rate_hz"], 1e-6) / max(kb["rate_hz"], 1e-6)))
    return {
        "rm": map_similarity(map_a, map_b),
        "dori": dori,
        "dscale_rel": dsc,
        "same_size": same_size,
        "log_rate_ratio": lr,
    }


_HAFTING_PREFIX = re.compile(r"^rat\d+_")


def hafting_label(raw_unit_key: str) -> str:
    """Tetrode and cell label of a Hafting et al. (2005) unit from its manifest ``raw_unit_key``.

    ``"Hafting_Fig2c_Trial1.mat:rat11015_t5c1_timeStamps"`` gives ``"t5c1"``: the variable name
    without the animal tag and the ``_timeStamps`` suffix.
    """
    var = str(raw_unit_key).split(":", 1)[-1]
    return _HAFTING_PREFIX.sub("", var).replace("_timeStamps", "")


def link_tetrode(keys, rate_maps: np.ndarray, labels_dataset: str = "hafting2005"):
    """Link recordings of the tetrode datasets across sessions of one animal.

    Parameters
    ----------
    keys : pandas.DataFrame
        One row per recording in canonical order: ``row``, ``animal_id``, ``session_id``,
        ``cell_id``, ``dataset``, ``orientation_deg``, ``scale_cm``, ``Ex_cm``, ``Ey_cm``,
        ``rate_hz`` and ``label`` (tetrode and cell label, or None).
    rate_maps : ndarray (n, 50, 50)
        Rate maps of the recordings (:func:`tetrode_rate_map`), row-aligned with ``keys``.
    labels_dataset : str
        Dataset whose sessions hold the same labelled cells (the Hafting et al. 2005 trials).

    Returns
    -------
    pairs : pandas.DataFrame
        Every compared pair of recordings in different sessions of one animal, with the features of
        :func:`pair_features`, ``map_pass``, ``map_link`` (after one-to-one assignment),
        ``label_link`` and ``link``.
    thresholds : dict
        Per dataset: ``n_null`` (co-recorded pairs), ``rm_p99`` and ``rm_threshold``.
    """
    import pandas as pd

    K = keys.reset_index(drop=True)
    recs = K.to_dict("records")

    def feats(i, j):
        return pair_features(recs[i], recs[j], rate_maps[i], rate_maps[j])

    nulls = []
    for (an, sid), g in K.groupby(["animal_id", "session_id"]):
        for i, j in itertools.combinations(g.index.to_list(), 2):
            f_ = feats(i, j)
            f_.update({"dataset": recs[i]["dataset"], "animal_id": an, "session_id": sid})
            nulls.append(f_)
    NL = pd.DataFrame(
        nulls, columns=["rm", "dori", "dscale_rel", "same_size", "log_rate_ratio", "dataset", "animal_id", "session_id"]
    )
    thr: Dict[str, dict] = {}
    for ds, g in NL.groupby("dataset"):
        p99 = float(np.nanpercentile(g.rm, 99)) if g.rm.notna().any() else np.nan
        thr[ds] = {
            "n_null": int(len(g)),
            "rm_p99": p99,
            "rm_threshold": max(MIN_MAP_THRESHOLD, p99) if np.isfinite(p99) else MIN_MAP_THRESHOLD,
        }
    for ds in K.dataset.unique():
        thr.setdefault(ds, {"n_null": 0, "rm_p99": np.nan, "rm_threshold": MIN_MAP_THRESHOLD})

    rows = []
    for an, g in K.groupby("animal_id"):
        ss = sorted(g.session_id.unique(), key=session_number)
        for sa, sb in itertools.combinations(ss, 2):
            ia = g.index[g.session_id == sa].to_list()
            ib = g.index[g.session_id == sb].to_list()
            for i in ia:
                for j in ib:
                    f_ = feats(i, j)
                    ds = recs[i]["dataset"]
                    f_.update(
                        {
                            "dataset": ds,
                            "animal_id": an,
                            "session_a": sa,
                            "session_b": sb,
                            "cell_a": int(recs[i]["cell_id"]),
                            "cell_b": int(recs[j]["cell_id"]),
                            "row_a": int(recs[i]["row"]),
                            "row_b": int(recs[j]["row"]),
                            "i": i,
                            "j": j,
                            "label_a": recs[i]["label"],
                            "label_b": recs[j]["label"],
                        }
                    )
                    f_["map_pass"] = bool(
                        (f_["rm"] >= thr[ds]["rm_threshold"])
                        and f_["dori"] <= TETRODE_MAX_ORIENTATION_DIFF_DEG
                        and f_["dscale_rel"] <= TETRODE_MAX_RELATIVE_SCALE_DIFF
                        and f_["same_size"]
                    )
                    rows.append(f_)
    cols = [
        "rm",
        "dori",
        "dscale_rel",
        "same_size",
        "log_rate_ratio",
        "dataset",
        "animal_id",
        "session_a",
        "session_b",
        "cell_a",
        "cell_b",
        "row_a",
        "row_b",
        "i",
        "j",
        "label_a",
        "label_b",
        "map_pass",
    ]
    P = pd.DataFrame(rows, columns=cols)
    P["map_pass"] = P["map_pass"].astype(bool)
    P["map_link"] = False
    for (sa, sb), g in P[P.map_pass].groupby(["session_a", "session_b"]):
        ui = sorted(g.i.unique())
        uj = sorted(g.j.unique())
        cost = np.full((len(ui), len(uj)), 1e3)
        for r in g.itertuples():
            cost[ui.index(r.i), uj.index(r.j)] = -r.rm
        ri, ci = linear_sum_assignment(cost)
        for a_, b_ in zip(ri, ci):
            if cost[a_, b_] < 1e3:
                P.loc[(P.session_a == sa) & (P.session_b == sb) & (P.i == ui[a_]) & (P.j == uj[b_]), "map_link"] = True
    has_label = P.label_a.notna() & P.label_b.notna()
    P["label_link"] = (P.dataset == labels_dataset) & has_label & (P.label_a == P.label_b)
    P["link"] = P.map_link | P.label_link
    return P.drop(columns=["i", "j"]), thr


# ------------------------------------------------------------------------------ units and kept rows
class UnionFind:
    """Disjoint sets over hashable nodes (path halving)."""

    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        """Return the representative of the set holding ``x`` (adding ``x`` if new)."""
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b) -> None:
        """Merge the sets holding ``a`` and ``b``."""
        self.p[self.find(a)] = self.find(b)


def session_number(session_id: str) -> int:
    """Number of a session within its animal: the integer after the underscore (``V10_2`` gives 2)."""
    return int(str(session_id).split("_")[1])


def unit_components(
    entries: Sequence[Tuple[str, int]],
    vollan_links,
    vollan_session_files: Dict[str, str],
    entry_links: Sequence[Tuple[Tuple[str, int], Tuple[str, int]]],
) -> np.ndarray:
    """Group recordings into units (connected components of the links).

    Parameters
    ----------
    entries : sequence of (session_id, cell_id)
        Cohort recordings in canonical order.
    vollan_links : pandas.DataFrame
        Links between Vollan units (``file_a``, ``idx_a``, ``file_b``, ``idx_b``).
    vollan_session_files : dict
        ``{session_id: file}`` of the linked Vollan files; a cohort recording of such a session is
        joined to the unit ``(file, cell_id)``.
    entry_links : sequence of pairs of (session_id, cell_id)
        Links between cohort recordings (tetrode datasets).

    Returns
    -------
    ndarray of int64
        Unit number of each recording, numbered in order of first appearance.
    """
    uf = UnionFind()
    for e in entries:
        uf.find(("E", e))
    for r in vollan_links.itertuples():
        uf.union(("U", r.file_a, int(r.idx_a)), ("U", r.file_b, int(r.idx_b)))
    for s, c in entries:
        if s in vollan_session_files:
            uf.union(("E", (s, c)), ("U", vollan_session_files[s], int(c)))
    for ea, eb in entry_links:
        uf.union(("E", ea), ("E", eb))
    roots = [uf.find(("E", e)) for e in entries]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots], dtype=np.int64)


def kept_earliest(units: np.ndarray, session_numbers: Sequence[int]) -> np.ndarray:
    """Flag the earliest recording of each unit.

    The earliest recording has the lowest session number; among recordings with that number the
    first in the given (canonical) order is kept.

    Returns
    -------
    ndarray of bool
    """
    units = np.asarray(units)
    snum = np.asarray(session_numbers)
    first: Dict[int, int] = {}
    for u, s in zip(units.tolist(), snum.tolist()):
        if u not in first or s < first[u]:
            first[u] = s
    kept = np.zeros(len(units), dtype=bool)
    seen = set()
    for k, (u, s) in enumerate(zip(units.tolist(), snum.tolist())):
        if s == first[u] and u not in seen:
            kept[k] = True
            seen.add(u)
    return kept
