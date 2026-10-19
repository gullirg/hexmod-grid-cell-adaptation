"""Firing rate on successive field visits: the visit-sequence test of Fig 3F.

The test asks whether the firing rate on a field visit depends on how soon it follows the
previous visit, relative to the interval before that, using only the order of visits within a
cell.

**Field visits** (``visit_events_for_cell``). For every cohort cell the fields are delineated
again and the visits to them are extracted along the speed-filtered trajectory with
:func:`hexmod.rat.fields.extract_field_entry`. Each visit has a firing rate ``FR`` (spikes per
second in the field), an effective duration ``dur_eff`` (samples in the field times the sample
interval), its t_prev (time since the end of the previous visit), the number of spikes, the mean
running speed, the fraction of samples faster than 5 cm/s and the distance from the field centre
to the nearest edge of the area covered by the trajectory. Two extraction rules exist:

``"printed"`` (default, the published Fig 3F)
    Fields from a rate map of spike counts (``rate_map_spike_mode="counts"``) and the gap-aware
    t_prev rule (``tprev_rule="gap_aware"``): t_prev is undefined for the first visit, for
    intervals that are not positive and for intervals spanning a step of more than 5 s in the
    speed-filtered time axis.
``"library"``
    The settings of the other rat analyses: fields from the binary rate map (the fields of the
    cohort table) and the standard t_prev rule. Under this rule every t_prev is defined; the first
    visit and non-positive intervals are excluded by the gap window below.

**Sequences** (``triplet_rows_for_cell``). Every visit from the third on is taken as visit 3 of
a sequence with the two visits before it. With T12 the t_prev of visit 2 and T23 that of visit 3,
a sequence is kept when both are defined and lie in the gap window (0 < T <= 10 s), each of the
three visits lasts longer than 0.1 s, consecutive visits go to different fields, FR2 + FR3 > 0,
and the spike rule holds:

- ``"spike_ge2"``: at least 2 observed spikes in each of visits 1 to 3;
- ``"expected_ge2"``: at least 2 expected spikes in each of visits 1 to 3, where the expected
  count is the cell's mean in-field rate (all its visits) times the visit duration, so that the
  selection depends on exposure rather than on the observed response;
- ``"expected_ge2_fr23"``: as ``"expected_ge2"`` for visits 2 and 3 only, the two visits whose
  rates are compared (Fig 3F);
- ``"fr_positive"``: FR above 0 in visits 1 to 3.

For quadruplets (Fig 3F) a fourth visit must follow visit 3, with a defined t_prev T34 in
0 <= T34 <= 10 s (the lower limit is inclusive here, unlike that of T12 and T23), a duration above
0.1 s and a field other than that of visit 3, so that visit 3 is never the last visit of a run.

**Response.** The firing-rate ratio FRR = (FR2 - FR3) / (FR2 + FR3) is positive when visit 3 has
the lower rate. Sequences fall in two groups, ``T23<T12`` (visit 3 comes sooner after visit 2
than visit 2 after visit 1) and ``T23>T12``; ties are dropped. Adaptation predicts a positive
FRR in the first group and a negative FRR in the second.

**Cell values and tests** (``paired_cell_values``, ``visit_sequence_summary``). FRR is averaged
per cell and group; cells with sequences in both groups are paired cells. Fig 3F tests the
per-cell means of each group against zero and their paired difference (``T23<T12`` minus
``T23>T12``) with one-sample t-tests. ``nesting_sensitivity`` repeats the paired test on means
per session, animal and dataset, and ``sampling_diagnostic`` reports whether visit 3 differs
from visit 2 in speed, duration or wall distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .characterise import characterise_gridcell
from .fields import extract_field_entry
from .gridcell import GridCellAnalysis

__all__ = [
    "RULES", "extraction_settings", "SequenceSpec", "FIG3F_SPECS", "FIG3F_ANALYSIS", "VOLLAN_KEY",
    "EVENT_COLUMNS", "augment_event_table", "visit_events_for_cell", "cell_key", "events_frame",
    "triplet_rows_for_cell", "build_triplet_table", "paired_cell_values", "nesting_sensitivity",
    "sampling_diagnostic", "visit_sequence_summary", "fig3f_statistics", "nesting_value", "SPIKE_RULES",
]

#: Extraction settings of the two rules (see the module docstring).
RULES: Dict[str, dict] = {
    "printed": {"rate_map_spike_mode": "counts", "tprev_rule": "gap_aware", "tprev_cap": 10.0,
                "recording_gap_threshold": 5.0, "dt": 0.02},
    "library": {"rate_map_spike_mode": "binary", "tprev_rule": "standard", "tprev_cap": 10.0,
                "recording_gap_threshold": 5.0, "dt": 0.02},
}

#: Field delineation settings, as for the cohort (minimum peak distance 2/3 of the scale,
#: rate threshold at the 40th percentile of the map).
CHARACTERISATION = {"min_dist_btw_peaks_rel": 2 / 3, "prominence": 0.4}

#: Dataset key of the largest dataset, analysed separately in ``nesting_sensitivity``.
VOLLAN_KEY = "vollan2025"

#: Numeric columns of the visit table (one row per field visit), in the order they are stored.
EVENT_COLUMNS = (
    "event_idx", "t_enter", "t_exit", "fieldId", "t_prev", "t_prev_valid", "spans_recording_gap",
    "field_x", "field_y", "radius", "direction_prev", "dur_eff", "FR", "FR_prevOutfield",
    "mean_speed", "prop_speed_gt5", "spike_count", "wall_distance",
)

#: Identifier columns added to the visit table from the cohort table.
META_COLUMNS = ("row", "key", "animal_id", "session_id", "cell_id", "unique_cell_id", "dataset")

SPIKE_RULES = ("spike_ge2", "expected_ge2", "expected_ge2_fr23", "fr_positive")

#: Key and value columns of the paired-cell table (``paired_cell_values``).
PAIRED_COLUMNS = ["key", "frr_T23_lt_T12", "frr_T23_gt_T12", "n_events_T23_lt_T12", "n_events_T23_gt_T12",
                  "paired_difference", "antisymmetry_offset"]


def extraction_settings(rule: str = "printed") -> dict:
    """Return a copy of the extraction settings of a rule ("printed" or "library")."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {sorted(RULES)}, got {rule!r}")
    return dict(RULES[rule])


@dataclass(frozen=True)
class SequenceSpec:
    """Selection of visit sequences.

    Attributes
    ----------
    name : str
        Name of the analysis (used in the tables).
    min_gap, max_gap : float
        Gap window for T12 and T23 (s); with ``min_gap == 0`` the lower limit is exclusive.
    require_fourth : bool
        Require a fourth visit after visit 3 (quadruplet).
    spike_rule : str
        One of ``SPIKE_RULES``.
    exclude_consecutive_same_field : bool
        Drop sequences in which consecutive visits go to the same field.
    min_visit_dur : float
        Minimum duration of each visit (s, exclusive).
    """

    name: str
    min_gap: float
    max_gap: float
    require_fourth: bool
    spike_rule: str
    exclude_consecutive_same_field: bool = True
    min_visit_dur: float = 0.1


#: The three analyses of the diagnostics table; the last one is Fig 3F.
FIG3F_SPECS = (
    SequenceSpec("triplet_0to10_observed_ge2", 0.0, 10.0, False, "spike_ge2"),
    SequenceSpec("triplet_0to10_expected_ge2_fr23", 0.0, 10.0, False, "expected_ge2_fr23"),
    SequenceSpec("quadruplet_0to10_expected_ge2_fr23", 0.0, 10.0, True, "expected_ge2_fr23"),
)
FIG3F_ANALYSIS = "quadruplet_0to10_expected_ge2_fr23"


# ------------------------------------------------------------------------------------ field visits
def augment_event_table(ev_df: pd.DataFrame, gc_obj: GridCellAnalysis) -> pd.DataFrame:
    """Add spike count, speed and wall distance of each visit to the visit table.

    Parameters
    ----------
    ev_df : pandas.DataFrame
        ``ev_df`` of :func:`hexmod.rat.fields.extract_field_entry`.
    gc_obj : GridCellAnalysis
        The speed-filtered samples the visits were extracted from.

    Returns
    -------
    pandas.DataFrame
        Copy of ``ev_df`` with ``event_idx`` (visit number in time order), ``mean_speed``
        (cm/s), ``prop_speed_gt5`` (fraction of samples faster than 5 cm/s), ``spike_count`` and
        ``wall_distance`` (cm, from the field centre to the nearest edge of the bounding box of
        the trajectory; slightly negative when the centre, a rate-map bin centre, lies outside it).
    """
    ev = ev_df.copy().reset_index(drop=True)
    t = np.asarray(gc_obj.time)
    x = np.asarray(gc_obj.x)
    y = np.asarray(gc_obj.y)
    speed = np.asarray(gc_obj.speed)
    spikes = np.asarray(gc_obj.spike_array)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    # samples from the entry up to (not including) the exit; at least one sample
    on_idx = np.searchsorted(t, ev["t_enter"].to_numpy(), side="left")
    off_idx = np.searchsorted(t, ev["t_exit"].to_numpy(), side="left")
    off_idx = np.clip(off_idx, on_idx + 1, len(t))

    mean_speed = np.full(len(ev), np.nan)
    prop_speed_gt5 = np.full(len(ev), np.nan)
    spike_count = np.full(len(ev), np.nan)
    for i, (a, b) in enumerate(zip(on_idx, off_idx)):
        sl = slice(int(a), int(b))
        mean_speed[i] = np.nanmean(speed[sl]) if b > a else np.nan
        prop_speed_gt5[i] = np.nanmean(speed[sl] > 5.0) if b > a else np.nan
        spike_count[i] = np.nansum(spikes[sl]) if b > a else np.nan

    wall_distance = np.minimum.reduce(
        [
            ev["field_x"].to_numpy(dtype=float) - xmin,
            xmax - ev["field_x"].to_numpy(dtype=float),
            ev["field_y"].to_numpy(dtype=float) - ymin,
            ymax - ev["field_y"].to_numpy(dtype=float),
        ]
    )

    ev["event_idx"] = np.arange(len(ev), dtype=int)
    ev["mean_speed"] = mean_speed
    ev["prop_speed_gt5"] = prop_speed_gt5
    ev["spike_count"] = spike_count
    ev["wall_distance"] = wall_distance
    return ev


def visit_events_for_cell(item: dict) -> dict:
    """Field visits of one cell (worker function for ``hexmod.parallel.map_ordered``).

    Parameters
    ----------
    item : dict
        ``row`` (cohort row), ``cell_data`` (see :func:`hexmod.rat.gridcell.make_cell_data`)
        and the extraction settings of a rule (``rate_map_spike_mode``, ``tprev_rule``,
        ``tprev_cap``, ``recording_gap_threshold``, ``dt``; see ``RULES``).

    Returns
    -------
    dict
        ``row``; ``events`` (dict of ``EVENT_COLUMNS`` arrays, one entry per visit) or None;
        ``reason`` (why the cell has no visits, else None); ``error`` (the exception raised while
        extracting the visits, else None; the cell then has no visits and the caller decides
        whether the run can continue); ``gridness``, ``scale_cm``, ``orientation_deg`` and
        ``n_fields`` of the re-delineated rate map (NaN and 0 when the cell does not pass the
        gridness threshold on that map).
    """
    row = int(item["row"])
    out = {"row": row, "events": None, "reason": None, "error": None, "gridness": np.nan, "scale_cm": np.nan,
           "orientation_deg": np.nan, "n_fields": 0}
    try:
        cell_data = dict(item["cell_data"])
        cell_data["rate_map_spike_mode"] = item["rate_map_spike_mode"]
        char = characterise_gridcell({"cell_data": cell_data, **CHARACTERISATION,
                                      "rate_map_spike_mode": item["rate_map_spike_mode"]})
        if char is None or char.get("fieldLoc") is None or len(char["fieldLoc"]) == 0:
            out["reason"] = "no field on the re-delineated rate map"
            return out
        out.update(gridness=float(char["gridness"]), scale_cm=float(char["scale_cm"]),
                   orientation_deg=float(char["orientation_deg"]), n_fields=int(len(char["fieldLoc"])))
        gc_obj = GridCellAnalysis(cell_data, rate_map_spike_mode=item["rate_map_spike_mode"])
        field_loc = char["fieldLoc"]
        tmp = extract_field_entry(
            gc_obj.x, gc_obj.y, gc_obj.time, field_loc[:, 0], field_loc[:, 1], field_loc[:, 2],
            gc_obj.spike_array, dt=item["dt"], tprev_rule=item["tprev_rule"], tprev_cap=item["tprev_cap"],
            recording_gap_threshold=item["recording_gap_threshold"],
        )
        ev = augment_event_table(tmp["ev_df"], gc_obj)
        if "t_prev_valid" not in ev:
            # standard rule: every t_prev is defined (non-positive values fail the gap window)
            ev["t_prev_valid"] = np.ones(len(ev), dtype=bool)
            ev["spans_recording_gap"] = np.zeros(len(ev), dtype=bool)
        out["events"] = {col: ev[col].to_numpy() for col in EVENT_COLUMNS}
    except Exception as exc:  # reported per cell; step 06 stops unless failures are allowed
        out["error"] = repr(exc)
        out["reason"] = f"error: {exc!r}"
        out["events"] = None
    return out


def cell_key(animal_id, session_id, cell_id) -> str:
    """Return the text key ``"<animal>|<session>|<cell>"`` that identifies a cell in the tables."""
    return f"{animal_id}|{session_id}|{int(cell_id)}"


def events_frame(events: Dict[str, np.ndarray], cells: pd.DataFrame) -> pd.DataFrame:
    """Visit table with the identifiers of each cell.

    Parameters
    ----------
    events : dict
        ``row`` and the ``EVENT_COLUMNS`` arrays of all visits (as stored in the visit cache).
    cells : pandas.DataFrame
        Cohort table with ``row``, ``animal_id``, ``session_id``, ``cell_id``,
        ``unique_cell_id`` and ``dataset``.

    Returns
    -------
    pandas.DataFrame
        One row per visit with ``META_COLUMNS`` and ``EVENT_COLUMNS``.
    """
    rows = np.asarray(events["row"], dtype=np.int64)
    meta = cells.set_index("row")
    out = {"row": rows}
    keys = np.array([cell_key(a, s, c) for a, s, c in zip(meta["animal_id"], meta["session_id"], meta["cell_id"])],
                    dtype=object)
    pos = meta.index.get_indexer(rows)
    if (pos < 0).any():
        raise ValueError("visit table holds rows that are not in the cohort table")
    out["key"] = keys[pos]
    for col in ("animal_id", "session_id", "cell_id", "unique_cell_id", "dataset"):
        out[col] = meta[col].to_numpy()[pos]
    for col in EVENT_COLUMNS:
        out[col] = events[col]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------- sequences
def _valid_spike_rule(spike_rule: str, rows: Dict[str, np.ndarray]) -> np.ndarray:
    if spike_rule == "spike_ge2":
        return (rows["spike1"] >= 2) & (rows["spike2"] >= 2) & (rows["spike3"] >= 2)
    if spike_rule == "expected_ge2":
        return (rows["expected_spike1"] >= 2) & (rows["expected_spike2"] >= 2) & (rows["expected_spike3"] >= 2)
    if spike_rule == "expected_ge2_fr23":
        return (rows["expected_spike2"] >= 2) & (rows["expected_spike3"] >= 2)
    if spike_rule == "fr_positive":
        return (rows["fr1"] > 0) & (rows["fr2"] > 0) & (rows["fr3"] > 0)
    raise ValueError(f"Unknown spike rule: {spike_rule}")


def triplet_rows_for_cell(ev: pd.DataFrame, spec: SequenceSpec) -> pd.DataFrame:
    """Visit sequences of one cell that pass the selection of ``spec``.

    Parameters
    ----------
    ev : pandas.DataFrame
        Visit table of one cell (``events_frame`` rows of that cell).
    spec : SequenceSpec
        Selection.

    Returns
    -------
    pandas.DataFrame
        One row per kept sequence: the t_prev values ``t12``, ``t23`` (and ``t34``), rates
        ``fr1`` to ``fr3``, durations, spike counts, speeds and wall distances of the visits,
        ``group`` (``"T23<T12"`` or ``"T23>T12"``), ``frr_23`` and the visit 3 minus visit 2
        differences ``speed3_minus2``, ``dur3_minus2``, ``prop_speed_gt5_3_minus2``,
        ``wall3_minus2``, ``fr3_minus2``, plus the cell identifiers and ``analysis``. Empty when
        no sequence passes.
    """
    ev = ev.sort_values("event_idx").reset_index(drop=True)
    n = len(ev)
    if n < (4 if spec.require_fourth else 3):
        return pd.DataFrame()

    # mean in-field rate of the cell over all its visits (for the expected spike counts)
    dur_all = ev["dur_eff"].to_numpy(dtype=float)
    spike_all = ev["spike_count"].to_numpy(dtype=float)
    baseline_mask = np.isfinite(dur_all) & np.isfinite(spike_all) & (dur_all > 0)
    if baseline_mask.any() and np.nansum(dur_all[baseline_mask]) > 0:
        baseline_infield_rate = np.nansum(spike_all[baseline_mask]) / np.nansum(dur_all[baseline_mask])
    else:
        baseline_infield_rate = np.nan

    current_idx = np.arange(2, n - (1 if spec.require_fourth else 0), dtype=int)
    e1 = current_idx - 2
    e2 = current_idx - 1
    e3 = current_idx
    e4 = current_idx + 1

    rows = {
        "t12": ev.loc[e2, "t_prev"].to_numpy(dtype=float),
        "t23": ev.loc[e3, "t_prev"].to_numpy(dtype=float),
        "tprev_valid2": ev.loc[e2, "t_prev_valid"].to_numpy(dtype=bool),
        "tprev_valid3": ev.loc[e3, "t_prev_valid"].to_numpy(dtype=bool),
        "fr1": ev.loc[e1, "FR"].to_numpy(dtype=float),
        "fr2": ev.loc[e2, "FR"].to_numpy(dtype=float),
        "fr3": ev.loc[e3, "FR"].to_numpy(dtype=float),
        "dur1": ev.loc[e1, "dur_eff"].to_numpy(dtype=float),
        "dur2": ev.loc[e2, "dur_eff"].to_numpy(dtype=float),
        "dur3": ev.loc[e3, "dur_eff"].to_numpy(dtype=float),
        "spike1": ev.loc[e1, "spike_count"].to_numpy(dtype=float),
        "spike2": ev.loc[e2, "spike_count"].to_numpy(dtype=float),
        "spike3": ev.loc[e3, "spike_count"].to_numpy(dtype=float),
        "speed1": ev.loc[e1, "mean_speed"].to_numpy(dtype=float),
        "speed2": ev.loc[e2, "mean_speed"].to_numpy(dtype=float),
        "speed3": ev.loc[e3, "mean_speed"].to_numpy(dtype=float),
        "prop_speed_gt5_1": ev.loc[e1, "prop_speed_gt5"].to_numpy(dtype=float),
        "prop_speed_gt5_2": ev.loc[e2, "prop_speed_gt5"].to_numpy(dtype=float),
        "prop_speed_gt5_3": ev.loc[e3, "prop_speed_gt5"].to_numpy(dtype=float),
        "wall1": ev.loc[e1, "wall_distance"].to_numpy(dtype=float),
        "wall2": ev.loc[e2, "wall_distance"].to_numpy(dtype=float),
        "wall3": ev.loc[e3, "wall_distance"].to_numpy(dtype=float),
        "field1": ev.loc[e1, "fieldId"].to_numpy(dtype=int),
        "field2": ev.loc[e2, "fieldId"].to_numpy(dtype=int),
        "field3": ev.loc[e3, "fieldId"].to_numpy(dtype=int),
    }
    rows["expected_spike1"] = baseline_infield_rate * rows["dur1"]
    rows["expected_spike2"] = baseline_infield_rate * rows["dur2"]
    rows["expected_spike3"] = baseline_infield_rate * rows["dur3"]

    valid = (
        np.isfinite(rows["t12"])
        & np.isfinite(rows["t23"])
        & rows["tprev_valid2"]
        & rows["tprev_valid3"]
    )
    if spec.min_gap == 0:
        valid &= (rows["t12"] > 0) & (rows["t12"] <= spec.max_gap)
        valid &= (rows["t23"] > 0) & (rows["t23"] <= spec.max_gap)
    else:
        valid &= (rows["t12"] >= spec.min_gap) & (rows["t12"] <= spec.max_gap)
        valid &= (rows["t23"] >= spec.min_gap) & (rows["t23"] <= spec.max_gap)
    valid &= ((rows["dur1"] > spec.min_visit_dur) & (rows["dur2"] > spec.min_visit_dur)
              & (rows["dur3"] > spec.min_visit_dur))
    valid &= _valid_spike_rule(spec.spike_rule, rows)
    valid &= (rows["fr2"] + rows["fr3"]) > 0
    if spec.exclude_consecutive_same_field:
        valid &= (rows["field1"] != rows["field2"]) & (rows["field2"] != rows["field3"])

    out = pd.DataFrame({k: v[valid] for k, v in rows.items()})
    if out.empty:
        return out

    if spec.require_fourth:
        t34 = ev.loc[e4, "t_prev"].to_numpy(dtype=float)[valid]
        tprev_valid4 = ev.loc[e4, "t_prev_valid"].to_numpy(dtype=bool)[valid]
        dur4 = ev.loc[e4, "dur_eff"].to_numpy(dtype=float)[valid]
        field4 = ev.loc[e4, "fieldId"].to_numpy(dtype=int)[valid]
        fourth_valid = (
            np.isfinite(t34)
            & tprev_valid4
            & (t34 >= spec.min_gap)
            & (t34 <= spec.max_gap)
            & (dur4 > spec.min_visit_dur)
        )
        if spec.exclude_consecutive_same_field:
            fourth_valid &= out["field3"].to_numpy(dtype=int) != field4
        out["t34"] = t34
        out["dur4"] = dur4
        out["field4"] = field4
        out = out[fourth_valid].copy()
        if out.empty:
            return out

    out["group"] = np.where(out["t23"] < out["t12"], "T23<T12", np.where(out["t23"] > out["t12"], "T23>T12", "tie"))
    out = out[out["group"] != "tie"].copy()
    if out.empty:
        return out

    for col in META_COLUMNS:
        if col in ev:
            out[col] = ev[col].iloc[0]
    out["analysis"] = spec.name
    out["frr_23"] = (out["fr2"] - out["fr3"]) / (out["fr2"] + out["fr3"])
    out["speed3_minus2"] = out["speed3"] - out["speed2"]
    out["dur3_minus2"] = out["dur3"] - out["dur2"]
    out["prop_speed_gt5_3_minus2"] = out["prop_speed_gt5_3"] - out["prop_speed_gt5_2"]
    out["wall3_minus2"] = out["wall3"] - out["wall2"]
    out["fr3_minus2"] = out["fr3"] - out["fr2"]
    return out


#: Columns of an empty sequence table (no cell has a kept sequence).
SEQUENCE_COLUMNS = ("key", "row", "animal_id", "session_id", "cell_id", "unique_cell_id", "dataset", "analysis",
                    "group", "frr_23", "speed3_minus2", "dur3_minus2", "wall3_minus2")


def build_triplet_table(events: pd.DataFrame, specs: Iterable[SequenceSpec]) -> pd.DataFrame:
    """Kept sequences of every cell (``events_frame`` output) for each selection in ``specs``."""
    specs = list(specs)
    rows = []
    for _, ev_cell in events.groupby("key", sort=False):
        for spec in specs:
            tmp = triplet_rows_for_cell(ev_cell, spec)
            if not tmp.empty:
                rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(SEQUENCE_COLUMNS))


# ------------------------------------------------------------------------------- cell values, tests
def paired_cell_values(events: pd.DataFrame) -> pd.DataFrame:
    """Mean FRR per cell in each group, for the cells with sequences in both groups.

    Parameters
    ----------
    events : pandas.DataFrame
        Kept sequences of one analysis (``build_triplet_table`` rows with one ``analysis``).

    Returns
    -------
    pandas.DataFrame
        One row per paired cell, sorted by ``key``: ``key``, ``frr_T23_lt_T12``,
        ``frr_T23_gt_T12``, ``n_events_T23_lt_T12``, ``n_events_T23_gt_T12``,
        ``paired_difference`` (lt minus gt), ``antisymmetry_offset`` (lt plus gt), ``dataset``,
        ``animal_id``, ``session_id``, ``cell_id``, ``unique_cell_id`` and ``row`` when present.
    """
    meta_cols = ["key", "dataset", "animal_id", "session_id", "cell_id", "unique_cell_id"]
    if "row" in events:
        meta_cols.append("row")
    if events.empty or not {"T23<T12", "T23>T12"}.issubset(set(events["group"])):
        return pd.DataFrame(columns=PAIRED_COLUMNS + meta_cols[1:])  # no cell with both groups
    cell_means = (
        events.groupby(["key", "group"], observed=True)
        .agg(frr_23=("frr_23", "mean"), n_events=("frr_23", "size"))
        .reset_index()
    )
    pivot = cell_means.pivot_table(index="key", columns="group", values="frr_23", aggfunc="mean")
    pivot = pivot.dropna(subset=["T23<T12", "T23>T12"])
    counts = cell_means.pivot_table(index="key", columns="group", values="n_events", aggfunc="sum").reindex(pivot.index)

    paired = pd.DataFrame(
        {
            "key": pivot.index,
            "frr_T23_lt_T12": pivot["T23<T12"].to_numpy(dtype=float),
            "frr_T23_gt_T12": pivot["T23>T12"].to_numpy(dtype=float),
            "n_events_T23_lt_T12": counts["T23<T12"].fillna(0).to_numpy(dtype=int),
            "n_events_T23_gt_T12": counts["T23>T12"].fillna(0).to_numpy(dtype=int),
        }
    )
    paired["paired_difference"] = paired["frr_T23_lt_T12"] - paired["frr_T23_gt_T12"]
    paired["antisymmetry_offset"] = paired["frr_T23_lt_T12"] + paired["frr_T23_gt_T12"]
    metadata = events[meta_cols].drop_duplicates("key")
    paired = paired.merge(metadata, on="key", how="left", validate="one_to_one")
    return paired


def nesting_sensitivity(paired: pd.DataFrame) -> pd.DataFrame:
    """Paired test of Fig 3F on means per cell, session, animal and dataset.

    Repeated for all paired cells, for the cells of ``VOLLAN_KEY`` only and for the other
    datasets.

    Returns
    -------
    pandas.DataFrame
        ``cohort``, ``level``, ``n``, ``mean_paired_difference``, ``sem_paired_difference``,
        ``t`` and ``p_two_sided`` (one-sample t-test against 0; NaN with fewer than 2 values).
    """
    rows = []
    is_vollan = paired["dataset"].astype(str) == VOLLAN_KEY
    cohorts = {
        "all": paired,
        "vollan_only": paired[is_vollan],
        "without_vollan": paired[~is_vollan],
    }
    levels = {
        "cell": "key",
        "session": "session_id",
        "animal": "animal_id",
        "dataset": "dataset",
    }
    for cohort_name, cohort in cohorts.items():
        for level, identifier in levels.items():
            values = (
                cohort.groupby(identifier, observed=True)["paired_difference"].mean()
                if level != "cell"
                else cohort.set_index(identifier)["paired_difference"]
            )
            values = values.dropna().to_numpy(dtype=float)
            test = stats.ttest_1samp(values, 0.0) if len(values) > 1 else None
            rows.append(
                {
                    "cohort": cohort_name,
                    "level": level,
                    "n": int(len(values)),
                    "mean_paired_difference": float(values.mean()) if len(values) else float("nan"),
                    "sem_paired_difference": float(stats.sem(values)) if len(values) > 1 else float("nan"),
                    "t": float(test.statistic) if test is not None else float("nan"),
                    "p_two_sided": float(test.pvalue) if test is not None else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def sampling_diagnostic(events: pd.DataFrame, label: str) -> Dict[str, object]:
    """Summary of one analysis: paired test, antisymmetry offset and visit 3 minus visit 2 covariates.

    Parameters
    ----------
    events : pandas.DataFrame
        Kept sequences of one analysis.
    label : str
        Value of the ``analysis`` column.

    Returns
    -------
    dict
        The columns of ``rat/visits/diagnostics.csv``. The covariate differences (speed,
        duration, distance of the field from the wall) are averaged per cell over all sequences
        and tested against 0.
    """
    paired = paired_cell_values(events)
    difference = paired["paired_difference"].to_numpy(dtype=float)
    offset = paired["antisymmetry_offset"].to_numpy(dtype=float)
    role_means = (
        events.groupby("key", observed=True)
        .agg(
            speed3_minus2=("speed3_minus2", "mean"),
            dur3_minus2=("dur3_minus2", "mean"),
            wall3_minus2=("wall3_minus2", "mean"),
        )
        .reset_index()
    )

    def one_sample(values: pd.Series):
        values = values.to_numpy(dtype=float)
        test = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        return float(pd.Series(values).mean()), float(test.statistic), float(test.pvalue)

    speed_mean, speed_t, speed_p = one_sample(role_means["speed3_minus2"])
    duration_mean, duration_t, duration_p = one_sample(role_means["dur3_minus2"])
    wall_mean, wall_t, wall_p = one_sample(role_means["wall3_minus2"])
    difference_test = stats.ttest_1samp(difference, 0.0)
    offset_test = stats.ttest_1samp(offset, 0.0)
    return {
        "analysis": label,
        "n_events": int(len(events)),
        "n_cells_any_group": int(events["key"].nunique()),
        "n_paired_cells": int(len(paired)),
        "mean_T23_lt_T12": float(paired["frr_T23_lt_T12"].mean()),
        "mean_T23_gt_T12": float(paired["frr_T23_gt_T12"].mean()),
        "paired_difference": float(difference.mean()),
        "paired_t": float(difference_test.statistic),
        "paired_p": float(difference_test.pvalue),
        "antisymmetry_offset": float(offset.mean()),
        "antisymmetry_t": float(offset_test.statistic),
        "antisymmetry_p": float(offset_test.pvalue),
        "speed3_minus2_cm_s": speed_mean,
        "speed3_minus2_t": speed_t,
        "speed3_minus2_p": speed_p,
        "duration3_minus2_s": duration_mean,
        "duration3_minus2_t": duration_t,
        "duration3_minus2_p": duration_p,
        "field_wall_distance3_minus2_cm": wall_mean,
        "field_wall_distance3_minus2_t": wall_t,
        "field_wall_distance3_minus2_p": wall_p,
    }


def visit_sequence_summary(paired: pd.DataFrame) -> Dict[str, object]:
    """One-sample t-tests of Fig 3F on the paired cells.

    Parameters
    ----------
    paired : pandas.DataFrame
        ``paired_cell_values`` output (the order of the rows sets the order of summation).

    Returns
    -------
    dict
        The columns of ``rat/visits/summary.csv``: ``n_paired_cells``; mean, t, p and d of the
        ``T23<T12`` means (``*_lt``); mean, t and p of the ``T23>T12`` means (``*_gt``); mean
        (``paired_diff``), t, p and d of the paired difference; ``pct_positive`` (percent of
        cells with a positive difference). d is the mean over the SD (ddof = 1).
    """
    diff = paired["paired_difference"].to_numpy(float)
    lt = paired["frr_T23_lt_T12"].to_numpy(float)
    gt = paired["frr_T23_gt_T12"].to_numpy(float)
    t_lt, t_gt, t_diff = stats.ttest_1samp(lt, 0), stats.ttest_1samp(gt, 0), stats.ttest_1samp(diff, 0)
    return dict(
        n_paired_cells=int(len(paired)),
        mean_lt=float(lt.mean()), t_lt=float(t_lt.statistic), p_lt=float(t_lt.pvalue),
        d_lt=float(lt.mean() / lt.std(ddof=1)),
        mean_gt=float(gt.mean()), t_gt=float(t_gt.statistic), p_gt=float(t_gt.pvalue),
        paired_diff=float(diff.mean()), paired_t=float(t_diff.statistic), paired_p=float(t_diff.pvalue),
        paired_d=float(diff.mean() / diff.std(ddof=1)), pct_positive=float((diff > 0).mean() * 100),
    )


def fig3f_statistics(summary: Dict[str, object]) -> Dict[str, object]:
    """Statistic keys of Fig 3F (docs/PIPELINE.md, section 4) from ``visit_sequence_summary``."""
    n = int(summary["n_paired_cells"])
    return {
        "fig3f.n_cells": n,
        "fig3f.shorter.mean": summary["mean_lt"], "fig3f.shorter.t": summary["t_lt"],
        "fig3f.shorter.df": n - 1, "fig3f.shorter.p": summary["p_lt"],
        "fig3f.longer.mean": summary["mean_gt"], "fig3f.longer.t": summary["t_gt"],
        "fig3f.longer.df": n - 1, "fig3f.longer.p": summary["p_gt"],
        "fig3f.paired.mean_diff": summary["paired_diff"], "fig3f.paired.t": summary["paired_t"],
        "fig3f.paired.df": n - 1, "fig3f.paired.p": summary["paired_p"],
    }


def nesting_value(nesting: pd.DataFrame, cohort: str, level: str, column: str) -> Optional[float]:
    """Return one value of the ``nesting_sensitivity`` table (None when the row is absent)."""
    sel = nesting[(nesting["cohort"] == cohort) & (nesting["level"] == level)]
    return None if sel.empty else sel[column].iloc[0]
