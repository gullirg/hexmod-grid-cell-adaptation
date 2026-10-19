"""Printed statistics of the cohort, Table 1, Fig 3C, 3D, 4C, 4D, 5A to 5C and Supp Figs 1 and 2.

The functions take the per-cell results of steps 02 to 05 (docs/PIPELINE.md) and return flat
dicts that map statistic keys (PIPELINE.md, section 4) to numbers. The tests themselves are those
of :mod:`hexmod.stats`.

Grouping rules:

- **Thirds** (Fig 4D, Fig 5B and 5C): edges are the 0th, 33.3rd, 66.7th and 100th percentiles
  of the values (``numpy.percentile``, linear interpolation), the top edge is raised by 0.001 so
  that the largest value falls inside, and ``numpy.digitize`` assigns each cell to third 1, 2
  or 3 (:func:`thirds`).
- **Fig 4D**: thirds of the grid scale (1 small, 2 medium, 3 large); the quantity compared is
  the proximity difference ``align_tprev_infield - misalign_tprev_infield``.
- **Fig 5**: thirds of ``alpha_tprevXloc_all`` (1 strong adaptation, the most negative alpha;
  2 weak; 3 no adaptation) crossed with thirds of the proximity difference (1 bottom, 3 top);
  the quantity tested is the hexadirectional modulation ``beta_all``.
- **Conjunctive cells**: Rayleigh vector length of the movement-direction tuning above 0.3.

Per-cell values are used as written by the steps (read with :func:`hexmod.io.read_csv`, which
returns every float exactly). :func:`alpha_bounds` also gives the limits within which the
parametric fit keeps alpha (see :mod:`hexmod.rat.adaptation`), so that cells whose alpha sits on
a limit can be counted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import stats as _sps

from .. import stats as _st
from .. import datasets as _ds
from .adaptation import ADAPTATION_TAU_S, time_masks
from .fields import extract_field_entry, field_visit_input
from .gridcell import GridCellAnalysis

__all__ = [
    "CONJUNCTIVE_MIN_RAYLEIGH", "TABLE1_ORDER", "TPREV_BIN_LABELS", "THIRDS_TOP_MARGIN",
    "thirds", "conjunctive_mask", "cohort_statistics", "table1",
    "fig3c_statistics", "fig3d_statistics", "fig4c_statistics", "fig4d_statistics", "fig5_statistics",
    "supp1_statistics", "supp2_statistics", "example_shift_index", "core_statistics",
    "core_statistics_from_results", "alpha_bounds", "alpha_bound_counts",
]

#: Rayleigh vector length above which a cell counts as conjunctive (direction-tuned).
CONJUNCTIVE_MIN_RAYLEIGH = 0.3

#: Row order of Table 1.
TABLE1_ORDER = ["brun2008", "barry2012", "olafsdottir2016", "hafting2005", "sargolini2006", "vollan2025"]

#: Labels of the four t_prev bins (0 to 1, 1 to 2, 2 to 3 and above 3 s) used in statistic keys.
TPREV_BIN_LABELS = ("0_1s", "1_2s", "2_3s", "above_3s")

#: Amount added to the top percentile edge of the thirds, so that the largest value is included.
THIRDS_TOP_MARGIN = 0.001

#: Window (ms, exclusive) of the best time shift of the Supp Fig 2A example cell.
EXAMPLE_SHIFT_WINDOW_MS = (100, 150)


# ------------------------------------------------------------------------------------- helpers
def thirds(x):
    """Percentile thirds of ``x`` and the third of each value.

    Parameters
    ----------
    x : array_like
        Values, one per cell.

    Returns
    -------
    edges : ndarray, shape (4,)
        Percentiles 0, 100/3, 200/3 and 100 of ``x``, the last one raised by 0.001.
    category : ndarray of int
        ``numpy.digitize(x, edges)``: 1, 2 or 3.
    """
    b = np.percentile(x, np.linspace(0, 100, 3 + 1))
    b[-1] = b[-1] + THIRDS_TOP_MARGIN
    return b, np.digitize(x, b)


def conjunctive_mask(rayleigh_mag) -> np.ndarray:
    """Boolean mask of conjunctive cells (Rayleigh vector length above 0.3)."""
    return np.asarray(rayleigh_mag, float) > CONJUNCTIVE_MIN_RAYLEIGH


def _counts(df) -> Dict[str, int]:
    return {"n_animals": int(df["animal_id"].nunique()), "n_sessions": int(df["session_id"].nunique()),
            "n_cells": int(len(df))}


def _datasets_present(cells, dataset_keys: Optional[Iterable[str]]) -> List[str]:
    if dataset_keys is None:
        present = set(cells["dataset"])
        return [k for k in _ds.CANONICAL_ORDER if k in present]
    return list(dataset_keys)


# ------------------------------------------------------------------------------ cohort, Table 1
def cohort_statistics(cells, dataset_keys: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Cohort counts and the measured grid-scale range of each dataset (step 02 keys).

    Parameters
    ----------
    cells : pandas.DataFrame
        ``rat/cohort/cells.csv``.
    dataset_keys : iterable of str, optional
        Datasets to report (default: those present in ``cells``, in canonical order). A dataset
        without cells gets zero counts and no scale range.

    Returns
    -------
    dict
        ``cohort.n_animals``, ``cohort.n_sessions``, ``cohort.n_cells``;
        ``cohort.<dataset>.n_animals``, ``.n_sessions``, ``.n_cells``;
        ``cohort.conjunctive.n_cells``, ``.n_animals``, ``.n_sessions``;
        ``table1.<dataset>.scale_min_cm``, ``table1.<dataset>.scale_max_cm``.
    """
    out: Dict[str, float] = {}
    for name, value in _counts(cells).items():
        out[f"cohort.{name}"] = value
    for key in _datasets_present(cells, dataset_keys):
        sub = cells[cells["dataset"] == key]
        for name, value in _counts(sub).items():
            out[f"cohort.{key}.{name}"] = value
        if len(sub):
            scale = sub["scale_cm"].to_numpy(float)
            out[f"table1.{key}.scale_min_cm"] = float(scale.min())
            out[f"table1.{key}.scale_max_cm"] = float(scale.max())
    conj = cells[conjunctive_mask(cells["Rayleigh_mag"])]
    for name, value in _counts(conj).items():
        out[f"cohort.conjunctive.{name}"] = value
    return out


def table1(cells, text=None, dataset_keys: Optional[Iterable[str]] = None):
    """Table 1: animals, sessions, cells, conjunctive cells and measured scale range per dataset.

    Parameters
    ----------
    cells : pandas.DataFrame
        ``rat/cohort/cells.csv``.
    text : pandas.DataFrame, optional
        ``inputs/rat/table1_text.csv`` (columns ``dataset``, ``environment``,
        ``grid_scale_text``), the descriptive columns of the printed table.
    dataset_keys : iterable of str, optional
        Datasets analysed (default: those present in ``cells``).

    Returns
    -------
    pandas.DataFrame
        One row per dataset in the order of Table 1 and a final row ``total``: ``dataset``,
        ``label``, ``n_animals``, ``n_sessions``, ``n_cells``, ``n_conjunctive``,
        ``scale_min_cm``, ``scale_max_cm``, ``environment``, ``grid_scale_text``.
    """
    import pandas as pd

    keys = set(_datasets_present(cells, dataset_keys))
    lookup = {}
    if text is not None:
        lookup = {r["dataset"]: r for r in text.to_dict("records")}
    conj = conjunctive_mask(cells["Rayleigh_mag"])

    def row(name, label, sub, csub, env="", scale_text=""):
        scale = sub["scale_cm"].to_numpy(float)
        return {"dataset": name, "label": label, **_counts(sub), "n_conjunctive": int(len(csub)),
                "scale_min_cm": float(scale.min()) if len(scale) else float("nan"),
                "scale_max_cm": float(scale.max()) if len(scale) else float("nan"),
                "environment": env, "grid_scale_text": scale_text}

    rows = []
    for key in TABLE1_ORDER:
        if key not in keys:
            continue
        m = (cells["dataset"] == key).to_numpy()
        t = lookup.get(key, {})
        rows.append(row(key, _ds.label_of(key), cells[m], cells[m & conj], t.get("environment", ""),
                        t.get("grid_scale_text", "")))
    rows.append(row("total", "Total", cells, cells[conj]))
    cols = ["dataset", "label", "n_animals", "n_sessions", "n_cells", "n_conjunctive", "scale_min_cm",
            "scale_max_cm", "environment", "grid_scale_text"]
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------------------------- Fig 3, 4
def fig3c_statistics(fr_percent_change) -> Dict[str, float]:
    """Fig 3C: firing rate in the four t_prev bins (percent change from the mean rate).

    Cells with a missing value in any bin are left out. The repeated-measures ANOVA and the six
    paired post-hoc t-tests are those of :func:`hexmod.stats.rm_anova` and ``rm_pairwise``.

    Parameters
    ----------
    fr_percent_change : array_like, shape (n_cells, 4)
        ``rat/adaptation/fr_percent_change.npy``.

    Returns
    -------
    dict
        ``fig3c.mean_pct_change.<bin>``, ``fig3c.rm_anova.F``, ``.df1``, ``.df2``, ``.p`` and
        ``fig3c.posthoc.<a>_vs_<b>.t``, ``.df``, ``.p``.
    """
    X = np.asarray(fr_percent_change, float)
    a = _st.rm_anova(X)
    out: Dict[str, float] = {}
    for lab, m in zip(TPREV_BIN_LABELS, a["means"]):
        out[f"fig3c.mean_pct_change.{lab}"] = m
    for k in ("F", "df1", "df2", "p"):
        out[f"fig3c.rm_anova.{k}"] = a[k]
    for pair, r in _st.rm_pairwise(X, TPREV_BIN_LABELS).items():
        name = pair.replace(" vs ", "_vs_")
        for k in ("t", "df", "p"):
            out[f"fig3c.posthoc.{name}.{k}"] = r[k]
    return out


def fig3d_statistics(alpha) -> Dict[str, float]:
    """Fig 3D: one-sample t-test of the adaptation coefficient against 0.

    Parameters
    ----------
    alpha : array_like
        ``alpha_tprevXloc_all`` of every cell.

    Returns
    -------
    dict
        ``fig3d.n_cells``, ``.t``, ``.df``, ``.p``, ``.d`` (mean over SD, ddof 1) and
        ``.pct_negative`` (percentage of cells with alpha below 0).
    """
    alpha = np.asarray(alpha, float)
    r = _st.one_sample_t(alpha)
    return {"fig3d.n_cells": r["n"], "fig3d.t": r["t"], "fig3d.df": r["df"], "fig3d.p": r["p"],
            "fig3d.d": r["d"], "fig3d.pct_negative": float(np.mean(alpha < 0) * 100)}


def fig4c_statistics(align, misalign) -> Dict[str, float]:
    """Fig 4C: in-field proximity when running along a grid axis against away from the axes.

    Parameters
    ----------
    align, misalign : array_like
        ``align_tprev_infield`` and ``misalign_tprev_infield`` of every cell.

    Returns
    -------
    dict
        ``fig4c.mean_aligned``, ``.mean_misaligned``, ``.pct_higher`` (mean difference as a
        percentage of the misaligned mean), and the paired t-test ``.t``, ``.df``, ``.p``.
    """
    r = _st.paired_t(align, misalign)
    return {"fig4c.mean_aligned": r["mean_a"], "fig4c.mean_misaligned": r["mean_b"],
            "fig4c.pct_higher": float(r["mean_diff"] / r["mean_b"] * 100),
            "fig4c.t": r["t"], "fig4c.df": r["df"], "fig4c.p": r["p"]}


def fig4d_statistics(proximity_diff, scale_cm) -> Dict[str, float]:
    """Fig 4D: proximity difference in the three grid-scale thirds, one-way ANOVA.

    Parameters
    ----------
    proximity_diff : array_like
        ``align_tprev_infield - misalign_tprev_infield`` of every cell.
    scale_cm : array_like
        Grid scale of every cell (cm).

    Returns
    -------
    dict
        ``fig4d.scale_edges_cm.0`` to ``.3``, ``fig4d.n.small``, ``.medium``, ``.large`` and
        ``fig4d.anova.F``, ``.df1``, ``.df2``, ``.p``.
    """
    pdiff = np.asarray(proximity_diff, float)
    edges, cat = thirds(np.asarray(scale_cm, float))
    groups = [pdiff[(cat == k) & ~np.isnan(pdiff)] for k in (1, 2, 3)]
    a = _st.one_way_anova(*groups)
    out: Dict[str, float] = {f"fig4d.scale_edges_cm.{i}": float(e) for i, e in enumerate(edges)}
    for name, g in zip(("small", "medium", "large"), groups):
        out[f"fig4d.n.{name}"] = int(len(g))
    for k in ("F", "df1", "df2", "p"):
        out[f"fig4d.anova.{k}"] = a[k]
    return out


def fig5_statistics(alpha, proximity_diff, beta) -> Dict[str, float]:
    """Fig 5A to 5C: hexadirectional modulation overall and by adaptation and proximity thirds.

    Parameters
    ----------
    alpha : array_like
        ``alpha_tprevXloc_all`` (thirds: 1 strong, 2 weak, 3 no adaptation).
    proximity_diff : array_like
        ``align_tprev_infield - misalign_tprev_infield`` (thirds: 1 bottom to 3 top).
    beta : array_like
        ``beta_all``.

    Returns
    -------
    dict
        ``fig5.alpha_edges.0`` to ``.3``, ``fig5.proximity_edges.0`` to ``.3``;
        ``fig5a.t``, ``.df``, ``.p`` (all cells); for the top proximity third
        ``fig5b.strong.n``, ``.t``, ``.df``, ``.p``, ``.d``, ``fig5b.weak.*`` and ``fig5b.no.*``
        (``n``, ``t``, ``df``, ``p``) and the one-way ANOVA ``fig5b.anova.F``, ``.df1``,
        ``.df2``, ``.p``; ``fig5c.strong_bottom.n``, ``.t``, ``.df``, ``.p`` (strong adaptation,
        bottom proximity third) and the Student t-test of the strong top against the strong
        bottom group ``fig5c.strong_top_vs_bottom.t``, ``.df``, ``.p``.
    """
    alpha = np.asarray(alpha, float)
    pdiff = np.asarray(proximity_diff, float)
    beta = np.asarray(beta, float)
    ab, ac = thirds(alpha)
    pb, pc = thirds(pdiff)
    out: Dict[str, float] = {}
    for i, e in enumerate(ab):
        out[f"fig5.alpha_edges.{i}"] = float(e)
    for i, e in enumerate(pb):
        out[f"fig5.proximity_edges.{i}"] = float(e)
    r = _st.one_sample_t(beta)
    out.update({"fig5a.t": r["t"], "fig5a.df": r["df"], "fig5a.p": r["p"]})
    top = [beta[(pc == 3) & (ac == k)] for k in (1, 2, 3)]
    for name, g in zip(("strong", "weak", "no"), top):
        r = _st.one_sample_t(g)
        for k in ("n", "t", "df", "p"):
            out[f"fig5b.{name}.{k}"] = r[k]
        if name == "strong":
            out["fig5b.strong.d"] = r["d"]
    a = _st.one_way_anova(*top)
    for k in ("F", "df1", "df2", "p"):
        out[f"fig5b.anova.{k}"] = a[k]
    bottom_strong = beta[(pc == 1) & (ac == 1)]
    r = _st.one_sample_t(bottom_strong)
    for k in ("n", "t", "df", "p"):
        out[f"fig5c.strong_bottom.{k}"] = r[k]
    r = _st.two_sample_t(top[0], bottom_strong)
    for k in ("t", "df", "p"):
        out[f"fig5c.strong_top_vs_bottom.{k}"] = r[k]
    return out


# ---------------------------------------------------------------------------- Supp Figs 1 and 2
def supp1_statistics(alpha_odd, alpha_even) -> Dict[str, float]:
    """Supp Fig 1: adaptation coefficients fitted on odd and on even minutes of each session.

    Cells with a missing value in either half are left out.

    Returns
    -------
    dict
        ``supp1.odd.t``, ``.df``, ``.p`` and ``supp1.even.t``, ``.df``, ``.p`` (one-sample t-tests
        against 0), ``supp1.odd_vs_even.t``, ``.df``, ``.p`` (paired t-test), ``supp1.pearson_r``,
        ``supp1.pearson_df`` (n - 2) and ``supp1.pearson_p`` (two-sided).
    """
    a = np.asarray(alpha_odd, float)
    b = np.asarray(alpha_even, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    pe = _st.paired_t(a, b)
    odd, even = _st.one_sample_t(a[ok]), _st.one_sample_t(b[ok])
    r, p_r = _sps.pearsonr(a[ok], b[ok])
    return {"supp1.odd.t": odd["t"], "supp1.odd.df": odd["df"], "supp1.odd.p": odd["p"],
            "supp1.even.t": even["t"], "supp1.even.df": even["df"], "supp1.even.p": even["p"],
            "supp1.odd_vs_even.t": pe["t"], "supp1.odd_vs_even.df": pe["df"], "supp1.odd_vs_even.p": pe["p"],
            "supp1.pearson_r": float(r), "supp1.pearson_df": int(ok.sum() - 2), "supp1.pearson_p": float(p_r)}


def example_shift_index(peak_shift_ms) -> Optional[int]:
    """Row of the Supp Fig 2A example: the first cell whose best shift lies between 100 and 150 ms.

    Returns None when no cell qualifies.
    """
    peak = np.asarray(peak_shift_ms)
    lo, hi = EXAMPLE_SHIFT_WINDOW_MS
    idx = np.where((peak > lo) & (np.abs(peak) < hi))[0]
    return int(idx[0]) if len(idx) else None


def supp2_statistics(peak_shift_ms, alpha_original, alpha_shifted, fr_percent_change_shifted) -> Dict[str, float]:
    """Supp Fig 2: best time shifts, and adaptation after re-analysis with shifted spikes.

    Parameters
    ----------
    peak_shift_ms : array_like
        Best shift of every cohort cell (ms), ``rat/prospective/peak_shift.csv``.
    alpha_original, alpha_shifted : array_like
        ``alpha_original`` and ``alpha_tprevXloc_all`` of ``rat/prospective/shift_cells.csv``
        (the cells that remain grid cells with shifted spikes).
    fr_percent_change_shifted : array_like, shape (n_kept, 4)
        ``rat/prospective/fr_percent_change_shifted.npy``.

    Returns
    -------
    dict
        ``supp2.n_cells``, ``supp2.n_cells_kept``, ``supp2.median_best_shift_ms``,
        ``supp2.example_best_shift_ms`` (best shift of the cell of :func:`example_shift_index`,
        when there is one), ``supp2.rm_anova.F``, ``.df1``, ``.df2``, ``.p``,
        ``supp2.alpha_original_vs_shifted.t``, ``.df``, ``.p`` (paired t-test) and the effect
        sizes ``supp2.alpha_original.d``, ``supp2.alpha_shifted.d``.
    """
    peak = np.asarray(peak_shift_ms)
    a_o = np.asarray(alpha_original, float)
    a_s = np.asarray(alpha_shifted, float)
    out: Dict[str, float] = {"supp2.n_cells": int(len(peak)), "supp2.n_cells_kept": int(len(a_o)),
                             "supp2.median_best_shift_ms": float(np.median(peak))}
    i = example_shift_index(peak)
    if i is not None:
        out["supp2.example_best_shift_ms"] = int(peak[i])
    a = _st.rm_anova(fr_percent_change_shifted)
    for k in ("F", "df1", "df2", "p"):
        out[f"supp2.rm_anova.{k}"] = a[k]
    pr = _st.paired_t(a_o, a_s)
    for k in ("t", "df", "p"):
        out[f"supp2.alpha_original_vs_shifted.{k}"] = pr[k]
    ok = ~np.isnan(a_o) & ~np.isnan(a_s)
    out["supp2.alpha_original.d"] = _st.one_sample_t(a_o[ok])["d"]
    out["supp2.alpha_shifted.d"] = _st.one_sample_t(a_s[ok])["d"]
    return out


# ----------------------------------------------------------------------------- all core values
def core_statistics(cells, param, fr_percent_change, peak_shift_ms, shift_cells,
                    fr_percent_change_shifted) -> Dict[str, float]:
    """Statistics of Fig 3C, 3D, 4C, 4D, 5A to 5C and Supp Figs 1 and 2 from the step outputs.

    Parameters
    ----------
    cells : pandas.DataFrame
        ``rat/cohort/cells.csv``.
    param : pandas.DataFrame
        ``rat/adaptation/param_cells.csv`` (same rows as ``cells``).
    fr_percent_change : array_like, shape (n_cells, 4)
        ``rat/adaptation/fr_percent_change.npy``.
    peak_shift_ms : array_like
        ``peak_shift_ms`` of ``rat/prospective/peak_shift.csv``.
    shift_cells : pandas.DataFrame
        ``rat/prospective/shift_cells.csv``.
    fr_percent_change_shifted : array_like
        ``rat/prospective/fr_percent_change_shifted.npy``.

    Returns
    -------
    dict
        The keys of the functions above, in one flat dict.
    """
    if not np.array_equal(cells["row"].to_numpy(), param["row"].to_numpy()):
        raise ValueError("param_cells.csv and cells.csv hold different rows")
    alpha = param["alpha_tprevXloc_all"].to_numpy(float)
    pa = param["align_tprev_infield"].to_numpy(float)
    pm = param["misalign_tprev_infield"].to_numpy(float)
    pdiff = pa - pm
    out: Dict[str, float] = {}
    out.update(fig3c_statistics(fr_percent_change))
    out.update(fig3d_statistics(alpha))
    out.update(fig4c_statistics(pa, pm))
    out.update(fig4d_statistics(pdiff, cells["scale_cm"].to_numpy(float)))
    out.update(fig5_statistics(alpha, pdiff, param["beta_all"].to_numpy(float)))
    out.update(supp1_statistics(param["alpha_tprevXloc_oddMinute"], param["alpha_tprevXloc_evenMinute"]))
    out.update(supp2_statistics(peak_shift_ms, shift_cells["alpha_original"], shift_cells["alpha_tprevXloc_all"],
                                fr_percent_change_shifted))
    return out


def core_statistics_from_results(results_dir) -> Dict[str, float]:
    """Read the outputs of steps 02 to 05 under ``results/`` and return :func:`core_statistics`."""
    from ..io import load_npy, read_csv

    r = Path(results_dir)
    cells = read_csv(r / "rat" / "cohort" / "cells.csv")
    param = read_csv(r / "rat" / "adaptation" / "param_cells.csv")
    peak = read_csv(r / "rat" / "prospective" / "peak_shift.csv")
    if not np.array_equal(peak["row"].to_numpy(), cells["row"].to_numpy()):
        raise ValueError("peak_shift.csv and cells.csv hold different rows")
    return core_statistics(cells, param, load_npy(r / "rat" / "adaptation" / "fr_percent_change.npy"),
                           peak["peak_shift_ms"].to_numpy(), read_csv(r / "rat" / "prospective" / "shift_cells.csv"),
                           load_npy(r / "rat" / "prospective" / "fr_percent_change_shifted.npy"))


# ------------------------------------------------------------------------------- alpha limits
def alpha_bounds(inputs):
    """Limits of the adaptation coefficient in the parametric fit of one cell.

    :func:`hexmod.rat.adaptation.adaptation_parametric` clips alpha to the interval that keeps
    ``1 + alpha a`` between 0 and 2 for every sample of the mask (``|alpha a| <= 1``), where ``a`` is the normalised
    adaptation regressor: ``[max(-1 / max(a), 1 / min(a)), min(-1 / min(a), 1 / max(a))]``. This
    function builds the same regressors with the same arithmetic and returns these limits, so
    that a fitted alpha can be compared with them (it equals a limit exactly when the limit was
    reached).

    Parameters
    ----------
    inputs : dict
        The inputs of ``adaptation_parametric``: ``cell_data``, ``fieldLoc``, ``peakFR`` and
        optional ``tau_adapt`` (2 s), ``mask_type``, ``adapt_type`` and ``normaliseType``
        (same defaults).

    Returns
    -------
    dict
        ``alpha_lower_<T>_<M>`` and ``alpha_upper_<T>_<M>`` for each adaptation type T and mask
        M, plus ``session_id`` and ``cell_id``.
    """
    cell_data = inputs["cell_data"]
    fieldLoc = inputs["fieldLoc"]
    peakFR = inputs["peakFR"]
    tau_adapt = inputs.get("tau_adapt", ADAPTATION_TAU_S)
    mask_type = inputs.get("mask_type", ["infield", "outfield", "all"])
    adapt_type = inputs.get("adapt_type", ["tprev", "tprevXloc", "tprevXspike"])
    normaliseType = inputs.get("normaliseType", "zscore")
    gc_obj = GridCellAnalysis(cell_data)
    x, y, t, spikes, speed = gc_obj.x, gc_obj.y, gc_obj.time, gc_obj.spike_array, gc_obj.speed

    tmp = extract_field_entry(x, y, t, fieldLoc[:, 0], fieldLoc[:, 1], fieldLoc[:, 2], spikes)
    ev_df, t_prev_vec, in_field = tmp['ev_df'], tmp['t_prev_vec'], tmp['in_field']
    tprevFactor_raw = np.exp(-t_prev_vec/tau_adapt)
    tmp2 = field_visit_input(x, y, t, ev_df, peakFR, speed)
    weights = {"tprevXloc": tmp2['adaptWeight_loc'], "tprevXlocXspeed": tmp2['adaptWeight_locXspeed'],
               "tprevXspike": tmp2['adaptWeight_spikes']}
    mask_time = time_masks(t, in_field)

    result = dict()
    for mask_name in mask_type:
        mask = mask_time[mask_name]
        for adaptName in adapt_type:
            if adaptName == "tprev":
                adaptFactor_raw = tprevFactor_raw
            else:
                w = weights[adaptName]
                adaptFactor_raw = tprevFactor_raw*w/np.max(w)
            if normaliseType == 'meanSubtract':
                z = adaptFactor_raw[mask]-adaptFactor_raw[mask].mean()
            elif normaliseType == 'zscore':
                z = (adaptFactor_raw[mask]-adaptFactor_raw[mask].mean())/adaptFactor_raw[mask].std()
            else:
                raise ValueError('Invalid normaliseType')
            result[f"alpha_lower_{adaptName}_{mask_name}"] = np.max([-1/z.max(), 1/z.min()])
            result[f"alpha_upper_{adaptName}_{mask_name}"] = np.min([-1/z.min(), 1/z.max()])
    result['session_id'] = cell_data['session_id']
    result['cell_id'] = cell_data['cell_id']
    return result


def alpha_bound_counts(param, bounds, columns: Sequence[str] = ("tprevXloc_all",)) -> Dict[str, int]:
    """Count the cells whose fitted alpha equals a fit limit.

    Parameters
    ----------
    param : pandas.DataFrame
        Table with ``alpha_<T>_<M>`` columns (``rat/adaptation/param_cells.csv``).
    bounds : pandas.DataFrame
        Same rows, with the ``alpha_lower_<T>_<M>`` and ``alpha_upper_<T>_<M>`` columns of
        :func:`alpha_bounds`.
    columns : sequence of str
        The ``<T>_<M>`` combinations to count.

    Returns
    -------
    dict
        For each combination C: ``diagnostic.alpha_at_bound.C.n_cells``, ``.n_lower`` (alpha
        equal to the lower limit), ``.n_upper`` and ``.n_total``.
    """
    out: Dict[str, int] = {}
    for c in columns:
        a = param[f"alpha_{c}"].to_numpy(float)
        lo = bounds[f"alpha_lower_{c}"].to_numpy(float)
        hi = bounds[f"alpha_upper_{c}"].to_numpy(float)
        n_lo, n_hi = int(np.sum(a == lo)), int(np.sum(a == hi))
        base = f"diagnostic.alpha_at_bound.{c}"
        out.update({f"{base}.n_cells": int(np.sum(~np.isnan(a))), f"{base}.n_lower": n_lo,
                    f"{base}.n_upper": n_hi, f"{base}.n_total": n_lo + n_hi})
    return out
