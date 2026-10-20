"""Per-cell figure tables of the rat figures and the statistics that step 12 adds to those of
``hexmod.rat.paper_stats``.

Figure tables (written by step 12 under ``rat/figure_tables/``):

- ``cells_thirds``: thirds of the adaptation coefficient (Fig 5), of the proximity difference
  between aligned and misaligned running (Fig 4D, Fig 5) and of grid scale (Fig 4D), with the
  rule of ``paper_stats.thirds`` (percentile edges, top edge raised by 0.001, ``numpy.digitize``;
  third 1 holds the lowest values, for the adaptation coefficient the strongest adaptation).
- ``supp3bcd_arrays`` and ``gamma_fits``: deviations of field centres from a perfect lattice and
  field rates of small-scale cells (scale at most 0.4 of the environment size) and large-scale
  cells (scale at least 0.6 of it), both with gridness above 0.3, and their gamma fits with the
  location fixed at 0 (Supp Fig 3B to 3D).
- ``supp5_widths``: circular SD of the direction tuning of the conjunctive cells (Supp Fig 5B).

Statistics: ``posthoc_statistics`` gives the pairwise comparisons between the groups of Fig 3C,
4D, 5B and Supp Fig 2 (paired t-tests between t_prev bins, Student t-tests between thirds),
Holm-corrected within each panel; ``supp3_statistics`` and ``supp5_statistics`` give the Supp Fig 3
and Supp Fig 5 keys. The printed keys are listed in docs/STATISTICS.md.
"""

from __future__ import annotations

import itertools
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from .. import circular, stats
from .paper_stats import CONJUNCTIVE_MIN_RAYLEIGH, TPREV_BIN_LABELS, thirds

__all__ = [
    "SUPP3_GROUPS", "SUPP3_DISTRIBUTIONS", "cells_thirds", "supp3bcd_arrays", "gamma_fits", "supp5_widths",
    "paired_posthoc", "independent_posthoc", "posthoc_statistics", "supp3_statistics", "supp5_statistics",
]

#: Supp Fig 3 groups: name, relation of grid scale to the environment size, factor.
SUPP3_GROUPS = (("small", "at_most", 0.4), ("large", "at_least", 0.6))

#: Supp Fig 3 distributions: name in the results, array prefix in ``supp3bcd_arrays``.
SUPP3_DISTRIBUTIONS = (("deviation", "deviation"), ("rate_across_cells", "mean_rate"),
                       ("rate_within_cell", "rate_normalised"))


# ------------------------------------------------------------------------------------- thirds
def _check_rows(*tables) -> None:
    rows = [np.asarray(t["row"]) for t in tables]
    for r in rows[1:]:
        if len(r) != len(rows[0]) or not np.array_equal(r, rows[0]):
            raise ValueError("per-cell tables are not row-aligned with the cohort table")


def cells_thirds(cells, param, x_edges: Sequence[np.ndarray], rate_maps: Sequence[np.ndarray]):
    """Thirds of adaptation, proximity difference and grid scale for every cell (``paper_stats.thirds``).

    Parameters
    ----------
    cells : pandas.DataFrame
        ``rat/cohort/cells.csv``.
    param : pandas.DataFrame
        ``rat/adaptation/param_cells.csv`` (same rows).
    x_edges, rate_maps : sequences of arrays
        Rate-map bin edges along x and rate maps of the cells (``rat/cohort/cells_full.npz``).

    Returns
    -------
    table : pandas.DataFrame
        ``row``, ``alpha`` (``alpha_tprevXloc_all``), ``proximity_diff`` (``align_tprev_infield``
        minus ``misalign_tprev_infield``), ``beta`` (``beta_all``), ``adaptation_third``,
        ``proximity_third``, ``scale_third``, ``env_size_cm`` (last x edge of the rate map) and
        ``rate_map_max_hz``.
    edges : pandas.DataFrame
        ``quantity`` (``alpha``, ``proximity_diff``, ``scale_cm``) and ``edge_0`` to ``edge_3``.
    """
    import pandas as pd

    _check_rows(cells, param)
    alpha = param["alpha_tprevXloc_all"].to_numpy()
    proximity_diff = (param["align_tprev_infield"] - param["misalign_tprev_infield"]).to_numpy()
    beta = param["beta_all"].to_numpy()
    b_alpha, cat_alpha = thirds(alpha)
    b_prox, cat_prox = thirds(proximity_diff)
    b_scale, cat_scale = thirds(cells["scale_cm"].to_numpy())
    table = pd.DataFrame({
        "row": cells["row"].to_numpy(dtype=np.int64), "alpha": alpha, "proximity_diff": proximity_diff, "beta": beta,
        "adaptation_third": cat_alpha.astype(np.int64), "proximity_third": cat_prox.astype(np.int64),
        "scale_third": cat_scale.astype(np.int64),
        "env_size_cm": np.array([xe[-1] for xe in x_edges], float),
        "rate_map_max_hz": np.array([float(np.nanmax(r)) for r in rate_maps], float),
    })
    edges = pd.DataFrame({"quantity": ["alpha", "proximity_diff", "scale_cm"]})
    for i in range(4):
        edges[f"edge_{i}"] = [b_alpha[i], b_prox[i], b_scale[i]]
    return table, edges


# ------------------------------------------------------------------------------------ Supp Fig 3
def supp3bcd_arrays(cells, x_edges: Sequence[np.ndarray], field_deviation: Sequence[np.ndarray],
                    field_rate_hz: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Field deviations and field rates of the small-scale and large-scale cells (Supp Fig 3B to 3D).

    Small-scale cells have a grid scale of at most 0.4 times the environment size, large-scale
    cells at least 0.6 times it (environment size: last x edge of the rate map); both have
    gridness above 0.3.

    Returns
    -------
    dict
        For ``g`` in ``small``, ``large``: ``deviation_g`` (every field's deviation from the
        fitted lattice, cells concatenated in row order), ``rate_normalised_g`` (every field's
        rate divided by the mean field rate of its cell), ``mean_rate_g`` (mean field rate of each
        cell, Hz) and ``rows_g`` (cohort rows of the cells).
    """
    env = np.array([xe[-1] for xe in x_edges], float)
    scale = cells["scale_cm"].to_numpy()
    gridness = cells["gridness"].to_numpy()
    out: Dict[str, np.ndarray] = {}
    for name, relation, factor in SUPP3_GROUPS:
        if relation == "at_most":
            mask = (scale <= env * factor) & (gridness > 0.3)
        else:
            mask = (scale >= env * factor) & (gridness > 0.3)
        idx = np.flatnonzero(mask)
        mean_rate = np.array([np.asarray(field_rate_hz[i]).mean() for i in idx], float)
        out[f"deviation_{name}"] = (np.concatenate([np.asarray(field_deviation[i]) for i in idx])
                                    if len(idx) else np.zeros(0))
        out[f"rate_normalised_{name}"] = (np.concatenate([np.asarray(field_rate_hz[i]) / mean_rate[k]
                                                          for k, i in enumerate(idx)]) if len(idx) else np.zeros(0))
        out[f"mean_rate_{name}"] = mean_rate
        out[f"rows_{name}"] = cells["row"].to_numpy(dtype=np.int64)[idx]
    return out


def gamma_fits(arrays: Mapping[str, np.ndarray]):
    """Gamma fits (location fixed at 0) of the Supp Fig 3B to 3D distributions.

    Parameters
    ----------
    arrays : mapping
        Output of ``supp3bcd_arrays``.

    Returns
    -------
    pandas.DataFrame
        ``module`` (``small``, ``large``), ``distribution`` (``deviation``, ``rate_across_cells``,
        ``rate_within_cell``), ``shape``, ``scale`` and ``n_values``. Distributions with fewer
        than two values get NaN parameters.
    """
    import pandas as pd

    rows = []
    for name, _, _ in SUPP3_GROUPS:
        for dist, prefix in SUPP3_DISTRIBUTIONS:
            x = np.asarray(arrays[f"{prefix}_{name}"], float)
            if x.size >= 2:
                shape, _, scale = sps.gamma.fit(x, floc=0)
            else:
                shape, scale = np.nan, np.nan
            rows.append({"module": name, "distribution": dist, "shape": float(shape), "scale": float(scale),
                         "n_values": int(x.size)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ Supp Fig 5
def supp5_widths(cells):
    """Tuning widths of the conjunctive cells (Rayleigh vector length above 0.3).

    Returns
    -------
    pandas.DataFrame
        ``row``, ``Rayleigh_mag`` and ``circular_sd_deg`` (degrees(sqrt(-2 ln Rayleigh_mag))).
    """
    import pandas as pd

    conj = cells[cells["Rayleigh_mag"] > CONJUNCTIVE_MIN_RAYLEIGH]
    mag = conj["Rayleigh_mag"].to_numpy()
    return pd.DataFrame({"row": conj["row"].to_numpy(dtype=np.int64), "Rayleigh_mag": mag,
                         "circular_sd_deg": circular.circular_sd_deg(mag)})


# ---------------------------------------------------------------------------------- statistics
def _holm_keys(prefix: str, names: Sequence[str], results: Sequence[dict], pairs) -> Dict[str, float]:
    out: Dict[str, float] = {}
    adj = stats.holm([r["p"] for r in results])
    for (i, j), r, h in zip(pairs, results, adj):
        base = f"{prefix}.posthoc.{names[i]}_vs_{names[j]}"
        out[base + ".t"] = r["t"]
        out[base + ".df"] = r["df"]
        out[base + ".p"] = r["p"]
        out[base + ".p_holm"] = float(h)
    return out


def paired_posthoc(prefix: str, X) -> Dict[str, float]:
    """Paired t-tests between every pair of t_prev bins, Holm-corrected within the panel.

    Rows with a missing value are left out (``stats.rm_pairwise``).

    Returns
    -------
    dict
        ``<prefix>.posthoc.<a>_vs_<b>.t``, ``.df``, ``.p``, ``.p_holm`` with bin labels
        ``0_1s``, ``1_2s``, ``2_3s``, ``above_3s``.
    """
    pw = stats.rm_pairwise(X, TPREV_BIN_LABELS)
    pairs = list(itertools.combinations(range(len(TPREV_BIN_LABELS)), 2))
    results = [pw[f"{TPREV_BIN_LABELS[i]} vs {TPREV_BIN_LABELS[j]}"] for i, j in pairs]
    return _holm_keys(prefix, TPREV_BIN_LABELS, results, pairs)


def independent_posthoc(prefix: str, groups: Sequence[np.ndarray], names: Sequence[str]) -> Dict[str, float]:
    """Student t-tests between every pair of independent groups, Holm-corrected within the panel.

    Returns
    -------
    dict
        ``<prefix>.posthoc.<a>_vs_<b>.t``, ``.df``, ``.p``, ``.p_holm``.
    """
    pairs = list(itertools.combinations(range(len(groups)), 2))
    results = [stats.two_sample_t(groups[i], groups[j]) for i, j in pairs]
    return _holm_keys(prefix, names, results, pairs)


def posthoc_statistics(fr_percent_change, proximity_diff, scale_cm, alpha, beta,
                       fr_percent_change_shifted) -> Dict[str, float]:
    """Pairwise comparisons between the groups of Fig 3C, Fig 4D, Fig 5B and Supp Fig 2.

    Parameters
    ----------
    fr_percent_change : (n_cells, 4) ndarray
        ``rat/adaptation/fr_percent_change.npy`` (Fig 3C: the four t_prev bins).
    proximity_diff, scale_cm : array_like
        Proximity difference and grid scale of every cell (Fig 4D: thirds of grid scale).
    alpha, beta : array_like
        ``alpha_tprevXloc_all`` and ``beta_all`` (Fig 5B: adaptation thirds within the top third
        of the proximity difference).
    fr_percent_change_shifted : (n_kept, 4) ndarray
        ``rat/prospective/fr_percent_change_shifted.npy`` (Supp Fig 2).

    Returns
    -------
    dict
        ``fig3c.posthoc.<a>_vs_<b>.p_holm`` (the t, df and p of these pairs are written by
        ``paper_stats.fig3c_statistics``), ``fig4d.posthoc.<a>_vs_<b>.*`` (small, medium,
        large), ``fig5b.posthoc.<a>_vs_<b>.*`` (strong, weak, no) and
        ``supp2.posthoc.<a>_vs_<b>.*``, each with ``t``, ``df``, ``p`` and ``p_holm``.
    """
    out = {k: v for k, v in paired_posthoc("fig3c", fr_percent_change).items() if k.endswith(".p_holm")}
    pdiff = np.asarray(proximity_diff, float)
    _, scale_third = thirds(np.asarray(scale_cm, float))
    groups = [pdiff[(scale_third == k) & ~np.isnan(pdiff)] for k in (1, 2, 3)]
    out.update(independent_posthoc("fig4d", groups, ("small", "medium", "large")))
    beta = np.asarray(beta, float)
    _, a_third = thirds(np.asarray(alpha, float))
    _, p_third = thirds(pdiff)
    top = [beta[(p_third == 3) & (a_third == k)] for k in (1, 2, 3)]
    out.update(independent_posthoc("fig5b", top, ("strong", "weak", "no")))
    out.update(paired_posthoc("supp2", fr_percent_change_shifted))
    return out


def supp3_statistics(arrays: Mapping[str, np.ndarray], fits) -> Dict[str, float]:
    """Supp Fig 3B to 3D: numbers of cells and fields, and the gamma fit parameters."""
    out: Dict[str, float] = {}
    for name, _, _ in SUPP3_GROUPS:
        out[f"supp3.{name}.n_cells"] = int(len(arrays[f"rows_{name}"]))
        out[f"supp3.{name}.n_fields"] = int(len(arrays[f"deviation_{name}"]))
    for rec in fits.to_dict("records"):
        if np.isfinite(rec["shape"]):
            out[f"supp3.{rec['module']}.{rec['distribution']}.shape"] = float(rec["shape"])
            out[f"supp3.{rec['module']}.{rec['distribution']}.scale"] = float(rec["scale"])
    return out


def supp5_statistics(widths) -> Dict[str, float]:
    """Supp Fig 5: narrowest and median tuning width of the conjunctive cells and the six-fold
    component (percent of the mean rate) of a von Mises tuning curve of that width."""
    sd = widths["circular_sd_deg"].to_numpy(float)
    if not len(sd):
        return {"supp5.n_cells": 0}
    narrowest = float(sd.min())
    median = float(np.median(sd))
    return {"supp5.n_cells": int(len(sd)), "supp5.narrowest_sd_deg": narrowest,
            "supp5.narrowest_sixfold_pct": 100 * circular.sixfold_from_sd(narrowest),
            "supp5.median_sd_deg": median, "supp5.median_sixfold_pct": 100 * circular.sixfold_from_sd(median)}
