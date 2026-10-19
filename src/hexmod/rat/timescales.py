"""Timescale of firing-rate adaptation (Supp Fig 4) and the numbers quoted in the Methods.

Supp Fig 4 asks how quickly the firing rate recovers after a field visit. For every grid cell the
non-parametric model of :func:`hexmod.rat.adaptation.adaptation_nonparametric` (rate as a product
of location, speed and t_prev factors) is fitted with nine t_prev bins whose edges are spaced
logarithmically from 0.05 to 20.1 s (``exp(linspace(-3, 3, 10))``). Samples with t_prev outside
this range are left out (``mask_valid``). The t_prev factor of each bin is expressed as the percent
change from the cell's mean rate and averaged over cells. Two models are fitted to the nine bin
means, by least squares weighted by 1 / s.e.m.^2 of each bin mean (the printed fits) and
unweighted:

- one exponential, ``c + a exp(-t / tau)``;
- two exponentials, ``c + a_fast exp(-t / tau_fast) + a_slow exp(-t / tau_slow)``.

They are compared with AIC, AICc, BIC and the extra-sum-of-squares F test, and the uncertainty of
the time constants is estimated by resampling animals (all cells of each drawn animal) with
replacement.

The Methods numbers are the median interval between consecutive field visits in the small-scale
third of the cells, the mean running speed in the 100 cm arenas, and the ratio of the firing rate
in the first t_prev bin (0 to 1 s) to that in the last bin (above 3 s).

The fit settings (starting values, bounds, optimiser) are read from
``inputs/rat/timescale_fit_settings.json``. Several starting values may be listed per model; the
fit with the lowest residual sum of squares is kept (the first one listed on ties).

The model functions, fits and summaries need only numpy and scipy (the figure code can import
them); the per-cell functions import the rat analysis modules when they are called.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import optimize, stats

__all__ = [
    "FINE_TPREV_EDGES", "PUBLISHED_BOOTSTRAP_SEED", "N_BOOTSTRAP", "MAX_INTERVAL_S", "SPEED_RANGE_CM_S",
    "ARENA_100CM_RANGE",
    "bin_centres", "fine_bin_inputs", "fine_bin_cell", "percent_change", "bin_summary",
    "exp_single", "exp_double", "load_fit_settings", "fit_model", "fit_both", "model_comparison",
    "animal_bootstrap", "bootstrap_animals", "skip_bootstrap_draws",
    "interval_inputs", "field_visit_intervals", "interval_statistics",
    "session_speed", "arena_size_by_session", "running_speed_summary", "first_to_last_bin_ratio",
]

#: t_prev bin edges of Supp Fig 4 (s): nine bins, logarithmically spaced from 0.050 to 20.1 s.
FINE_TPREV_EDGES = np.exp(np.linspace(-3.0, 3.0, 10))

#: Seed and number of resamples of the animal bootstrap; the seed is the one used for the published
#: intervals. With 1000 resamples the upper ends of the 95% intervals vary by about 0.03 s (standard
#: deviation) from seed to seed.
PUBLISHED_BOOTSTRAP_SEED = 20260922
N_BOOTSTRAP = 1000

#: Field-visit intervals longer than this (s) are left out of the pooled interval statistics.
MAX_INTERVAL_S = 60.0

#: Running speeds (cm/s) counted as movement in the running-speed summary.
SPEED_RANGE_CM_S = (5.0, 150.0)

#: A session is in a 100 cm arena when the last rate-map bin edge (cm) lies strictly inside this range.
ARENA_100CM_RANGE = (85.0, 115.0)

_PARAMETERS = {"single": ("a", "tau", "c"), "double": ("a_fast", "tau_fast", "a_slow", "tau_slow", "c")}


# ------------------------------------------------------------------------------ fine-binned profile
def bin_centres(edges: np.ndarray = FINE_TPREV_EDGES) -> np.ndarray:
    """Arithmetic centres of consecutive bin edges (the abscissa of the fits, s)."""
    edges = np.asarray(edges, dtype=float)
    return (edges[1:] + edges[:-1]) / 2


def fine_bin_inputs(cell_data: dict, field_locations: np.ndarray) -> dict:
    """Inputs of :func:`hexmod.rat.adaptation.adaptation_nonparametric` for the nine-bin profile of one cell.

    Parameters
    ----------
    cell_data : dict
        As returned by :func:`hexmod.rat.gridcell.make_cell_data`.
    field_locations : (K, 3) ndarray
        Field centres and radii (cm).

    Returns
    -------
    dict
        8 speed bins, 5 cm location bins, mask ``all``, the nine t_prev bins and the valid-range
        mask.
    """
    return {"cell_data": cell_data, "fieldLoc": field_locations, "tprev_bins": FINE_TPREV_EDGES,
            "n_bins_speed": 8, "spatialResolution": 5, "mask_type": ["all"], "mask_valid": True}


def fine_bin_cell(inputs: dict) -> Dict[str, np.ndarray]:
    """Nine-bin non-parametric fit of one cell (worker function).

    Parameters
    ----------
    inputs : dict
        As returned by :func:`fine_bin_inputs`.

    Returns
    -------
    dict
        ``rate_hz`` (9,) t_prev factor of each bin, ``mean_rate_hz`` mean rate of the samples
        used, ``samples_per_bin`` (9,) int.
    """
    from .adaptation import adaptation_nonparametric

    out = adaptation_nonparametric(inputs)
    return {"rate_hz": np.asarray(out["tprev_nonparam_all"], dtype=float),
            "mean_rate_hz": float(out["meanFR_all"]),
            "samples_per_bin": np.asarray(out["tprev_bincounts_all"], dtype=np.int64)}


def percent_change(rates: Sequence[np.ndarray], mean_rates: Sequence[float]) -> np.ndarray:
    """Per-cell rate in each bin as percent change from the cell's mean rate.

    Parameters
    ----------
    rates : sequence of (n_bins,) arrays
        Rate per bin of each cell (Hz).
    mean_rates : sequence of float
        Mean rate of each cell (Hz).

    Returns
    -------
    (n_cells, n_bins) ndarray
    """
    mean_rates = np.asarray(mean_rates, dtype=float)
    return np.row_stack(rates) / mean_rates[:, np.newaxis] * 100 - 100


def bin_summary(pct: np.ndarray, edges: np.ndarray = FINE_TPREV_EDGES) -> Dict[str, np.ndarray]:
    """Population mean and standard error of the percent change in each bin.

    The mean is taken over the cells with a value in the bin; the standard error divides the
    standard deviation (ddof 1) by the square root of the number of cells with a value in every
    bin.

    Returns
    -------
    dict
        ``edge_low_s``, ``edge_high_s``, ``x_s`` (bin centres), ``cells_in_bin``,
        ``mean_pct_change``, ``sem_pct_change``, ``ci95_low``, ``ci95_high`` (mean -/+ 1.96 s.e.m.),
        ``weight`` (1 / s.e.m.^2), ``n_cells``, ``n_cells_complete``.
    """
    pct = np.asarray(pct, dtype=float)
    edges = np.asarray(edges, dtype=float)
    complete = ~np.isnan(pct).any(axis=1)
    mean = np.nanmean(pct, axis=0)
    sem = np.nanstd(pct, axis=0, ddof=1) / np.sqrt(complete.sum())
    return {"edge_low_s": edges[:-1], "edge_high_s": edges[1:], "x_s": bin_centres(edges),
            "cells_in_bin": (~np.isnan(pct)).sum(axis=0), "mean_pct_change": mean, "sem_pct_change": sem,
            "ci95_low": mean - 1.96 * sem, "ci95_high": mean + 1.96 * sem, "weight": 1 / sem ** 2,
            "n_cells": int(len(pct)), "n_cells_complete": int(complete.sum())}


# ------------------------------------------------------------------------------------------- fits
def exp_single(t, a, tau, c):
    """One exponential, ``c + a exp(-t / tau)``."""
    return c + a * np.exp(-t / tau)


def exp_double(t, a1, t1, a2, t2, c):
    """Two exponentials, ``c + a1 exp(-t / t1) + a2 exp(-t / t2)``."""
    return c + a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)


_MODELS = {"single": exp_single, "double": exp_double}


def load_fit_settings(path: Optional[Path] = None) -> dict:
    """Read the fit settings (default ``inputs/rat/timescale_fit_settings.json``)."""
    if path is None:
        from .. import config

        path = config.inputs_dir() / "rat" / "timescale_fit_settings.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _starting_values(p0: Sequence, mean: np.ndarray) -> List[float]:
    return [float(np.mean(mean)) if v == "mean_of_bin_means" else float(v) for v in p0]


def fit_model(model: str, x: np.ndarray, mean: np.ndarray, sem: Optional[np.ndarray], settings: dict) -> dict:
    """Least-squares fit of one model to the bin means.

    Parameters
    ----------
    model : {"single", "double"}
        One or two exponentials.
    x, mean : ndarray
        Bin centres (s) and bin means (%).
    sem : ndarray or None
        Standard errors of the bin means; the fit is weighted by 1 / sem^2 (``curve_fit`` with
        ``sigma=sem``, ``absolute_sigma=True``). None gives the unweighted fit.
    settings : dict
        Fit settings of the model: ``p0`` (starting values; the string ``"mean_of_bin_means"``
        stands for the mean of the bin means), optional ``extra_starts`` (further starting values),
        ``bounds`` (lower and upper bounds, null for none), ``method`` and ``max_nfev``.

    Returns
    -------
    dict
        The parameters (two exponentials ordered so that ``tau_fast <= tau_slow``) and ``SS``,
        the residual sum of squares (weighted: the chi-square). The fit with the lowest ``SS``
        over the starting values is returned; None when every start fails.
    """
    fn = _MODELS[model]
    bounds = settings["bounds"]
    lower = [-np.inf if v is None else v for v in bounds[0]]
    upper = [np.inf if v is None else v for v in bounds[1]]
    best = None
    starts = [settings["p0"]] + list(settings.get("extra_starts", []))
    for p0 in starts:
        kw = dict(p0=_starting_values(p0, mean), bounds=(lower, upper), method=settings["method"],
                  max_nfev=settings.get("max_nfev", 20000))
        if sem is not None:
            kw.update(sigma=sem, absolute_sigma=True)
        try:
            p, _ = optimize.curve_fit(fn, x, mean, **kw)
        except (RuntimeError, ValueError):
            continue
        p = np.array(p, float)
        if len(p) == 5 and p[1] > p[3]:
            p = p[[2, 3, 0, 1, 4]]
        if sem is not None:
            ss = float(np.sum(((mean - fn(x, *p)) / sem) ** 2))
        else:
            ss = float(np.sum((mean - fn(x, *p)) ** 2))
        if best is None or ss < best["SS"]:
            best = dict(zip(_PARAMETERS[model], (float(v) for v in p)))
            best["SS"] = ss
    return best


def fit_both(x: np.ndarray, mean: np.ndarray, sem: Optional[np.ndarray], settings: dict) -> Dict[str, dict]:
    """Fit one and two exponentials (see :func:`fit_model`); returns ``{"single": ..., "double": ...}``."""
    return {m: fit_model(m, x, mean, sem, settings[m]) for m in ("single", "double")}


def model_comparison(x: np.ndarray, mean: np.ndarray, sem: np.ndarray, settings: dict, weighted: bool) -> dict:
    """Fits of both models and their comparison.

    With weights the residual sum of squares is the chi-square and is used directly as -2 log
    likelihood (up to a constant); without weights -2 log likelihood is ``n log(SS / n)``.

    Returns
    -------
    dict
        ``single``, ``double`` (parameters and ``SS``), ``AIC_*``, ``AICc_*``, ``BIC_*`` for both
        models, ``dAIC_double_minus_single``, ``dAICc``, ``dBIC``, ``extra_SS_F`` and
        ``extra_SS_p`` (nested F test of two against one exponential) and ``n_points``. When a
        model cannot be fitted, only ``single``, ``double`` (None for that model) and ``n_points``.
    """
    fits = fit_both(x, mean, sem if weighted else None, settings)
    n = len(mean)
    if fits["single"] is None or fits["double"] is None:   # a model could not be fitted
        return {"single": fits["single"], "double": fits["double"], "n_points": int(n)}
    k1, k2 = len(_PARAMETERS["single"]), len(_PARAMETERS["double"])
    ss1, ss2 = fits["single"]["SS"], fits["double"]["SS"]
    if weighted:
        ll1, ll2 = ss1, ss2
    else:
        ll1, ll2 = n * np.log(ss1 / n), n * np.log(ss2 / n)
    aic1, aic2 = ll1 + 2 * k1, ll2 + 2 * k2
    aicc1, aicc2 = aic1 + 2 * k1 * (k1 + 1) / (n - k1 - 1), aic2 + 2 * k2 * (k2 + 1) / (n - k2 - 1)
    bic1, bic2 = ll1 + k1 * np.log(n), ll2 + k2 * np.log(n)
    F = ((ss1 - ss2) / (k2 - k1)) / (ss2 / (n - k2))
    out = {"single": fits["single"], "double": fits["double"],
           "AIC_single": float(aic1), "AIC_double": float(aic2),
           "AICc_single": float(aicc1), "AICc_double": float(aicc2),
           "BIC_single": float(bic1), "BIC_double": float(bic2),
           "dAIC_double_minus_single": float(aic2 - aic1), "dAICc": float(aicc2 - aicc1), "dBIC": float(bic2 - bic1),
           "extra_SS_F": float(F), "extra_SS_p": float(stats.f.sf(F, k2 - k1, n - k2)), "n_points": int(n)}
    return out


def animal_bootstrap(pct: np.ndarray, animal_ids: Sequence[str], settings: dict, x: Optional[np.ndarray] = None,
                     n_boot: int = N_BOOTSTRAP, rng: Optional[np.random.Generator] = None) -> dict:
    """Animal bootstrap of the weighted fits.

    Only cells with a value in every bin are used. In each resample as many animals as there are
    drawn with replacement (``rng.choice`` over the sorted animal labels), all cells of each drawn
    animal are pooled, the bin means and standard errors are recomputed and both models are
    fitted with weights 1 / s.e.m.^2.

    Parameters
    ----------
    pct : (n_cells, n_bins) ndarray
        Per-cell percent change.
    animal_ids : sequence of str
        Animal of each cell.
    settings : dict
        Fit settings (see :func:`fit_model`).
    x : ndarray, optional
        Bin centres (default: centres of ``FINE_TPREV_EDGES``).
    n_boot : int
        Number of resamples.
    rng : numpy.random.Generator, optional
        Random generator (default ``default_rng(PUBLISHED_BOOTSTRAP_SEED)``). Pass the same generator to
        successive calls to continue one random stream.

    Returns
    -------
    dict
        ``n_cells``, ``n_animals``, ``n_boot_ok`` (resamples in which both fits converged),
        ``tau_single_95``, ``tau_fast_95``, ``tau_slow_95``, ``dAIC_weighted_95`` (2.5 and 97.5
        percentiles), ``tau_slow_median``, ``share_slow_over_10s``,
        ``share_boot_double_preferred_by_AIC`` (fractions), ``n_double_preferred`` (int), and
        ``draws`` (per-resample values, for the record).
    """
    if rng is None:
        rng = np.random.default_rng(PUBLISHED_BOOTSTRAP_SEED)
    if x is None:
        x = bin_centres()
    pct = np.asarray(pct, dtype=float)
    animal_ids = np.asarray(animal_ids)
    complete = ~np.isnan(pct).any(axis=1)
    X = pct[complete]
    animals = animal_ids[complete]
    labels = np.unique(animals)   # as bootstrap_animals
    rows_of = {a: np.where(animals == a)[0] for a in labels}
    k1, k2 = len(_PARAMETERS["single"]), len(_PARAMETERS["double"])
    tau1 = np.full(n_boot, np.nan)
    tau_fast = np.full(n_boot, np.nan)
    tau_slow = np.full(n_boot, np.nan)
    d_aic = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(labels, len(labels), replace=True)
        rows = np.concatenate([rows_of[a] for a in pick])
        Xb = X[rows]
        mean = Xb.mean(axis=0)
        sem = Xb.std(axis=0, ddof=1) / np.sqrt(len(rows))
        f1 = fit_model("single", x, mean, sem, settings["single"])
        f2 = fit_model("double", x, mean, sem, settings["double"])
        if f1 is None or f2 is None:
            continue
        tau1[b] = f1["tau"]
        tau_fast[b] = f2["tau_fast"]
        tau_slow[b] = f2["tau_slow"]
        d_aic[b] = (f2["SS"] + 2 * k2) - (f1["SS"] + 2 * k1)
    ok = ~np.isnan(d_aic)
    n_ok = int(ok.sum())
    n_pref = int((d_aic[ok] < 0).sum())

    def pct95(v):
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    return {"n_cells": int(len(X)), "n_animals": int(len(labels)), "n_boot_ok": n_ok,
            "tau_single_95": pct95(tau1), "tau_fast_95": pct95(tau_fast), "tau_slow_95": pct95(tau_slow),
            "dAIC_weighted_95": pct95(d_aic), "tau_slow_median": float(np.nanmedian(tau_slow)),
            "share_slow_over_10s": float((tau_slow[ok] > 10).sum() / n_ok) if n_ok else math.nan,
            "share_boot_double_preferred_by_AIC": float(n_pref / n_ok) if n_ok else math.nan,
            "n_double_preferred": n_pref,
            "draws": {"tau_single": tau1, "tau_fast": tau_fast, "tau_slow": tau_slow, "dAIC_weighted": d_aic}}


def bootstrap_animals(pct: np.ndarray, animal_ids: Sequence[str]) -> np.ndarray:
    """Animals drawn from by ``animal_bootstrap``: the sorted labels of the cells with a value in every bin."""
    pct = np.asarray(pct, dtype=float)
    complete = ~np.isnan(pct).any(axis=1)
    return np.unique(np.asarray(animal_ids)[complete])


def skip_bootstrap_draws(rng: np.random.Generator, n_animals: int, n_boot: int) -> None:
    """Advance ``rng`` by the draws of an ``animal_bootstrap`` over ``n_animals`` animals, without fitting.

    The draws of a resample depend only on the number of animals, so a later bootstrap on the same
    generator continues exactly as after the full bootstrap.
    """
    for _ in range(int(n_boot)):
        rng.choice(np.arange(int(n_animals)), int(n_animals), replace=True)


# -------------------------------------------------------------------------------- Methods numbers
def interval_inputs(cell_data: dict, field_locations: np.ndarray) -> dict:
    """Input item of :func:`field_visit_intervals` for one cell."""
    return {"cell_data": cell_data, "fieldLoc": field_locations}


def field_visit_intervals(inputs: dict) -> np.ndarray:
    """Intervals between consecutive field visits of one cell (worker function).

    The interval of a visit is its t_prev at field entry (standard rule: from the exit of the
    previous visit to the entry of this one, plus one sample). The first visit, which has no
    previous visit, and non-positive or undefined values are dropped.

    Parameters
    ----------
    inputs : dict
        ``cell_data`` and ``fieldLoc`` ((K, 3) field centres and radii, cm).

    Returns
    -------
    ndarray
        Intervals (s) in visit order.
    """
    from .fields import extract_field_entry
    from .gridcell import GridCellAnalysis

    g = GridCellAnalysis(inputs["cell_data"])
    fl = inputs["fieldLoc"]
    r = extract_field_entry(g.x, g.y, g.time, fl[:, 0], fl[:, 1], fl[:, 2], g.spike_array)
    tp = r["ev_df"]["t_prev"].to_numpy(float)[1:]
    return tp[np.isfinite(tp) & (tp > 0)]


def interval_statistics(intervals: Sequence[np.ndarray], scale_cm: np.ndarray,
                        max_interval_s: float = MAX_INTERVAL_S) -> dict:
    """Field-visit intervals by third of grid scale.

    The thirds are cut at the 33.3 and 66.7 percentiles of the grid scale (``numpy.digitize``).
    Pooled statistics use the intervals of all cells of a third up to ``max_interval_s``; the
    per-cell median and the event count use all intervals.

    Returns
    -------
    dict
        ``scale_third_edges_cm`` and ``intervals_by_scale_third.<third>`` with ``n_cells``,
        ``scale_range_cm``, ``n_events``, ``pooled_median_s``, ``pooled_iqr_s``,
        ``median_of_cell_medians_s``, ``pooled_share_under_1s`` and ``pooled_share_under_2s``
        (only ``n_cells`` and ``n_events`` for a third without intervals).
    """
    scale_cm = np.asarray(scale_cm, dtype=float)
    medians = np.array([float(np.median(tp)) if len(tp) else np.nan for tp in intervals])
    n_events = np.array([len(tp) for tp in intervals])
    capped = [tp[tp <= max_interval_s] for tp in intervals]
    edges = np.percentile(scale_cm, [100 / 3, 200 / 3])
    third = np.digitize(scale_cm, edges) + 1
    out = {"scale_third_edges_cm": edges.tolist(), "intervals_by_scale_third": {}}
    for k, name in ((1, "small"), (2, "medium"), (3, "large")):
        m = third == k
        ev = np.concatenate([capped[i] for i in np.where(m)[0]] + [np.zeros(0)])
        if not len(ev):   # no cell or no interval in this third
            out["intervals_by_scale_third"][name] = {"n_cells": int(m.sum()), "n_events": int(n_events[m].sum())}
            continue
        out["intervals_by_scale_third"][name] = {
            "n_cells": int(m.sum()), "scale_range_cm": [float(scale_cm[m].min()), float(scale_cm[m].max())],
            "n_events": int(n_events[m].sum()), "median_of_cell_medians_s": float(np.nanmedian(medians[m])),
            "pooled_median_s": float(np.median(ev)), "pooled_iqr_s": np.percentile(ev, [25, 75]).tolist(),
            "pooled_share_under_1s": float((ev < 1).mean()), "pooled_share_under_2s": float((ev < 2).mean())}
    return out


def session_speed(position: dict, speed_range=SPEED_RANGE_CM_S) -> dict:
    """Running speed of one session.

    Parameters
    ----------
    position : dict
        ``position`` of a session of the merged trajectories (``speed`` in cm/s).
    speed_range : (float, float)
        Speeds counted as movement (inclusive).

    Returns
    -------
    dict
        ``n_samples``, ``n_moving`` and ``mean_speed_moving_cm_s`` (mean over the moving samples).
    """
    sp = np.asarray(position["speed"], float)
    sp = sp[np.isfinite(sp)]
    mv = sp[(sp >= speed_range[0]) & (sp <= speed_range[1])]
    return {"n_samples": int(len(sp)), "n_moving": int(len(mv)), "mean_speed_moving_cm_s": float(mv.mean())}


def arena_size_by_session(session_ids: Sequence[str], x_edges: Sequence[np.ndarray]) -> Dict[str, float]:
    """Arena size of each session: median over its cells of the last x edge of the rate map (cm)."""
    import pandas as pd

    env = pd.Series([float(e[-1]) for e in x_edges], index=pd.Index(list(session_ids), name="session_id"))
    return env.groupby(level=0, sort=True).median().to_dict()


def running_speed_summary(sessions: Sequence[str], speed: Dict[str, dict], arena_cm: Dict[str, float],
                          arena_range=ARENA_100CM_RANGE, speed_range=SPEED_RANGE_CM_S) -> dict:
    """Mean running speed over the sessions in the 100 cm arenas.

    Parameters
    ----------
    sessions : sequence of str
        Cohort sessions in canonical order.
    speed : dict
        ``{session_id: session_speed(...)}``.
    arena_cm : dict
        ``{session_id: arena size (cm)}``.

    Returns
    -------
    dict
        ``n_sessions``, ``speed_range_cm_s`` and ``mean_speed_cm_s`` (mean over sessions of the
        per-session mean speed while moving; absent when no session qualifies).
    """
    import pandas as pd

    sel = [s for s in sessions if arena_range[0] < arena_cm[s] < arena_range[1]]
    out = {"n_sessions": len(sel),
           "speed_range_cm_s": [int(v) if float(v).is_integer() else float(v) for v in speed_range]}
    if sel:
        out["mean_speed_cm_s"] = float(pd.Series([speed[s]["mean_speed_moving_cm_s"] for s in sel]).mean())
    return out


def first_to_last_bin_ratio(pct: np.ndarray) -> float:
    """Mean over cells of the rate in the first t_prev bin divided by the rate in the last bin.

    Parameters
    ----------
    pct : (n_cells, n_bins) ndarray
        Percent change from the mean rate (``rat/adaptation/fr_percent_change.npy``); the rate
        ratio is ``(100 + first) / (100 + last)``.
    """
    pct = np.asarray(pct, dtype=float)
    return float(np.nanmean((pct[:, 0] + 100) / (pct[:, -1] + 100)))
