"""Preferred movement direction of grid cells relative to the grid axes (Fig 2B to 2E).

A grid cell is conjunctive when the resultant length of its movement-direction tuning curve
(``Rayleigh_mag``) exceeds 0.3. The analyses of Fig 2 ask whether preferred directions are aligned
with the grid axes, as reported for human entorhinal cortex by Doeller et al. (2010):

- **Fig 2B**: the deviation of each conjunctive cell's preferred direction from the nearest grid
  axis, ``wrap60(preferred direction - grid orientation)`` in [-30, 30) degrees, in 5-degree bins;
  cells within 15 degrees of an axis count as aligned (two-sided exact binomial test against one
  half).
- **Fig 2C**: tuning curves rotated so that the bin nearest the preferred direction comes first,
  averaged per animal (animals with at least 5 conjunctive cells), then across animals (band:
  standard error across animals); and the tuning width of each cell as the circular standard
  deviation ``sqrt(-2 ln R)``.
- **Fig 2D**: the selection of Doeller et al. (2010) applied to all grid cells: the
  Kullback-Leibler divergence from uniform of the tuning curve smoothed by a circular Gaussian at
  least 0.15 (smoothing SD 9.01 degrees, which with the 15-degree boxcar already applied to the
  curves gives a total SD of 10 degrees), one recording per cell (``same_cell``). The relation of
  preferred direction (modulo 60) to grid orientation is the Fisher and Lee (1983)
  circular-circular correlation of the angles multiplied by 6, tested against 20,000 permutations
  of directions across cells (one-sided).
- **Fig 2E**: within each animal of Vollan et al. (2025) with at least 5 conjunctive cells (one
  recording per cell), the clustering of preferred directions on a 60-degree cycle in room
  coordinates: Rayleigh test (Zar's approximation) on 6 x preferred direction, Holm correction over
  animals, and the chance band of the resultant length (95th percentile for uniform directions:
  100,000 draws with seed 5600 + n for up to 60 cells, ``sqrt(-ln 0.05 / n)`` above).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .. import stats
from ..circular import wrap60

__all__ = [
    "CONJUNCTIVE_MIN_R",
    "ALIGNED_MAX_ABS_DEV_DEG",
    "FIG2B_BIN_EDGES",
    "MIN_CELLS_PER_ANIMAL",
    "KL_SMOOTH_SD_DEG",
    "KL_MIN",
    "FIG2D_PERMUTATIONS",
    "FIG2D_KEY",
    "CHANCE_BAND_N",
    "deviation_from_axes",
    "fig2b_bins",
    "tuning_width_deg",
    "aligned_tuning_curves",
    "per_animal_curves",
    "gaussian_smooth_circular",
    "kl_from_uniform",
    "tuning_kl",
    "direction_orientation_correlation",
    "per_animal_rayleigh",
    "chance_band",
    "fig2_results",
]

#: Resultant length above which a grid cell counts as conjunctive.
CONJUNCTIVE_MIN_R = 0.3
#: Largest absolute deviation from the nearest grid axis of an aligned cell (deg, excluded).
ALIGNED_MAX_ABS_DEV_DEG = 15.0
#: Edges of the deviation histogram of Fig 2B (deg).
FIG2B_BIN_EDGES = np.arange(-30, 31, 5)
#: Fewest conjunctive cells of an animal in the per-animal analyses (Fig 2C, 2E).
MIN_CELLS_PER_ANIMAL = 5
#: SD of the circular Gaussian applied before the divergence measure of Fig 2D (deg).
KL_SMOOTH_SD_DEG = math.sqrt(10.0**2 - 15.0**2 / 12.0)
#: Smallest divergence from uniform of a directional cell in Fig 2D (nats).
KL_MIN = 0.15
#: Permutations of the Fig 2D test and the text key of its random seed (``stats.seed_of``).
FIG2D_PERMUTATIONS = 20000
FIG2D_KEY = "q2sel_kl|one|all|dir_mean"
#: Range of cell numbers over which the chance band of Fig 2E is drawn.
CHANCE_BAND_N = (5, 300)

N_BINS = 72
BIN_CENTRES_DEG = (np.arange(N_BINS) + 0.5) * 5.0


# --------------------------------------------------------------------------------------- Fig 2B
def deviation_from_axes(direction_deg, orientation_deg) -> np.ndarray:
    """Deviation of a preferred direction from the nearest grid axis, in [-30, 30) degrees."""
    return wrap60(np.asarray(direction_deg, float) - np.asarray(orientation_deg, float))


def fig2b_bins(deviation_deg, edges: np.ndarray = FIG2B_BIN_EDGES):
    """Histogram of deviations in 5-degree bins (lower edge included).

    The six bins from -15 to 15 degrees form the aligned half. (A deviation of exactly -15 degrees
    would fall in the aligned half of the histogram but not count as aligned; none occurs.)

    Returns
    -------
    pandas.DataFrame
        ``bin`` (from 1), ``deviation_from_deg``, ``deviation_to_deg``, ``half`` (``aligned`` for
        bins within 15 degrees of the axis, else ``misaligned``) and ``count``.
    """
    import pandas as pd

    counts = np.histogram(np.asarray(deviation_deg, float), bins=edges)[0]
    lo, hi = edges[:-1], edges[1:]
    half = np.where((lo >= -ALIGNED_MAX_ABS_DEV_DEG) & (hi <= ALIGNED_MAX_ABS_DEV_DEG), "aligned", "misaligned")
    return pd.DataFrame(
        {
            "bin": np.arange(1, len(counts) + 1, dtype=np.int64),
            "deviation_from_deg": lo.astype(np.int64),
            "deviation_to_deg": hi.astype(np.int64),
            "half": half,
            "count": counts.astype(np.int64),
        }
    )


# --------------------------------------------------------------------------------------- Fig 2C
def tuning_width_deg(R) -> np.ndarray:
    """Circular standard deviation of a tuning curve with resultant length R, in degrees.

    Computed as ``sqrt(-2 ln R) * 180 / pi`` (the order of operations of the published values).
    """
    return np.sqrt(-2 * np.log(np.asarray(R, float))) * 180 / np.pi


def aligned_tuning_curves(dir_maps: np.ndarray, preferred_deg) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate each tuning curve so that the bin nearest its preferred direction comes first.

    Parameters
    ----------
    dir_maps : ndarray (n, 72)
        Movement-direction tuning curves (5-degree bins from 0 degrees).
    preferred_deg : array_like (n,)
        Preferred direction of each cell (deg).

    Returns
    -------
    aligned : ndarray (n, 73)
        Rotated curves with the first bin repeated at the end.
    angles : ndarray (73,)
        Bin centres of the rotated curves (2.5 to 357.5 deg, then 2.5 again).
    """
    raw = np.asarray(dir_maps, float)
    pref = np.asarray(preferred_deg, float)
    bin_size = 360 / raw.shape[1]
    angles_mid = np.arange(0, 360, bin_size) + bin_size / 2
    aligned = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        shift = -np.argmin(np.abs(angles_mid - pref[i]))
        aligned[i] = np.roll(raw[i], shift)
    return np.c_[aligned, aligned[:, 0]], np.r_[angles_mid, angles_mid[0]]


def per_animal_curves(
    aligned: np.ndarray, animal_ids: Sequence[str], angles: np.ndarray, min_cells: int = MIN_CELLS_PER_ANIMAL
):
    """Mean rotated tuning curve per animal and the mean and s.e.m. across animals.

    Two forms are computed: ``not_normalised`` (curves as they are) and
    ``normalised_to_cell_mean`` (each cell divided by its own mean over the 72 bins). Animals with at
    least ``min_cells`` cells enter the mean across animals; the others are listed with
    ``in_mean_across_animals`` False.

    Parameters
    ----------
    aligned : ndarray (n, 73)
        Output of :func:`aligned_tuning_curves`.
    animal_ids : sequence of str (n,)
    angles : ndarray (73,)
        Bin centres of the rotated curves.

    Returns
    -------
    curves : pandas.DataFrame
        ``form``, ``animal_id``, ``n_cells``, ``in_mean_across_animals``, ``bin``, ``direction_deg``,
        ``mean_curve``; forms in the order above, animals by decreasing cell count (then label), the
        included animals first.
    mean : pandas.DataFrame
        ``form`` (``normalised_to_cell_mean`` first), ``bin``, ``direction_deg``,
        ``mean_across_animals``, ``sem_across_animals`` (SD with ddof=1 over sqrt of the number of
        animals).
    summary : dict
        ``n_animals`` and ``n_cells`` in the mean across animals.
    """
    import pandas as pd

    A = np.asarray(aligned, float)
    ids = np.asarray(animal_ids).astype(str)
    names, counts = np.unique(ids, return_counts=True)
    cnt = dict(zip(names.tolist(), counts.tolist()))
    keep = sorted([a for a in cnt if cnt[a] >= min_cells], key=lambda a: (-cnt[a], a))
    drop = sorted([a for a in cnt if cnt[a] < min_cells], key=lambda a: (-cnt[a], a))
    forms = {"not_normalised": A, "normalised_to_cell_mean": A / A[:, :72].mean(1, keepdims=True)}
    rows, mean_rows = [], {}
    for name, X in forms.items():
        animal_means = {}
        for a in keep + drop:
            sel = ids == a
            animal_means[a] = X[sel].mean(0)
            rows += [
                dict(
                    form=name,
                    animal_id=a,
                    n_cells=int(sel.sum()),
                    in_mean_across_animals=a in keep,
                    bin=b,
                    direction_deg=float(angles[b]),
                    mean_curve=float(animal_means[a][b]),
                )
                for b in range(A.shape[1])
            ]
        if keep:
            M = np.array([animal_means[a] for a in keep])
            n = len(keep)
            mean_an = M.mean(0)
            sem_an = M.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(A.shape[1], np.nan)
            mean_rows[name] = pd.DataFrame(
                {
                    "form": name,
                    "bin": np.arange(A.shape[1], dtype=np.int64),
                    "direction_deg": angles,
                    "mean_across_animals": mean_an,
                    "sem_across_animals": sem_an,
                }
            )
    curves = pd.DataFrame(
        rows, columns=["form", "animal_id", "n_cells", "in_mean_across_animals", "bin", "direction_deg", "mean_curve"]
    )
    order = [f for f in ("normalised_to_cell_mean", "not_normalised") if f in mean_rows]
    mean = (
        pd.concat([mean_rows[f] for f in order], ignore_index=True)
        if order
        else pd.DataFrame(columns=["form", "bin", "direction_deg", "mean_across_animals", "sem_across_animals"])
    )
    summary = {"n_animals": len(keep), "n_cells": int(sum(cnt[a] for a in keep))}
    return curves, mean, summary


# --------------------------------------------------------------------------------------- Fig 2D
def gaussian_smooth_circular(M: np.ndarray, sd_deg: float, bin_deg: float = 5.0) -> np.ndarray:
    """Smooth each row (a circular tuning curve) with a Gaussian of SD ``sd_deg`` truncated at 5 SD."""
    h = int(math.ceil(5 * sd_deg / bin_deg))
    k = np.arange(-h, h + 1)
    g = np.exp(-0.5 * (k * bin_deg / sd_deg) ** 2)
    g /= g.sum()
    return np.stack([np.convolve(np.r_[r[-h:], r, r[:h]], g, mode="valid") for r in np.asarray(M, float)])


def kl_from_uniform(M: np.ndarray) -> np.ndarray:
    """Kullback-Leibler divergence (nats) of each row, normalised to a distribution, from uniform."""
    M = np.asarray(M, float)
    p = M / M.sum(1, keepdims=True)
    return (p * np.log(np.clip(p, 1e-300, None) * M.shape[1])).sum(1)


def tuning_kl(dir_maps: np.ndarray) -> np.ndarray:
    """Divergence from uniform of the tuning curves after circular Gaussian smoothing (Fig 2D selection)."""
    return kl_from_uniform(gaussian_smooth_circular(dir_maps, KL_SMOOTH_SD_DEG))


def direction_orientation_correlation(
    orientation_deg, direction_deg, n_perm: int = FIG2D_PERMUTATIONS, key: str = FIG2D_KEY
) -> Dict[str, float]:
    """Fisher-Lee correlation of grid orientation and preferred direction modulo 60 degrees.

    Both angles are multiplied by 6 (one 60-degree cycle becomes a full circle). The permutation
    null shuffles directions across cells with ``numpy.random.default_rng(stats.seed_of(key))``,
    so the result depends on the order of the cells (canonical order).

    Returns
    -------
    dict
        ``r``, ``n``, ``p_one_sided_positive``, ``p_one_sided_negative``, ``p_two_sided``, ``null_sd``.
    """
    a6 = np.radians(6 * np.asarray(orientation_deg, float))
    b6 = np.radians(6 * (np.asarray(direction_deg, float) % 60))
    return stats.fisher_lee_permutation(a6, b6, n_perm, key=key)


# --------------------------------------------------------------------------------------- Fig 2E
def per_animal_rayleigh(animal_ids: Sequence[str], direction_deg, min_cells: int = MIN_CELLS_PER_ANIMAL):
    """Rayleigh test of six-fold clustering of preferred directions within each animal.

    Parameters
    ----------
    animal_ids : sequence of str
    direction_deg : array_like
        Preferred directions (deg), one per cell.
    min_cells : int
        Fewest cells of an animal that is tested.

    Returns
    -------
    pandas.DataFrame
        One row per tested animal, sorted by ``animal_id`` as text: ``animal_id``, ``n``, ``R6``
        (resultant length of 6 x direction), ``Z`` (n R6^2), ``p_rayleigh`` (Zar), ``R_crit_p05``
        (resultant length at which p = 0.05), ``phase6_deg`` (mean direction on the 60-degree
        cycle, [0, 60)), ``p_holm`` (Holm over the tested animals) and ``R6_chance95`` (95th
        percentile of R6 for uniform directions).
    """
    import pandas as pd

    ids = np.asarray(animal_ids).astype(str)
    d = np.asarray(direction_deg, float)
    rows = []
    for a in sorted(set(ids.tolist())):
        sel = ids == a
        if sel.sum() < min_cells:
            continue
        r = stats.rayleigh_zar(np.deg2rad(6 * d[sel]))
        rows.append(
            {
                "animal_id": a,
                "n": r["n"],
                "R6": r["R"],
                "Z": r["Z"],
                "p_rayleigh": r["p"],
                "R_crit_p05": stats.rayleigh_r_crit(r["n"]),
                "phase6_deg": float(np.degrees(r["mean_rad"]) / 6 % 60),
            }
        )
    cols = ["animal_id", "n", "R6", "Z", "p_rayleigh", "R_crit_p05", "phase6_deg", "p_holm", "R6_chance95"]
    pa = pd.DataFrame(rows, columns=cols[:7])
    pa["p_holm"] = stats.holm(pa["p_rayleigh"].to_numpy(float)) if len(pa) else np.zeros(0)
    pa["R6_chance95"] = [stats.rayleigh_monte_carlo(0.0, int(n))[2] for n in pa["n"]]
    pa["n"] = pa["n"].astype(np.int64)
    return pa[cols]


def chance_band(n_values: Optional[Sequence[int]] = None):
    """Chance level of the resultant length of n uniform directions, for each n.

    Returns
    -------
    pandas.DataFrame
        ``n``, ``R_chance_mean`` and ``R_chance95`` (Monte Carlo for n <= 60, closed form above;
        ``stats.rayleigh_monte_carlo``). Default n: every whole number from 5 to 300.
    """
    import pandas as pd

    if n_values is None:
        n_values = np.arange(CHANCE_BAND_N[0], CHANCE_BAND_N[1] + 1)
    vals = [stats.rayleigh_monte_carlo(0.0, int(n)) for n in n_values]
    return pd.DataFrame(
        {
            "n": np.asarray(n_values, dtype=np.int64),
            "R_chance_mean": [v[1] for v in vals],
            "R_chance95": [v[2] for v in vals],
        }
    )


# ------------------------------------------------------------------------------- all Fig 2 results
def _finite_numbers(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if isinstance(v, (bool, np.bool_, int, np.integer)):
            out[k] = int(v)
        elif v is not None and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def fig2_results(
    cells, dir_maps: np.ndarray, kept_earliest, module_bins: Optional[dict] = None, vollan_key: str = "vollan2025"
):
    """Tables and statistics of Fig 2B to 2E from the cohort.

    Parameters
    ----------
    cells : pandas.DataFrame
        Cohort table (``rat/cohort/cells.csv``) in canonical order.
    dir_maps : ndarray (n, 72)
        Tuning curves, row-aligned with ``cells``.
    kept_earliest : array_like of bool (n,)
        Earliest recording of each unit (``rat/same_cell/entries.csv``).
    module_bins : dict, optional
        Module bins (``preprocessing.labels.read_module_bins``; read from the shipped table by default).
    vollan_key : str
        Dataset of the per-animal analysis of Fig 2E.

    Returns
    -------
    tables : dict
        ``{file name: DataFrame}`` for the files of ``rat/conjunctive/``.
    results : dict
        Statistics keyed as in ``stats/step10_conjunctive_cells.json`` (finite numbers only).
    details : dict
        The full output of the Fig 2D test, and ``statistics_not_finite``: the keys left out of
        ``results`` because their value is not a finite number.

    Notes
    -----
    Grid orientation, preferred direction, resultant length and scale enter Fig 2B, 2D and 2E
    rounded as after two CSV round trips with pandas' default float parser, and one more for the
    per-animal tests of Fig 2E (``io.csv_float_roundtrip``; docs/PIPELINE.md, section 2.7), which
    reproduces every published value bit for bit. Fig 2C uses the exact values.
    """
    from ..io import csv_float_roundtrip
    from ..preprocessing import labels

    ident = ["row", "animal_id", "session_id", "cell_id", "dataset"]
    kept = np.asarray(kept_earliest, bool)
    ori2 = csv_float_roundtrip(cells["orientation_deg"].to_numpy(float), 2)
    ang2 = csv_float_roundtrip(cells["Rayleigh_angle"].to_numpy(float), 2)
    mag2 = csv_float_roundtrip(cells["Rayleigh_mag"].to_numpy(float), 2)
    scale2 = csv_float_roundtrip(cells["scale_cm"].to_numpy(float), 2)
    conj = cells["Rayleigh_mag"].to_numpy(float) > CONJUNCTIVE_MIN_R
    S: Dict[str, float] = {}
    details: Dict[str, float] = {}

    # Fig 2B: deviation from the grid axes
    b = cells.loc[conj, ident].copy()
    b["orientation_deg"] = ori2[conj]
    b["Rayleigh_angle"] = ang2[conj]
    b["Rayleigh_mag"] = mag2[conj]
    b["deviation_deg"] = deviation_from_axes(b["Rayleigh_angle"], b["orientation_deg"])
    b["aligned"] = np.abs(b["deviation_deg"].to_numpy()) < ALIGNED_MAX_ABS_DEV_DEG
    b["kept_earliest"] = kept[conj]
    n_b, n_al = int(len(b)), int(b["aligned"].sum())
    S.update({"fig2b.n_cells": n_b, "fig2b.n_aligned": n_al, "fig2b.n_misaligned": n_b - n_al})
    if n_b:
        S["fig2b.pct_aligned"] = 100 * (n_al / n_b)
        S["fig2b.binomial.p"] = stats.binomial_test(n_al, n_b, 0.5, alternative="two-sided")

    # Fig 2C: rotated tuning curves by animal, tuning widths
    aligned, angles = aligned_tuning_curves(np.asarray(dir_maps)[conj], cells["Rayleigh_angle"].to_numpy(float)[conj])
    curves, mean, csum = per_animal_curves(aligned, cells["animal_id"].to_numpy()[conj], angles)
    width = tuning_width_deg(cells["Rayleigh_mag"].to_numpy(float)[conj])
    S.update({"fig2c.n_animals": csum["n_animals"], "fig2c.n_cells": csum["n_cells"]})
    if len(width):
        S.update(
            {
                "fig2c.circular_sd.median_deg": float(np.median(width)),
                "fig2c.circular_sd.q25_deg": float(np.percentile(width, 25)),
                "fig2c.circular_sd.q75_deg": float(np.percentile(width, 75)),
            }
        )

    # Fig 2D: selection of Doeller et al. (2010), one recording per cell, Fisher-Lee correlation
    kl = tuning_kl(dir_maps)
    module = labels.assign_modules(cells["animal_id"].to_numpy(), scale2, module_bins)
    sel = (kl >= KL_MIN) & kept & np.isfinite(ang2)
    d = cells.loc[sel, ident].copy()
    d["module"] = module[sel]
    d["orientation_deg"] = ori2[sel]
    d["direction_deg"] = ang2[sel]
    d["kl"] = kl[sel]
    d["deviation_deg"] = deviation_from_axes(d["direction_deg"], d["orientation_deg"])
    d["conjunctive"] = conj[sel]
    S.update(
        {
            "fig2d.n_cells": int(len(d)),
            "fig2d.n_animals": int(d["animal_id"].nunique()),
            "fig2d.n_sessions": int(d["session_id"].nunique()),
            "fig2d.n_conjunctive": int(d["conjunctive"].sum()),
        }
    )
    if len(d) >= 3:
        details = direction_orientation_correlation(d["orientation_deg"], d["direction_deg"])
        S.update(
            {
                "fig2d.fisher_lee.r": details["r"],
                "fig2d.permutation.p": details["p_one_sided_positive"],
                "fig2d.permutation.n_perm": FIG2D_PERMUTATIONS,
            }
        )

    # Fig 2E: clustering of preferred directions within animals (Vollan et al. 2025)
    esel = conj & kept & (cells["dataset"].to_numpy() == vollan_key)
    e = cells.loc[esel, ["row", "animal_id"]].copy()
    e["Rayleigh_angle"] = ang2[esel]
    pa = per_animal_rayleigh(e["animal_id"], csv_float_roundtrip(e["Rayleigh_angle"].to_numpy(float), 1))
    e = e[e["animal_id"].isin(pa["animal_id"])]
    p_all = pa["p_rayleigh"].to_numpy(float)
    below = np.sort(p_all[p_all < 0.05])
    S.update(
        {"fig2e.n_animals": int(len(pa)), "fig2e.n_cells": int(pa["n"].sum()), "fig2e.n_p_below_0_05": int(len(below))}
    )
    for k, p in enumerate(below, start=1):
        S[f"fig2e.p_below_0_05.{k}"] = float(p)
    if len(pa):
        S["fig2e.min_p_holm"] = float(pa["p_holm"].min())
        S["fig2e.binomial.p"] = stats.binomial_test(len(below), len(pa), 0.05, alternative="greater")

    tables = {
        "fig2b_cells.csv": b,
        "fig2b_bins.csv": fig2b_bins(b["deviation_deg"]),
        "fig2c_per_animal_curves.csv": curves,
        "fig2c_mean.csv": mean,
        "fig2d_cells.csv": d,
        "fig2e_per_animal.csv": pa,
        "fig2e_cells.csv": e,
        "fig2e_band.csv": chance_band(),
    }
    finite = _finite_numbers(S)
    details = dict(details, statistics_not_finite=[k for k in S if k not in finite])
    return tables, finite, details
