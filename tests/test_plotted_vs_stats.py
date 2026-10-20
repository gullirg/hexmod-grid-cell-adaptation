"""The statistics recomputed from the values each figure draws equal the statistics files.

Every figure writes the values it draws to ``figures/<Figure>/source_data/``. The checks below recompute the
statistics of each figure from these tables, with the functions of ``hexmod.stats``,
``hexmod.rat.paper_stats``, ``hexmod.rat.figure_tables``, ``hexmod.rat.conjunctive`` and
``hexmod.rat.timescales``, and compare them with the keys of ``stats/*.json`` in the same results folder
(docs/STATISTICS.md). They show that each figure draws the values its statistics were computed
from. A few numbers are not recomputed but read from a statistics table of the figure: the Fig 7A and 7D
tests (computed on firing rates in Hz, drawn as percent change) and the Fig 8 optimal speed (a closed form).

The tests check the results folder of a complete run, given in ``HEXMOD_RESULTS`` (the folder that holds
``figures/`` and ``stats/``), for example ``HEXMOD_RESULTS=results pytest -m results tests/test_plotted_vs_stats.py``.
They are skipped when no such folder is given.

Counts must be equal; other numbers agree within ``RTOL`` (relative) and ``ATOL`` (absolute): statistics
recomputed in another cell order (Fig 3F) or from per-cell values rounded by a CSV round trip (docs/PIPELINE.md,
Rounded per-cell inputs) differ in the last digits only.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest
import scipy.stats as sps

from hexmod.figures import FIGURES

REPO = Path(__file__).resolve().parents[1]
#: Relative and absolute tolerance of the comparison of non-integer statistics.
RTOL = 1e-12
ATOL = 1e-15
#: Looser relative tolerance where a statistic is recomputed through a different but equivalent formula
#: (chi-square from the drawn fit curves).
RTOL_RECOMPUTED = 1e-9

MODULES = {"Small module": "small", "Large module": "large"}
METHODS = {"simple": "simple", "location-controlled": "location_controlled"}
TPREV_COLUMNS = {
    "Fig3C": ["pct_change_0_to_1_s", "pct_change_1_to_2_s", "pct_change_2_to_3_s", "pct_change_above_3_s"],
    "SuppFig2D": ["t_prev_0_to_1_s", "t_prev_1_to_2_s", "t_prev_2_to_3_s", "t_prev_above_3_s"],
}

Entry = Tuple[str, float, float]   # key, recomputed value, relative tolerance
CHECKS: Dict[str, Callable] = {}


def check(figure: str):
    """Register the function that recomputes the statistics of ``figure``."""
    def register(func):
        CHECKS[figure] = func
        return func
    return register


class Source:
    """Reader of the source data tables of one figure."""

    def __init__(self, folder: Path):
        self.folder = folder

    def __call__(self, name: str):
        from hexmod import io

        path = self.folder / "source_data" / f"{name}.csv"
        assert path.is_file(), f"source data missing: {path.name}"
        return io.read_csv(path)


def entries(values: Dict[str, float], rtol: float = RTOL) -> List[Entry]:
    return [(k, v, rtol) for k, v in values.items()]


# ------------------------------------------------------------------------------------------- figures
@check("Fig2")
def fig2(src: Source) -> List[Entry]:
    from hexmod import stats
    from hexmod.rat import conjunctive

    out: List[Entry] = []
    b = src("Fig2B_deviation_histogram")
    n = int(b["count"].sum())
    n_al = int(b.loc[b["half"] == "aligned", "count"].sum())
    out += [("fig2b.n_cells", n, 0), ("fig2b.n_aligned", n_al, 0), ("fig2b.n_misaligned", n - n_al, 0)]
    if n:
        out += [("fig2b.pct_aligned", 100 * (n_al / n), RTOL),
                ("fig2b.binomial.p", stats.binomial_test(n_al, n, 0.5, alternative="two-sided"), RTOL)]
    c = src("Fig2C_per_animal_curves_and_mean")
    animals = [col for col in c.columns if col.startswith("animal_")]
    out += [("fig2c.n_animals", len(animals), 0),
            ("fig2c.n_cells", sum(int(col.rsplit("_n", 1)[1]) for col in animals), 0)]
    d = src("Fig2D_cells")
    out += [("fig2d.n_cells", len(d), 0), ("fig2d.n_animals", d["animal_id"].nunique(), 0),
            ("fig2d.n_sessions", d["session_id"].nunique(), 0)]
    if len(d) >= 3:
        r = conjunctive.direction_orientation_correlation(d["orientation_deg"], d["preferred_direction_deg"])
        out += [("fig2d.fisher_lee.r", r["r"], RTOL), ("fig2d.permutation.p", r["p_one_sided_positive"], RTOL)]
    e = src("Fig2E_animals")
    p = np.array([min(1.0, stats.zar_p(int(k), float(R)))
                  for k, R in zip(e["n_conjunctive_cells"], e["rayleigh_vector_length_6x"])])
    below = np.sort(p[p < 0.05])
    out += [("fig2e.n_animals", len(e), 0), ("fig2e.n_cells", int(e["n_conjunctive_cells"].sum()), 0),
            ("fig2e.n_p_below_0_05", len(below), 0)]
    out += [(f"fig2e.p_below_0_05.{k}", v, RTOL) for k, v in enumerate(below, start=1)]
    if len(e):
        out += [("fig2e.min_p_holm", float(np.min(stats.holm(p))), RTOL),
                ("fig2e.binomial.p", stats.binomial_test(len(below), len(e), 0.05, alternative="greater"), RTOL)]
    return out


@check("Fig3")
def fig3(src: Source) -> List[Entry]:
    from hexmod import stats
    from hexmod.rat import figure_tables, paper_stats

    X = src("Fig3C_firing_rate_change_pct")[TPREV_COLUMNS["Fig3C"]].to_numpy(float)
    out = entries(paper_stats.fig3c_statistics(X))
    out += [(k, v, RTOL) for k, v in figure_tables.paired_posthoc("fig3c", X).items() if k.endswith(".p_holm")]
    out += entries(paper_stats.fig3d_statistics(src("Fig3D_adaptation_coefficient")["adaptation_coefficient"]))
    f = src("Fig3F_firing_rate_ratio")
    lt, gt = f["frr_T23_lt_T12"].to_numpy(float), f["frr_T23_gt_T12"].to_numpy(float)
    out.append(("fig3f.n_cells", len(f), 0))
    for name, x in (("shorter", lt), ("longer", gt)):
        r = stats.one_sample_t(x)
        out += [(f"fig3f.{name}.mean", r["mean"], RTOL), (f"fig3f.{name}.t", r["t"], RTOL),
                (f"fig3f.{name}.df", r["df"], 0), (f"fig3f.{name}.p", r["p"], RTOL)]
    r = stats.paired_t(lt, gt)
    out += [("fig3f.paired.mean_diff", r["mean_diff"], RTOL), ("fig3f.paired.t", r["t"], RTOL),
            ("fig3f.paired.df", r["df"], 0), ("fig3f.paired.p", r["p"], RTOL)]
    return out


@check("Fig4")
def fig4(src: Source) -> List[Entry]:
    from hexmod.rat import figure_tables, paper_stats

    d = src("Fig4CD_cells")
    out = entries(paper_stats.fig4c_statistics(d["proximity_aligned"], d["proximity_misaligned"]))
    pdiff = d["proximity_difference"].to_numpy(float)
    out += entries(paper_stats.fig4d_statistics(pdiff, d["grid_scale_cm"]))
    _, third = paper_stats.thirds(d["grid_scale_cm"].to_numpy(float))
    groups = [pdiff[(third == k) & ~np.isnan(pdiff)] for k in (1, 2, 3)]
    out += entries(figure_tables.independent_posthoc("fig4d", groups, ("small", "medium", "large")))
    return out


@check("Fig5")
def fig5(src: Source) -> List[Entry]:
    from hexmod.rat import figure_tables, paper_stats

    d = src("Fig5_cells")
    alpha = d["adaptation_coefficient_alpha"].to_numpy(float)
    pdiff = d["proximity_aligned_minus_misaligned"].to_numpy(float)
    beta = d["HexMod_beta"].to_numpy(float)
    out = entries(paper_stats.fig5_statistics(alpha, pdiff, beta))
    _, a_third = paper_stats.thirds(alpha)
    _, p_third = paper_stats.thirds(pdiff)
    top = [beta[(p_third == 3) & (a_third == k)] for k in (1, 2, 3)]
    out += entries(figure_tables.independent_posthoc("fig5b", top, ("strong", "weak", "no")))
    return out


@check("Fig6")
def fig6(src: Source) -> List[Entry]:
    from hexmod import stats

    out: List[Entry] = []
    d = src("Fig6D_proximity")
    for module, g in d.groupby("module", sort=False):
        al = g["proximity_aligned"].to_numpy(float)
        mis = g["proximity_misaligned"].to_numpy(float)
        r = stats.paired_t(al, mis)
        diff = al - mis
        base = f"fig6d.{module}"
        out += [(base + ".mean_aligned", al.mean(), RTOL), (base + ".mean_misaligned", mis.mean(), RTOL),
                (base + ".t", r["t"], RTOL), (base + ".df", len(diff) - 1, 0),
                (base + ".d", diff.mean() / diff.std(ddof=1), RTOL)]
    m = src("Fig6BC_maps")
    sel = (m["module"] == "large") & m["map"].str.startswith("summed") & m["map"].str.contains("seed 0")
    v = m.loc[sel, "value"].to_numpy(float)
    assert len(v), "summed map of the first large-module participant not drawn"
    out += [("fig6c.large_p0.max_pct", v.max() - 100, RTOL), ("fig6c.large_p0.min_pct", v.min() - 100, RTOL),
            ("fig6c.large_p0.sd_pct", v.std(), RTOL)]
    return out


@check("Fig7")
def fig7(src: Source) -> List[Entry]:
    from hexmod import stats

    out: List[Entry] = []
    be = src("Fig7BE_beta_by_symmetry")
    for (module, method, k), g in be.groupby(["module", "method", "x"], sort=False):
        prefix = "fig7b" if method == "simple" else "fig7e"
        r = stats.one_sample_t(g["value"])
        base = f"{prefix}.{MODULES[module]}.fold_{int(k)}"
        out += [(base + ".t", r["t"], RTOL), (base + ".df", r["df"], 0), (base + ".p", r["p"], RTOL),
                (base + ".significant", int(r["p"] < 0.05), 0)]
    c = src("Fig7C_orientation")
    for (module, method), g in c.groupby(["module", "method"], sort=False):
        six = 6 * g["orientation_deg_modulo_60"].to_numpy(float)
        counts = np.bincount((np.floor((six + 15) / 30) % 12).astype(int), minlength=12)
        out += [(f"fig7c.{MODULES[module]}.{METHODS[method]}.count_{30 * b}", int(counts[b]), 0) for b in range(12)]
    f = src("Fig7F_p_per_experiment")
    n_exp = f["experiment"].nunique()
    out.append(("fig7f.n_experiments", n_exp, 0))
    for (module, method, k), g in f.groupby(["module", "method", "symmetry_n_fold"], sort=False):
        count = int((g["p_one_sided"] < 0.05).sum())
        base = f"fig7f.{MODULES[module]}.{METHODS[method]}.fold_{int(k)}"
        out += [(base + ".count", count, 0), (base + ".rate", count / g["experiment"].nunique(), RTOL)]
    st = src("Fig7_statistics")
    for _, row in st[st["panel"].isin(["A", "D"])].iterrows():
        base = f"fig7{row['panel'].lower()}.{MODULES[row['module']]}"
        out += [(base + ".t", row["statistic"], RTOL), (base + ".df", int(row["df"]), 0),
                (base + ".p", row["p"], RTOL), (base + ".n", int(row["n"]), 0)]
    return out


@check("Fig8")
def fig8(src: Source) -> List[Entry]:
    s = src("Fig8_statistics")
    v = s.loc[s["quantity"] == "speed of the largest proximity difference, in units of grid scale / tau", "value"]
    assert len(v) == 1
    return [("fig8.v_opt_scale_per_tau", float(v.iloc[0]), RTOL), ("fig8d.n_runs", len(src("Fig8D_speed_sweep")), 0)]


@check("SuppFig1")
def supp1(src: Source) -> List[Entry]:
    from hexmod.rat import paper_stats

    d = src("SuppFig1_adaptation_coefficient_odd_even_minutes")
    return entries(paper_stats.supp1_statistics(d["alpha_odd_minutes"], d["alpha_even_minutes"]))


@check("SuppFig2")
def supp2(src: Source) -> List[Entry]:
    from hexmod.rat import figure_tables, paper_stats

    peak = src("SuppFig2C_best_time_shift")["best_time_shift_ms"].to_numpy()
    X = src("SuppFig2D_firing_rate_change_pct")[TPREV_COLUMNS["SuppFig2D"]].to_numpy(float)
    e = src("SuppFig2E_adaptation_coefficient")
    out = entries(paper_stats.supp2_statistics(peak, e["alpha_original"], e["alpha_shifted"], X))
    out += entries(figure_tables.paired_posthoc("supp2", X))
    si = src("SuppFig2B_spatial_information_example_cell")
    best = si["shift_ms"].to_numpy()[np.argmax(si["spatial_information_bits_per_spike"].to_numpy(float))]
    out.append(("supp2.example_best_shift_ms", int(best), 0))
    return out


@check("SuppFig3")
def supp3(src: Source) -> List[Entry]:
    out: List[Entry] = []
    v = src("SuppFig3BCD_values")
    module = {"small scale": "small", "large scale": "large"}
    distribution = {"B": "deviation", "C": "rate_across_cells", "D": "rate_within_cell"}
    for (panel, group), g in v.groupby(["panel", "group"], sort=False):
        x = g["value"].to_numpy(float)
        m = module[group]
        if panel == "B":
            out.append((f"supp3.{m}.n_fields", len(x), 0))
        if panel == "C":
            out.append((f"supp3.{m}.n_cells", len(x), 0))
        if len(x) >= 2:
            shape, _, scale = sps.gamma.fit(x, floc=0)
            out += [(f"supp3.{m}.{distribution[panel]}.shape", shape, RTOL),
                    (f"supp3.{m}.{distribution[panel]}.scale", scale, RTOL)]
    return out


@check("SuppFig4")
def supp4(src: Source) -> List[Entry]:
    from hexmod.rat import timescales

    p = src("SuppFig4_points")
    f = src("SuppFig4_fits")
    x = p["t_prev_s_plotted"].to_numpy(float)
    y = p["mean_firing_rate_change_pct"].to_numpy(float)
    sem = p["sem_pct"].to_numpy(float)
    one, two = f.iloc[0], f.iloc[1]
    chi1, chi2 = float(one["weighted_chi_square"]), float(two["weighted_chi_square"])
    fit1 = timescales.exp_single(x, one["a_or_a1_pct"], one["tau_or_tau1_s"], one["c_pct"])
    fit2 = timescales.exp_double(x, two["a_or_a1_pct"], two["tau_or_tau1_s"], two["a2_pct"], two["tau2_s"],
                                 two["c_pct"])
    n, k1, k2 = len(x), 3, 5
    F = ((chi1 - chi2) / (k2 - k1)) / (chi2 / (n - k2))
    taus = sorted([float(two["tau_or_tau1_s"]), float(two["tau2_s"])])
    return [("supp4.n_cells_last_bin", int(p["cells"].iloc[-1]), 0),
            ("supp4.single.tau_s", float(one["tau_or_tau1_s"]), RTOL),
            ("supp4.single.chi2", chi1, RTOL),
            ("supp4.single.chi2", float(np.sum(((y - fit1) / sem) ** 2)), RTOL_RECOMPUTED),
            ("supp4.double.tau_fast_s", taus[0], RTOL), ("supp4.double.tau_slow_s", taus[1], RTOL),
            ("supp4.double.chi2", chi2, RTOL),
            ("supp4.double.chi2", float(np.sum(((y - fit2) / sem) ** 2)), RTOL_RECOMPUTED),
            ("supp4.delta_aic", (chi2 + 2 * k2) - (chi1 + 2 * k1), RTOL),
            ("supp4.delta_bic", (chi2 + k2 * np.log(n)) - (chi1 + k1 * np.log(n)), RTOL),
            ("supp4.f_test.F", F, RTOL), ("supp4.f_test.df1", k2 - k1, 0), ("supp4.f_test.df2", n - k2, 0),
            ("supp4.f_test.p", float(sps.f.sf(F, k2 - k1, n - k2)), RTOL)]


@check("SuppFig5")
def supp5(src: Source) -> List[Entry]:
    from hexmod.rat import figure_tables

    w = src("SuppFig5B_conjunctive_cell_widths")
    sd = w["circular_sd_deg"].to_numpy(float)
    np.testing.assert_allclose(sd, np.degrees(np.sqrt(-2 * np.log(w["Rayleigh_mag"].to_numpy(float)))), rtol=1e-12)
    out = entries(figure_tables.supp5_statistics(w))
    if len(sd):
        out += [("fig2c.circular_sd.median_deg", float(np.median(sd)), RTOL),
                ("fig2c.circular_sd.q25_deg", float(np.percentile(sd, 25)), RTOL),
                ("fig2c.circular_sd.q75_deg", float(np.percentile(sd, 75)), RTOL)]
    m = src("SuppFig5_marked_points").set_index("point")
    # panel A marks the narrow tuning curve of Fig 1C
    out += [("fig1.narrow.circular_sd_deg", m.loc["narrow tuning of Fig 1C", "circular_sd_deg"], RTOL),
            ("fig1.narrow.sixfold", m.loc["narrow tuning of Fig 1C", "sixfold_percent_of_mean"] / 100, RTOL)]
    for point, name in (("narrowest conjunctive cell", "narrowest"), ("median conjunctive cell", "median")):
        if point in m.index:
            out += [(f"supp5.{name}_sd_deg", m.loc[point, "circular_sd_deg"], RTOL),
                    (f"supp5.{name}_sixfold_pct", m.loc[point, "sixfold_percent_of_mean"], RTOL)]
    return out


# ---------------------------------------------------------------------------------------- comparison
def read_statistics(results: Path) -> Dict[str, float]:
    """All statistics of a results folder (``stats/*.json`` merged)."""
    from hexmod import io

    values: Dict[str, float] = {}
    for path in sorted((results / "stats").glob("*.json")):
        if path.name.startswith("statistics_check"):
            continue
        values.update(io.read_json(path))
    return values


def compare(figure: str, results: Path) -> Tuple[List[str], int]:
    """Recompute the statistics of ``figure`` and compare them with the statistics files.

    Statistics missing from the files are passed over.

    Returns
    -------
    problems : list of str
        Disagreements.
    n_compared : int
        Number of statistics compared.
    """
    folder = results / "figures" / figure
    assert folder.is_dir(), f"{figure} was not drawn in {results}"
    stats_values = read_statistics(results)
    problems, n_compared = [], 0
    for key, value, rtol in CHECKS[figure](Source(folder)):
        value = float(value)
        if key not in stats_values:
            continue
        ref = stats_values[key]
        n_compared += 1
        if isinstance(ref, int) and not isinstance(ref, bool) and rtol == 0:
            ok = value == ref
        else:
            ok = math.isfinite(value) and abs(value - ref) <= ATOL + rtol * abs(ref)
        if not ok:
            problems.append(f"{key}: recomputed {value!r}, statistics file {ref!r}")
    return problems, n_compared


def test_every_figure_has_a_check():
    assert sorted(CHECKS) == sorted(FIGURES)


# ----------------------------------------------------------------------------- results of a run
def _results_folder() -> Path:
    value = os.environ.get("HEXMOD_RESULTS")
    if value:
        folder = Path(value).expanduser()
        if (folder / "figures").is_dir() and (folder / "stats").is_dir():
            return folder
    pytest.skip("no results folder with figures and statistics (set HEXMOD_RESULTS)")


@pytest.mark.results
@pytest.mark.parametrize("figure", FIGURES)
def test_plotted_values_give_the_statistics(figure):
    problems, n_compared = compare(figure, _results_folder())
    assert not problems, "\n".join(problems)
    assert n_compared > 0, f"no statistic of {figure} in the statistics files"
