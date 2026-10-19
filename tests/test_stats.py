"""Tests of hexmod.stats.

Unit tests use textbook examples (Student's sleep data, the plant-growth ANOVA, hand-computed
sums of squares, series expansions of the Rayleigh distribution). Tests marked ``frozen``
recompute printed statistics from reference results laid out like ``results/`` (folder in
``HEXMOD_FROZEN``) and compare them with the published values at full precision.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest
from scipy import stats as sps

from hexmod import stats

# Student's sleep data (Cushny and Peebles 1905; R dataset "sleep"): extra hours of sleep
SLEEP_1 = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
SLEEP_2 = np.array([1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])

# R dataset "PlantGrowth": dried weight of plants under a control and two treatments
PLANT = [
    np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]),
    np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]),
    np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]),
]


# ------------------------------------------------------------------------------------- t-tests
def test_paired_t_sleep_data():
    r = stats.paired_t(SLEEP_1, SLEEP_2)
    assert r["n"] == 10 and r["df"] == 9
    assert r["t"] == pytest.approx(-4.0621, abs=1e-4)
    assert r["p"] == pytest.approx(0.002833, abs=1e-6)
    d = SLEEP_1 - SLEEP_2
    assert r["d"] == pytest.approx(d.mean() / d.std(ddof=1), rel=1e-12)
    assert r["mean_diff"] == pytest.approx(-1.58, abs=1e-12)


def test_one_sample_t_equals_paired_t_on_differences():
    d = SLEEP_1 - SLEEP_2
    one = stats.one_sample_t(d)
    pair = stats.paired_t(SLEEP_1, SLEEP_2)
    assert one["t"] == pytest.approx(pair["t"], rel=1e-12)
    assert one["p"] == pytest.approx(pair["p"], rel=1e-12)
    assert one["d"] == pytest.approx(pair["d"], rel=1e-12)


def test_one_sample_t_drops_nan_and_matches_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    r = stats.one_sample_t(x)
    assert r["n"] == 5 and r["df"] == 4 and r["mean"] == 3.0
    assert r["t"] == pytest.approx(3.0 / (math.sqrt(2.5) / math.sqrt(5)), rel=1e-12)
    assert r["p"] == pytest.approx(2 * sps.t.sf(r["t"], 4), rel=1e-10)
    assert r["d"] == pytest.approx(3.0 / math.sqrt(2.5), rel=1e-12)
    assert stats.cohens_d([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(3.0 / math.sqrt(2.5), rel=1e-12)


def test_two_sample_t_student_and_welch_sleep_data():
    student = stats.two_sample_t(SLEEP_1, SLEEP_2, equal_var=True)
    assert student["t"] == pytest.approx(-1.8608, abs=1e-4)
    assert student["df"] == 18
    assert student["p"] == pytest.approx(0.07919, abs=1e-5)
    welch = stats.two_sample_t(SLEEP_1, SLEEP_2, equal_var=False)
    assert welch["t"] == pytest.approx(-1.8608, abs=1e-4)
    assert welch["df"] == pytest.approx(17.776, abs=1e-3)
    assert welch["p"] == pytest.approx(0.07939, abs=1e-5)


# ------------------------------------------------------------------------------------- ANOVAs
def test_one_way_anova_plant_growth():
    r = stats.one_way_anova(*PLANT)
    assert (r["df1"], r["df2"]) == (2, 27)
    assert r["F"] == pytest.approx(4.846, abs=1e-3)
    assert r["p"] == pytest.approx(0.01591, abs=1e-5)


RM_DATA = np.array([[8, 7, 1], [9, 5, 2], [6, 2, 3], [5, 3, 1], [8, 4, 5]], float)


def test_rm_anova_hand_computed():
    # grand mean 4.6; SS_treatment 58.8, SS_subjects 254/15, SS_total 95.6, SS_error 298/15
    r = stats.rm_anova(RM_DATA)
    assert (r["df1"], r["df2"], r["n"]) == (2, 8, 5)
    assert r["F"] == pytest.approx((58.8 / 2) / ((298 / 15) / 8), rel=1e-12)
    assert r["p"] == pytest.approx(sps.f.sf(r["F"], 2, 8), rel=1e-12)
    assert r["means"] == pytest.approx([7.2, 4.2, 2.4], rel=1e-12)


def test_rm_anova_two_conditions_equals_paired_t_squared():
    X = np.column_stack([SLEEP_1, SLEEP_2])
    r = stats.rm_anova(X)
    t = stats.paired_t(SLEEP_1, SLEEP_2)
    assert r["F"] == pytest.approx(t["t"] ** 2, rel=1e-10)
    assert r["p"] == pytest.approx(t["p"], rel=1e-8)


def test_rm_anova_drops_rows_with_nan():
    X = np.vstack([RM_DATA, [np.nan, 1.0, 2.0]])
    assert stats.rm_anova(X)["F"] == stats.rm_anova(RM_DATA)["F"]


def test_rm_pairwise():
    out = stats.rm_pairwise(RM_DATA, ["a", "b", "c"])
    assert list(out) == ["a vs b", "a vs c", "b vs c"]
    ref = sps.ttest_rel(RM_DATA[:, 0], RM_DATA[:, 2])
    assert out["a vs c"]["t"] == pytest.approx(ref.statistic, rel=1e-12)
    assert out["a vs c"]["df"] == 4


# --------------------------------------------------------------------------- Holm and binomial
def test_holm_textbook():
    adj = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02], rel=1e-12)
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_binomial_textbook():
    assert stats.binomial_test(7, 10, 0.5) == pytest.approx(2 * 176 / 1024, rel=1e-12)
    # P(X >= 3) for X ~ Binomial(12, 0.05): the chance of 3 of 12 animals below 0.05
    exact = 1 - sum(math.comb(12, k) * 0.05 ** k * 0.95 ** (12 - k) for k in range(3))
    assert stats.binomial_test(3, 12, 0.05, "greater") == pytest.approx(exact, rel=1e-10)
    assert round(exact, 3) == 0.020


# ----------------------------------------------------------------------------------- Rayleigh
def _rayleigh_series_p(n, Z):
    """Rayleigh p value from the series expansion (Greenwood and Durand 1955; Mardia and Jupp)."""
    return math.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n)
                           - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2))


@pytest.mark.parametrize("n,Z", [(50, 3.0), (100, 2.0), (200, 4.5), (1000, 6.0)])
def test_zar_p_matches_series_expansion(n, Z):
    R = math.sqrt(Z / n)
    assert stats.zar_p(n, R) == pytest.approx(_rayleigh_series_p(n, Z), rel=2e-3)


def test_rayleigh_zar_on_angles():
    rng = np.random.default_rng(0)
    a = rng.vonmises(1.0, 0.5, size=40)
    r = stats.rayleigh_zar(a)
    z = np.exp(1j * a).mean()
    assert r["n"] == 40
    assert r["R"] == pytest.approx(abs(z), rel=1e-12)
    assert r["mean_rad"] == pytest.approx(np.angle(z), rel=1e-12)
    assert r["Z"] == pytest.approx(40 * abs(z) ** 2, rel=1e-12)
    assert r["p"] == pytest.approx(min(1.0, stats.zar_p(40, abs(z))), rel=1e-12)
    # p is capped at 1 for a perfectly balanced sample
    assert stats.rayleigh_zar(np.array([0.0, np.pi]))["p"] == 1.0


@pytest.mark.parametrize("n", [6, 17, 112, 238])
def test_rayleigh_r_crit(n):
    rc = stats.rayleigh_r_crit(n)
    assert stats.zar_p(n, rc) == pytest.approx(0.05, rel=1e-9)
    if n > 100:
        assert rc == pytest.approx(math.sqrt(-math.log(0.05) / n), rel=0.02)


def test_rayleigh_monte_carlo_agrees_with_zar():
    n = 30
    nul = stats.rayleigh_chance(n, ndraw=100000)
    assert len(nul) == 100000 and np.all(np.diff(nul) >= 0)
    assert nul.mean() == pytest.approx(math.sqrt(math.pi / (4 * n)), rel=0.03)
    p_mc, chance_mean, chance95 = stats.rayleigh_monte_carlo(0.35, n)
    assert p_mc == pytest.approx(stats.zar_p(n, 0.35), abs=0.003)
    assert chance95 == pytest.approx(stats.rayleigh_r_crit(n), abs=0.01)
    # the same seed gives the same null; above 60 directions the asymptotic band is used
    assert stats.rayleigh_monte_carlo(0.35, n) == (p_mc, chance_mean, chance95)
    p, m, q = stats.rayleigh_monte_carlo(0.1, 175)
    assert q == math.sqrt(-math.log(0.05) / 175) and m == math.sqrt(math.pi / (4 * 175))
    assert p == min(1.0, stats.zar_p(175, 0.1))


def test_v_test():
    a = np.full(8, 0.3)
    r = stats.v_test(a, mu=0.3)
    assert r["mean_cos"] == pytest.approx(1.0) and r["u"] == pytest.approx(4.0)
    assert r["p_toward"] == pytest.approx(sps.norm.sf(4.0))


# --------------------------------------------------------------------------------- Fisher-Lee
def _fisher_lee_pairs(a, b):
    """Definition of Fisher and Lee (1983) through all pairs of observations."""
    i, j = np.triu_indices(len(a), 1)
    num = (np.sin(a[i] - a[j]) * np.sin(b[i] - b[j])).sum()
    return num / math.sqrt((np.sin(a[i] - a[j]) ** 2).sum() * (np.sin(b[i] - b[j]) ** 2).sum())


def test_fisher_lee_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 2 * np.pi, 60)
    b = a + rng.vonmises(0.0, 2.0, 60)
    assert stats.fisher_lee(a, b) == pytest.approx(_fisher_lee_pairs(a, b), abs=1e-12)
    assert stats.fisher_lee(a, a) == pytest.approx(1.0, abs=1e-12)
    assert stats.fisher_lee(a, -a) == pytest.approx(-1.0, abs=1e-12)
    assert stats.fisher_lee(a, a + 1.3) == pytest.approx(1.0, abs=1e-12)


def test_seed_of_crc32():
    assert stats.seed_of("") == 0
    assert stats.seed_of("The quick brown fox jumps over the lazy dog") == 0x414FA339
    assert stats.seed_of("q2") == zlib.crc32(b"q2") % 2 ** 31


def test_fisher_lee_permutation_reproducible():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 2 * np.pi, 50)
    b = rng.uniform(0, 2 * np.pi, 50)
    r1 = stats.fisher_lee_permutation(a, b, n_perm=500, key="example")
    r2 = stats.fisher_lee_permutation(a, b, n_perm=500, key="example")
    assert r1 == r2
    null = stats.fisher_lee_permutation_null(a, b, 500, np.random.default_rng(stats.seed_of("example")))
    perm = np.random.default_rng(stats.seed_of("example")).permutation(50)
    assert null[0] == pytest.approx(stats.fisher_lee(a, b[perm]), abs=1e-12)
    assert r1["p_one_sided_positive"] == (1 + np.sum(null >= r1["r"])) / 501
    strong = stats.fisher_lee_permutation(a, a + 0.1 * rng.standard_normal(50), n_perm=500, key="x")
    assert strong["p_one_sided_positive"] == pytest.approx(1 / 501)


# ------------------------------------------------------------------- frozen: printed statistics
def _cells(frozen_path):
    from hexmod.io import read_csv

    return read_csv(frozen_path("rat/cohort/cells.csv"))


@pytest.mark.frozen
def test_frozen_fig3c(frozen_path):
    from hexmod.io import load_npy

    X = load_npy(frozen_path("rat/adaptation/fr_percent_change.npy"))
    assert X.shape == (3938, 4)
    r = stats.rm_anova(X)
    assert (r["df1"], r["df2"], r["n"]) == (3, 11811, 3938)
    assert r["F"] == pytest.approx(375.7525820528324, rel=1e-13)
    assert round(r["F"], 1) == 375.8
    assert r["p"] == pytest.approx(4.120668499662508e-233, rel=1e-9)
    assert r["means"] == pytest.approx([-6.188826559123555, 0.38888380845650705, 2.3888770543049613,
                                        2.8399806894079536], rel=1e-13)
    pw = stats.rm_pairwise(X, ["0-1", "1-2", "2-3", ">3"])
    assert pw["2-3 vs >3"]["t"] == pytest.approx(-1.5531461396793789, rel=1e-12)
    assert pw["2-3 vs >3"]["p"] == pytest.approx(0.12046860285454972, rel=1e-10)
    assert max(v["p"] for k, v in pw.items() if k != "2-3 vs >3") < 1e-9


@pytest.mark.frozen
def test_frozen_fig3d_fig4_fig5_supp1(frozen_path):
    from hexmod.io import read_csv

    cells = _cells(frozen_path)
    par = read_csv(frozen_path("rat/adaptation/param_cells.csv"))
    assert len(par) == 3938 and (par["row"].to_numpy() == np.arange(3938)).all()
    m = cells.merge(par[[c for c in par.columns if c not in cells.columns or c == "row"]], on="row")
    alpha = m["alpha_tprevXloc_all"].to_numpy(float)
    r = stats.one_sample_t(alpha)
    assert r["t"] == pytest.approx(-41.08719970089978, rel=1e-13)
    assert r["d"] == pytest.approx(-0.6547397180098896, rel=1e-13)
    assert (alpha < 0).mean() * 100 == pytest.approx(77.04418486541391, rel=1e-13)
    # Supp Fig 1
    odd, even = m["alpha_tprevXloc_oddMinute"].to_numpy(float), m["alpha_tprevXloc_evenMinute"].to_numpy(float)
    assert stats.one_sample_t(odd)["t"] == pytest.approx(-35.40040117227108, rel=1e-13)
    assert stats.one_sample_t(even)["t"] == pytest.approx(-31.229263144227485, rel=1e-13)
    pe = stats.paired_t(odd, even)
    assert pe["t"] == pytest.approx(-2.399088973236987, rel=1e-10) and pe["df"] == 3937
    assert pe["p"] == pytest.approx(0.016482139205538857, rel=1e-8)
    assert sps.pearsonr(odd, even)[0] == pytest.approx(0.30957060461914476, rel=1e-12)
    # Fig 4C
    pa, pm = m["align_tprev_infield"].to_numpy(float), m["misalign_tprev_infield"].to_numpy(float)
    r4c = stats.paired_t(pa, pm)
    assert r4c["t"] == pytest.approx(11.068996081948974, rel=1e-13) and r4c["df"] == 3937
    assert r4c["mean_a"] == pytest.approx(0.3228464801535668, rel=1e-13)
    assert r4c["mean_b"] == pytest.approx(0.3206788378601611, rel=1e-13)
    # Fig 4D: scale tertiles by percentile, top edge raised by 0.001
    pdiff = pa - pm
    sc = m["scale_cm"].to_numpy(float)
    sb = np.percentile(sc, [0, 100 / 3, 200 / 3, 100])
    sb[-1] += 0.001
    scat = np.digitize(sc, sb)
    groups = [pdiff[(scat == k) & ~np.isnan(pdiff)] for k in (1, 2, 3)]
    assert [len(g) for g in groups] == [1313, 1312, 1313]
    assert sb[:3] == pytest.approx([27.59647279816268, 61.792851394097596, 86.10474521499107], rel=1e-14)
    a4d = stats.one_way_anova(*groups)
    assert (a4d["df1"], a4d["df2"]) == (2, 3935)
    assert a4d["F"] == pytest.approx(7.055496902834318, rel=1e-12)
    assert a4d["p"] == pytest.approx(0.0008736100192217434, rel=1e-10)
    # Fig 5
    beta = m["beta_all"].to_numpy(float)
    assert stats.one_sample_t(beta)["t"] == pytest.approx(-0.23781936939235346, rel=1e-12)
    ab = np.percentile(alpha, [0, 100 / 3, 200 / 3, 100])
    ab[-1] += 0.001
    ac = np.digitize(alpha, ab)
    pb = np.percentile(pdiff, [0, 100 / 3, 200 / 3, 100])
    pb[-1] += 0.001
    pc = np.digitize(pdiff, pb)
    top = {name: stats.one_sample_t(beta[(pc == 3) & (ac == k)]) for name, k in (("strong", 1), ("weak", 2), ("no", 3))}
    assert [top[k]["n"] for k in ("strong", "weak", "no")] == [435, 430, 448]
    assert top["strong"]["t"] == pytest.approx(-2.8448374277402873, rel=1e-12)
    assert top["strong"]["p"] == pytest.approx(0.004653649269882926, rel=1e-10)
    assert top["strong"]["d"] == pytest.approx(-0.1363995145293546, rel=1e-12)
    assert top["weak"]["t"] == pytest.approx(0.08254509286249098, rel=1e-10)
    assert top["no"]["t"] == pytest.approx(0.8194625630264761, rel=1e-12)
    a5b = stats.one_way_anova(*[beta[(pc == 3) & (ac == k)] for k in (1, 2, 3)])
    assert a5b["F"] == pytest.approx(4.013330304625063, rel=1e-12) and a5b["df2"] == 1310
    assert a5b["p"] == pytest.approx(0.018295776898235454, rel=1e-10)
    bottom_strong = stats.one_sample_t(beta[(pc == 1) & (ac == 1)])
    assert bottom_strong["n"] == 443 and bottom_strong["t"] == pytest.approx(1.363854024549391, rel=1e-12)
    st = stats.two_sample_t(beta[(pc == 3) & (ac == 1)], beta[(pc == 1) & (ac == 1)])
    assert st["df"] == 876 and st["t"] == pytest.approx(-2.9957671327691107, rel=1e-12)
    assert st["p"] == pytest.approx(0.002814675813280782, rel=1e-10)


@pytest.mark.frozen
def test_frozen_supp2(frozen_path):
    from hexmod.io import load_npy, read_csv

    X = load_npy(frozen_path("rat/prospective/fr_percent_change_shifted.npy"))
    r = stats.rm_anova(X)
    assert (r["n"], r["df2"]) == (3754, 11259)
    assert r["F"] == pytest.approx(294.6744444161788, rel=1e-12)
    sh = read_csv(frozen_path("rat/prospective/shift_cells.csv"))
    pr = stats.paired_t(sh["alpha_original"], sh["alpha_tprevXloc_all"])
    assert pr["df"] == 3753 and pr["t"] == pytest.approx(-4.130142488147495, rel=1e-12)
    assert stats.one_sample_t(sh["alpha_original"])["d"] == pytest.approx(-0.6666826349413769, rel=1e-12)
    assert stats.one_sample_t(sh["alpha_tprevXloc_all"])["d"] == pytest.approx(-0.6106926243822308, rel=1e-12)


@pytest.mark.frozen
def test_frozen_fig3f(frozen_path):
    from hexmod.io import read_csv

    pc = read_csv(frozen_path("rat/visits/paired_cells.csv"))
    assert len(pc) == 3576
    lt = stats.one_sample_t(pc["frr_T23_lt_T12"])
    gt = stats.one_sample_t(pc["frr_T23_gt_T12"])
    pd_ = stats.paired_t(pc["frr_T23_lt_T12"], pc["frr_T23_gt_T12"])
    assert lt["t"] == pytest.approx(2.525943153241181, rel=1e-10)
    assert gt["t"] == pytest.approx(-2.1005184387217954, rel=1e-10)
    assert pd_["t"] == pytest.approx(2.9043756581724347, rel=1e-10) and pd_["df"] == 3575
    assert pd_["p"] == pytest.approx(0.0037024248653153, rel=1e-8)


@pytest.mark.frozen
def test_frozen_fig2b_fig2c(frozen_path):
    from hexmod.circular import circular_sd_deg

    cells = _cells(frozen_path)
    conj = cells[cells["Rayleigh_mag"] > 0.3]
    assert len(conj) == 1171
    assert (conj["animal_id"].nunique(), conj["session_id"].nunique()) == (21, 55)
    pref = conj["Rayleigh_angle"].to_numpy(float)
    gori = conj["orientation_deg"].to_numpy(float)
    aligned = int(np.round((np.cos(6.0 * np.deg2rad(pref - gori)) > 0).mean() * len(conj)))
    assert aligned == 564
    assert stats.binomial_test(aligned, len(conj), 0.5) == pytest.approx(0.21967191561522467, rel=1e-12)
    csd = circular_sd_deg(conj["Rayleigh_mag"].to_numpy(float))
    assert np.median(csd) == pytest.approx(73.05863103970445, rel=1e-13)
    assert np.percentile(csd, [25, 75]) == pytest.approx([65.96672386661956, 80.01894576684913], rel=1e-12)


@pytest.mark.frozen
def test_frozen_fig2d(frozen_path):
    from hexmod.io import read_csv

    d = read_csv(frozen_path("rat/conjunctive/fig2d_cells.csv"))
    assert len(d) == 890 and d["animal_id"].nunique() == 21 and d["session_id"].nunique() == 47
    a6 = np.radians(6 * d["orientation_deg"].to_numpy(float))
    b6 = np.radians(6 * (d["direction_deg"].to_numpy(float) % 60))
    r = stats.fisher_lee_permutation(a6, b6, n_perm=20000, key="q2sel_kl|one|all|dir_mean")
    assert r["r"] == pytest.approx(0.0012622152849992038, rel=1e-10)
    assert r["p_one_sided_positive"] == pytest.approx(0.1639918004099795, abs=1e-15)


@pytest.mark.frozen
def test_frozen_fig2e(frozen_path):
    from hexmod.io import read_csv

    cells = _cells(frozen_path)
    ent = read_csv(frozen_path("rat/same_cell/entries.csv"))
    assert (ent["row"].to_numpy() == cells["row"].to_numpy()).all()
    keep = (cells["dataset"] == "vollan2025") & (cells["Rayleigh_mag"] > 0.3) & ent["kept_earliest"].astype(bool)
    v = cells[keep.to_numpy()]
    ref = read_csv(frozen_path("rat/conjunctive/fig2e_per_animal.csv")).set_index("animal_id")
    rows = []
    for animal, sub in v.groupby("animal_id"):
        if len(sub) < 5:
            continue
        r = stats.rayleigh_zar(np.deg2rad(6 * sub["Rayleigh_angle"].to_numpy(float)))
        rows.append((animal, r["n"], r["R"], r["p"], stats.rayleigh_r_crit(r["n"]),
                     stats.rayleigh_monte_carlo(r["R"], r["n"])[2]))
    assert len(rows) == 12 and sum(x[1] for x in rows) == 1082
    p_holm = stats.holm([x[3] for x in rows])
    for (animal, n, R, p, rc, band), ph in zip(rows, p_holm):
        e = ref.loc[animal]
        assert n == e["n"]
        assert R == pytest.approx(e["R6"], rel=1e-12)
        assert p == pytest.approx(e["p_rayleigh"], rel=1e-12)
        assert rc == pytest.approx(e["R_crit_p05"], rel=1e-10)
        assert ph == pytest.approx(e["p_holm"], rel=1e-12)
        assert band == pytest.approx(e["R6_chance95"], rel=1e-12)
    # with the CSV round trips of the published analysis (docs/PIPELINE.md, 2.7) the values agree bit for bit
    from hexmod.io import csv_float_roundtrip

    ang = csv_float_roundtrip(cells["Rayleigh_angle"], times=3)[keep.to_numpy()]
    exact = []
    for animal, idx in v.assign(_ang=ang).groupby("animal_id")["_ang"]:
        if len(idx) >= 5:
            exact.append((animal, stats.rayleigh_zar(np.deg2rad(6 * idx.to_numpy(float)))))
    holm_exact = stats.holm([r["p"] for _, r in exact])
    for (animal, r), ph in zip(exact, holm_exact):
        assert (r["R"], r["p"], ph) == (ref.loc[animal, "R6"], ref.loc[animal, "p_rayleigh"], ref.loc[animal, "p_holm"])
    below = sorted(round(x[3], 3) for x in rows if x[3] < 0.05)
    assert below == [0.019, 0.037, 0.046]
    assert round(float(min(p_holm)), 3) == 0.231
    assert round(stats.binomial_test(3, 12, 0.05, "greater"), 3) == 0.020
