"""Tests of hexmod.stats.

Unit tests use textbook examples (Student's sleep data, the plant-growth ANOVA, hand-computed
sums of squares, series expansions of the Rayleigh distribution).
"""

from __future__ import annotations

import math

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


def test_fisher_lee_permutation_reproducible():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 2 * np.pi, 50)
    b = rng.uniform(0, 2 * np.pi, 50)
    r1 = stats.fisher_lee_permutation(a, b, n_perm=500, seed=11)
    r2 = stats.fisher_lee_permutation(a, b, n_perm=500, seed=11)
    assert r1 == r2
    null = stats.fisher_lee_permutation_null(a, b, 500, np.random.default_rng(11))
    perm = np.random.default_rng(11).permutation(50)
    assert null[0] == pytest.approx(stats.fisher_lee(a, b[perm]), abs=1e-12)
    assert r1["p_one_sided_positive"] == (1 + np.sum(null >= r1["r"])) / 501
    strong = stats.fisher_lee_permutation(a, a + 0.1 * rng.standard_normal(50), n_perm=500, seed=12)
    assert strong["p_one_sided_positive"] == pytest.approx(1 / 501)
