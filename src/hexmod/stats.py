"""Statistical tests used for the printed statistics.

- one-sample, paired and two-sample t-tests and Cohen's d (``one_sample_t``, ``paired_t``,
  ``two_sample_t``);
- one-way ANOVA (``one_way_anova``) and a repeated-measures ANOVA written out by hand
  (``rm_anova``, with the paired post-hoc tests of ``rm_pairwise``);
- Holm correction (``holm``) and the exact binomial test (``binomial_test``);
- Rayleigh test with Zar's approximation of the p value and its critical resultant length
  (``rayleigh_zar``, ``zar_p``, ``rayleigh_r_crit``), the V test (``v_test``) and the
  Monte Carlo Rayleigh null used for chance bands (``rayleigh_chance``);
- the Fisher and Lee (1983) circular-circular correlation and its seeded cell-permutation test
  (``fisher_lee``, ``fisher_lee_permutation``).

Each function keeps the order of operations of the reference values
(``reference/expected_statistics.json``), because another order can change the last digit.
Results are returned as plain dicts of Python floats and ints so they can be written to JSON.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import optimize, stats

__all__ = [
    "one_sample_t", "paired_t", "two_sample_t", "cohens_d", "one_way_anova", "rm_anova",
    "rm_pairwise", "holm", "binomial_test", "zar_p", "rayleigh_zar", "rayleigh_r_crit", "v_test",
    "rayleigh_chance", "rayleigh_monte_carlo", "fisher_lee", "fisher_lee_permutation_null",
    "permutation_pvalues", "fisher_lee_permutation",
]


# --------------------------------------------------------------------------------------- t-tests
def cohens_d(x) -> float:
    """One-sample Cohen's d: mean divided by the standard deviation with ddof=1."""
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1))


def one_sample_t(x, popmean: float = 0.0) -> Dict[str, float]:
    """Two-sided one-sample t-test against ``popmean`` after dropping NaN.

    Parameters
    ----------
    x : array_like
        Values (NaN are removed first).
    popmean : float
        Hypothesised mean (default 0).

    Returns
    -------
    dict
        ``n``, ``mean``, ``t``, ``df`` (n - 1), ``p`` (two-sided) and ``d`` (mean / SD with
        ddof=1; the effect size is computed relative to zero, as in the paper).
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    r = stats.ttest_1samp(x, popmean)
    return dict(n=int(len(x)), mean=float(x.mean()), t=float(r.statistic), df=int(len(x) - 1),
                p=float(r.pvalue), d=float(x.mean() / x.std(ddof=1)))


def paired_t(a, b) -> Dict[str, float]:
    """Two-sided paired t-test of ``a`` against ``b`` on the pairs where ``a - b`` is not NaN.

    Returns
    -------
    dict
        ``n``, ``t``, ``df`` (n - 1), ``p``, ``mean_a``, ``mean_b``, ``mean_diff`` and ``d``
        (mean difference over its SD with ddof=1).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~np.isnan(a - b)
    r = stats.ttest_rel(a[ok], b[ok])
    dd = a[ok] - b[ok]
    return dict(n=int(ok.sum()), t=float(r.statistic), df=int(ok.sum() - 1), p=float(r.pvalue),
                mean_a=float(a[ok].mean()), mean_b=float(b[ok].mean()), mean_diff=float(dd.mean()),
                d=float(dd.mean() / dd.std(ddof=1)))


def two_sample_t(a, b, equal_var: bool = True) -> Dict[str, float]:
    """Two-sided two-sample t-test (Student when ``equal_var``, Welch otherwise).

    The arrays are used as given (no NaN removal).

    Returns
    -------
    dict
        ``t``, ``df`` (n_a + n_b - 2 for Student; Welch-Satterthwaite for Welch), ``p``,
        ``n_a``, ``n_b``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    r = stats.ttest_ind(a, b, equal_var=equal_var)
    na, nb = len(a), len(b)
    if equal_var:
        df: float = int(na + nb - 2)
    else:
        va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
        df = float((va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)))
    return dict(t=float(r.statistic), df=df, p=float(r.pvalue), n_a=int(na), n_b=int(nb))


# ------------------------------------------------------------------------------------- ANOVAs
def one_way_anova(*groups) -> Dict[str, float]:
    """One-way ANOVA (``scipy.stats.f_oneway``).

    Returns
    -------
    dict
        ``F``, ``df1`` (k - 1), ``df2`` (N - k), ``p``.
    """
    fo = stats.f_oneway(*groups)
    k = len(groups)
    return dict(F=float(fo.statistic), df1=int(k - 1), df2=int(sum(map(len, groups)) - k), p=float(fo.pvalue))


def rm_anova(X) -> Dict[str, object]:
    """One-way repeated-measures ANOVA on a cells x conditions array.

    Rows with any NaN are removed. The sums of squares are computed directly:
    treatment ``n * sum_j (mean_j - grand)^2``, subjects ``k * sum_i (mean_i - grand)^2`` and
    error = total - treatment - subjects; ``F = (SS_tr / (k - 1)) / (SS_err / ((k - 1)(n - 1)))``.

    Parameters
    ----------
    X : array_like, shape (n, k)

    Returns
    -------
    dict
        ``F``, ``df1``, ``df2``, ``p``, ``n`` (rows used) and ``means`` (list of k column means).
    """
    X = np.asarray(X, float)
    ok = ~np.isnan(X).any(axis=1)
    X = X[ok]
    n, k = X.shape
    gm = X.mean()
    ss_tr = n * ((X.mean(0) - gm) ** 2).sum()
    ss_sub = k * ((X.mean(1) - gm) ** 2).sum()
    ss_err = ((X - gm) ** 2).sum() - ss_tr - ss_sub
    F = (ss_tr / (k - 1)) / (ss_err / ((k - 1) * (n - 1)))
    pF = float(stats.f.sf(F, k - 1, (k - 1) * (n - 1)))
    return dict(F=float(F), df1=int(k - 1), df2=int((k - 1) * (n - 1)), p=pF, n=int(n),
                means=[float(x) for x in X.mean(0)])


def rm_pairwise(X, labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Paired t-tests between every pair of columns (rows with any NaN removed).

    Parameters
    ----------
    X : array_like, shape (n, k)
    labels : sequence of str
        Column names; result keys are ``"<label_i> vs <label_j>"`` for i < j.

    Returns
    -------
    dict
        For each pair: ``t``, ``df``, ``p`` and ``d`` (mean difference over its SD, ddof=1).
    """
    X = np.asarray(X, float)
    X = X[~np.isnan(X).any(axis=1)]
    n = X.shape[0]
    out = {}
    for i, j in itertools.combinations(range(X.shape[1]), 2):
        r = stats.ttest_rel(X[:, i], X[:, j])
        dd = X[:, i] - X[:, j]
        out[f"{labels[i]} vs {labels[j]}"] = {"t": float(r.statistic), "df": int(n - 1), "p": float(r.pvalue),
                                              "d": float(dd.mean() / dd.std(ddof=1))}
    return out


# --------------------------------------------------------------------------- corrections, binomial
def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p values (same order as the input)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def binomial_test(k: int, n: int, p: float = 0.5, alternative: str = "two-sided") -> float:
    """Exact binomial test p value (``scipy.stats.binomtest``)."""
    return float(stats.binomtest(int(k), int(n), p, alternative=alternative).pvalue)


# ----------------------------------------------------------------------------------- Rayleigh
def zar_p(n: int, R: float) -> float:
    """Rayleigh test p value from Zar's approximation (not clipped at 1).

    ``p = exp(sqrt(1 + 4 n + 4 (n^2 - (n R)^2)) - (1 + 2 n))`` for n directions with resultant
    length R (Zar, Biostatistical Analysis).
    """
    Rn = n * R
    return math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - Rn * Rn)) - (1 + 2 * n))


def rayleigh_zar(ang_rad) -> Dict[str, float]:
    """Rayleigh test of uniformity with Zar's approximation.

    Parameters
    ----------
    ang_rad : array_like
        Angles in radians (for six-fold clustering pass ``6 * theta``).

    Returns
    -------
    dict
        ``n``, ``R`` (mean resultant length), ``mean_rad``, ``Z`` (n R^2) and ``p`` (clipped at 1).
    """
    a = np.asarray(ang_rad, float)
    n = len(a)
    C, S = np.cos(a).sum(), np.sin(a).sum()
    Rn = math.hypot(C, S)
    R = Rn / n
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - Rn * Rn)) - (1 + 2 * n))
    return {"n": int(n), "R": float(R), "mean_rad": float(math.atan2(S, C)), "Z": float(n * R * R),
            "p": float(min(1.0, p))}


def rayleigh_r_crit(n: int, alpha: float = 0.05) -> float:
    """Resultant length at which Zar's Rayleigh p value equals ``alpha`` for n directions."""
    return float(optimize.brentq(lambda R: zar_p(n, R) - alpha, 1e-9, 1 - 1e-12))


def v_test(ang_rad, mu: float = 0.0) -> Dict[str, float]:
    """V test for clustering toward the direction ``mu`` (radians).

    Returns
    -------
    dict
        ``n``, ``mean_cos``, ``u`` (sqrt(2 n) mean_cos), ``p_toward`` (normal upper tail) and
        ``p_away`` (lower tail).
    """
    a = np.asarray(ang_rad, float)
    n = len(a)
    V = float(np.cos(a - mu).mean())
    u = math.sqrt(2 * n) * V
    return {"n": int(n), "mean_cos": V, "u": float(u), "p_toward": float(stats.norm.sf(u)),
            "p_away": float(stats.norm.cdf(u))}


_RAYLEIGH_NULL: Dict[tuple, np.ndarray] = {}


def rayleigh_chance(n: int, ndraw: int = 100000, seed: int = 5600) -> np.ndarray:
    """Sorted resultant lengths of ``ndraw`` sets of n uniform directions.

    The generator is ``numpy.random.default_rng(seed + n)``; draws are made in blocks of
    ``max(1, 2000000 // n)`` sets. Results are cached per (n, ndraw, seed).
    """
    key = (int(n), int(ndraw), int(seed))
    if key not in _RAYLEIGH_NULL:
        rng = np.random.default_rng(seed + n)
        out = np.empty(ndraw)
        c = max(1, 2000000 // n)
        for s in range(0, ndraw, c):
            e = min(ndraw, s + c)
            a = rng.uniform(0, 2 * np.pi, (e - s, n))
            out[s:e] = np.abs(np.exp(1j * a).mean(1))
        _RAYLEIGH_NULL[key] = np.sort(out)
    return _RAYLEIGH_NULL[key]


def rayleigh_monte_carlo(R: float, n: int, ndraw: int = 100000, seed: int = 5600, max_n_mc: int = 60):
    """Rayleigh p value, chance mean and chance 95th percentile of R for n directions.

    For ``n <= max_n_mc`` all three come from ``rayleigh_chance(n)`` (p = (1 + #null >= R) /
    (ndraw + 1)); above it, p is Zar's approximation, the chance mean ``sqrt(pi / (4 n))`` and the
    95th percentile ``sqrt(-ln(0.05) / n)``. This is the rule that defines the chance band of
    Fig 2E.

    Returns
    -------
    tuple of float
        ``(p, chance_mean, chance_95)``.
    """
    if n <= max_n_mc:
        nul = rayleigh_chance(n, ndraw, seed)
        p = (1 + np.sum(nul >= R - 1e-12)) / (len(nul) + 1)
        return float(p), float(nul.mean()), float(np.quantile(nul, 0.95))
    Rn = n * R
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - Rn * Rn)) - (1 + 2 * n))
    return float(min(1.0, p)), math.sqrt(math.pi / (4 * n)), math.sqrt(-math.log(0.05) / n)


# ---------------------------------------------------------------- circular-circular correlation
def fisher_lee(a, b) -> float:
    """Fisher and Lee (1983) circular-circular correlation of angles ``a`` and ``b`` (radians).

    For directions on a 60-degree cycle pass the angles multiplied by 6.
    Returns NaN when a denominator term is zero.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    Aa, Bb, Cc, Dd = ca @ cb, sa @ sb, ca @ sb, sa @ cb
    E, F, G, Hh = np.cos(2 * a).sum(), np.sin(2 * a).sum(), np.cos(2 * b).sum(), np.sin(2 * b).sum()
    den = (n * n - E * E - F * F) * (n * n - G * G - Hh * Hh)
    return float(4 * (Aa * Bb - Cc * Dd) / math.sqrt(den)) if den > 0 else float("nan")


def fisher_lee_permutation_null(a, b, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Fisher-Lee coefficients after ``n_perm`` random permutations of ``b`` across cells.

    Each permutation is ``rng.permutation(n)``, drawn in sequence from ``rng``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    E, F, G, Hh = np.cos(2 * a).sum(), np.sin(2 * a).sum(), np.cos(2 * b).sum(), np.sin(2 * b).sum()
    den = math.sqrt((n * n - E * E - F * F) * (n * n - G * G - Hh * Hh))
    out = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        cbp = cb[p]
        sbp = sb[p]
        out[k] = 4 * ((ca @ cbp) * (sa @ sbp) - (ca @ sbp) * (sa @ cbp)) / den
    return out


def permutation_pvalues(null, obs: float) -> Dict[str, float]:
    """Permutation p values with the (1 + count) / (n + 1) rule.

    Returns
    -------
    dict
        ``p_one_sided_positive`` (null >= obs), ``p_one_sided_negative`` (null <= obs),
        ``p_two_sided`` (|null| >= |obs|) and ``null_sd``.
    """
    null = np.asarray(null, float)
    n = len(null)
    return {"p_one_sided_positive": float((1 + np.sum(null >= obs)) / (n + 1)),
            "p_one_sided_negative": float((1 + np.sum(null <= obs)) / (n + 1)),
            "p_two_sided": float((1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1)),
            "null_sd": float(np.std(null))}


def fisher_lee_permutation(a, b, n_perm: int = 20000, seed: Optional[int] = None,
                           rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Fisher-Lee correlation and its cell-permutation test.

    Parameters
    ----------
    a, b : array_like
        Angles in radians (already multiplied by 6 for 60-degree cycles), one per cell, in the
        canonical cell order.
    n_perm : int
        Number of permutations (20,000 for Fig 2D).
    seed : int, optional
        Seed of the generator, ``numpy.random.default_rng(seed)``.
    rng : numpy Generator, optional
        Generator to use instead of ``seed``.

    Returns
    -------
    dict
        ``r``, ``n`` and the entries of ``permutation_pvalues``.
    """
    if rng is None:
        if seed is None:
            raise ValueError("give either seed or rng")
        rng = np.random.default_rng(seed)
    r = fisher_lee(a, b)
    null = fisher_lee_permutation_null(a, b, n_perm, rng)
    return {"r": r, "n": int(len(np.asarray(a))), **permutation_pvalues(null, r)}
