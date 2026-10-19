"""Tests of hexmod.circular against definitions and the analytic values printed in the paper."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, special

from hexmod import circular


def test_wrap60_and_wrap():
    x = np.array([0.0, 29.9, 30.0, -30.0, 45.0, -45.0, 59.0, 61.0, 360.0, -90.0])
    w = circular.wrap60(x)
    assert w == pytest.approx([0.0, 29.9, -30.0, -30.0, -15.0, 15.0, -1.0, 1.0, 0.0, -30.0], abs=1e-12)
    assert np.all((w >= -30) & (w < 30))
    assert circular.wrap(190.0, 360.0) == pytest.approx(-170.0)
    assert circular.wrap60(x) == pytest.approx(circular.wrap(x, 60.0), abs=0)


def test_circular_sd():
    for sd in (5.0, 15.0, 43.3, 73.06):
        R = circular.resultant_from_sd(sd)
        assert circular.circular_sd_deg(R) == pytest.approx(sd, rel=1e-12)
    # cosine tuning has resultant length 1/2: circular SD 67.46 degrees
    assert circular.circular_sd_deg(0.5) == pytest.approx(67.46062504653993, rel=1e-13)
    assert round(float(circular.circular_sd_deg(0.5)), 2) == 67.46


def test_von_mises_values_printed_for_fig1():
    k = circular.kappa_from_sd(15.0)
    assert round(k, 4) == 15.1055
    assert special.i1e(k) / special.i0e(k) == pytest.approx(circular.resultant_from_sd(15.0), rel=1e-12)
    assert round(circular.sixfold_from_sd(15.0), 4) == 0.5929
    assert circular.sixfold_component(k) == circular.sixfold_from_sd(15.0)


def test_sixfold_values_printed_for_supp5():
    assert round(100 * circular.sixfold_from_sd(43.3), 1) == 0.3      # narrowest conjunctive cell
    assert round(100 * circular.sixfold_from_sd(73.06), 3) == 0.003   # median width


def test_harmonic_amplitude_is_fourier_coefficient():
    """2 I_k / I_0 equals the k-th cosine coefficient of exp(kappa cos theta) / I0 relative to its mean."""
    theta = np.linspace(0, 2 * np.pi, 200000, endpoint=False)
    for kappa in (0.5, 3.0, 15.1055):
        f = np.exp(kappa * (np.cos(theta) - 1.0))
        for k in (1, 2, 6):
            coef = 2 * (f * np.cos(k * theta)).mean() / f.mean()
            assert circular.harmonic_amplitude(k, kappa) == pytest.approx(coef, rel=1e-9, abs=1e-14)


def test_sixfold_implementations_agree():
    """Unscaled and scaled Bessel-function forms of the same formula agree to rounding error."""
    def unscaled(sd):
        r = circular.resultant_from_sd(sd)
        k = optimize.brentq(lambda q: special.i1(q) / special.i0(q) - r, 1e-9, 700.0)
        return 2.0 * special.iv(6, k) / special.i0(k)

    def scaled(sd):
        r = circular.resultant_from_sd(sd)
        k = optimize.brentq(lambda q: special.ive(1, q) / special.ive(0, q) - r, 1e-9, 5e4)
        return 2 * special.ive(6, k) / special.ive(0, k)

    sds = np.linspace(2, 95, 940)
    ours = np.array([circular.sixfold_from_sd(s) for s in sds])
    a = np.array([scaled(s) for s in sds])
    assert np.max(np.abs(ours - a)) < 1e-12
    b = np.array([unscaled(s) for s in sds[sds > 6]])   # the unscaled form overflows at small widths
    assert np.max(np.abs(ours[sds > 6] - b)) < 1e-12
