"""Tests of the closed-form results in hexmod.sim.analytic (Fig 1 legend, Fig 8B and 8C, Supp Fig 5)."""

from __future__ import annotations

import numpy as np
import pytest

from hexmod.sim import analytic


def test_von_mises_values_printed_for_fig1():
    """Circular SD 15 deg: kappa 15.1055 and six-fold component 2 I6 / I0 = 0.5929."""
    k = analytic.kappa_from_sd(15.0)
    assert round(k, 4) == 15.1055
    assert round(analytic.sixfold_component(k), 4) == 0.5929
    assert round(analytic.sixfold_from_sd(15.0), 4) == 0.5929


def test_sixfold_curve():
    sds = analytic.SIXFOLD_SDS
    assert len(sds) == 940 and sds[0] == 2 and sds[-1] == 95
    y = analytic.sixfold_curve(sds[::47])
    assert np.all(np.diff(y) < 0)
    assert y[0] == pytest.approx(analytic.sixfold_from_sd(2.0))


def test_optimal_speed():
    """Aligned minus misaligned proximity peaks at v = (sqrt 3 - 1) / ln(sqrt 3) * scale / tau."""
    assert analytic.SPEED_OPT_FACTOR == pytest.approx(1.3326827218660933, rel=1e-15)
    assert f"{analytic.optimal_speed(30, 2):.1f}" == "20.0"
    assert f"{analytic.optimal_speed(70, 2):.1f}" == "46.6"
    v = np.linspace(1, 300, 299001)
    for scale, tau in ((30, 2.0), (70, 2.0), (40, 1.0)):
        d = analytic.proximity_difference(scale, v, tau)
        assert v[np.argmax(d)] == pytest.approx(analytic.optimal_speed(scale, tau), abs=2e-3)
        assert d.max() == pytest.approx(analytic.max_proximity_difference(), rel=1e-9)


def test_proximity_curves():
    v = analytic.FIG8BC_SPEEDS
    assert v[0] == 5 and v[-1] == 149 and len(v) == 73
    for scale in analytic.FIG8BC_SCALES:
        a = analytic.proximity_aligned(scale, v, 2)
        m = analytic.proximity_misaligned(scale, v, 2)
        assert np.all((0 < m) & (m < a) & (a < 1))
        assert np.all(np.diff(a) > 0) and np.all(np.diff(m) > 0)
        assert a == pytest.approx(np.exp(-scale / (v * 2.0)), rel=1e-14)
        assert m == pytest.approx(np.exp(-np.sqrt(3) * scale / (v * 2.0)), rel=1e-14)
        assert analytic.proximity_difference(scale, v, 2) == pytest.approx(a - m, abs=1e-15)
