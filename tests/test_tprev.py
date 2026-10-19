"""Time since the last field visit (t_prev) on a hand-made visit sequence: standard and gap-aware rules.

One field of radius 10 cm at the origin. The trajectory (50 Hz samples) is inside the field for the
samples [10, 21), [40, 51), [100, 111), [130, 141) and [700, 711) and 50 cm away otherwise. The time axis
jumps by 8 s between samples 119 and 120, as after a period of slow running removed by the speed filter.
"""

import numpy as np
import pytest

from hexmod.rat.fields import extract_field_entry

DT = 0.02
N = 750
VISITS = [(10, 21), (40, 51), (100, 111), (130, 141), (700, 711)]


def sequence():
    t = np.arange(N) * DT
    t[120:] += 8.0  # gap of 8.02 s between samples 119 and 120
    x = np.full(N, 50.0)
    for a, b in VISITS:
        x[a:b] = 0.0
    y = np.zeros(N)
    spikes = np.zeros(N, dtype=int)
    spikes[[12, 15, 44, 45, 46, 105, 133, 705]] = 1
    return x, y, t, spikes


def entry(rule, **kw):
    x, y, t, spikes = sequence()
    return t, extract_field_entry(x, y, t, np.array([0.0]), np.array([0.0]), np.array([10.0]), spikes, dt=DT,
                                  tprev_rule=rule, **kw)


def expected_intervals(t):
    """Interval from the previous exit (first sample outside) to the entry, plus one sample."""
    return np.array([np.nan] + [t[VISITS[k][0]] - t[VISITS[k - 1][1]] + DT for k in range(1, len(VISITS))])


def test_visits_found():
    t, r = entry("standard")
    ev = r["ev_df"]
    assert list(ev.t_enter) == [t[a] for a, b in VISITS]
    assert list(ev.t_exit) == [t[b] for a, b in VISITS]
    np.testing.assert_allclose(ev.dur_eff, 11 * DT)
    # in-field rate of a visit: spikes / duration (the first visit is not evaluated)
    np.testing.assert_allclose(ev.FR, [0.0, 3 / (11 * DT), 1 / (11 * DT), 1 / (11 * DT), 1 / (11 * DT)])


def test_standard_rule():
    t, r = entry("standard")
    ev, tp, inf = r["ev_df"], r["t_prev_vec"], r["in_field"]
    want = expected_intervals(t)
    want[0] = 0.0
    np.testing.assert_allclose(ev.t_prev, want, rtol=0, atol=1e-12)
    np.testing.assert_allclose(want[1:], [0.40, 1.00, 8.40, 11.20], atol=1e-9)
    # before and during the first visit: 0
    assert np.all(tp[:21] == 0)
    # between visits: time since the previous exit plus one sample; no cap, gaps ignored
    for k in range(1, len(VISITS)):
        a_prev, b_prev = VISITS[k - 1]
        a, b = VISITS[k]
        np.testing.assert_allclose(tp[b_prev:a], t[b_prev:a] - t[b_prev] + DT, rtol=0, atol=1e-12)
        assert np.all(tp[a:b] == ev.t_prev[k])
    assert tp[129] == pytest.approx(t[129] - t[111] + DT)  # ramp across the 8 s gap
    assert tp[699] == pytest.approx(t[699] - t[141] + DT)  # 11.2 s, not capped
    # after the last visit: 0
    assert np.all(tp[711:] == 0)
    # the first visit is not marked as in-field; later visits are split into halves
    assert np.all(inf[10:21] == 0)
    assert list(inf[40:51]) == [1] * 6 + [2] * 5
    assert "t_prev_valid" not in ev.columns and "t_prev_valid_vec" not in r


def test_gap_aware_rule():
    t, r = entry("gap_aware")
    ev, tp, inf = r["ev_df"], r["t_prev_vec"], r["in_field"]
    want = expected_intervals(t)
    np.testing.assert_allclose(ev.t_prev, want, rtol=0, atol=1e-12, equal_nan=True)
    assert list(ev.spans_recording_gap) == [False, False, False, True, False]
    assert list(ev.t_prev_valid) == [False, True, True, False, True]
    # undefined before the second visit
    assert np.all(np.isnan(tp[:21]))
    # valid intervals as in the standard rule
    np.testing.assert_allclose(tp[21:40], t[21:40] - t[21] + DT, rtol=0, atol=1e-12)
    assert np.all(tp[40:51] == ev.t_prev[1])
    np.testing.assert_allclose(tp[51:100], t[51:100] - t[51] + DT, rtol=0, atol=1e-12)
    # the interval before the fourth visit spans the 8 s gap: undefined, also during that visit
    assert np.all(np.isnan(tp[111:141]))
    # long interval: ramp and visit value capped at 10 s
    ramp = t[141:700] - t[141] + DT
    np.testing.assert_allclose(tp[141:700], np.minimum(ramp, 10.0), rtol=0, atol=1e-12)
    assert tp[699] == 10.0 and np.all(tp[700:711] == 10.0)
    assert np.all(np.isnan(tp[711:]))
    np.testing.assert_array_equal(r["t_prev_valid_vec"], np.isfinite(tp))
    # every visit, including the first, is marked as in-field
    assert list(inf[10:21]) == [1] * 6 + [2] * 5


def test_gap_aware_options():
    t, r = entry("gap_aware", tprev_cap=None, recording_gap_threshold=10.0)
    ev, tp = r["ev_df"], r["t_prev_vec"]
    assert not ev.spans_recording_gap.any()  # the 8 s gap is below the 10 s threshold
    assert list(ev.t_prev_valid) == [False, True, True, True, True]
    assert tp[705] == pytest.approx(11.20)  # no cap
    np.testing.assert_allclose(tp[111:130], t[111:130] - t[111] + DT, rtol=0, atol=1e-12)


def test_rules_agree_on_valid_visits():
    t, standard = entry("standard")
    _, gap = entry("gap_aware", tprev_cap=None)
    ok = gap["ev_df"].t_prev_valid.to_numpy()
    np.testing.assert_array_equal(standard["ev_df"].t_prev.to_numpy()[ok], gap["ev_df"].t_prev.to_numpy()[ok])


def test_unknown_rule():
    x, y, t, spikes = sequence()
    with pytest.raises(ValueError):
        extract_field_entry(x, y, t, np.array([0.0]), np.array([0.0]), np.array([10.0]), spikes, tprev_rule="capped")
