"""The reference values of the statistics check (reference/expected_statistics.json).

- every printed reference value rounds to the number printed for it (or satisfies the printed
  bound, word or bracket asterisks), except the known differences between the text and the analysis
  listed in ``KNOWN_PRINTED_DIFFERENCES`` of step 40, which must still differ;
- the file holds summary statistics only: no single-cell values and no diagnostics of the rat
  analyses, and every rat statistic that the paper does not print is a constant of the analysis;
- entries are complete and in the documented order.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "reference" / "expected_statistics.json"
RAT_STEPS = {"step02_cohort", "step03_adaptation_parametric", "step04_adaptation_nonparametric",
             "step05_prospective_coding", "step06_visit_sequences", "step07_adaptation_timescales",
             "step08_methods_numbers", "step09_same_cell_links", "step10_conjunctive_cells",
             "step11_example_cells", "step12_rat_statistics"}
#: Rat statistics that are not printed but are constants of the analysis.
RAT_CONSTANTS = {"cohort.rule.min_samples_above_speed", "cohort.rule.speed_threshold_cm_s"}


def step40():
    name = "step40_check_statistics"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(scope="module")
def entries():
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert data["schema"] == "hexmod-expected-statistics/2"
    return data["statistics"]


def test_entries_are_complete_and_ordered(entries):
    fields = {"key", "kind", "location", "printed", "step", "tolerance", "value"}
    assert all(set(e) == fields for e in entries)
    keys = [(e["step"], e["key"]) for e in entries]
    assert keys == sorted(keys) and len({k for _, k in keys}) == len(keys)
    for e in entries:
        assert e["kind"] in ("printed", "reported")
        assert e["tolerance"] >= 0
        if e["kind"] == "printed":
            assert e["location"] and e["printed"], e["key"]


def test_printed_values_round_to_the_printed_numbers(entries):
    s40 = step40()
    known = s40.KNOWN_PRINTED_DIFFERENCES
    bad = []
    for e in entries:
        if e["kind"] != "printed":
            continue
        status, detail = s40.printed_agreement(e["key"], e["printed"], e["value"])
        expected = "differs" if e["key"] in known else "agrees"
        if status != expected:
            bad.append(f"{e['key']}: printed {e['printed']!r}, value {e['value']!r}: {status} ({detail})")
    assert not bad, "\n".join(bad)
    printed_keys = {e["key"] for e in entries if e["kind"] == "printed"}
    assert set(known) <= printed_keys


def test_printed_agreement_rules():
    pa = step40().printed_agreement
    assert pa("x.t", "t(29)=-23.6", -23.62570895799755)[0] == "agrees"
    assert pa("x.t", "t(29)=-23.6", -23.66)[0] == "differs"
    assert pa("x.df", "F(3,11811)", 11811)[0] == "agrees" and pa("x.df1", "F(3,11811)", 3)[0] == "agrees"
    assert pa("x.n", "3,938 grid cells", 3938)[0] == "agrees"
    assert pa("x.p", "p<0.001", 1e-20)[0] == "agrees" and pa("x.p", "p<0.001", 0.002)[0] == "differs"
    assert pa("x.high_s", "more than 10 s", 50.0)[0] == "agrees"
    assert pa("x.significant", "not significant", 0)[0] == "agrees"
    assert pa("x.p_holm", "***", 1e-5)[0] == "agrees" and pa("x.p_holm", "no bracket", 0.2)[0] == "agrees"
    assert pa("x.n", "Three", 3)[0] == "agrees" and pa("x.y", "no number here", 1.0)[0] == "not compared"
    assert pa("fig1.narrow.sixfold", "59%", 0.5929256133949723)[0] == "agrees"


def test_only_summary_statistics_of_the_recordings(entries):
    bad = []
    for e in entries:
        k = e["key"]
        if k.startswith(("table1.", "diagnostic.")):
            bad.append(f"{k}: single-cell values and diagnostics are not reference values")
        if e["step"] in RAT_STEPS and e["kind"] == "reported" and k not in RAT_CONSTANTS:
            bad.append(f"{k}: a rat statistic that the paper does not print")
    assert not bad, "\n".join(bad)
