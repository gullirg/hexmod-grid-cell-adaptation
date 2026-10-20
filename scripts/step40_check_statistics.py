"""Step 40: check every statistic against its reference value and the number printed in the paper.

Example::

    python scripts/step40_check_statistics.py

The step merges the statistics files ``results/stats/*.json`` written by the other steps
(one number per key; a key written by two steps is an error) and compares every key with the
entry of the same key in ``reference/expected_statistics.json``. A statistic passes when
``abs(computed - expected) <= tolerance`` (the tolerances are explained in reference/README.md).
A few numbers printed in the paper follow from other statistics by a fixed formula (Cohen's d of
a one-sample or paired t-test, percentages of counts, optimal speeds for a given grid scale); when
no step writes them, they are computed here from the statistics they depend on (``DERIVED``
below) and checked like the others. The step also checks that every computed value of a printed
statistic rounds to the number printed in the paper (``printed_agreement``); the known exceptions
are listed with their reasons in ``KNOWN_PRINTED_DIFFERENCES``. This check is reported, not
enforced (the animal bootstrap of Supp Fig 4 can move the last printed digit on another machine);
``tests/test_reference_statistics.py`` enforces it for the reference values.

Writes to ``results/stats/``:

- ``statistics_check.csv``: ``key``, ``step``, ``printed``, ``expected``, ``computed``,
  ``abs_diff``, ``tolerance``, ``status`` (``pass``, ``fail``, ``missing``: in the reference but not
  computed, ``no reference``: computed but not in the reference);
- ``statistics_check.md``: the same as a readable report, failures first, with the printed values
  that do not round to their computed values and the sources of the simulation results.

The exit status is 1 when a statistic fails or is missing.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hexmod import io as hio  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step40_check_statistics"
SCHEMA = "hexmod-expected-statistics/2"
STATUS_ORDER = {"fail": 0, "missing": 1, "pass": 2, "no reference": 3}
#: Order of the statistics files in the reports (pipeline order); other files follow by name.
STEP_ORDER = ("step02_cohort", "step03_adaptation_parametric", "step04_adaptation_nonparametric",
              "step05_prospective_coding", "step06_visit_sequences", "step07_adaptation_timescales",
              "step08_methods_numbers", "step09_same_cell_links", "step10_conjunctive_cells",
              "step11_example_cells", "step12_rat_statistics", "step27_simulation_statistics", STEP)

#: Adaptation time constant of the simulations [s] (the optimal running speeds of the Discussion).
SIM_TAU_S = 2.0


# ------------------------------------------------------------------------------ derived statistics
def _cohen_d_one_sample(t: float, df: float) -> float:
    """Cohen's d (mean over SD, ddof 1) of a one-sample or paired t-test: t / sqrt(n)."""
    return t / math.sqrt(df + 1.0)


def _cohen_dz_ddof0(t: float, df: float) -> float:
    """Cohen's d_z of a paired t-test with the SD of the differences computed with ddof 0: t / sqrt(n - 1).

    This is the effect size of Fig 7A (also ``cohens_dz_ddof0`` in the figure's statistics table);
    with ddof 1, as for the other effect sizes of the paper, it would be t / sqrt(n).
    """
    return t / math.sqrt(df)


def _six_largest_animals(results: Path) -> Optional[int]:
    """Cells of the six animals with the most cells among the Fig 2D cells."""
    path = results / "rat" / "conjunctive" / "fig2d_cells.csv"
    if not path.is_file():
        return None
    table = hio.read_csv(path, usecols=["animal_id"])
    counts = table["animal_id"].value_counts(sort=True)
    return int(counts.iloc[:6].sum())


def _vonmises_sixfold_pct(sd_deg: float) -> float:
    """Six-fold component [percent of the mean] of a von Mises curve of circular SD ``sd_deg``."""
    from hexmod.sim import analytic

    return 100.0 * float(analytic.sixfold_from_sd(sd_deg))


def _max_sd_of_conjunctive_deg() -> float:
    """Largest circular SD [deg] of a tuning curve whose Rayleigh vector length exceeds the threshold."""
    from hexmod.rat.conjunctive import CONJUNCTIVE_MIN_R

    return math.degrees(math.sqrt(-2.0 * math.log(CONJUNCTIVE_MIN_R)))


#: key, statistics it is computed from, function of those values, description.
DERIVED: List[Tuple[str, Tuple[str, ...], Callable[..., float], str]] = [
    ("fig2b.pct_misaligned", ("fig2b.n_misaligned", "fig2b.n_cells"), lambda m, n: 100.0 * m / n,
     "misaligned conjunctive cells in percent"),
    ("fig2e.expected_below_0_05", ("fig2e.n_animals",), lambda n: 0.05 * n,
     "animals expected below p = 0.05 by chance"),
    ("fig3f.shorter.d", ("fig3f.shorter.t", "fig3f.shorter.df"), _cohen_d_one_sample, "Cohen's d, t / sqrt(n)"),
    ("fig3f.longer.d", ("fig3f.longer.t", "fig3f.longer.df"), _cohen_d_one_sample, "Cohen's d, t / sqrt(n)"),
    ("fig3f.paired.d", ("fig3f.paired.t", "fig3f.paired.df"), _cohen_d_one_sample, "Cohen's d, t / sqrt(n)"),
    ("fig7a.small.d", ("fig7a.small.t", "fig7a.small.df"), _cohen_dz_ddof0,
     "Cohen's d_z (paired, SD with ddof 0), t / sqrt(n - 1)"),
    ("fig7a.large.d", ("fig7a.large.t", "fig7a.large.df"), _cohen_dz_ddof0,
     "Cohen's d_z (paired, SD with ddof 0), t / sqrt(n - 1)"),
    ("supp5.cosine_width_vonmises_sixfold_pct", ("fig1.broad.circular_sd_deg",), _vonmises_sixfold_pct,
     "six-fold component of a von Mises curve as wide as cosine tuning"),
    ("supp5.max_conjunctive_sd_deg", (), _max_sd_of_conjunctive_deg,
     "widest tuning (circular SD) compatible with the conjunctive threshold"),
    ("discussion.optimal_speed_8m_grid_m_s", ("fig8.v_opt_scale_per_tau",), lambda f: f * 8.0 / SIM_TAU_S,
     "optimal running speed for an 8 m grid, tau 2 s"),
    ("discussion.optimal_speed_4m_grid_m_s", ("fig8.v_opt_scale_per_tau",), lambda f: f * 4.0 / SIM_TAU_S,
     "optimal running speed for a 4 m grid, tau 2 s"),
]
#: Derived statistics computed from result files rather than from other statistics.
DERIVED_FROM_FILES: List[Tuple[str, Callable[[Path], Optional[float]], str]] = [
    ("fig2d.n_cells_six_largest_animals", _six_largest_animals,
     "Fig 2D cells of the six animals with the most cells (coloured in the figure)"),
]


def derived_statistics(values: Dict[str, Any], results: Path) -> Dict[str, Any]:
    """Derived statistics that no step wrote, computed from ``values`` and the result files."""
    out: Dict[str, Any] = {}
    for key, inputs, func, _ in DERIVED:
        if key in values or not all(k in values for k in inputs):
            continue
        out[key] = float(func(*[values[k] for k in inputs]))
    for key, func, _ in DERIVED_FROM_FILES:
        if key in values:
            continue
        value = func(results)
        if value is not None:
            out[key] = value
    return out


# ------------------------------------------------------------------------------- printed values
#: Printed statistics whose computed value does not round to the printed number, and why.
KNOWN_PRINTED_DIFFERENCES: Dict[str, str] = {
    "methods.running_speed_cm_s": "the Methods print 19 cm/s; the mean running speed is 19.5 cm/s",
    "methods.shift_max_ms": "the text gives shifts up to +400 ms; the largest shift analysed is +380 ms "
                            "(40 shifts of 20 ms from -400 ms)",
    "supp3.large.rate_across_cells.shape": "the Supp Fig 3C legend prints shape=3.3; the fit gives 3.35",
    "supp4.bootstrap.tau_fast_high_s": "the Supp Fig 4 legend prints 0.56 s; this bootstrap gives 0.568 s",
}
#: Statistics stored in another unit than they are printed in: value as printed.
PRINTED_TRANSFORMS: Dict[str, Callable[[float], float]] = {
    "fig1.narrow.sixfold": lambda v: 100.0 * v,                  # fraction, printed in percent
    "fig1.broad.sixfold": lambda v: 100.0 * v,
    "methods.ratio_0_1s_to_above_3s": lambda v: 100.0 * (1.0 - v),   # ratio, printed as a reduction in percent
}
_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10}
_NUMBER = re.compile(r"(?<![\d.,])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\d.,])[-+]?\d+(?:\.\d+)?")
_GROUP = re.compile(r"(?<=[A-Za-z\u00b2])\(([^)]*)\)")         # t(3937), F(2, 4), r(3936)
_INEQUALITY = re.compile(r"(<|>|\u2264|\u2265|more than|less than)\s*([-+]?\d+(?:\.\d+)?)")
_STARS = {"***": (0.0, 0.001), "**": (0.001, 0.01), "*": (0.01, 0.05), "no bracket": (0.05, math.inf)}


def _rounds_to(value: float, number: str) -> bool:
    text = number.replace(",", "")
    decimals = len(text.split(".")[1]) if "." in text else 0
    rounded = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    return float(rounded) == float(text)


def printed_agreement(key: str, printed: str, value: Any) -> Tuple[str, str]:
    """Whether a value rounds to the number printed for it.

    Parameters
    ----------
    key : str
        Statistic key (selects ``PRINTED_TRANSFORMS`` and, for degrees of freedom, the numbers in
        brackets such as ``t(3937)``).
    printed : str
        The text as printed (``reference/expected_statistics.json``).
    value : number
        Computed or reference value.

    Returns
    -------
    status : str
        ``"agrees"`` (a number of the text equals the value rounded to that number's decimals, or
        the value satisfies a printed bound such as p < 0.001, a significance word or the asterisks
        of a drawn bracket), ``"differs"``, or ``"not compared"`` (no number in the text).
    detail : str
        The printed number or bound that agrees, or the numbers that were tried.
    """
    if not printed or not _is_number(value) or math.isnan(float(value)):
        return "not compared", ""
    v = float(PRINTED_TRANSFORMS.get(key, float)(float(value)))
    text = printed.replace("\u2212", "-").strip()
    low = text.lower()
    if low in ("significant", "not significant"):
        ok = v == (1.0 if low == "significant" else 0.0)
        return ("agrees" if ok else "differs"), low
    if low.startswith("significantly") and key.split(".")[-1] == "p":   # a p value described in words
        return ("agrees" if v < 0.05 else "differs"), "p < 0.05"
    if low in _STARS:
        lo, hi = _STARS[low]
        return ("agrees" if lo <= v < hi else "differs"), f"drawn mark {text!r}"
    is_df = key.split(".")[-1] in ("df", "df1", "df2") or key.endswith("_df")
    candidates = []
    for m in _GROUP.finditer(text):
        if is_df:
            candidates += re.findall(r"\d+", m.group(1))
    body = _GROUP.sub(" ", text)
    candidates += _NUMBER.findall(body)
    candidates += [str(n) for w, n in _WORDS.items() if re.search(rf"\b{w}\b", low)]
    for number in candidates:
        if _rounds_to(v, number):
            return "agrees", number
    for op, number in _INEQUALITY.findall(body):
        bound = float(number)
        if (op in ("<", "less than") and v < bound) or (op == "\u2264" and v <= bound) \
                or (op in (">", "more than") and v > bound) or (op == "\u2265" and v >= bound):
            return "agrees", f"{op} {number}"
    if not candidates:
        return "not compared", ""
    return "differs", ", ".join(candidates)


def printed_check(table: pd.DataFrame) -> pd.DataFrame:
    """Printed statistics whose computed value does not round to the printed number."""
    rows = []
    for _, r in table.iterrows():
        if r["kind"] != "printed" or not r["printed"] or r["computed"] is None:
            continue
        status, detail = printed_agreement(r["key"], r["printed"], r["computed"])
        if status == "differs":
            rows.append({"key": r["key"], "printed": r["printed"], "computed": r["computed"],
                         "known": r["key"] in KNOWN_PRINTED_DIFFERENCES,
                         "reason": KNOWN_PRINTED_DIFFERENCES.get(r["key"], "not a known difference")})
    table = pd.DataFrame(rows, columns=["key", "printed", "computed", "known", "reason"])
    table["known"] = table["known"].astype(bool)
    return table


def simulation_sources(results: Path) -> List[str]:
    """Where the simulation results behind steps 25 and 27 came from (their run records)."""
    out = []
    for step, note in (("step25_simulation_collect", "source"), ("step27_simulation_statistics", "sources")):
        path = results / "logs" / f"{step}.json"
        if not path.is_file():
            continue
        try:
            value = hio.read_json(path).get("notes", {}).get(note)
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            value = "; ".join(f"{k}: {v}" for k, v in sorted(value.items()))
        if value:
            out.append(f"{step}: {value}")
    return out


# ------------------------------------------------------------------------------------- statistics
def statistics_files(stats_dir: Path) -> List[Path]:
    """Statistics files of the steps (``stats/*.json``), in pipeline order."""
    files = [p for p in stats_dir.glob("*.json") if p.stem != STEP]
    rank = {s: i for i, s in enumerate(STEP_ORDER)}
    return sorted(files, key=lambda p: (rank.get(p.stem, len(rank)), p.name))


def collect_statistics(results: Path) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """Merge ``results/stats/*.json``; return values, the step of each key and duplicated keys."""
    values: Dict[str, Any] = {}
    steps: Dict[str, str] = {}
    duplicates: List[str] = []
    for path in statistics_files(results / "stats"):
        data = hio.read_json(path)
        if not isinstance(data, dict):
            raise SystemExit(f"error: {path} does not hold a flat JSON object")
        for key, value in data.items():
            if key in values:
                duplicates.append(f"{key} ({steps[key]} and {path.stem})")
                continue
            values[key] = value
            steps[key] = path.stem
    for key, value in derived_statistics(values, results).items():
        values[key] = value
        steps[key] = STEP
    return values, steps, duplicates


def load_reference(path: Path) -> List[Dict[str, Any]]:
    """Entries of ``reference/expected_statistics.json``."""
    data = hio.read_json(path)
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"error: {path} is not a {SCHEMA} file")
    return list(data["statistics"])


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def compare_value(computed: Any, expected: Any, tolerance: float) -> Tuple[Optional[float], bool]:
    """Absolute difference and whether ``computed`` agrees with ``expected`` within ``tolerance``."""
    if isinstance(computed, bool):
        computed = int(computed)
    if not (_is_number(computed) and _is_number(expected)):
        return None, computed == expected
    c, e = float(computed), float(expected)
    if math.isnan(c) or math.isnan(e):
        return None, math.isnan(c) and math.isnan(e)
    if isinstance(computed, (int, np.integer)) and isinstance(expected, (int, np.integer)):
        diff = abs(int(computed) - int(expected))
        return float(diff), diff <= tolerance
    diff = abs(c - e)
    return diff, diff <= tolerance


def check_statistics(values: Dict[str, Any], steps: Dict[str, str],
                     reference: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """One row per reference entry and per computed key without a reference."""
    rows = []
    seen = set()
    for entry in reference:
        key = entry["key"]
        seen.add(key)
        row = {"key": key, "step": entry.get("step", ""), "printed": entry.get("printed", ""),
               "expected": entry["value"], "computed": None, "abs_diff": None,
               "tolerance": entry["tolerance"], "status": "missing",
               "kind": entry.get("kind", ""), "location": entry.get("location", "")}
        if key in values:
            diff, ok = compare_value(values[key], entry["value"], float(entry["tolerance"]))
            row.update(computed=values[key], abs_diff=diff, status="pass" if ok else "fail")
            if steps.get(key) and steps[key] != row["step"]:
                row["step"] = f"{steps[key]} (reference: {row['step']})"
        rows.append(row)
    for key in values:
        if key not in seen:
            rows.append({"key": key, "step": steps[key], "printed": "", "expected": None,
                         "computed": values[key], "abs_diff": None, "tolerance": None,
                         "status": "no reference", "kind": "", "location": ""})
    return pd.DataFrame(rows, columns=["key", "step", "printed", "expected", "computed", "abs_diff", "tolerance",
                                       "status", "kind", "location"])


def _fmt(value: Any, digits: int = 17) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_check_reports(table: pd.DataFrame, reference_path: Path, duplicates: Sequence[str],
                        stats_dir: Path, sources: Sequence[str] = ()) -> Tuple[Path, Path]:
    """Write ``statistics_check.csv`` and ``statistics_check.md``."""
    csv_cols = ["key", "step", "printed", "expected", "computed", "abs_diff", "tolerance", "status"]
    out = table[csv_cols].copy()
    for col in ("expected", "computed", "abs_diff", "tolerance"):
        out[col] = [_fmt(v) for v in out[col]]
    csv_path = hio.write_csv(out, stats_dir / "statistics_check.csv")

    counts = table["status"].value_counts().to_dict()
    lines = ["# Statistics check", ""]
    lines += [f"Reference: `reference/{reference_path.name}`; statistics: `results/stats/*.json`. "
              "A statistic passes when the absolute difference between the computed and the reference "
              "value is at most the tolerance.", "",
              "| Status | Statistics |", "|---|---|"]
    for status in ("pass", "fail", "missing", "no reference"):
        lines.append(f"| {status} | {counts.get(status, 0)} |")
    printed = table[table["kind"] == "printed"]
    lines += ["", f"Statistics printed in the paper: {len(printed)}, of which "
              f"{int((printed['status'] == 'pass').sum())} pass.", ""]
    differ = printed_check(table)
    lines += [f"Printed statistics whose computed value does not round to the printed number: {len(differ)} "
              f"({int(differ['known'].sum())} known differences between the text and the analysis).", ""]
    if len(differ):
        lines += ["| Key | Printed | Computed | Reason |", "|---|---|---|---|"]
        for _, r in differ.iterrows():
            lines.append(f"| `{r['key']}` | {_md_escape(r['printed'])} | {_fmt(r['computed'], 6)} "
                         f"| {_md_escape(r['reason'])} |")
        lines.append("")
    if sources:
        lines += ["Sources of the simulation results:", ""] + [f"- {s}" for s in sources] + [""]
    if duplicates:
        lines += ["Keys written by more than one step (the first file is used):", ""]
        lines += [f"- {d}" for d in duplicates] + [""]
    bad = table[table["status"].isin(["fail", "missing"])]
    if len(bad):
        lines += ["## Failed or missing", "",
                  "| Key | Printed | Reference | Computed | Difference | Tolerance | Status |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in bad.iterrows():
            lines.append(f"| `{r['key']}` | {_md_escape(r['printed'])} | {_fmt(r['expected'])} | {_fmt(r['computed'])} "
                         f"| {_fmt(r['abs_diff'], 3)} | {_fmt(r['tolerance'], 3)} | {r['status']} |")
        lines.append("")
    lines += ["## All statistics", ""]
    table = table.assign(_step=[s.split(" ")[0] for s in table["step"]])
    rank = {s: i for i, s in enumerate(STEP_ORDER)}
    for step in sorted(table["_step"].unique(), key=lambda s: (rank.get(s, len(rank)), s)):
        part = table[table["_step"] == step]
        lines += [f"### {step}", "",
                  "| Key | Kind | Printed | Reference | Computed | Difference | Tolerance | Status |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            lines.append(f"| `{r['key']}` | {r['kind']} | {_md_escape(r['printed'])} | {_fmt(r['expected'])} "
                         f"| {_fmt(r['computed'])} | {_fmt(r['abs_diff'], 3)} | {_fmt(r['tolerance'], 3)} "
                         f"| {r['status']} |")
        lines.append("")
    md_path = stats_dir / "statistics_check.md"
    with hio.atomic_path(md_path, suffix=".md") as tmp:
        Path(tmp).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


# ------------------------------------------------------------------------------------------- main
def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    parse_args(argv)
    results = config.results_dir()
    stats_dir = results / "stats"
    if not stats_dir.is_dir():
        raise SystemExit(f"error: {stats_dir} not found; run the analysis steps first")
    reference_path = config.repo_root() / "reference" / "expected_statistics.json"
    failed = False
    with RunRecord(STEP, argv=argv) as rec:
        reference = load_reference(reference_path)
        rec.add_input(reference_path)
        values, steps, duplicates = collect_statistics(results)
        for path in statistics_files(stats_dir):
            rec.add_input(path)
        table = check_statistics(values, steps, reference)
        sources = simulation_sources(results)
        for path in write_check_reports(table, reference_path, duplicates, stats_dir, sources):
            rec.add_output(path)
        counts = {s: int((table["status"] == s).sum()) for s in STATUS_ORDER}
        rec.note("statistics", counts)
        rec.note("derived", sorted(k for k, s in steps.items() if s == STEP))
        rec.note("simulation_sources", sources)
        failed = counts["fail"] > 0 or counts["missing"] > 0 or bool(duplicates)
        print(f"statistics: {counts['pass']} pass, {counts['fail']} fail, {counts['missing']} missing, "
              f"{counts['no reference']} without reference", flush=True)
        if duplicates:
            print(f"keys written by more than one step: {', '.join(duplicates)}", flush=True)
        differ = printed_check(table)
        known = differ["known"].to_numpy(dtype=bool)
        unknown = differ.loc[~known]
        rec.note("printed_values_differing", {"known": differ.loc[known, "key"].tolist(),
                                              "not_known": unknown["key"].tolist()})
        print(f"printed values that do not round to the computed value: {len(differ)} "
              f"({len(differ) - len(unknown)} known)" + (f"; not known: {', '.join(unknown['key'])}"
                                                        if len(unknown) else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
