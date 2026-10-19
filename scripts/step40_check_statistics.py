"""Step 40: check every statistic against the published values.

Examples::

    python scripts/step40_check_statistics.py
    python scripts/step40_check_statistics.py --compare-frozen DIR

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
  that do not round to their computed values and the sources of the simulation results;
- with ``--compare-frozen DIR``: ``frozen_comparison.csv`` and ``frozen_comparison.md``, every file
  of ``DIR`` (a results folder in the layout of ``results/``) compared with the file of the
  same path here: integers, booleans and text exactly, floating-point values with the tolerances of
  ``FROZEN_TOLERANCES``; status ``equal`` (identical values), ``close`` (within tolerance),
  ``different`` or ``missing`` (not written here). Parts of the reference that this pipeline does not
  write (``NOT_COMPARED``) are left out of the comparison and named in the report.

The exit status is 1 when a statistic fails or is missing, or when a compared file is different or
missing; ``--smoke`` writes the reports for the synthetic smoke results and always ends with 0
(synthetic values are not expected to match the reference, so most statistics are reported as
failing in a smoke run).
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import pickletools
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
OWN_OUTPUTS = ("statistics_check.csv", "statistics_check.md", "frozen_comparison.csv", "frozen_comparison.md")
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
                        stats_dir: Path, smoke: bool, sources: Sequence[str] = ()) -> Tuple[Path, Path]:
    """Write ``statistics_check.csv`` and ``statistics_check.md``."""
    csv_cols = ["key", "step", "printed", "expected", "computed", "abs_diff", "tolerance", "status"]
    out = table[csv_cols].copy()
    for col in ("expected", "computed", "abs_diff", "tolerance"):
        out[col] = [_fmt(v) for v in out[col]]
    csv_path = hio.write_csv(out, stats_dir / "statistics_check.csv")

    counts = table["status"].value_counts().to_dict()
    lines = ["# Statistics check", ""]
    if smoke:
        lines += ["Smoke run on synthetic inputs: the values are not expected to match the reference.", ""]
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
    if not smoke:
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


# ------------------------------------------------------------------------------ frozen comparison
#: (glob pattern of the path, relative tolerance, absolute tolerance, reason). Floating-point values
#: are close when ``abs(a - b) <= atol + rtol * abs(b)``; the first matching pattern applies.
FROZEN_TOLERANCES: List[Tuple[str, float, float, str]] = [
    ("rat/timescales/bootstrap.json", 0.25, 0.1,
     "animal bootstrap: the two-exponential refits depend on their starting values (flat minimum of the slow "
     "time constant); measured differences up to 0.025 s in the intervals, 0.075 s in the median slow time "
     "constant, 0.49 in the delta AIC interval with shifted spikes and 0.008 in the shares"),
    ("rat/timescales/fits.json", 1e-3, 1e-9,
     "unweighted two-exponential fits (not printed) sit at a flat minimum of the fit"),
    ("stats/step07_adaptation_timescales.json", 0.0, 0.1,
     "bootstrap intervals (up to 0.017 s) and share (0.1 percentage points), as rat/timescales/bootstrap.json"),
    ("*", 1e-9, 1e-12, "last-digit differences (text round trips of the reference results, numerical libraries)"),
]
ALLOWED_PICKLE_MODULES = ("numpy", "builtins", "collections")
_PICKLE_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8", "STRING", "BINSTRING",
                      "SHORT_BINSTRING"}


def check_pickle(path: Path) -> None:
    """Refuse a pickle that names a global outside numpy and the Python builtins.

    The opcodes are read with ``pickletools`` before anything is loaded, following the strings and
    memo entries that name the module of each ``STACK_GLOBAL``.
    """
    memo: Dict[int, Optional[str]] = {}
    recent: List[Optional[str]] = []

    def refuse_unless_allowed(module: Optional[str], what: str) -> None:
        if module is None or module.split(".")[0] not in ALLOWED_PICKLE_MODULES:
            raise ValueError(f"{path.name}: refusing to load {what}")

    with open(path, "rb") as f:
        for op, arg, _ in pickletools.genops(f):
            name = op.name
            if name in _PICKLE_STRING_OPS:
                recent.append(arg if isinstance(arg, str) else None)
            elif name in ("BINGET", "LONG_BINGET", "GET"):
                recent.append(memo.get(int(arg)))
            elif name == "MEMOIZE":
                memo[len(memo)] = recent[-1] if recent else None
            elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
                memo[int(arg)] = recent[-1] if recent else None
            elif name in ("GLOBAL", "INST"):
                refuse_unless_allowed(str(arg).split(" ")[0], str(arg))
                recent.append(None)
            elif name == "STACK_GLOBAL":
                module = recent[-2] if len(recent) >= 2 else None
                refuse_unless_allowed(module, f"{module}.{recent[-1] if recent else ''}")
                recent.append(None)
            elif name != "FRAME":
                # any other opcode, including one that only pops (POP), breaks the pair of
                # strings that a following STACK_GLOBAL takes from the top of the stack
                recent.append(None)
            recent = recent[-4:]


def tolerance_for(rel: str) -> Tuple[float, float, str]:
    for pattern, rtol, atol, reason in FROZEN_TOLERANCES:
        if fnmatch.fnmatch(rel, pattern):
            return rtol, atol, reason
    return 0.0, 0.0, ""


class Diff:
    """Accumulates the comparison of two objects."""

    def __init__(self, rtol: float, atol: float):
        self.rtol, self.atol = rtol, atol
        self.equal = True
        self.close = True
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.notes: List[str] = []

    def note(self, text: str) -> None:
        if len(self.notes) < 12:
            self.notes.append(text)

    def mismatch(self, where: str, text: str) -> None:
        self.equal = self.close = False
        self.note(f"{where}: {text}")

    def floats(self, where: str, a: np.ndarray, b: np.ndarray) -> None:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            self.mismatch(where, f"shape {a.shape} against {b.shape}")
            return
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        if not np.array_equal(nan_a, nan_b):
            self.mismatch(where, f"{int((nan_a != nan_b).sum())} NaN positions differ")
            return
        ok = ~nan_a
        if not ok.any():
            return
        inf = np.isinf(a) | np.isinf(b)
        if inf.any():
            if not np.array_equal(a[inf], b[inf]):
                self.mismatch(where, "infinite values differ")
                return
            ok &= ~inf
        d = np.abs(a[ok] - b[ok])
        if d.size == 0 or not d.any():
            return
        self.equal = False
        m = float(d.max())
        self.max_abs = max(self.max_abs, m)
        scale = np.abs(b[ok])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(scale > 0, d / scale, np.inf)
        self.max_rel = max(self.max_rel, float(rel.max()))
        if (d > self.atol + self.rtol * scale).any():
            self.close = False
            self.note(f"{where}: max abs difference {m:.3g}")
        else:
            self.note(f"{where}: within tolerance (max abs {m:.3g})")

    def exact(self, where: str, a: np.ndarray, b: np.ndarray) -> None:
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            self.mismatch(where, f"shape {a.shape} against {b.shape}")
        elif not np.array_equal(a, b):
            self.mismatch(where, f"{int((a != b).sum())} of {a.size} values differ")

    def arrays(self, where: str, a: Any, b: Any) -> None:
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            if a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
                self.floats(where, a, b)
            else:
                self.mismatch(where, f"dtype {a.dtype} against {b.dtype}")
        elif a.dtype.kind in "biu" and b.dtype.kind in "biu":
            self.exact(where, a.astype(np.int64), b.astype(np.int64))
        else:
            self.exact(where, a.astype(str) if a.dtype.kind in "OUS" else a,
                       b.astype(str) if b.dtype.kind in "OUS" else b)

    def objects(self, where: str, a: Any, b: Any, extra_keys_ok: bool = True) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            ka, kb = set(a), set(b)
            for k in sorted(kb - ka, key=str):
                self.mismatch(f"{where}[{k!r}]", "missing here")
            if ka - kb and not extra_keys_ok:
                self.mismatch(where, f"{len(ka - kb)} keys not in the reference")
            elif ka - kb:
                self.note(f"{where}: {len(ka - kb)} keys not in the reference (not compared)")
            for k in [k for k in b if k in a]:
                self.objects(f"{where}[{k!r}]", a[k], b[k], extra_keys_ok)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                self.mismatch(where, f"length {len(a)} against {len(b)}")
                return
            if all(_is_number(x) or isinstance(x, bool) for x in list(a) + list(b)):
                self.arrays(where, np.asarray(a), np.asarray(b))
                return
            for i, (x, y) in enumerate(zip(a, b)):
                self.objects(f"{where}[{i}]", x, y, extra_keys_ok)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            self.arrays(where, a, b)
        elif isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            both = isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_))
            if not both or bool(a) != bool(b):
                self.mismatch(where, f"{a!r} against {b!r}")
        elif _is_number(a) and _is_number(b):
            if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                if int(a) != int(b):
                    self.mismatch(where, f"{a} against {b}")
            else:
                self.floats(where, np.array([a]), np.array([b]))
        elif a is None and b is None:
            return
        elif a != b:
            self.mismatch(where, f"{str(a)[:40]!r} against {str(b)[:40]!r}")

    def frames(self, a: pd.DataFrame, b: pd.DataFrame) -> None:
        missing = [c for c in b.columns if c not in a.columns]
        if missing:
            self.mismatch("columns", f"missing here: {missing[:6]}")
        extra = [c for c in a.columns if c not in b.columns]
        if extra:
            self.note(f"columns not in the reference (not compared): {extra[:6]}")
        if len(a) != len(b):
            self.mismatch("rows", f"{len(a)} against {len(b)}")
            return
        for c in [c for c in b.columns if c in a.columns]:
            x, y = a[c], b[c]
            num_x = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
            num_y = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
            if num_x and num_y:
                if pd.api.types.is_integer_dtype(x) and pd.api.types.is_integer_dtype(y):
                    self.exact(c, x.to_numpy(), y.to_numpy())
                else:
                    self.floats(c, x.to_numpy(dtype=float), y.to_numpy(dtype=float))
            else:
                xs = np.array(["" if pd.isna(v) else str(v) for v in x])
                ys = np.array(["" if pd.isna(v) else str(v) for v in y])
                self.exact(c, xs, ys)

    @property
    def status(self) -> str:
        return "equal" if self.equal else ("close" if self.close else "different")


def _load_any(path: Path) -> Any:
    name = path.name
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return hio.read_csv(path)
    if name.endswith(".json"):
        return hio.read_json(path)
    if name.endswith(".npy"):
        return hio.load_npy(path)
    if name.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if name.endswith(".pkl"):
        check_pickle(path)
        return hio.load_cache(path)
    return path.read_bytes()


#: Parts of a reference results folder that this pipeline does not write; they are left out of the
#: comparison and named in its report. ``keys``: entries of the session dicts of the preprocessed
#: recordings; ``sessions_outside_manifest``: sessions of the preprocessed recordings that are not in
#: ``inputs/manifests/sessions.csv`` (the analysis reads only the sessions of the manifest). Columns of a
#: reference table that the table of this run does not have are left out and named in the same way.
NOT_COMPARED: Dict[str, Any] = {
    "keys": {"theta_phases": "theta phase of the local field potential (no analysis uses it)"},
    "sessions_outside_manifest": True,
}


def _drop_keys(obj: Any, names: Sequence[str]) -> Tuple[Any, int]:
    """Copy of nested dicts without the entries named ``names``, and how many were dropped."""
    if not isinstance(obj, dict):
        return obj, 0
    out, n = {}, 0
    for k, v in obj.items():
        if k in names:
            n += 1
            continue
        out[k], m = _drop_keys(v, names)
        n += m
    return out, n


def reference_parts_compared(rel: str, ref: Any, diff: "Diff", ours: Any = None) -> Any:
    """The reference content without the parts that this run does not write (noted in ``diff``).

    These are the parts listed in ``NOT_COMPARED`` and, for a table, the columns that ``ours`` (the table of
    this run) does not have.
    """
    if isinstance(ref, pd.DataFrame):
        drop = [c for c in ref.columns if isinstance(ours, pd.DataFrame) and c not in ours.columns]
        if drop:
            diff.note(f"reference columns not written by this pipeline (not compared): {drop}")
            ref = ref.drop(columns=drop)
        return ref
    if not (rel.startswith("preprocessed/") and rel.endswith(".pkl") and isinstance(ref, dict)):
        return ref
    if NOT_COMPARED["sessions_outside_manifest"]:
        from hexmod.preprocessing import labels

        known = set(labels.read_sessions()["session_id"])
        outside = [s for sessions in ref.values() if isinstance(sessions, dict) for s in sessions if s not in known]
        if outside:
            diff.note(f"reference sessions not in the session manifest (not compared): {', '.join(map(str, outside))}")
            ref = {a: {s: v for s, v in sessions.items() if s in known} for a, sessions in ref.items()}
            ref = {a: sessions for a, sessions in ref.items() if sessions}
    ref, n = _drop_keys(ref, list(NOT_COMPARED["keys"]))
    if n:
        diff.note(f"reference entries not written by this pipeline (not compared): {', '.join(NOT_COMPARED['keys'])}"
                  f" ({n} sessions)")
    return ref


def compare_file(rel: str, ours: Path, theirs: Path) -> Dict[str, Any]:
    """Compare one result file with the reference file of the same path."""
    rtol, atol, reason = tolerance_for(rel)
    row = {"path": rel, "status": "missing", "max_abs_diff": None, "max_rel_diff": None,
           "rtol": rtol, "atol": atol, "detail": ""}
    if not ours.is_file():
        row["detail"] = "not written by this run"
        return row
    if hio.sha256_file(ours) == hio.sha256_file(theirs):
        row.update(status="equal", max_abs_diff=0.0, max_rel_diff=0.0, detail="identical files")
        return row
    diff = Diff(rtol, atol)
    try:
        a, b = _load_any(ours), _load_any(theirs)
        b = reference_parts_compared(rel, b, diff, ours=a)
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            diff.frames(a, b)
        elif isinstance(a, bytes) or isinstance(b, bytes):
            if a != b:
                diff.mismatch("bytes", "files differ")
        else:
            diff.objects("", a, b, extra_keys_ok=True)
    except Exception as exc:  # unreadable or refused file
        diff.mismatch("read", f"{type(exc).__name__}: {exc}")
    row.update(status=diff.status, max_abs_diff=diff.max_abs, max_rel_diff=diff.max_rel,
               detail="; ".join(diff.notes) + (f" [{reason}]" if diff.status == "close" and reason else ""))
    return row


def compare_frozen(results: Path, frozen: Path) -> pd.DataFrame:
    """Compare every file of ``frozen`` (except run records) with the file of the same path in ``results``."""
    rows = []
    for theirs in sorted(p for p in frozen.rglob("*") if p.is_file()):
        rel = theirs.relative_to(frozen).as_posix()
        if rel.startswith("logs/") or any(part.startswith(".") for part in Path(rel).parts) or rel.endswith(".md"):
            continue
        if rel.startswith("stats/") and Path(rel).name in OWN_OUTPUTS:
            continue
        rows.append(compare_file(rel, results / rel, theirs))
        print(f"  {rows[-1]['status']:<9} {rel}", flush=True)
    return pd.DataFrame(rows, columns=["path", "status", "max_abs_diff", "max_rel_diff", "rtol", "atol", "detail"])


def write_frozen_reports(table: pd.DataFrame, frozen: Path, stats_dir: Path) -> Tuple[Path, Path]:
    out = table.copy()
    for col in ("max_abs_diff", "max_rel_diff", "rtol", "atol"):
        out[col] = [_fmt(v, 6) for v in out[col]]
    csv_path = hio.write_csv(out, stats_dir / "frozen_comparison.csv")
    counts = table["status"].value_counts().to_dict()
    lines = ["# Comparison with reference results", "",
             f"Every file of the reference folder `{frozen.name}` compared with the file of the same path "
             "in this results folder. Integers, booleans and text must be identical; floating-point values "
             "are close when `abs(a - b) <= atol + rtol * abs(reference)`.", "",
             "| Status | Files |", "|---|---|"]
    for status in ("equal", "close", "different", "missing"):
        lines.append(f"| {status} | {counts.get(status, 0)} |")
    lines += ["", "| File | Status | Max abs difference | Max rel difference | rtol | atol | Detail |",
              "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| `{r['path']}` | {r['status']} | {_fmt(r['max_abs_diff'], 3)} | {_fmt(r['max_rel_diff'], 3)} "
                     f"| {_fmt(r['rtol'], 3)} | {_fmt(r['atol'], 3)} | {_md_escape(r['detail'])} |")
    md_path = stats_dir / "frozen_comparison.md"
    with hio.atomic_path(md_path, suffix=".md") as tmp:
        Path(tmp).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


# ------------------------------------------------------------------------------------------- main
def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", default=None, metavar="FILE",
                   help="expected statistics (default reference/expected_statistics.json)")
    p.add_argument("--compare-frozen", nargs="?", const="", default=None, metavar="DIR",
                   help="also compare every file of DIR (layout of results/; default HEXMOD_FROZEN) "
                        "with this results folder")
    p.add_argument("--smoke", action="store_true", help="report only (synthetic results); exit status 0")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config.guard_smoke_results(args.smoke)
    results = config.results_dir()
    stats_dir = results / "stats"
    if not stats_dir.is_dir():
        if args.smoke:
            print(f"smoke run: {stats_dir} not found, nothing to check")
            return 0
        raise SystemExit(f"error: {stats_dir} not found; run the analysis steps first")
    reference_path = Path(args.reference) if args.reference else config.repo_root() / "reference" / \
        "expected_statistics.json"
    failed = False
    with RunRecord(STEP, argv=argv) as rec:
        reference = load_reference(reference_path)
        rec.add_input(reference_path)
        values, steps, duplicates = collect_statistics(results)
        for path in statistics_files(stats_dir):
            rec.add_input(path)
        table = check_statistics(values, steps, reference)
        sources = simulation_sources(results)
        for path in write_check_reports(table, reference_path, duplicates, stats_dir, args.smoke, sources):
            rec.add_output(path)
        counts = {s: int((table["status"] == s).sum()) for s in STATUS_ORDER}
        rec.note("statistics", counts)
        rec.note("derived", sorted(k for k, s in steps.items() if s == STEP))
        rec.note("simulation_sources", sources)
        failed = counts["fail"] > 0 or counts["missing"] > 0 or bool(duplicates)
        print(f"statistics: {counts['pass']} pass, {counts['fail']} fail, {counts['missing']} missing, "
              f"{counts['no reference']} without reference"
              + (" (smoke run: synthetic values are not expected to match the reference)" if args.smoke else ""),
              flush=True)
        if duplicates:
            print(f"keys written by more than one step: {', '.join(duplicates)}", flush=True)
        if not args.smoke:
            differ = printed_check(table)
            known = differ["known"].to_numpy(dtype=bool)
            unknown = differ.loc[~known]
            rec.note("printed_values_differing", {"known": differ.loc[known, "key"].tolist(),
                                                  "not_known": unknown["key"].tolist()})
            print(f"printed values that do not round to the computed value: {len(differ)} "
                  f"({len(differ) - len(unknown)} known)" + (f"; not known: {', '.join(unknown['key'])}"
                                                            if len(unknown) else ""), flush=True)

        if args.compare_frozen is not None:
            frozen = Path(args.compare_frozen) if args.compare_frozen else config.frozen_dir()
            if frozen is None or not frozen.is_dir():
                raise SystemExit("error: --compare-frozen needs a folder (or HEXMOD_FROZEN)")
            print(f"comparing with {frozen}", flush=True)
            ftable = compare_frozen(results, frozen)
            for path in write_frozen_reports(ftable, frozen, stats_dir):
                rec.add_output(path)
            fcounts = {s: int((ftable["status"] == s).sum()) for s in ("equal", "close", "different", "missing")}
            rec.note("frozen_comparison", fcounts)
            print("frozen comparison: " + ", ".join(f"{v} {k}" for k, v in fcounts.items()), flush=True)
            failed = failed or fcounts["different"] > 0 or fcounts["missing"] > 0
    if args.smoke:
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
