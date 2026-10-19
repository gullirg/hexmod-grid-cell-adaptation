"""Text and file rules for everything in the repository (see docs/CONVENTIONS.md).

1. No en dash, em dash or similar dash characters in any text file (the minus sign U+2212 is
   allowed). Use a hyphen, "to" or a comma instead.
2. No names or initials of the paper's authors or team in file names or file contents, except
   in CITATION.cff, in the README sections on citation and contact, and in the full references of
   the datasets (the "Citing the data" section of docs/DATA.md and the ``citation`` entries of
   ``hexmod.datasets``). Published work is otherwise cited as "First author et al. (year)".
3. No absolute or machine-specific paths and no references to unpublished material. Eight-digit
   numbers of the form of a date are reported; the two fixed integer seeds of the published
   analysis that have this form are allowed only as the named constants listed in
   ``SEED_CONSTANTS``.
4. No notebooks, pickles, raw-data files or font files are committed; no file above 25 MB.
5. Analysis modules (everything in src/hexmod outside hexmod.figures) do not import matplotlib,
   and no code refers to datasets that are not part of the study.

The names are stored as salted sha256 digests (``_digest``), so that this file passes its own checks
without listing them; a word, or a part of a word, is reported when its digest is in a list.
"""

from __future__ import annotations

import ast
import functools
import gzip
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


_SALT = "hexmod/text-rules/1:"


def _digest(text: str) -> str:
    """Salted sha256 digest (16 hex digits) of a name, the form in which the name lists are stored."""
    return hashlib.sha256((_SALT + text).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------------------------------ rules
DASHES = {"\u2012": "figure dash", "\u2013": "en dash", "\u2014": "em dash", "\u2015": "horizontal bar",
          "\u2e3a": "two-em dash", "\u2e3b": "three-em dash", "\ufe58": "small em dash"}

# Author and team names, matched case-insensitively anywhere in a word (digests of the lower-case names),
# and the lengths of these names.
NAME_SUBSTRINGS = frozenset({"0e924c783a85263c", "144b11754aaa1d89", "2a0a151e26f81e58", "4947818a8efa2524",
                             "4c9146c7590e6b09", "5e4238bae7d7fff7", "94aa3fe1d062d7cb", "f169d577ee206245"})
NAME_SUBSTRING_LENGTHS = (5, 6, 7, 9)
# names matched as whole words, case-sensitive
NAME_WORDS = frozenset({"2419fb589a2618e8", "791a1cf9befd449f"})
# A surname that is also part of a dataset key (case-insensitive): allowed only in that key (followed by 2012)
# and as "<name> et al.".
DATASET_SURNAME = frozenset({"a35d2e3bbe156b93"})
DATASET_SURNAME_LENGTHS = (5,)
DATASET_SURNAME_ALLOWED_AFTER = re.compile(r"(?i)(2012|\s+et\s+al\.)")
_M, _K = "M", "K"
NAME_PATTERNS = [
    re.compile("(?<![A-Z])" + _M + _K + "(?![A-Za-z])"),          # initials as a token or camel-case suffix
    re.compile("(?i)(?<![a-z0-9])" + _M + _K + "(?![a-z0-9])"),    # standalone, any case
    re.compile("(?i)(?<![a-z0-9])" + _M + _K + "_"),               # prefix of a name
]
# the repository address on GitHub is allowed
ALLOWED_SNIPPETS = ["gullirg/hexmod-grid-cell-adaptation"]

# eight-digit numbers of the form of a date
DATE_LIKE_NUMBER = re.compile(r"\b202[56](0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])\b")

# Seeds of the published analysis that DATE_LIKE_NUMBER would report. Each entry allows exactly one
# line in one file: the assignment of a named integer constant (``NAME = <digits>``, optionally followed by a
# comment). Every other date-like number is still reported.
SEED_CONSTANTS = [
    ("src/hexmod/rat/timescales.py", "PUBLISHED_BOOTSTRAP_SEED"),     # animal bootstrap of Supp Fig 4
    ("src/hexmod/figures/fig2.py", "PUBLISHED_DRAW_ORDER_SEED"),      # drawing order of the dots of Fig 2D
]

# absolute paths under these top-level folders; a relative path that contains one of these names
# (for example data/tmp/x.mat) is allowed (written in parts so that this file passes its own check)
ABSOLUTE_ROOTS = ["/" + d + "/" for d in ("home", "Users", "private", "tmp", "mnt", "Volumes", "opt")]
NOTEBOOK_SUFFIX = "." + "ipynb"
PATH_PATTERNS = [
    re.compile(r"(?i)(?<![\w.~-])" + re.escape(root)) for root in ABSOLUTE_ROOTS
] + [
    re.compile(r"(?<![\w.~])~/[A-Za-z_]"),                    # home-relative paths
    re.compile(r"(?<![A-Za-z])[A-Z]:\\\\?(Users|Documents|Program)"),  # Windows drive paths
    DATE_LIKE_NUMBER,                                          # eight-digit numbers of the form of a date
    re.compile(r"\b202[56]-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b"),   # calendar dates
    re.compile(r"(?i)\bblock[ _]?[0-9]{2}\b"),               # "block NN" references
    re.compile("(?i)" + re.escape(NOTEBOOK_SUFFIX) + r"\b"),    # notebooks
]

FORBIDDEN_SUFFIXES = {NOTEBOOK_SUFFIX, ".pkl", ".pickle", ".mat", ".ttf", ".otf", ".woff", ".woff2", ".h5", ".hdf5"}
MAX_BYTES = 25 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".json", ".csv", ".tsv",
                 ".cff", ".svg", ".html", ".sh", ".bib", ".tex", ".rst", ".in"}
EXCLUDED_DIRS = {".git", "data", "results", "__pycache__", ".pytest_cache", "build", "dist", ".cache"}
NAME_EXEMPT_FILES = {"CITATION.cff"}
#: Files of ignore patterns: they name the file types that are never committed.
PATH_EXEMPT_FILES = {".gitignore"}
README_EXEMPT_HEADINGS = re.compile(r"(?i)^#+\s.*(contact|cite|citation|citing)")
#: Markdown files whose citation and contact sections are exempt from the name rule (besides READMEs).
SECTION_EXEMPT_FILES = {"docs/DATA.md"}
#: Python files whose ``"citation": (...)`` entries (full dataset references) are exempt from the name rule.
CITATION_EXEMPT_FILES = {"src/hexmod/datasets.py"}
_CITATION_ENTRY = re.compile(r'"citation":\s*\((.*?)\),', re.S)
PAPER_TITLE = "Firing rate adaptation in grid cells produces macroscopic hexadirectional modulation"
# Datasets that are not part of the study, matched case-insensitively anywhere in a word (digests of the
# lower-case names), and the lengths of these names.
DATASETS_OUTSIDE_STUDY = frozenset({"5a66ba92f6382042", "d0386f68765fa0fa"})
DATASETS_OUTSIDE_STUDY_LENGTHS = (6, 7)


# ---------------------------------------------------------------------------------------- helpers
def repository_files(root: Path = REPO) -> List[Path]:
    """Files that would be committed: git's tracked and untracked-but-not-ignored files."""
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root,
                             capture_output=True, check=True)
        files = [root / p for p in out.stdout.decode("utf-8").split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.endswith(".egg-info")]
            files.extend(Path(dirpath) / f for f in filenames)
    keep = []
    for f in files:
        rel = f.relative_to(root)
        if any(p in EXCLUDED_DIRS or p.endswith(".egg-info") for p in rel.parts):
            continue
        if f.is_file():
            keep.append(f)
    return sorted(keep)


def read_text(path: Path):
    """Return the text of a file, or None for binary files."""
    name = path.name
    if name.endswith(".gz"):
        if path.stat().st_size > 50 * 1024 * 1024:
            return None
        try:
            return gzip.decompress(path.read_bytes()).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
    if path.suffix.lower() in TEXT_SUFFIXES or path.suffix == "" or name.startswith("."):
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def npz_strings(path: Path) -> List[str]:
    """String values stored in an npz file (for the dash check)."""
    out = []
    try:
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                a = z[key]
                if a.dtype.kind in "US":
                    out.extend(str(v) for v in a.ravel())
                out.append(key)
    except Exception:
        pass
    return out


def find_dashes(text: str) -> List[Tuple[int, str]]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for ch, label in DASHES.items():
            if ch in line:
                hits.append((lineno, label))
    return hits


def _strip_allowed(line: str) -> str:
    for snippet in ALLOWED_SNIPPETS:
        line = line.replace(snippet, "")
    return line


_LETTER_RUN = re.compile(r"[A-Za-z]+")
_WORD = re.compile(r"\w+")


@functools.lru_cache(maxsize=None)
def _run_matches(run: str, digests: frozenset, lengths: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    """(start, end) of the parts of a lower-case letter run whose digest is in ``digests``."""
    return tuple((i, i + n) for n in lengths for i in range(len(run) - n + 1) if _digest(run[i:i + n]) in digests)


def substring_hits(line: str, digests: frozenset, lengths: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """(start, end) of the parts of the words of a line whose lower-case digest is in ``digests``."""
    return [(m.start() + i, m.start() + j) for m in _LETTER_RUN.finditer(line)
            for i, j in _run_matches(m.group().lower(), digests, lengths)]


def word_hits(line: str, digests: frozenset) -> List[Tuple[int, int]]:
    """(start, end) of the whole words of a line (case-sensitive) whose digest is in ``digests``."""
    return [m.span() for m in _WORD.finditer(line) if _digest(m.group()) in digests]


def name_hits(text: str, substrings=NAME_SUBSTRINGS, lengths=NAME_SUBSTRING_LENGTHS, words=NAME_WORDS,
              surname=DATASET_SURNAME) -> List[Tuple[int, str]]:
    """(line number, text) of the author and team names in a text, at most one per rule and line.

    The name lists can be replaced (``test_rule_patterns_work`` checks the rules with invented names).
    """
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = _strip_allowed(line)
        surname_spans = [(i, j) for i, j in substring_hits(line, surname, DATASET_SURNAME_LENGTHS)
                         if not DATASET_SURNAME_ALLOWED_AFTER.match(line, j)]
        for spans in (substring_hits(line, substrings, lengths), word_hits(line, words), surname_spans):
            if spans:
                i, j = min(spans)
                hits.append((lineno, line[i:j]))
        for pat in NAME_PATTERNS:
            m = pat.search(line)
            if m:
                hits.append((lineno, m.group(0)))
    return hits


def readme_name_hits(text: str, **lists) -> List[Tuple[int, str]]:
    """Name hits in a Markdown file, ignoring the citation and contact sections and the citation line."""
    hits = []
    exempt = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("#"):
            exempt = bool(README_EXEMPT_HEADINGS.match(line))
            continue
        if exempt or PAPER_TITLE in line:
            continue
        for _, word in name_hits(line, **lists):
            hits.append((lineno, word))
    return hits


def without_citation_entries(text: str) -> str:
    """Blank the ``"citation": (...)`` entries of a Python source, keeping its line numbers."""
    return _CITATION_ENTRY.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def seed_constant_lines(text: str, names) -> List[int]:
    """Line numbers of the assignments ``NAME = <integer>`` of the given constant names."""
    if not names:
        return []
    pattern = re.compile(r"^(" + "|".join(map(re.escape, names)) + r") = [0-9]+(\s+#.*)?$")
    return [lineno for lineno, line in enumerate(text.splitlines(), 1) if pattern.match(line)]


def path_hits(text: str, seed_constants=()) -> List[Tuple[int, str]]:
    """Private references in a text; date-like numbers are allowed on the lines of the given seed constants."""
    allowed = set(seed_constant_lines(text, seed_constants))
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = _strip_allowed(line)
        for pat in PATH_PATTERNS:
            if pat is DATE_LIKE_NUMBER and lineno in allowed:
                continue
            m = pat.search(line)
            if m:
                hits.append((lineno, m.group(0)))
    return hits


def _seed_constants_of(path: Path) -> List[str]:
    rel = path.relative_to(REPO).as_posix()
    return [name for file, name in SEED_CONSTANTS if file == rel]


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


FILES = repository_files()


# ------------------------------------------------------------------------------------------ tests
def test_repository_has_files():
    assert any(p.name == "pyproject.toml" for p in FILES)


def test_no_dash_characters():
    bad = []
    for path in FILES:
        text = read_text(path)
        if text is None and path.suffix == ".npz":
            text = "\n".join(npz_strings(path))
        if text is None:
            continue
        bad.extend(f"{_rel(path)}:{ln}: {label}" for ln, label in find_dashes(text))
    assert not bad, "dash characters found (use a hyphen, 'to' or a comma):\n" + "\n".join(bad[:50])


def test_no_names_in_file_names():
    bad = [_rel(p) for p in FILES if name_hits(_rel(p))]
    assert not bad, "file names with author or team names:\n" + "\n".join(bad)


def test_no_names_in_file_contents():
    bad = []
    for path in FILES:
        if path.name in NAME_EXEMPT_FILES:
            continue
        text = read_text(path)
        if text is None:
            continue
        rel = _rel(path)
        if rel in CITATION_EXEMPT_FILES:
            text = without_citation_entries(text)
        if path.name.lower().startswith("readme") or rel in SECTION_EXEMPT_FILES:
            hits = readme_name_hits(text)
        else:
            hits = name_hits(text)
        bad.extend(f"{_rel(path)}:{ln}: {word!r}" for ln, word in hits)
    assert not bad, "author or team names found:\n" + "\n".join(bad[:50])


def test_no_private_paths_or_history():
    bad = []
    for path in FILES:
        if path.name in NAME_EXEMPT_FILES or path.name in PATH_EXEMPT_FILES:
            continue
        text = read_text(path)
        if text is None:
            continue
        bad.extend(f"{_rel(path)}:{ln}: {snippet!r}" for ln, snippet in path_hits(text, _seed_constants_of(path)))
    assert not bad, "absolute paths or private references found:\n" + "\n".join(bad[:50])


def test_seed_constants_are_each_used_once():
    """Every entry of SEED_CONSTANTS names a file of the repository and allows exactly one assignment in it."""
    files = {_rel(p): p for p in FILES}
    bad = []
    for file, name in SEED_CONSTANTS:
        if file not in files:
            bad.append(f"{file}: not in the repository")
            continue
        n = len(seed_constant_lines(files[file].read_text(encoding="utf-8"), [name]))
        if n != 1:
            bad.append(f"{file}: {name} assigned on {n} lines (expected 1)")
    assert not bad, "\n".join(bad)


def test_no_forbidden_or_large_files():
    bad = [f"{_rel(p)}: forbidden file type" for p in FILES if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    bad += [f"{_rel(p)}: {p.stat().st_size} bytes" for p in FILES if p.stat().st_size > MAX_BYTES]
    assert not bad, "\n".join(bad)


def test_analysis_modules_do_not_import_matplotlib():
    bad = []
    pkg = REPO / "src" / "hexmod"
    for path in sorted(pkg.rglob("*.py")):
        if "figures" in path.relative_to(pkg).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "matplotlib" or n.startswith("matplotlib.") or n == "pylab" for n in names):
                bad.append(f"{_rel(path)}:{node.lineno}")
    assert not bad, "analysis modules must not import matplotlib:\n" + "\n".join(bad)


def test_no_code_for_datasets_outside_the_study():
    """No Python file names a dataset outside the study (the full dataset references are not code)."""
    bad = []
    for path in FILES:
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        if _rel(path) in CITATION_EXEMPT_FILES:
            text = without_citation_entries(text)
        for ln, line in enumerate(text.splitlines(), 1):
            if substring_hits(line, DATASETS_OUTSIDE_STUDY, DATASETS_OUTSIDE_STUDY_LENGTHS):
                bad.append(f"{_rel(path)}:{ln}")
    assert not bad, "\n".join(bad)


def test_python_files_parse():
    bad = []
    for path in FILES:
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                bad.append(f"{_rel(path)}: {exc}")
    assert not bad, "\n".join(bad)


def _citation_names() -> Tuple[List[str], List[str]]:
    """Family and given names of the authors listed in CITATION.cff (a public source of names to test with)."""
    text = (REPO / "CITATION.cff").read_text(encoding="utf-8").split("preferred-citation:")[0]
    return re.findall(r"family-names: (\S+)", text), re.findall(r"given-names: (\S+)", text)


def test_rule_patterns_work():
    """The checks catch what they should and leave ordinary text alone."""
    assert find_dashes("a \u2013 b") and find_dashes("a \u2014 b") and not find_dashes("a \u2212 b - c")
    assert name_hits("def ttestplot" + _M + _K + "2(x): pass")
    assert name_hits("import " + _M + _K + "_core")
    assert name_hits("load_" + "m" + "k" + "_cohort()")
    assert not name_hits("os.makedirs(p); Path(p).mkdir(); MKL_NUM_THREADS=1; Doeller et al. 2010")
    assert not name_hits("https://github.com/" + ALLOWED_SNIPPETS[0])
    assert name_hits("user " + ALLOWED_SNIPPETS[0].split("/")[0])     # the handle outside the address
    # the digest rules, with invented names
    test_lists = dict(substrings=frozenset({_digest("quillon")}), lengths=(7,), words=frozenset({_digest("Zed")}),
                      surname=frozenset({_digest("vexor")}))
    assert name_hits("by Quillon", **test_lists) and name_hits("load_quillon_cells()", **test_lists)
    assert name_hits("QuillonCohort; xquillonx", **test_lists) == [(1, "Quillon")]
    assert name_hits("by Zed", **test_lists) and not name_hits("Zedd; zed; Zed_x", **test_lists)
    assert name_hits("def test_vexor_theta_phase():", **test_lists) and name_hits("Vexor C", **test_lists)
    assert not name_hits("def test_vexor2012_theta_phase():  # Vexor et al. (2012)", **test_lists)
    assert not name_hits("quillo; Ze d; vexo", **test_lists)
    assert readme_name_hits("# Title\n\nCode by Quillon\n", **test_lists)
    assert not readme_name_hits("# Contact\n\nQuillon\n", **test_lists)
    assert not readme_name_hits("# Data\n\n## Citing the data\n\nQuillon\n", **test_lists)
    # the stored lists catch the authors' names of CITATION.cff (all family names longer than two letters)
    family, given = _citation_names()
    assert family and all(name_hits("by " + name) for name in family if len(name) > 2)
    assert sum(bool(name_hits("by " + name)) for name in given) >= 4
    assert path_hits("see " + ABSOLUTE_ROOTS[0] + "user/data") and path_hits("cd " + "~/" + "data")
    assert path_hits("in " + ABSOLUTE_ROOTS[2] + "x/y") and path_hits("'" + ABSOLUTE_ROOTS[3] + "x'")
    assert not path_hits("data" + ABSOLUTE_ROOTS[2] + "y.mat; " + ABSOLUTE_ROOTS[2][1:] + "x.mat")
    assert not path_hits("results" + ABSOLUTE_ROOTS[3] + "x.csv")
    assert path_hits("run of " + "2025" + "0101") and path_hits("from " + "block" + " 12")
    seed = "SEED = " + "2025" + "0101"
    assert not path_hits(seed + "  # seed of the published panel", ["SEED"])
    assert path_hits(seed, ["OTHER"]) and path_hits("x = f(" + "2025" + "0101)", ["SEED"])
    assert path_hits(seed + "\nrng(" + "2025" + "0101)", ["SEED"]) == [(2, "2025" + "0101")]
    cited = 'X = {"citation": (\n    "Quillon, N. (2012)."\n),\n "label": 1}\n'
    assert not name_hits(without_citation_entries(cited), **test_lists) and name_hits(cited, **test_lists)
    assert without_citation_entries(cited).count("\n") == cited.count("\n")
    assert not path_hits("results/rat/cohort/cells.csv; 4 blocks of 600 s; v1.2.3; 11,811")


@pytest.mark.parametrize("pattern", ["*.svg", "positions.csv", "*.csv"])
def test_figure_outputs_have_no_dashes(pattern):
    """Figure files written by step30 (when present under HEXMOD_RESULTS) contain no dashes."""
    from hexmod import config

    files = sorted((config.results_dir() / "figures").rglob(pattern))
    if not files:
        pytest.skip("no figure outputs found")
    bad = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        bad.extend(f"{path}:{ln}: {label}" for ln, label in find_dashes(text))
    assert not bad, "\n".join(bad[:50])
