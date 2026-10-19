"""Step 30: draw the figures from the result files.

Examples::

    python scripts/step30_figures.py                                   # every figure
    python scripts/step30_figures.py --figure Fig3 --figure SuppFig2   # selected figures
    python scripts/step30_figures.py --precomputed                     # simulation figures from the shipped results
    python scripts/step30_figures.py --list

Each figure is drawn by ``hexmod.figures.<module>.build(results_dir, out_dir)`` (``fig1`` to ``fig8``,
``supp1`` to ``supp5``) and written to ``results/figures/<Figure>/``: the figure as PDF (editable TrueType
text), SVG (text kept as text) and PNG (600 dpi), ``positions.csv`` and ``source_data/*.csv``. A figure is drawn
into a temporary folder that replaces the previous output only when the build succeeds.

The figures read only ``results/`` and, for the simulation results, the shipped ``results_precomputed/``
(used when ``results/sim`` lacks a file, or always with ``--precomputed``). A figure whose input files are
missing is reported as skipped, so that the simulation figures can be drawn without the recorded data.

Fonts: the figures are set in Arial. Set ``HEXMOD_FONT_DIR`` (or ``--font-dir``) to a folder with the Arial
TrueType files when Arial is not installed; otherwise Liberation Sans or DejaVu Sans is used, with a warning.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

from hexmod.figures import FIGURES, figure_name, module_name  # noqa: E402

STEP = "step30_figures"


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", action="append", default=None, metavar="NAME",
                   help="figure to draw (Fig1 to Fig8, SuppFig1 to SuppFig5, or 'all'); repeat or separate with "
                        "commas (default all)")
    p.add_argument("--results", default=None, metavar="DIR",
                   help="results folder (default HEXMOD_RESULTS or <repo>/results); figures go to "
                        "DIR/figures")
    p.add_argument("--precomputed", nargs="?", const="", default=None, metavar="DIR",
                   help="take the simulation results from the shipped results_precomputed/ (or DIR) even when "
                        "results/sim holds them")
    p.add_argument("--font-dir", default=None, metavar="DIR", help="folder with Arial TrueType files "
                   "(default HEXMOD_FONT_DIR)")
    p.add_argument("--strict", action="store_true",
                   help="count figures that cannot be drawn (module or input files missing) as failures")
    p.add_argument("--smoke", action="store_true", help="accepted for the smoke run (draws whatever the synthetic "
                   "results allow)")
    p.add_argument("--list", action="store_true", help="list the figures and their modules and exit")
    return p.parse_args(argv)


def selected_figures(values: Optional[List[str]]) -> List[str]:
    """Return the figures named on the command line, in the order of the paper."""
    if not values:
        return list(FIGURES)
    names = [v.strip() for value in values for v in value.split(",") if v.strip()]
    if any(n.lower() == "all" for n in names):
        return list(FIGURES)
    chosen = {figure_name(n) for n in names}
    return [f for f in FIGURES if f in chosen]


def module_available(figure: str) -> bool:
    """Return True when the module of a figure can be found."""
    return importlib.util.find_spec(module_name(figure)) is not None


def output_problems(folder: Path) -> List[str]:
    """Return problems with the files a figure build wrote (missing files, text-rule violations)."""
    from hexmod.figures.common import forbidden_characters

    problems = []
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    stems = {p.stem for p in files if p.suffix == ".pdf" and p.parent == folder}
    if not stems:
        problems.append("no PDF written")
    for stem in stems:
        for ext in (".svg", ".png"):
            if not (folder / (stem + ext)).is_file():
                problems.append(f"{stem}{ext} missing")
    if not (folder / "positions.csv").is_file():
        problems.append("positions.csv missing")
    for p in files:
        if p.suffix in (".svg", ".csv", ".json", ".txt"):
            found = forbidden_characters(p.read_text(encoding="utf-8", errors="replace"))
            if found:
                problems.append(f"{p.relative_to(folder).as_posix()}: {', '.join(found)}")
    return problems


def call_build(build, results_dir: Path, out_dir: Path) -> None:
    """Call a figure module's ``build(results_dir, out_dir)``."""
    build(results_dir, out_dir)


def build_one(figure: str, results_dir: Path) -> dict:
    """Draw one figure into a temporary folder and move it into place; return a status record."""
    import matplotlib as mpl
    from hexmod.figures import common

    figures_dir = results_dir / "figures"
    out_dir = common.figure_dir(results_dir, figure)
    tmp_dir = figures_dir / f".{figure}.building"
    assert out_dir.parent == figures_dir and tmp_dir.parent == figures_dir
    t0 = time.perf_counter()
    status = {"figure": figure, "module": module_name(figure)}
    try:
        module = importlib.import_module(module_name(figure))
    except ModuleNotFoundError as exc:
        if exc.name == module_name(figure):
            status.update(status="skipped", reason="module not available")
        else:
            status.update(status="failed", reason=f"import error: {exc}", traceback=traceback.format_exc())
        return status
    except Exception as exc:
        status.update(status="failed", reason=f"import error: {type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        return status
    if not hasattr(module, "build"):
        status.update(status="failed", reason=f"{module_name(figure)} has no build(results_dir, out_dir)")
        return status
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    common.reset_used_inputs()
    try:
        with mpl.rc_context():
            call_build(module.build, results_dir, tmp_dir)
        problems = output_problems(tmp_dir)
        if problems:
            raise RuntimeError("; ".join(problems))
    except common.MissingInput as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        status.update(status="skipped", reason=f"missing input: {exc}")
        if out_dir.exists():   # an output of an earlier build is kept; it may not match the current results
            status["stale_output"] = str(out_dir)
        return status
    except Exception as exc:  # report the failure and keep the previous output
        shutil.rmtree(tmp_dir, ignore_errors=True)
        status.update(status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return status
    finally:
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    status.update(status="ok", runtime_s=round(time.perf_counter() - t0, 2),
                  inputs=[str(p) for p in common.used_inputs()],
                  outputs=sorted(str(p) for p in out_dir.rglob("*") if p.is_file()))
    return status


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        figures = selected_figures(args.figure)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.list:
        for figure in FIGURES:
            present = "" if module_available(figure) else "   (not yet available)"
            print(f"{figure:<9} {module_name(figure)}{present}")
        return 0
    if args.results:
        os.environ["HEXMOD_RESULTS"] = str(Path(args.results).expanduser().resolve())
    if args.font_dir:
        os.environ["HEXMOD_FONT_DIR"] = str(Path(args.font_dir).expanduser().resolve())

    from hexmod.figures import common, style
    from hexmod.runlog import RunRecord

    config.guard_smoke_results(args.smoke)
    results_dir = config.results_dir()
    if args.precomputed is not None:
        common.set_precomputed(args.precomputed or None, prefer=True)
    font = style.house_font()
    print(f"results: {results_dir}; font: {font}; figures: {', '.join(figures)}", flush=True)

    statuses = []
    with RunRecord(STEP, workers=1, argv=argv if argv is not None else sys.argv[1:]) as rec:
        for figure in figures:
            st = build_one(figure, results_dir)
            statuses.append(st)
            line = f"[{figure}] {st['status']}"
            if st["status"] == "ok":
                line += f" in {st['runtime_s']:.1f} s"
                for p in st["inputs"]:
                    rec.add_input(Path(p))
                for p in st["outputs"]:
                    rec.add_output(Path(p))
            else:
                line += f": {st['reason']}"
                if st.get("stale_output"):
                    line += f" (the output of an earlier build in {st['stale_output']} is kept and may be stale)"
            print(line, flush=True)
            if st["status"] == "failed" and st.get("traceback"):
                print(st["traceback"], file=sys.stderr, flush=True)
        import matplotlib

        rec.note("font", font)
        rec.note("font_files", [Path(f).name for f in style.font_files(font)])
        rec.note("matplotlib", matplotlib.__version__)
        rec.note("precomputed_preferred", args.precomputed is not None)
        rec.note("figures", {st["figure"]: {k: v for k, v in st.items()
                                            if k in ("status", "reason", "runtime_s", "stale_output")}
                             for st in statuses})
    failed = [st["figure"] for st in statuses if st["status"] == "failed"]
    skipped = [st["figure"] for st in statuses if st["status"] == "skipped"]
    if args.strict:   # a figure that cannot be drawn counts as failed
        failed += skipped
    print(f"drawn: {sum(st['status'] == 'ok' for st in statuses)}; skipped: {', '.join(skipped) or 'none'}; "
          f"failed: {', '.join(failed) or 'none'}" + (" (--strict: skipped figures count as failed)"
                                                     if args.strict and skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
