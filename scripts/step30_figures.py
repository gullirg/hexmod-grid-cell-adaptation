"""Step 30: draw the figures from the result files.

Examples::

    python scripts/step30_figures.py                                   # every figure
    python scripts/step30_figures.py --figure Fig3 --figure SuppFig2   # selected figures
    python scripts/step30_figures.py --list

Each figure is drawn by ``hexmod.figures.<module>.build(results_dir, out_dir)`` (``fig2`` to ``fig8``,
``supp1`` to ``supp5``) and written to ``results/figures/<Figure>/``: the figure as PDF (editable TrueType
text), SVG (text kept as text) and PNG (600 dpi), and ``source_data/*.csv``. The folder of a figure is emptied
before the figure is drawn. Figure 1 and panel E of Figure 3 are schematics and are not drawn by the code.

The figures read only ``results/`` and, for the simulation results, the shipped ``results_precomputed/``
(used when ``results/sim`` lacks a file). A figure whose input files are missing is reported as skipped, so
that the simulation figures can be drawn without the recorded data.

Fonts: the figures are set in Arial. Set ``HEXMOD_FONT_DIR`` (or ``--font-dir``) to a folder with the Arial
TrueType files when Arial is not installed; otherwise Liberation Sans or DejaVu Sans is used, with a warning.
"""

from __future__ import annotations

import argparse
import importlib
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
                   help="figure to draw (Fig2 to Fig8, SuppFig1 to SuppFig5, or 'all'); repeat or separate with "
                        "commas (default all)")
    p.add_argument("--font-dir", default=None, metavar="DIR", help="folder with Arial TrueType files "
                   "(default HEXMOD_FONT_DIR)")
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


def build_one(figure: str, results_dir: Path) -> dict:
    """Draw one figure into its emptied folder; return a status record."""
    import matplotlib as mpl
    from hexmod.figures import common

    out_dir = common.figure_dir(results_dir, figure)
    t0 = time.perf_counter()
    status = {"figure": figure, "module": module_name(figure)}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    common.reset_used_inputs()
    try:
        module = importlib.import_module(module_name(figure))
        with mpl.rc_context():
            module.build(results_dir, out_dir)
    except common.MissingInput as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        status.update(status="skipped", reason=f"missing input: {exc}")
        return status
    except Exception as exc:  # report the failure and leave no partial output
        shutil.rmtree(out_dir, ignore_errors=True)
        status.update(status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return status
    finally:
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")
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
            print(f"{figure:<9} {module_name(figure)}")
        return 0
    if args.font_dir:
        os.environ["HEXMOD_FONT_DIR"] = str(Path(args.font_dir).expanduser().resolve())

    from hexmod.figures import style
    from hexmod.runlog import RunRecord

    results_dir = config.results_dir()
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
            print(line, flush=True)
            if st["status"] == "failed" and st.get("traceback"):
                print(st["traceback"], file=sys.stderr, flush=True)
        import matplotlib

        rec.note("font", font)
        rec.note("font_files", [Path(f).name for f in style.font_files(font)])
        rec.note("matplotlib", matplotlib.__version__)
        rec.note("figures", {st["figure"]: {k: v for k, v in st.items()
                                            if k in ("status", "reason", "runtime_s")}
                             for st in statuses})
    failed = [st["figure"] for st in statuses if st["status"] == "failed"]
    skipped = [st["figure"] for st in statuses if st["status"] == "skipped"]
    print(f"drawn: {sum(st['status'] == 'ok' for st in statuses)}; skipped: {', '.join(skipped) or 'none'}; "
          f"failed: {', '.join(failed) or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
