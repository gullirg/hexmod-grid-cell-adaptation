"""Step 21: the two pools of simulated grid cells.

Examples::

    python scripts/step21_simulation_pools.py                      # small pool rebuilt, large pool from inputs/
    python scripts/step21_simulation_pools.py --regenerate-large   # also rebuild the large pool (about 5 min)

Small module (grid scale 40 vm): 100,000 cells built with ``hexmod.sim.pool.set_gridcells_pool``
(``module_param("small")``, base seed 0; about 6 minutes on one core). Its column checksums
(``hexmod.sim.io_pool.column_checksums``) are compared with ``inputs/simulation/pool_checksums.json``;
they are equal when the pool is the one of the paper. These checksums are taken over values rounded
to six significant digits (``significant_digits`` in that file), because builds of the numerical
libraries can differ in the last digit of a few values (an OpenBLAS and an MKL build of numpy give
gridness values that differ by up to 7e-16 in 38 of the 100,000 cells).

Large module (grid scale 70 vm): copied from ``inputs/simulation/large_module_pool_70cm.npz`` and
checked against the same file of checksums. With ``--regenerate-large`` it is rebuilt with
``hexmod.sim.pool_large`` instead; the rebuilt pool agrees with the shipped one to within
rounding (measured: field centres 3e-14 vm, peak rates 2e-16 Hz, gridness 1e-11; peaks,
radii and redraw counts exactly), so it is compared with tolerances that leave room for other machines
(field centres 1e-12 vm, peak rates 1e-14 Hz, gridness 1e-9; other columns exactly). The orientation estimate
``gridori_est``, which no step uses, can differ by a few degrees and is only reported. Steps 22 to 24
read the pools written here, so after ``--regenerate-large`` their large-module results can differ
from the published ones in the last digits.

Writes ``sim/pools/small_pool_40cm.npz``, ``sim/pools/large_pool_70cm.npz`` (layout of
``hexmod.sim.io_pool``) and ``sim/pools/pool_checksums.json`` (one entry per pool, as written by
``io_pool.pool_summary``, with checksums of the exact values; step 25 uses them to check that the
participant files were computed from these pools). A pool that does not match the reference
checksums or tolerances is written as ``<name>.failed.npz`` instead (any earlier file of the
standard name is removed, so that steps 22 to 24 cannot use it), and the step ends with exit
status 1.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

import numpy as np  # noqa: E402

from hexmod import io as hio  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step21_simulation_pools"
N_CELLS = 100000
BASE_SEED = 0
SMALL_FILE = "small_pool_40cm.npz"
LARGE_FILE = "large_pool_70cm.npz"
SMALL_KEY = "small_module_pool_40cm"
LARGE_KEY = "large_module_pool_70cm"

#: Tolerances of a rebuilt large pool against the shipped one (absolute; columns not listed are
#: compared exactly, ``gridori_est`` is reported only).
LARGE_TOLERANCES = {"centers": 1e-12, "peakFR_jitter": 1e-14, "gridness": 1e-9}
LARGE_REPORT_ONLY = ("gridori_est",)


def shipped_pool_path() -> Path:
    """The shipped large-module pool."""
    return config.inputs_dir() / "simulation" / "large_module_pool_70cm.npz"


def reference_checksums() -> dict:
    """Content of ``inputs/simulation/pool_checksums.json``."""
    return hio.read_json(config.inputs_dir() / "simulation" / "pool_checksums.json")


def pools_dir() -> Path:
    """``results/sim/pools``."""
    return config.results_dir() / "sim" / "pools"


def write_pool(table, out: Path, name: str, problems: List[str], rec: RunRecord) -> Path:
    """Write a pool under its standard name when it passed its checks, else as ``<name>.failed.npz``."""
    from hexmod.sim import io_pool

    path = out / name
    if problems:
        if path.is_file():
            path.unlink()
        path = out / name.replace(".npz", ".failed.npz")
    io_pool.save_pool(table, path)
    rec.add_output(path)
    return path


def compare_checksums(table, reference: dict) -> List[str]:
    """Columns whose checksum differs from ``reference`` (an entry of pool_checksums.json).

    The values are rounded as for the reference checksums (its ``significant_digits``, if any).
    """
    from hexmod.sim import io_pool

    got = io_pool.column_checksums(table, reference.get("significant_digits"))
    problems = []
    if list(got) != list(reference["column_order"]):
        problems.append(f"column order {list(got)} differs from {reference['column_order']}")
    if len(table) != reference.get("n_cells", len(table)):
        problems.append(f"{len(table)} cells, expected {reference['n_cells']}")
    for name, digest in got.items():
        ref = reference["columns"].get(name, {}).get("sha256")
        if ref != digest:
            problems.append(f"column {name}: checksum differs")
    return problems


def compare_with_tolerance(rebuilt, shipped) -> Dict[str, dict]:
    """Largest absolute difference per column between two pools with the same cells.

    Returns ``{column: {"max_abs_diff", "n_cells_differ", "tolerance", "ok"}}``; ragged columns are
    compared array by array (shapes must agree).
    """
    out = {}
    for col in shipped.columns:
        a, b = rebuilt[col].to_numpy(), shipped[col].to_numpy()
        tol = LARGE_TOLERANCES.get(col, 0.0)
        if a.dtype != object:
            if col in LARGE_REPORT_ONLY:  # an angle: circular difference in degrees
                d = np.abs((a.astype(float) - b.astype(float) + 180.0) % 360.0 - 180.0)
            else:
                d = np.abs(a.astype(float) - b.astype(float))
            worst, n_diff, shapes_ok = float(np.max(d)) if d.size else 0.0, int(np.sum(d > 0)), True
        else:
            worst, n_diff, shapes_ok = 0.0, 0, True
            for u, v in zip(a, b):
                u, v = np.asarray(u, float), np.asarray(v, float)
                if u.shape != v.shape:
                    shapes_ok = False
                    n_diff += 1
                    continue
                if u.size:
                    m = float(np.max(np.abs(u - v)))
                    worst = max(worst, m)
                    n_diff += int(m > 0)
        ok = shapes_ok and (col in LARGE_REPORT_ONLY or worst <= tol)
        out[col] = {"max_abs_diff": worst, "n_cells_differ": n_diff,
                    "tolerance": None if col in LARGE_REPORT_ONLY else tol, "ok": bool(ok)}
    return out


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regenerate-large", action="store_true",
                   help="rebuild the large-module pool instead of copying the shipped one (about 5 min)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    from hexmod.sim import io_pool, pool, pool_large

    out = pools_dir()
    out.mkdir(parents=True, exist_ok=True)
    reference = reference_checksums()
    equal_note = "equal to inputs/simulation/pool_checksums.json"
    failures: List[str] = []
    with RunRecord(STEP, argv=argv) as rec:
        rec.add_input(config.inputs_dir() / "simulation" / "pool_checksums.json")

        # small module: rebuilt
        small = pool.set_gridcells_pool(pool.module_param("small"), N_CELLS, BASE_SEED)
        problems = compare_checksums(small, reference[SMALL_KEY])
        rec.note("small_pool_checksums", equal_note if not problems else problems)
        failures += [f"small pool: {p}" for p in problems]
        small_path = write_pool(small, out, SMALL_FILE, problems, rec)

        # large module: shipped, or rebuilt with the large-module builder
        shipped_path = shipped_pool_path()
        rec.add_input(shipped_path)
        shipped = io_pool.load_pool(shipped_path)
        problems = compare_checksums(shipped, reference[LARGE_KEY])
        failures += [f"shipped large pool: {p}" for p in problems]
        rec.note("shipped_large_pool_checksums", equal_note if not problems else problems)
        if args.regenerate_large:
            large = pool_large.set_gridcells_pool_large(
                pool_large.large_pool_param(), pool_large.LARGE_POOL_N_CELLS,
                pool_large.LARGE_POOL_BASE_SEED)
            diffs = compare_with_tolerance(large, shipped)
            rec.note("large_pool", "rebuilt with hexmod.sim.pool_large")
            rec.note("large_pool_differences_from_shipped", diffs)
            large_problems = [f"rebuilt large pool, column {c}: difference {d['max_abs_diff']:.3g} above "
                              f"{d['tolerance']}" for c, d in diffs.items() if not d["ok"]]
            failures += large_problems
            large_path = write_pool(large, out, LARGE_FILE, problems + large_problems, rec)
        else:
            large = shipped
            large_path = out / (LARGE_FILE if not problems else LARGE_FILE.replace(".npz", ".failed.npz"))
            if problems and (out / LARGE_FILE).is_file():
                (out / LARGE_FILE).unlink()
            shutil.copyfile(shipped_path, large_path)
            rec.add_output(large_path)
            rec.note("large_pool", "copied from inputs/simulation/large_module_pool_70cm.npz")

        summary = {SMALL_KEY: io_pool.pool_summary(small, small_path),
                   LARGE_KEY: io_pool.pool_summary(large, large_path)}
        hio.write_json(summary, out / "pool_checksums.json")
        rec.add_output(out / "pool_checksums.json")
        rec.note("n_cells", {"small": int(len(small)), "large": int(len(large))})
        rec.note("problems", failures)
    for f in failures:
        print(f"error: {f}", file=sys.stderr)
    print(f"pools written to {out} (small {len(small)} cells, large {len(large)} cells)"
          + ("" if not failures else f"; {len(failures)} check(s) failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
