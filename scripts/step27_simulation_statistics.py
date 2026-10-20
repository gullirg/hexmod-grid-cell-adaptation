"""Step 27: the simulation statistics printed in the paper (Fig 1 legend, Figs 6, 7 and 8).

Example::

    python scripts/step27_simulation_statistics.py

Reads the outputs of steps 22 (``sim/fig6/maps.npz``), 23 (``sim/fig6/fig6d_*.csv``), 25
(``sim/fig7/fig7_values.json``) and 26 (``sim/fig8/fig8d_sweep.csv``). A missing file stops the
step, unless ``--allow-precomputed`` is given: it is then taken from the shipped
``results_precomputed/sim`` (the run record lists the source of every input, and step 40 reports
it). Computes:

- Fig 1 legend (closed form, ``hexmod.sim.analytic``): the six-fold component 2 I6(kappa) /
  I0(kappa) of a von Mises tuning curve of circular SD 15 deg (``fig1.narrow.*``), and the circular
  SD of a cosine tuning curve (resultant length 0.5; ``fig1.broad.circular_sd_deg``) and its six-fold
  component, which is zero because a cosine curve has no six-fold Fourier term
  (``fig1.broad.sixfold``);
- Fig 6C: maximum, minimum and SD of the summed rate map of participant 0 of the large module, in
  percent of its mean (100 subtracted from maximum and minimum);
- Fig 6D: per module the mean aligned and misaligned proximity, the paired t-test over the 10,000
  cells and Cohen's d of the paired differences (mean over SD with ddof 1);
- Fig 7: every value of ``fig7_values.json``, unchanged;
- Fig 8: v_opt / (scale / tau) = (sqrt(3) - 1) / ln(sqrt(3)) (Fig 8B, 8C) and the number of runs of
  the speed sweep (Fig 8D).

Writes ``stats/step27_simulation_statistics.json`` (flat ``{key: number}``, keys in docs/PIPELINE.md,
section 4).
"""

from __future__ import annotations

import argparse
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
from scipy import stats  # noqa: E402

from hexmod import io as hio  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step27_simulation_statistics"
MODULES = ("small", "large")
NARROW_SD_DEG = 15.0
COSINE_RESULTANT = 0.5


def sim_file(relpath: str, rec: RunRecord, sources: Dict[str, str], allow_precomputed: bool = False) -> Path:
    """``results/sim/<relpath>``, or (with ``allow_precomputed``) the shipped file when no step wrote it."""
    path = config.results_dir() / "sim" / relpath
    if path.is_file():
        sources[relpath] = "results"
    elif not allow_precomputed:
        raise SystemExit(f"error: {path} not found; run the simulation steps (22, 23, 25, 26) first, or pass "
                         "--allow-precomputed to use the shipped results_precomputed/sim")
    else:
        shipped = config.precomputed_dir() / "sim" / relpath
        if not shipped.is_file() and relpath.endswith(".csv"):
            shipped = shipped.with_name(shipped.name + ".gz")
        if not shipped.is_file():
            raise SystemExit(f"error: {path} not found and no shipped copy; run the simulation steps first")
        path = shipped
        sources[relpath] = "results_precomputed"
    rec.add_input(path)
    return path


def fig1_values() -> Dict[str, float]:
    from hexmod.circular import circular_sd_deg
    from hexmod.sim import analytic

    kappa = analytic.kappa_from_sd(NARROW_SD_DEG)
    return {"fig1.narrow.circular_sd_deg": NARROW_SD_DEG, "fig1.narrow.kappa": kappa,
            "fig1.narrow.sixfold": analytic.sixfold_component(kappa),
            "fig1.broad.circular_sd_deg": float(circular_sd_deg(COSINE_RESULTANT)),
            "fig1.broad.sixfold": 0.0}


def fig6c_values(maps: Dict[str, np.ndarray]) -> Dict[str, float]:
    nm = maps["seed0__large"]
    return {"fig6c.large_p0.max_pct": float(nm.max() - 100), "fig6c.large_p0.min_pct": float(nm.min() - 100),
            "fig6c.large_p0.sd_pct": float(nm.std())}


def fig6d_values(table, module: str) -> Dict[str, float]:
    aligned = table["aligned"].to_numpy(float)
    misaligned = table["misaligned"].to_numpy(float)
    t, _ = stats.ttest_rel(aligned, misaligned)
    diff = aligned - misaligned
    return {f"fig6d.{module}.mean_aligned": float(aligned.mean()),
            f"fig6d.{module}.mean_misaligned": float(misaligned.mean()), f"fig6d.{module}.t": float(t),
            f"fig6d.{module}.df": int(len(diff) - 1), f"fig6d.{module}.d": float(diff.mean() / diff.std(ddof=1))}


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--allow-precomputed", action="store_true",
                   help="take inputs that no simulation step has written from the shipped results_precomputed/sim")
    p.add_argument("--smoke", action="store_true", help="accepted for the smoke run (same computation)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config.guard_smoke_results(args.smoke)
    from hexmod.sim import analytic

    out = config.results_dir() / "stats" / f"{STEP}.json"
    sources: Dict[str, str] = {}
    with RunRecord(STEP, argv=argv) as rec:
        values: Dict[str, float] = {}
        values.update(fig1_values())
        allow = args.allow_precomputed
        values.update(fig6c_values(hio.load_npz(sim_file("fig6/maps.npz", rec, sources, allow))))
        for m in MODULES:
            values.update(fig6d_values(hio.read_csv(sim_file(f"fig6/fig6d_{m}.csv", rec, sources, allow)), m))
        fig7 = hio.read_json(sim_file("fig7/fig7_values.json", rec, sources, allow))
        clash = set(fig7) & set(values)
        if clash:
            raise ValueError(f"keys written twice: {sorted(clash)}")
        values.update(fig7)
        values["fig8.v_opt_scale_per_tau"] = float(analytic.SPEED_OPT_FACTOR)
        values["fig8d.n_runs"] = int(len(hio.read_csv(sim_file("fig8/fig8d_sweep.csv", rec, sources, allow))))
        bad = [k for k, v in values.items() if not isinstance(v, (int, float)) or isinstance(v, bool)
               or (isinstance(v, float) and not np.isfinite(v))]
        if bad:
            raise ValueError(f"statistics that are not finite numbers: {bad}")
        hio.write_json(values, out)
        rec.add_output(out)
        rec.note("sources", sources)
        rec.note("n_keys", len(values))
    print(f"{len(values)} statistics written to {out}; inputs from {sorted(set(sources.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
