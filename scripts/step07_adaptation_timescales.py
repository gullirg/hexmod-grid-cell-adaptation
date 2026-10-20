"""Step 07: timescale of firing-rate adaptation (Supp Fig 4).

Examples::

    python scripts/step07_adaptation_timescales.py --workers 4
    python scripts/step07_adaptation_timescales.py --variant unshifted --n-boot 200

For every grid cell the firing rate is estimated in nine t_prev bins from 0.05 to 20.1 s
(non-parametric model of location, speed and t_prev; samples with t_prev outside the bins are left
out) and expressed as percent change from the cell's mean rate. One and two exponentials are fitted
to the population means, weighted by 1 / s.e.m.^2 (the printed fits) and unweighted, and compared
with AIC, AICc, BIC and the extra-sum-of-squares F test. Animals are resampled with replacement to
obtain intervals for the time constants (fixed seed; seed and number of resamples are written to the
run record).

Two variants: ``unshifted`` (all cohort cells, spikes as recorded; the printed values) and
``shifted`` (the cells kept by step 05, every spike shifted by the cell's best shift and the fields
of the re-characterised cell; reported only). The bootstrap of the shifted variant continues the
random stream of the unshifted one; with ``--variant shifted`` alone the stream is first advanced by
the draws of the unshifted bootstrap (``hexmod.rat.timescales.skip_bootstrap_draws``, over the
animals of ``rat/timescales/finebin_percent_change.npy`` written by an earlier run of the unshifted
variant), so that the shifted bootstrap is the same in both cases.

Reads ``preprocessed/trajectories.pkl``, ``rat/cohort/cells.csv``, ``rat/cohort/field_locations.npz``
and, for the shifted variant, ``rat/prospective/shift_cells.csv`` and
``rat/prospective/shift_field_locations.npz``. Writes ``rat/timescales/`` (see docs/PIPELINE.md, step
07) and ``stats/step07_adaptation_timescales.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from hexmod import config  # noqa: E402  (after the path set-up)

config.limit_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hexmod import io as hio  # noqa: E402
from hexmod.parallel import map_ordered  # noqa: E402
from hexmod.rat import timescales as ts  # noqa: E402
from hexmod.rat.gridcell import make_cell_data  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402

STEP = "step07_adaptation_timescales"
VARIANTS = ("unshifted", "shifted")
WEIGHTINGS = (("weighted_by_SEM", True), ("unweighted", False))


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=None, help="worker processes (default HEXMOD_WORKERS or 4)")
    p.add_argument("--variant", choices=["both", *VARIANTS], default="both",
                   help="cells and spikes to analyse (default both)")
    p.add_argument("--n-boot", type=int, default=ts.N_BOOTSTRAP,
                   help=f"animal bootstrap resamples (default {ts.N_BOOTSTRAP})")
    p.add_argument("--seed", type=int, default=ts.PUBLISHED_BOOTSTRAP_SEED, help="seed of the animal bootstrap")
    return p.parse_args(argv)


def fine_bin_profile(traj: dict, ids: pd.DataFrame, field_locations: list, shift_ms: Optional[np.ndarray],
                     workers: int):
    """Percent change per t_prev bin of every listed cell (rows in the order of ``ids``)."""
    items = []
    for i, row in enumerate(ids.itertuples(index=False)):
        session = traj[row.animal_id][row.session_id]
        shift = None if shift_ms is None else float(shift_ms[i]) / 1000.0
        cd = make_cell_data(session, int(row.cell_id), row.animal_id, row.session_id, spike_shift_s=shift)
        items.append(ts.fine_bin_inputs(cd, field_locations[i]))
    out = map_ordered(ts.fine_bin_cell, items, workers, desc="cells")
    del items
    pct = ts.percent_change([o["rate_hz"] for o in out], [o["mean_rate_hz"] for o in out])
    counts = np.row_stack([o["samples_per_bin"] for o in out])
    return pct, counts


def points_table(summary: dict) -> pd.DataFrame:
    """Supp Fig 4 points: one row per t_prev bin (bins numbered from 1)."""
    n = len(summary["x_s"])
    cols = ["edge_low_s", "edge_high_s", "x_s", "cells_in_bin", "mean_pct_change", "sem_pct_change",
            "ci95_low", "ci95_high", "weight"]
    df = pd.DataFrame({"bin": np.arange(1, n + 1)})
    for c in cols:
        df[c] = summary[c]
    return df


def fits_table(weighted: dict) -> pd.DataFrame:
    """Supp Fig 4 fit parameters (weighted fits)."""
    s, d = weighted["single"], weighted["double"]
    rows = [("one exponential", "a_pct", s["a"]), ("one exponential", "tau_s", s["tau"]),
            ("one exponential", "c_pct", s["c"]), ("one exponential", "chi2_weighted", s["SS"]),
            ("two exponentials", "a_fast_pct", d["a_fast"]), ("two exponentials", "tau_fast_s", d["tau_fast"]),
            ("two exponentials", "a_slow_pct", d["a_slow"]), ("two exponentials", "tau_slow_s", d["tau_slow"]),
            ("two exponentials", "c_pct", d["c"]), ("two exponentials", "chi2_weighted", d["SS"])]
    return pd.DataFrame(rows, columns=["model", "parameter", "value"])


def statistics(n_cells: int, summary: dict, weighted: dict, boot: Optional[dict], n_boot: int) -> dict:
    """Printed Supp Fig 4 statistics (docs/STATISTICS.md)."""
    k_single, k_double = 3, 5
    out = {"supp4.n_cells": int(n_cells), "supp4.n_cells_last_bin": int(summary["cells_in_bin"][-1]),
           "supp4.n_bins": int(len(summary["x_s"])), "supp4.bin_edges.first_s": float(ts.FINE_TPREV_EDGES[0]),
           "supp4.bin_edges.last_s": float(ts.FINE_TPREV_EDGES[-1])}
    if "extra_SS_F" not in weighted:   # a model could not be fitted
        return out
    out.update({
        "supp4.single.tau_s": weighted["single"]["tau"],
        "supp4.single.chi2": weighted["single"]["SS"],
        "supp4.double.tau_fast_s": weighted["double"]["tau_fast"],
        "supp4.double.tau_slow_s": weighted["double"]["tau_slow"],
        "supp4.double.chi2": weighted["double"]["SS"],
        "supp4.delta_aic": weighted["dAIC_double_minus_single"],
        "supp4.delta_bic": weighted["dBIC"],
        "supp4.f_test.F": weighted["extra_SS_F"],
        "supp4.f_test.df1": k_double - k_single,
        "supp4.f_test.df2": int(weighted["n_points"]) - k_double,
        "supp4.f_test.p": weighted["extra_SS_p"],
        "supp4.single.dof": int(weighted["n_points"]) - k_single,
        "supp4.double.dof": int(weighted["n_points"]) - k_double,
    })
    if boot is not None and boot["n_boot_ok"]:
        for key, name in (("tau_single_95", "tau_single"), ("tau_fast_95", "tau_fast"), ("tau_slow_95", "tau_slow")):
            out[f"supp4.bootstrap.{name}_low_s"] = boot[key][0]
            out[f"supp4.bootstrap.{name}_high_s"] = boot[key][1]
        out["supp4.bootstrap.pct_double_preferred"] = 100.0 * boot["n_double_preferred"] / boot["n_boot_ok"]
        out["supp4.bootstrap.n_resamples"] = int(n_boot)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workers = config.workers(args.workers)
    n_boot = args.n_boot
    res = config.results_dir()
    out_dir = res / "rat" / "timescales"
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    with RunRecord(STEP, workers=workers, argv=argv) as rec:
        settings_path = config.inputs_dir() / "rat" / "timescale_fit_settings.json"
        settings = ts.load_fit_settings(settings_path)
        rec.add_input(settings_path)
        traj_path = res / "preprocessed" / "trajectories.pkl"
        cells_path = res / "rat" / "cohort" / "cells.csv"
        fl_path = res / "rat" / "cohort" / "field_locations.npz"
        shift_cells_path = res / "rat" / "prospective" / "shift_cells.csv"
        shift_fl_path = res / "rat" / "prospective" / "shift_field_locations.npz"
        for p in (traj_path, cells_path, fl_path):
            rec.add_input(p)
        traj = hio.load_cache(traj_path)
        cohort = {}
        cells = hio.read_csv(cells_path)
        fl = hio.load_npz(fl_path)
        if not np.array_equal(fl["row"], cells["row"].to_numpy()):
            raise ValueError("field_locations.npz is not row-aligned with cells.csv")
        cohort["unshifted"] = (cells, fl["field_locations"], None)
        if "shifted" in variants:
            if shift_cells_path.is_file() and shift_fl_path.is_file():
                rec.add_input(shift_cells_path)
                rec.add_input(shift_fl_path)
                sc = hio.read_csv(shift_cells_path)
                sfl = hio.load_npz(shift_fl_path)
                if not np.array_equal(sfl["row"], sc["row"].to_numpy()):
                    raise ValueError("shift_field_locations.npz is not row-aligned with shift_cells.csv")
                cohort["shifted"] = (sc, sfl["field_locations"], sc["peakShift_SI"].to_numpy(float))
            else:
                print("shifted variant skipped: step 05 outputs not found", file=sys.stderr)
                rec.note("shifted_variant", "skipped: step 05 outputs not found")

        fits, boots, rng = {}, {}, np.random.default_rng(args.seed)
        stats_out = None
        stream = "one generator, unshifted variant first"
        if "shifted" in cohort and "unshifted" not in variants:
            unshifted_pct = out_dir / "finebin_percent_change.npy"
            if not unshifted_pct.is_file():
                raise FileNotFoundError(f"{unshifted_pct} is missing; the shifted bootstrap continues the random "
                                        "stream of the unshifted one, so run --variant unshifted (or both) first")
            rec.add_input(unshifted_pct)
            n_animals = len(ts.bootstrap_animals(hio.load_npy(unshifted_pct), cells["animal_id"].to_numpy()))
            ts.skip_bootstrap_draws(rng, n_animals, n_boot)
            stream += f"; the draws of the unshifted bootstrap ({n_animals} animals) were skipped"
        for variant in VARIANTS:
            if variant not in cohort or variant not in variants:
                continue
            ids, field_locations, shift_ms = cohort[variant]
            pct, counts = fine_bin_profile(traj, ids[["animal_id", "session_id", "cell_id"]], field_locations,
                                           shift_ms, workers)
            summary = ts.bin_summary(pct)
            x = summary["x_s"]
            fits[variant] = {name: ts.model_comparison(x, summary["mean_pct_change"], summary["sem_pct_change"],
                                                       settings, weighted)
                             for name, weighted in WEIGHTINGS}
            boot = ts.animal_bootstrap(pct, ids["animal_id"].to_numpy(), settings, x=x, n_boot=n_boot, rng=rng)
            draws = boot.pop("draws")
            boots[variant] = {k: v for k, v in boot.items() if k != "n_double_preferred"}
            p = hio.save_npz(out_dir / f"bootstrap_draws_{variant}.npz", arrays=draws)
            rec.add_output(p)
            rec.note(f"{variant}.n_cells", int(len(pct)))
            rec.note(f"{variant}.n_cells_complete", summary["n_cells_complete"])
            rec.note(f"{variant}.samples_per_bin_total", counts.sum(axis=0).tolist())
            if variant == "unshifted":
                p = hio.save_npy(out_dir / "finebin_percent_change.npy", pct)
                rec.add_output(p)
                p = hio.write_csv(points_table(summary), out_dir / "supp4_points.csv")
                rec.add_output(p)
                if "extra_SS_F" in fits[variant]["weighted_by_SEM"]:
                    p = hio.write_csv(fits_table(fits[variant]["weighted_by_SEM"]), out_dir / "supp4_fits.csv")
                    rec.add_output(p)
                stats_out = statistics(len(pct), summary, fits[variant]["weighted_by_SEM"], boot, n_boot)
            else:
                p = hio.save_npy(out_dir / "finebin_percent_change_shifted.npy", pct)
                rec.add_output(p)
        p = hio.write_json(fits, out_dir / "fits.json")
        rec.add_output(p)
        p = hio.write_json(boots, out_dir / "bootstrap.json")
        rec.add_output(p)
        if stats_out is not None:
            p = hio.write_json(stats_out, res / "stats" / f"{STEP}.json")
            rec.add_output(p)
        rec.note("bootstrap_seed", int(args.seed))
        rec.note("bootstrap_resamples", int(n_boot))
        rec.note("bootstrap_stream", stream)
        rec.note("tprev_bin_edges_s", ts.FINE_TPREV_EDGES.tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
