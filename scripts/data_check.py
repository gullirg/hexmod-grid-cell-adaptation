"""Check that the raw data of the six datasets are complete and unchanged.

Examples::

    python scripts/data_check.py            # sizes and sha256 checksums of every file (seconds to minutes)
    python scripts/data_check.py --quick    # sizes only (seconds)

Every file listed in ``inputs/manifests/files_*.csv`` is looked for under the data folder
(``--data``, default ``HEXMOD_DATA`` or ``<repo>/data``) and compared with its size and sha256
checksum (and its md5 where the manifest gives one). The folder of the Sargolini recordings must
hold exactly the listed files.

Writes ``results/data_check.json`` (``HEXMOD_RESULTS`` or ``<repo>/results``) and the run record
``results/logs/data_check.json``. The exit code is 0 when every dataset passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from hexmod import config  # noqa: E402
from hexmod import data_sources as dsrc  # noqa: E402
from hexmod import datasets as ds  # noqa: E402
from hexmod.io import write_json  # noqa: E402
from hexmod.runlog import RunRecord  # noqa: E402


def check_all(base: Path, checksum: str, keys: List[str], workers: int = 2) -> dict:
    """Check every dataset in ``keys``; return the content of ``data_check.json``."""
    results = {k: dsrc.check_dataset(k, base, checksum, workers=workers) for k in keys}
    return {"ok": all(r["status"] == "ok" for r in results.values()), "checksums": checksum, "datasets": results}


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--data", default=None, help="data folder (default HEXMOD_DATA or <repo>/data)")
    p.add_argument("--quick", action="store_true", help="compare sizes only")
    p.add_argument("--workers", type=int, default=2, help="files checked in parallel (default 2)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base = Path(args.data).expanduser() if args.data else config.data_dir()
    checksum = "size" if args.quick else "sha256"
    keys = list(ds.CANONICAL_ORDER)
    out_path = config.results_dir() / "data_check.json"
    with RunRecord("data_check", hash_inputs=True, argv=argv if argv is not None else sys.argv[1:]) as rec:
        for manifest in dict.fromkeys(dsrc.manifest_path(k) for k in keys):
            rec.add_input(manifest)
        report = check_all(base, checksum, keys, args.workers)
        write_json(report, out_path)
        rec.add_output(out_path)
        rec.note("data_folder", "data" if base.resolve() == config.data_dir().resolve() else "given with --data")
        rec.note("ok", report["ok"])
        rec.note("checksums", checksum)
        rec.note("status", {k: v["status"] for k, v in report["datasets"].items()})
    print(f"data folder: {base}  (checked: {'sizes' if checksum == 'size' else 'sizes and sha256 checksums'})")
    for key in keys:
        r = report["datasets"][key]
        print(f"  {key:<16} {r['status']:<18} {r['n_ok']:>4} of {r['n_expected']:>4} files, "
              f"{r['bytes']:>15,} bytes")
        lines = [f"{p}: {why}" for p, why in r["problems"].items()]
        for line in lines[:10] + [f"{p}: not in the manifest" for p in r["extra_files"][:10]]:
            print(f"      {line}")
        hidden = max(len(lines) - 10, 0) + max(len(r["extra_files"]) - 10, 0)
        if hidden > 0:
            print(f"      ... and {hidden} more")
    if report["ok"]:
        print("all six datasets pass")
    else:
        print("not every dataset passes; see the files listed above and run scripts/data_download.py")
    print(f"written: {out_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
