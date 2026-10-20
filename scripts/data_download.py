"""Download the six datasets and verify every file against the shipped manifests.

Examples::

    python scripts/data_download.py --list                         # what would be downloaded, sizes and addresses
    python scripts/data_download.py                                # all six datasets (about 13.6 GB)
    python scripts/data_download.py --datasets brun2008,hafting2005
    python scripts/data_download.py --datasets vollan2025 --only 28063_4.mat   # one file

Files go into the data folder (``--data``, default ``HEXMOD_DATA`` or ``<repo>/data``) in the
layout described in docs/DATA.md. Interrupted downloads resume where they stopped; every file is
checked against its size and sha256 checksum (and md5 where the archive publishes one) before it
is accepted. Running the script again skips files that are already present and valid.

``brun2008`` is distributed as one zip archive, from which the six recordings of the square-box
experiment are extracted. ``sargolini2006`` is a zip archive split into three volumes; the 150
files the analysis reads are exactly the members stored in the last volume, so only that volume
is downloaded (``--sargolini-full-archive`` downloads all three volumes instead and extracts the
same files). Archives are kept after extraction unless ``--delete-archives`` is given.
``barry2012`` and ``olafsdottir2016`` are one HDF5 file each, from a Zenodo record.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from hexmod import config  # noqa: E402
from hexmod import data_sources as dsrc  # noqa: E402
from hexmod import datasets as ds  # noqa: E402

_print_lock = threading.Lock()


def say(text: str) -> None:
    """Print one line (safe from several download threads)."""
    with _print_lock:
        print(text, flush=True)


def gb(n: int) -> str:
    """Format a byte count."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e3:.1f} kB"


def parse_datasets(value: str) -> List[str]:
    """Turn the ``--datasets`` value into dataset keys in canonical order."""
    value = value.strip().lower()
    if value == "all":
        return list(ds.CANONICAL_ORDER)
    keys = [k.strip() for k in value.split(",") if k.strip()]
    unknown = [k for k in keys if k not in ds.DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {', '.join(ds.CANONICAL_ORDER)} or all")
    return [k for k in ds.CANONICAL_ORDER if k in keys]


class Progress:
    """Prints the progress of one download at most every ``every`` seconds."""

    def __init__(self, label: str, every: float = 15.0):
        self.label, self.every = label, every
        self.t0 = self.last = time.time()
        self.start_bytes: Optional[int] = None

    def __call__(self, done: int, size: int) -> None:
        now = time.time()
        if self.start_bytes is None:
            self.start_bytes = done
        if now - self.last >= self.every:
            rate = (done - self.start_bytes) / max(now - self.t0, 1e-9)
            say(f"    {self.label}: {gb(done)} of {gb(size)} ({rate / 1e6:.1f} MB/s)")
            self.last = now


def used_files(rows) -> str:
    """Describe the files of a manifest that the analysis uses, for example '6 files used by the analysis'."""
    final = dsrc.final_files(rows)
    notes = [r for r in final if str(r["path"]).endswith(".md")]       # file descriptions shipped with the data
    n = len(final) - len(notes)
    text = f"{n} {'file' if n == 1 else 'files'} used by the analysis"
    if notes:
        text += " and the file description " + ", ".join(Path(str(r["path"])).name for r in notes)
    return text


def list_downloads(keys: List[str], dest: Path, full_archive: bool) -> int:
    """Print what would be downloaded (dry run)."""
    total = 0
    for key in keys:
        meta = ds.DATASETS[key]
        rows = dsrc.read_manifest(key)
        todo = dsrc.download_rows(rows, full_archive)
        size = sum(r["bytes"] for r in todo)
        total += size
        say(f"{key}: {len(todo)} download(s), {size:,} bytes ({gb(size)}); {used_files(rows)}; "
            f"{meta['licence']}, {dsrc.SOURCES[key]['landing_page']}")
        for r in todo:
            target = dest / r["path"]
            present = " [present]" if target.is_file() and target.stat().st_size == r["bytes"] else ""
            say(f"  {r['bytes']:>14,}  {r['path']}{present}")
            say(f"  {'':>14}  {dsrc.object_url(key, r['source'])}")
        if key == "sargolini2006" and not full_archive:
            whole = sum(r["bytes"] for r in rows if r["role"] in ("volume", "archive"))
            say(f"  (the whole split archive is {whole:,} bytes; --sargolini-full-archive downloads it)")
    say(f"total: {total:,} bytes ({gb(total)})")
    return 0


def present_and_valid(path: Path, row: dict) -> bool:
    """True when a file is present with the right size and checksum."""
    return dsrc.verify_file(path, row) == "ok"


def fetch(key: str, row: dict, dest: Path, args) -> str:
    """Download one manifest row unless it is already present and valid."""
    target = dest / row["path"]
    if present_and_valid(target, row):
        return "present"
    if target.exists():
        say(f"  {row['path']}: present but invalid, downloading again")
        target.unlink()
    say(f"  {row['path']}: downloading {gb(row['bytes'])}")
    t0 = time.time()
    dsrc.download_file(lambda: dsrc.resolve_url(key, row["source"]), target, row["bytes"], row["sha256"],
                       row.get("md5", ""), retries=args.retries, progress=Progress(row["name"]))
    dt = max(time.time() - t0, 1e-9)
    say(f"  {row['path']}: done and verified ({row['bytes'] / dt / 1e6:.1f} MB/s)")
    return "downloaded"


def download_dataset(key: str, dest: Path, args, pool: ThreadPoolExecutor) -> bool:
    """Download, unpack and verify one dataset; return True when every file is valid."""
    rows = dsrc.read_manifest(key)
    members = [r for r in rows if r["role"] == "member"]
    say(f"{key}: {ds.DATASETS[key]['label']}; {ds.DATASETS[key]['licence']}")
    if members and all(present_and_valid(dest / r["path"], r) for r in members):
        say(f"  all {len(members)} extracted files are present and valid")
        todo = [r for r in rows if r["role"] == "file"]
    else:
        todo = dsrc.download_rows(rows, args.sargolini_full_archive)
    if args.only:
        todo = [r for r in todo if any(fnmatch(r["name"], g) or fnmatch(r["path"], g) for g in args.only)]
        if not todo:
            say(f"  no download matches {' or '.join(args.only)}")
            return True
    futures = [pool.submit(fetch, key, r, dest, args) for r in todo]
    ok = True
    for fut in futures:
        try:
            fut.result()
        except Exception as exc:
            say(f"  ERROR: {exc}")
            ok = False
    if not ok:
        return False
    if members and (not args.only or any(r["role"] == "archive" for r in todo)):
        try:
            dsrc.extract_members(key, dest, rows, log=say)
        except dsrc.MissingVolumeError as exc:
            say(f"  ERROR: {exc}; run again with --sargolini-full-archive")
            return False
        if args.delete_archives:
            for r in rows:
                if r["role"] in ("archive", "volume") and (dest / r["path"]).is_file():
                    (dest / r["path"]).unlink()
                    say(f"  removed {r['path']}")
    if args.only:
        say(f"  {len(todo)} selected download(s) present and verified")
        return True
    result = dsrc.check_dataset(key, dest, "size", rows)
    say(f"  {key}: {result['n_ok']} of {result['n_expected']} files valid ({result['status']})")
    for name in [f"{p}: {why}" for p, why in result["problems"].items()] + \
            [f"{p}: not in the manifest" for p in result["extra_files"]]:
        say(f"    {name}")
    return result["status"] == "ok"


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1] if "\n\n" in __doc__ else None)
    p.add_argument("--datasets", default="all", help="'all' (default) or comma-separated dataset keys")
    p.add_argument("--data", default=None, metavar="DIR", help="data folder (default HEXMOD_DATA or <repo>/data)")
    p.add_argument("--list", action="store_true", help="dry run: list files, sizes and addresses, then exit")
    p.add_argument("--sargolini-full-archive", action="store_true",
                   help="download all three volumes of the split Sargolini archive (2.37 GB) instead of the "
                        "last volume only (270 MB)")
    p.add_argument("--only", action="append", metavar="PATTERN",
                   help="download only the files whose name or path matches this shell pattern "
                        "(may be repeated; the completeness check is then skipped)")
    p.add_argument("--delete-archives", action="store_true", help="remove archives after successful extraction")
    p.add_argument("--jobs", type=int, default=2, help="parallel downloads (default 2)")
    p.add_argument("--retries", type=int, default=5, help="further attempts per file after a failure (default 5)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    keys = parse_datasets(args.datasets)
    dest = Path(args.data).expanduser() if args.data else config.data_dir()
    if args.list:
        return list_downloads(keys, dest, args.sargolini_full_archive)
    say(f"data folder: {dest}")
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for key in keys:
            if not download_dataset(key, dest, args, pool):
                failed.append(key)
    if failed:
        say(f"not complete: {', '.join(failed)} (run the command again to resume)")
        return 1
    say("all requested datasets are present and verified; run scripts/data_check.py for a full check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
