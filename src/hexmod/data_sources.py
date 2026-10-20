"""Where the raw datasets come from, and how they are downloaded, unpacked and verified.

The six datasets are fetched from three archives, without login:

- ``vollan2025`` from EBRAINS (doi:10.25493/R5FR-EDG). Files are requested through the EBRAINS
  data proxy, which returns the address of the object in the EBRAINS object store.
- ``brun2008``, ``hafting2005`` and ``sargolini2006`` from the NIRD Research Data Archive,
  through its public S3 endpoint (path-style addresses, unsigned requests).
- ``barry2012`` and ``olafsdottir2016`` from Zenodo (doi:10.5281/zenodo.23477461): one HDF5 file
  per dataset with the positions and spike times of the sessions analysed in the paper.

Every file that the analysis reads is listed, with its size and sha256 checksum, in a manifest
``inputs/manifests/files_<key>.csv``. Manifest columns:

``dataset``
    dataset key.
``path``
    location of the file relative to the data folder (``HEXMOD_DATA``).
``name``
    file name (the last part of ``path``).
``role``
    ``file``: downloaded as it is and read by the analysis; ``archive``: a zip archive that is
    downloaded and from which members are extracted; ``volume``: an earlier volume of a split
    zip archive (downloaded only when the whole split archive is requested); ``member``: a file
    extracted from an archive and read by the analysis.
``bytes``, ``sha256``
    size and sha256 checksum.
``md5``
    md5 checksum published by the archive (empty where none is published).
``source``
    for ``file``, ``archive`` and ``volume`` rows, the object key at the archive (for Zenodo,
    the file name in the record); for ``member`` rows, the name of the member inside the archive.
``archive``
    for ``member`` rows, the ``path`` of the archive the member comes from.

Some manifests have extra columns (``used_in_cohort`` for ``vollan2025`` and ``brun2008``,
``listdir_order`` for ``sargolini2006``; see docs/DATA.md).

Zip archives are read by a small reader in this module rather than by :mod:`zipfile`, because one
archive (``sargolini2006``) is a split archive in three volumes, which :mod:`zipfile` cannot
read. The files the analysis uses are exactly the members stored in the last volume, so only
that volume (270 MB of the 2.37 GB archive) has to be downloaded; the whole split archive can be
requested instead and gives the same files.
"""

from __future__ import annotations

import csv
import hashlib
import os
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config
from . import datasets as _ds

# ------------------------------------------------------------------------------------- locations
EBRAINS_PROXY = "https://data-proxy.ebrains.eu/api/v1/buckets"
EBRAINS_OBJECT_STORE = "https://rgw.cscs.ch"
NIRD_S3 = "https://s3.nird.sigma2.no"
NIRD_BUCKET = "archive-ro"
ZENODO = "https://zenodo.org"

#: Download location of each dataset.
SOURCES: Dict[str, dict] = {
    "vollan2025": {
        "host": "ebrains",
        "bucket": "d-4080b78d-edc5-4ae4-8144-7f6de79930ea",
        "landing_page": "https://doi.org/10.25493/R5FR-EDG",
    },
    "brun2008": {
        "host": "nird",
        "identifier": "54d73ada-d6ad-40d7-9dcb-85310168228e",
        "landing_page": "https://doi.org/10.11582/2018.00015",
    },
    "hafting2005": {
        "host": "nird",
        "identifier": "60452682-5815-41eb-bd0a-fb11b914623d",
        "landing_page": "https://doi.org/10.11582/2017.00017",
    },
    "sargolini2006": {
        "host": "nird",
        "identifier": "dbb6250a-9793-4d9b-9265-a90f886451d4",
        "landing_page": "https://doi.org/10.11582/2017.00019",
    },
    "barry2012": {
        "host": "zenodo",
        "record": "23477461",
        "landing_page": "https://doi.org/10.5281/zenodo.23477461",
    },
    "olafsdottir2016": {
        "host": "zenodo",
        "record": "23477461",
        "landing_page": "https://doi.org/10.5281/zenodo.23477461",
    },
}

MANIFEST_COLUMNS = ["dataset", "path", "name", "role", "bytes", "sha256", "md5", "source", "archive"]
FINAL_ROLES = ("file", "member")
USER_AGENT = "hexmod-data-download/1.0"


def object_url(key: str, source: str) -> str:
    """Return the direct address of an object of a dataset.

    Parameters
    ----------
    key : str
        Dataset key.
    source : str
        Object key at the archive (the ``source`` column of a manifest row).

    Returns
    -------
    str
        For EBRAINS, the address in the object store; for NIRD, the path-style S3 address; for
        Zenodo, the download address of the file in the record.
    """
    src = SOURCES[key]
    if src["host"] == "ebrains":
        return f"{EBRAINS_OBJECT_STORE}/ebrains:{src['bucket']}/{source}"
    if src["host"] == "zenodo":
        return f"{ZENODO}/records/{src['record']}/files/{source}?download=1"
    return f"{NIRD_S3}/{NIRD_BUCKET}/{src['identifier']}/{source}"


def proxy_url(key: str, source: str) -> Optional[str]:
    """Return the EBRAINS data-proxy address of an object (None for other archives)."""
    src = SOURCES[key]
    if src["host"] != "ebrains":
        return None
    return f"{EBRAINS_PROXY}/{src['bucket']}/{source}"


def resolve_url(key: str, source: str, session=None, timeout: float = 30.0) -> str:
    """Return the address from which an object is downloaded.

    For EBRAINS the data proxy is asked for the current address of the object (a small JSON
    reply); if that fails, the direct object-store address is used. For NIRD and Zenodo the
    address is returned directly.
    """
    proxy = proxy_url(key, source)
    if proxy is None:
        return object_url(key, source)
    try:
        session = session or _session()
        reply = session.get(proxy, params={"redirect": "false"}, timeout=timeout)
        reply.raise_for_status()
        url = reply.json().get("url")
        if url:
            return url
    except Exception:  # network or format problem: fall back to the direct address
        pass
    return object_url(key, source)


# ------------------------------------------------------------------------------------- manifests
def manifest_path(key: str) -> Path:
    """Return the manifest file of a dataset (``inputs/manifests/files_<key>.csv``)."""
    return config.repo_root() / _ds.DATASETS[key]["manifest"]


def read_manifest(key: str, path: Optional[Path] = None) -> List[dict]:
    """Return the manifest rows of one dataset as dicts (``bytes`` as int, other values as str).

    Parameters
    ----------
    key : str
        Dataset key.
    path : Path, optional
        Manifest file (default: the dataset's shipped manifest).

    Returns
    -------
    list of dict
        Rows of that dataset in manifest order.
    """
    path = Path(path) if path is not None else manifest_path(key)
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == key]
    for r in rows:
        r["bytes"] = int(r["bytes"])
    return rows


def final_files(rows: Iterable[dict]) -> List[dict]:
    """Return the rows of the files the analysis reads (roles ``file`` and ``member``)."""
    return [r for r in rows if r["role"] in FINAL_ROLES]


def download_rows(rows: Sequence[dict], full_archive: bool = False) -> List[dict]:
    """Return the rows that are downloaded (``file`` and ``archive``; ``volume`` too if requested)."""
    roles = ("file", "archive", "volume") if full_archive else ("file", "archive")
    return [r for r in rows if r["role"] in roles]


def archive_volumes(rows: Sequence[dict], archive_path: str) -> List[dict]:
    """Return the rows of all volumes of an archive in disk order (earlier volumes, then the archive).

    Earlier volumes of a split archive have the same stem as the archive and the suffixes
    ``.z01``, ``.z02`` and so on.
    """
    stem = archive_path.rsplit(".", 1)[0]
    vols = sorted((r for r in rows if r["role"] == "volume" and r["path"].rsplit(".", 1)[0] == stem),
                  key=lambda r: r["path"])
    last = [r for r in rows if r["path"] == archive_path]
    if len(last) != 1:
        raise KeyError(f"archive {archive_path} is not in the manifest")
    return vols + last


# -------------------------------------------------------------------------------------- checksums
def file_digests(path: Path, algorithms: Sequence[str] = ("sha256",), chunk: int = 1 << 22) -> Dict[str, str]:
    """Return hex digests of a file for the given hashlib algorithms, in one pass."""
    hashes = {a: hashlib.new(a) for a in algorithms}
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            for h in hashes.values():
                h.update(block)
    return {a: h.hexdigest() for a, h in hashes.items()}


def verify_file(path: Path, row: dict, checksum: str = "sha256") -> str:
    """Check one file against its manifest row.

    Parameters
    ----------
    path : Path
        File to check.
    row : dict
        Manifest row (``bytes``, ``sha256``, ``md5``).
    checksum : {"sha256", "size"}
        Compare the sha256 checksum (and the md5 when the manifest has one), or the size only.

    Returns
    -------
    str
        ``"ok"``, ``"missing"``, ``"wrong size"`` or ``"checksum mismatch"``.
    """
    path = Path(path)
    if not path.is_file():
        return "missing"
    if path.stat().st_size != int(row["bytes"]):
        return "wrong size"
    if checksum == "size":
        return "ok"
    algos = ["sha256"] + (["md5"] if row.get("md5") else [])
    got = file_digests(path, algos)
    if any(got[a] != row[a] for a in algos):
        return "checksum mismatch"
    return "ok"


def check_dataset(key: str, base: Optional[Path] = None, checksum: str = "sha256",
                  rows: Optional[Sequence[dict]] = None, workers: int = 1) -> dict:
    """Check all files of one dataset under the data folder.

    Parameters
    ----------
    key : str
        Dataset key.
    base : Path, optional
        Data folder (default ``config.data_dir()``).
    checksum : {"sha256", "size"}
        What is compared besides presence.
    rows : sequence of dict, optional
        Manifest rows (default: the shipped manifest).
    workers : int
        Files checked in parallel (threads).

    Returns
    -------
    dict
        ``status`` (``ok``, ``missing``, ``incomplete``, ``checksum mismatch`` or
        ``unexpected files``), ``n_expected``, ``n_ok``, ``bad_files`` (paths of the files that
        failed, relative to the data folder), ``problems`` (``{path: reason}``), ``extra_files``
        (files in a folder that must hold exactly the listed files) and ``bytes`` (total size of
        the verified files).
    """
    from concurrent.futures import ThreadPoolExecutor

    base = Path(base) if base is not None else config.data_dir()
    rows = final_files(rows if rows is not None else read_manifest(key))
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        statuses = list(pool.map(lambda r: verify_file(base / r["path"], r, checksum), rows))
    n_ok = sum(s == "ok" for s in statuses)
    total = sum(int(r["bytes"]) for r, s in zip(rows, statuses) if s == "ok")
    problems = {r["path"]: s for r, s in zip(rows, statuses) if s != "ok"}
    reasons = set(problems.values())
    extra = sorted(unexpected_files(key, base, rows))
    if n_ok == len(rows) and not extra:
        status = "ok"
    elif n_ok == len(rows):
        status = "unexpected files"
    elif reasons == {"missing"} and n_ok == 0:
        status = "missing"
    elif "checksum mismatch" in reasons:
        status = "checksum mismatch"
    else:
        status = "incomplete"
    return {"status": status, "n_expected": len(rows), "n_ok": n_ok, "bad_files": list(problems),
            "problems": problems, "extra_files": extra, "bytes": total}


#: Folders that must hold exactly the files of the manifest. The Sargolini archive holds more
#: recordings than the analysis uses; extra files there would change the cohort.
EXACT_FOLDERS = {"sargolini2006": ["sargolini2006/subset"]}


def unexpected_files(key: str, base: Path, rows: Sequence[dict]) -> List[str]:
    """Return files found in an exact folder of a dataset that the manifest does not list."""
    listed = {r["path"] for r in rows}
    out = []
    for folder in EXACT_FOLDERS.get(key, []):
        d = Path(base) / folder
        if not d.is_dir():
            continue
        for name in sorted(os.listdir(d)):
            rel = f"{folder}/{name}"
            if rel not in listed and not name.endswith(".part"):
                out.append(rel)
    return out


# --------------------------------------------------------------------------------- zip archives
class ZipFormatError(ValueError):
    """The archive is damaged or uses a feature the reader does not support."""


class MissingVolumeError(FileNotFoundError):
    """A member is stored (partly) in a volume of a split archive that is not available."""


@dataclass
class ZipEntry:
    """One entry of a zip central directory."""

    name: str
    disk: int
    offset: int
    method: int
    flags: int
    crc32: int
    compressed_size: int
    size: int


class ZipVolumes:
    """Random access to a zip archive stored in one or more volumes.

    Parameters
    ----------
    volumes : sequence of Path or None
        Volume files in disk order (for a split archive ``x.z01``, ``x.z02``, ..., ``x.zip``;
        for an ordinary archive just the ``.zip``). A volume that is not available is given as
        None; members stored in it cannot be read, but all other members can.
    """

    def __init__(self, volumes: Sequence[Optional[Path]]):
        self.volumes = [Path(v) if v is not None else None for v in volumes]
        if not self.volumes or self.volumes[-1] is None:
            raise ValueError("the last volume (the .zip file) is required")
        self.sizes = [v.stat().st_size if v is not None else None for v in self.volumes]
        self._handles: Dict[int, object] = {}

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def __enter__(self) -> "ZipVolumes":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def _handle(self, disk: int):
        if disk >= len(self.volumes):
            raise ZipFormatError(f"the archive refers to volume {disk + 1}, but has {len(self.volumes)} volumes")
        if self.volumes[disk] is None:
            raise MissingVolumeError(f"volume {disk + 1} of {len(self.volumes)} is not available")
        if disk not in self._handles:
            self._handles[disk] = open(self.volumes[disk], "rb")
        return self._handles[disk]

    def read(self, disk: int, offset: int, n: int) -> bytes:
        """Read ``n`` bytes starting at ``offset`` of volume ``disk``, continuing into later volumes."""
        out = bytearray()
        while n > 0:
            f = self._handle(disk)
            size = self.sizes[disk]
            if offset >= size:
                disk, offset = disk + 1, offset - size
                continue
            f.seek(offset)
            chunk = f.read(min(n, size - offset))
            if not chunk:
                raise ZipFormatError("unexpected end of volume")
            out += chunk
            n -= len(chunk)
            offset += len(chunk)
        return bytes(out)

    def iter_range(self, disk: int, offset: int, n: int, chunk: int = 1 << 20):
        """Yield the bytes of a range in pieces of at most ``chunk`` bytes."""
        while n > 0:
            size = self.sizes[disk] if disk < len(self.sizes) else None
            if size is not None and offset >= size:
                disk, offset = disk + 1, offset - size
                continue
            k = min(n, chunk) if size is None else min(n, chunk, size - offset)
            block = self.read(disk, offset, k)
            yield block
            n -= len(block)
            offset += len(block)


_EOCD = 0x06054B50
_EOCD64 = 0x06064B50
_EOCD64_LOCATOR = 0x07064B50
_CENTRAL = 0x02014B50
_LOCAL = 0x04034B50


def read_central_directory(archive: ZipVolumes) -> Tuple[List[ZipEntry], int]:
    """Read the central directory of a (possibly split) zip archive.

    Returns
    -------
    entries : list of ZipEntry
    n_disks : int
        Number of volumes the archive declares.
    """
    last = len(archive.volumes) - 1
    size = archive.sizes[last]
    tail_len = min(size, 22 + 65535 + 20)
    tail = archive.read(last, size - tail_len, tail_len)
    pos = tail.rfind(struct.pack("<I", _EOCD))
    if pos < 0:
        raise ZipFormatError("end of central directory not found (not a zip archive?)")
    (_, disk_no, cd_disk, _, n_total, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", tail[pos:pos + 22])
    if 0xFFFF in (disk_no, cd_disk, n_total) or 0xFFFFFFFF in (cd_size, cd_offset):
        loc = tail[pos - 20:pos]
        if len(loc) != 20 or struct.unpack("<I", loc[:4])[0] != _EOCD64_LOCATOR:
            raise ZipFormatError("zip64 end of central directory locator not found")
        _, rec_disk, rec_offset, n_disks64 = struct.unpack("<IIQI", loc)
        rec = archive.read(rec_disk, rec_offset, 56)
        if struct.unpack("<I", rec[:4])[0] != _EOCD64:
            raise ZipFormatError("zip64 end of central directory record not found")
        (_, _, _, _, disk_no, cd_disk, _, n_total, cd_size, cd_offset) = struct.unpack("<IQHHIIQQQQ", rec)
    n_disks = disk_no + 1
    if n_disks != len(archive.volumes):
        raise ZipFormatError(f"the archive has {n_disks} volumes, {len(archive.volumes)} were given")
    data = archive.read(cd_disk, cd_offset, cd_size)
    entries, p = [], 0
    while p < len(data):
        if struct.unpack("<I", data[p:p + 4])[0] != _CENTRAL:
            raise ZipFormatError("damaged central directory")
        (_, _, _, flags, method, _, _, crc, csize, usize, fnl, exl, cml, disk, _, _, lho) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", data[p:p + 46])
        raw_name = data[p + 46:p + 46 + fnl]
        extra = data[p + 46 + fnl:p + 46 + fnl + exl]
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        if 0xFFFFFFFF in (usize, csize, lho) or disk == 0xFFFF:
            usize, csize, lho, disk = _zip64_extra(extra, usize, csize, lho, disk)
        entries.append(ZipEntry(name, disk, lho, method, flags, crc, csize, usize))
        p += 46 + fnl + exl + cml
    if len(entries) != n_total:
        raise ZipFormatError(f"central directory lists {len(entries)} entries, expected {n_total}")
    return entries, n_disks


def _zip64_extra(extra: bytes, usize: int, csize: int, lho: int, disk: int):
    p = 0
    while p + 4 <= len(extra):
        tag, n = struct.unpack("<HH", extra[p:p + 4])
        body = extra[p + 4:p + 4 + n]
        if tag == 0x0001:
            q = 0
            if usize == 0xFFFFFFFF:
                usize = struct.unpack("<Q", body[q:q + 8])[0]
                q += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", body[q:q + 8])[0]
                q += 8
            if lho == 0xFFFFFFFF:
                lho = struct.unpack("<Q", body[q:q + 8])[0]
                q += 8
            if disk == 0xFFFF:
                disk = struct.unpack("<I", body[q:q + 4])[0]
            break
        p += 4 + n
    return usize, csize, lho, disk


def iter_member(archive: ZipVolumes, entry: ZipEntry, chunk: int = 1 << 20):
    """Yield the uncompressed bytes of one member, checking its size and CRC-32.

    Raises
    ------
    MissingVolumeError
        If the member is stored in a volume that is not available.
    ZipFormatError
        For encrypted members, unsupported compression methods or a failed check.
    """
    if entry.flags & 0x1:
        raise ZipFormatError(f"{entry.name}: encrypted members are not supported")
    if entry.method not in (0, 8):
        raise ZipFormatError(f"{entry.name}: compression method {entry.method} is not supported")
    header = archive.read(entry.disk, entry.offset, 30)
    if struct.unpack("<I", header[:4])[0] != _LOCAL:
        raise ZipFormatError(f"{entry.name}: local header not found")
    fnl, exl = struct.unpack("<HH", header[26:30])
    start = entry.offset + 30 + fnl + exl
    crc, n = 0, 0
    inflater = zlib.decompressobj(-15) if entry.method == 8 else None
    for block in archive.iter_range(entry.disk, start, entry.compressed_size, chunk):
        out = inflater.decompress(block) if inflater is not None else block
        if out:
            crc = zlib.crc32(out, crc)
            n += len(out)
            yield out
    if inflater is not None:
        out = inflater.flush()
        if out:
            crc = zlib.crc32(out, crc)
            n += len(out)
            yield out
    if n != entry.size or (crc & 0xFFFFFFFF) != entry.crc32:
        raise ZipFormatError(f"{entry.name}: size or CRC-32 check failed")


def extract_member(archive: ZipVolumes, entry: ZipEntry, target: Path) -> Dict[str, str]:
    """Extract one member to ``target`` (written atomically) and return its sha256 and md5."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    try:
        with open(tmp, "wb") as f:
            for block in iter_member(archive, entry):
                f.write(block)
                sha.update(block)
                md5.update(block)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def extract_members(key: str, base: Optional[Path] = None, rows: Optional[Sequence[dict]] = None,
                    log: Callable[[str], None] = print) -> Dict[str, str]:
    """Extract the listed members of a dataset's archives into the data folder and verify them.

    Members that are already present with the right checksum are left alone. The volumes of a
    split archive that are present are used; members stored in a missing volume raise
    :class:`MissingVolumeError`.

    Returns
    -------
    dict
        ``{member path: "extracted" or "present"}``.
    """
    base = Path(base) if base is not None else config.data_dir()
    rows = list(rows if rows is not None else read_manifest(key))
    members = [r for r in rows if r["role"] == "member"]
    result: Dict[str, str] = {}
    for archive_path in sorted({r["archive"] for r in members}):
        todo = []
        for r in members:
            if r["archive"] != archive_path:
                continue
            if verify_file(base / r["path"], r) == "ok":
                result[r["path"]] = "present"
            else:
                todo.append(r)
        if not todo:
            continue
        vols = [base / v["path"] for v in archive_volumes(rows, archive_path)]
        n_done = 0
        with ZipVolumes([v if v.is_file() else None for v in vols]) as z:
            entries, _ = read_central_directory(z)
            by_name = {e.name: e for e in entries}
            for r in todo:
                entry = by_name.get(r["source"])
                if entry is None:
                    raise ZipFormatError(f"{r['source']} is not in {archive_path}")
                got = extract_member(z, entry, base / r["path"])
                if got["sha256"] != r["sha256"] or (r.get("md5") and got["md5"] != r["md5"]):
                    (base / r["path"]).unlink()
                    raise ZipFormatError(f"{r['path']}: checksum differs from the manifest")
                result[r["path"]] = "extracted"
                n_done += 1
        log(f"  {archive_path}: {n_done} files extracted and verified")
    return result


# ------------------------------------------------------------------------------------- downloads
class DownloadError(RuntimeError):
    """A file could not be downloaded and verified."""


_local = threading.local()


def _new_session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _session():
    if getattr(_local, "session", None) is None:
        _local.session = _new_session()
    return _local.session


def download_file(url: "str | Callable[[], str]", target: Path, size: int, sha256: str, md5: str = "",
                  retries: int = 5, timeout: Tuple[float, float] = (30.0, 120.0), chunk: int = 1 << 20,
                  progress: Optional[Callable[[int, int], None]] = None, backoff: float = 5.0) -> str:
    """Download one file with resume, retries and checksum verification.

    Data are written to ``<target>.part``; an interrupted download resumes from the end of that
    file with an HTTP Range request. The file is renamed to ``target`` only when its size and
    checksums match.

    Parameters
    ----------
    url : str or callable
        Address of the file, or a function returning it (called before every attempt).
    target : Path
        Final location.
    size : int
        Expected size in bytes.
    sha256 : str
        Expected sha256 checksum.
    md5 : str, optional
        Expected md5 checksum (checked when given).
    retries : int
        Further attempts after a failed one (network errors resume where they stopped; a
        checksum mismatch restarts from zero).
    timeout : (float, float)
        Connect and read timeouts in seconds.
    chunk : int
        Bytes per write.
    progress : callable, optional
        Called as ``progress(bytes_done, size)`` while downloading.
    backoff : float
        Seconds to wait after the first failed attempt; doubled after each further failure.

    Returns
    -------
    str
        ``"downloaded"``.
    """
    import requests

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff * 2 ** (attempt - 1))
        try:
            start = part.stat().st_size if part.exists() else 0
            if start > size:
                part.unlink()
                start = 0
            if start < size:
                address = url() if callable(url) else url
                headers = {"Range": f"bytes={start}-"} if start else {}
                with _session().get(address, headers=headers, stream=True, timeout=timeout) as reply:
                    if start and reply.status_code == 200:
                        start = 0  # the server ignored the range request: start again
                    elif start and reply.status_code == 206:
                        got = reply.headers.get("Content-Range", "")
                        if not got.startswith(f"bytes {start}-"):
                            raise DownloadError(f"unexpected Content-Range {got!r}")
                    reply.raise_for_status()
                    done = start
                    with open(part, "ab" if start else "wb") as f:
                        for block in reply.iter_content(chunk):
                            f.write(block)
                            done += len(block)
                            if progress is not None:
                                progress(done, size)
            got_size = part.stat().st_size
            if got_size != size:
                raise DownloadError(f"incomplete download: {got_size} of {size} bytes")
            algos = ["sha256"] + (["md5"] if md5 else [])
            digests = file_digests(part, algos)
            if digests["sha256"] != sha256 or (md5 and digests["md5"] != md5):
                part.unlink()
                raise DownloadError("checksum mismatch; the download is restarted")
            os.replace(part, target)
            return "downloaded"
        except (requests.RequestException, OSError, DownloadError) as exc:
            last_error = exc
    raise DownloadError(f"{target.name}: failed after {retries + 1} attempts: {last_error}")
