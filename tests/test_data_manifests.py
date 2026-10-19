"""Tests of the data manifests, the archive reader, the downloader and the data check.

No test downloads anything: downloads are tested against a local HTTP server, and split zip
archives are made with the ``zip`` command when it is available.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import struct
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hexmod import data_sources as dsrc
from hexmod import datasets as ds

REPO = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX32 = re.compile(r"^[0-9a-f]{32}$")
ALL_KEYS = list(ds.CANONICAL_ORDER)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------- manifests
@pytest.mark.parametrize("key", ALL_KEYS)
def test_manifest_rows_are_well_formed(key):
    rows = dsrc.read_manifest(key)
    assert rows, key
    with open(dsrc.manifest_path(key), encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header[:len(dsrc.MANIFEST_COLUMNS)] == dsrc.MANIFEST_COLUMNS
    top = f"{key}/"
    paths = [r["path"] for r in rows]
    assert len(set(paths)) == len(paths)
    archives = {r["path"] for r in rows if r["role"] == "archive"}
    for r in rows:
        assert r["path"].startswith(top) and "\\" not in r["path"] and ".." not in r["path"]
        assert r["name"] == r["path"].rsplit("/", 1)[-1]
        assert r["role"] in ("file", "archive", "volume", "member")
        assert r["bytes"] > 0
        assert HEX64.match(r["sha256"]), r["path"]
        assert r["md5"] == "" or HEX32.match(r["md5"]), r["path"]
        if r["role"] == "member":
            assert r["archive"] in archives and r["source"]
        else:
            assert r["archive"] == ""
        if r["role"] != "member":
            assert r["source"], r["path"]


def test_manifest_paths_agree_with_dataset_metadata():
    for key in ALL_KEYS:
        meta = ds.DATASETS[key]
        assert (REPO / meta["manifest"]).is_file()
        for r in dsrc.final_files(dsrc.read_manifest(key)):
            assert any(r["path"] == p or r["path"].startswith(p.rstrip("/") + "/") for p in meta["raw_subpaths"]), r


def test_vollan_manifest():
    rows = dsrc.read_manifest("vollan2025")
    of = [r for r in rows if r["source"].startswith("sharing_v4/navigation/of/")]
    assert len(of) == 31 and sum(r["bytes"] for r in of) == 11_852_351_426
    used = [r for r in of if r["used_in_cohort"] == "True"]
    assert len(used) == 24 and sum(r["bytes"] for r in used) == 9_452_057_661
    assert {r["used_in_cohort"] for r in of} == {"True", "False"}
    doc = [r for r in rows if r["name"] == "mat_files.md"]
    assert len(doc) == 1 and doc[0]["path"] == "vollan2025/sharing_v4/documentation/mat_files.md"
    for r in rows:  # where the archive publishes a plain md5 as the object hash, it is the md5
        if HEX32.match(r["etag"]):
            assert r["md5"] == r["etag"]
        else:
            assert r["md5"] == "" and re.match(r"^[0-9a-f]{32}-\d+$", r["etag"])


def test_brun_manifest():
    rows = dsrc.read_manifest("brun2008")
    (archive,) = [r for r in rows if r["role"] == "archive"]
    assert archive["bytes"] == 1_436_379_107 and archive["md5"]
    members = [r for r in rows if r["role"] == "member"]
    assert len(members) == 6
    assert all(r["path"].startswith("brun2008/Brun2008-v3.0.0/square/recordings/") for r in members)
    assert all(r["source"] == r["path"][len("brun2008/"):] for r in members)
    assert sorted(r["name"] for r in members if r["used_in_cohort"] == "True") == [
        "11468_07040601+02.mat", "11468_16030601+02.mat"]


def test_hafting_manifest():
    rows = dsrc.read_manifest("hafting2005")
    assert [r["name"] for r in rows] == ["Hafting_Fig2c_Trial1.mat", "Hafting_Fig2c_Trial2.mat",
                                         "Hafting_Fig2d_Trial1.mat", "Hafting_Fig2d_Trial2.mat", "rat_10925.mat"]
    assert sum(r["bytes"] for r in rows) == 6_054_104 and all(r["md5"] for r in rows)


def test_sargolini_manifest():
    rows = dsrc.read_manifest("sargolini2006")
    vols = [r for r in rows if r["role"] in ("volume", "archive")]
    assert [r["name"] for r in vols] == ["gridcell_data_sargolini.z01", "gridcell_data_sargolini.z02",
                                         "gridcell_data_sargolini.zip"]
    assert sum(r["bytes"] for r in vols) == 2_367_241_994 and vols[-1]["bytes"] == 270_089_994
    assert [r["role"] for r in vols] == ["volume", "volume", "archive"]
    members = [r for r in rows if r["role"] == "member"]
    assert len(members) == 150
    assert [int(r["listdir_order"]) for r in members] == list(range(150))
    assert all(r["path"] == "sargolini2006/subset/" + r["name"] for r in members)
    assert all(r["source"] == "Data_sargolini/" + r["name"] for r in members)
    assert {r["archive"] for r in members} == {vols[-1]["path"]}
    kinds = {"POS": 0, "unit": 0, "EEG": 0}
    for r in members:
        if r["name"].endswith("_POS.mat"):
            kinds["POS"] += 1
        elif re.search(r"_T\d+C\d+\.mat$", r["name"]):
            kinds["unit"] += 1
        else:
            kinds["EEG"] += 1
    assert kinds == {"POS": 13, "unit": 109, "EEG": 28}


def test_zenodo_manifests():
    """barry2012 and olafsdottir2016: one HDF5 file each, from one Zenodo record, with md5 and sha256."""
    expected = {"barry2012": ("barry2012_familiar_box_sessions.h5", 47_170_260),
                "olafsdottir2016": ("olafsdottir2016_open_field_sessions.h5", 27_785_094)}
    for key, (name, size) in expected.items():
        (r,) = dsrc.read_manifest(key)
        assert (r["path"], r["name"], r["role"], r["bytes"], r["source"]) == (f"{key}/{name}", name, "file", size, name)
        assert HEX32.match(r["md5"]) and HEX64.match(r["sha256"])
        assert ds.DATASETS[key]["raw_subpaths"] == [r["path"]]
        assert dsrc.SOURCES[key]["host"] == "zenodo" and dsrc.SOURCES[key]["record"] == "23477461"
        assert dsrc.object_url(key, name) == f"https://zenodo.org/records/23477461/files/{name}?download=1"
        assert dsrc.resolve_url(key, name) == dsrc.object_url(key, name)


def test_zenodo_listing_is_parsed(monkeypatch):
    """The record's file listing (a small JSON reply of the Zenodo API) gives sizes and md5 checksums."""
    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"files": [{"key": "a.h5", "size": 12, "checksum": "md5:" + "0" * 32},
                              {"key": "README.md", "size": 3, "checksum": "md5:" + "1" * 32}]}

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, **kwargs):
            self.urls.append(url)
            return Reply()

    session = Session()
    assert dsrc.list_remote("barry2012", session=session) == {"a.h5": (12, "0" * 32), "README.md": (3, "1" * 32)}
    assert session.urls == ["https://zenodo.org/api/records/23477461"]


def test_sources_and_addresses():
    assert set(dsrc.SOURCES) == set(ALL_KEYS)
    assert dsrc.object_url("hafting2005", "rat_10925.mat") == (
        "https://s3.nird.sigma2.no/archive-ro/60452682-5815-41eb-bd0a-fb11b914623d/rat_10925.mat")
    assert dsrc.object_url("vollan2025", "sharing_v4/navigation/of/24365_2.mat").startswith(
        "https://rgw.cscs.ch/ebrains:d-4080b78d-edc5-4ae4-8144-7f6de79930ea/sharing_v4/")
    assert dsrc.proxy_url("brun2008", "x") is None
    assert dsrc.proxy_url("vollan2025", "a/b.mat").endswith("/d-4080b78d-edc5-4ae4-8144-7f6de79930ea/a/b.mat")


def test_download_list_totals(capsys):
    """The dry run lists every download; totals equal the manifests."""
    dl = load_script("data_download")
    totals = {}
    for full in (False, True):
        assert dl.main(["--list"] + (["--sargolini-full-archive"] if full else [])) == 0
        out = capsys.readouterr().out
        totals[full] = int(re.search(r"^total: ([\d,]+) bytes", out, re.M).group(1).replace(",", ""))
    expected = 0
    for key in ALL_KEYS:
        expected += sum(r["bytes"] for r in dsrc.download_rows(dsrc.read_manifest(key)))
    assert totals[False] == expected == 13_639_845_431
    assert totals[True] == 15_736_997_431


# ------------------------------------------------------------------------------- archive reader
def _blob(n, seed):
    return random.Random(seed).randbytes(n) if hasattr(random.Random, "randbytes") else os.urandom(n)


def test_zip_reader_single_volume(tmp_path):
    files = {"a/x.bin": _blob(50_000, 1), "a/y.txt": b"hello " * 5000, "b/empty": b""}
    path = tmp_path / "t.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("a/x.bin", files["a/x.bin"], compress_type=zipfile.ZIP_STORED)
        z.writestr("a/y.txt", files["a/y.txt"], compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("b/empty", b"", compress_type=zipfile.ZIP_DEFLATED)
    with dsrc.ZipVolumes([path]) as zv:
        entries, n_disks = dsrc.read_central_directory(zv)
        assert n_disks == 1 and [e.name for e in entries] == list(files)
        for e in entries:
            assert b"".join(dsrc.iter_member(zv, e, chunk=4096)) == files[e.name]
    # a damaged member fails its CRC-32 check
    data = bytearray(path.read_bytes())
    pos = data.find(files["a/x.bin"][:64])
    data[pos + 100] ^= 0xFF
    bad = tmp_path / "bad.zip"
    bad.write_bytes(bytes(data))
    with dsrc.ZipVolumes([bad]) as zv:
        entries, _ = dsrc.read_central_directory(zv)
        with pytest.raises(dsrc.ZipFormatError):
            list(dsrc.iter_member(zv, entries[0]))


def test_zip64_extra_field():
    body = struct.pack("<QQQI", 5_000_000_000, 4_000_000_000, 6_000_000_000, 3)
    extra = struct.pack("<HH", 0x9999, 2) + b"xx" + struct.pack("<HH", 1, len(body)) + body
    assert dsrc._zip64_extra(extra, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF) == (
        5_000_000_000, 4_000_000_000, 6_000_000_000, 3)
    assert dsrc._zip64_extra(struct.pack("<HH", 1, 8) + struct.pack("<Q", 7), 0xFFFFFFFF, 10, 20, 0) == (7, 10, 20, 0)


@pytest.mark.skipif(shutil.which("zip") is None, reason="the zip command is not available")
def test_split_archive_last_volume_only(tmp_path):
    """Members stored in the last volume of a split archive are read from that volume alone."""
    src = tmp_path / "src" / "Data"
    src.mkdir(parents=True)
    files = {}
    sizes_k = [23, 30, 37, 44, 51, 58, 65, 72, 3, 3, 3, 3]  # the last members fit in the last volume
    for i, size_k in enumerate(sizes_k):
        name = f"f{i:02d}.bin"
        files[f"Data/{name}"] = _blob(size_k * 1000, i)
        (src / name).write_bytes(files[f"Data/{name}"])
    subprocess.run(["zip", "-q", "-r", "-0", "-s", "100k", str(tmp_path / "arch.zip"), "Data"],
                   cwd=tmp_path / "src", check=True)
    vols = sorted(tmp_path.glob("arch.z[0-9][0-9]")) + [tmp_path / "arch.zip"]
    assert len(vols) >= 3
    sizes = [v.stat().st_size for v in vols]
    with dsrc.ZipVolumes(vols) as zv:
        entries, n_disks = dsrc.read_central_directory(zv)
        assert n_disks == len(vols)
        spanning = 0
        for e in entries:
            if e.name.endswith("/"):
                continue
            assert b"".join(dsrc.iter_member(zv, e)) == files[e.name]
            fnl, exl = struct.unpack("<HH", zv.read(e.disk, e.offset, 30)[26:30])
            spanning += e.offset + 30 + fnl + exl + e.compressed_size > sizes[e.disk]
        assert spanning >= 1  # some members cross a volume boundary
    last = len(vols) - 1
    on_last = {e.name for e in entries if e.disk == last and not e.name.endswith("/")}
    assert on_last
    with dsrc.ZipVolumes([None] * last + [vols[-1]]) as zv:
        entries2, _ = dsrc.read_central_directory(zv)
        for e in entries2:
            if e.name.endswith("/"):
                continue
            if e.disk == last:
                assert b"".join(dsrc.iter_member(zv, e)) == files[e.name]
            else:
                with pytest.raises(dsrc.MissingVolumeError):
                    b"".join(dsrc.iter_member(zv, e))

    # the same through the manifest-driven extraction, with an exact folder and a stray file
    base = tmp_path / "data"
    arch = base / "sargolini2006" / "archive"
    arch.mkdir(parents=True)
    shutil.copy(vols[-1], arch / "arch.zip")
    rows = [{"dataset": "sargolini2006", "path": f"sargolini2006/archive/{v.name}", "name": v.name,
             "role": "archive" if v.suffix == ".zip" else "volume", "bytes": v.stat().st_size,
             "sha256": sha256(v.read_bytes()), "md5": "", "source": v.name, "archive": ""} for v in vols]
    for name in sorted(on_last):
        short = name.split("/", 1)[1]
        rows.append({"dataset": "sargolini2006", "path": f"sargolini2006/subset/{short}", "name": short,
                     "role": "member", "bytes": len(files[name]), "sha256": sha256(files[name]), "md5": "",
                     "source": name, "archive": "sargolini2006/archive/arch.zip"})
    result = dsrc.extract_members("sargolini2006", base, rows, log=lambda s: None)
    assert sorted(result.values()) == ["extracted"] * len(on_last)
    assert dsrc.check_dataset("sargolini2006", base, rows=rows)["status"] == "ok"
    assert dsrc.extract_members("sargolini2006", base, rows, log=lambda s: None) == {
        r["path"]: "present" for r in rows if r["role"] == "member"}
    (base / "sargolini2006" / "subset" / "stray.mat").write_bytes(b"x")
    res = dsrc.check_dataset("sargolini2006", base, rows=rows)
    assert res["status"] == "unexpected files" and res["extra_files"] == ["sargolini2006/subset/stray.mat"]


# ---------------------------------------------------------------------------------- verification
def test_verify_file_and_check_dataset(tmp_path):
    data = b"0123456789" * 100
    row = {"dataset": "hafting2005", "path": "hafting2005/a.mat", "name": "a.mat", "role": "file",
           "bytes": len(data), "sha256": sha256(data), "md5": hashlib.md5(data).hexdigest(), "source": "a.mat",
           "archive": ""}
    path = tmp_path / row["path"]
    assert dsrc.verify_file(path, row) == "missing"
    path.parent.mkdir(parents=True)
    path.write_bytes(data[:-1])
    assert dsrc.verify_file(path, row) == "wrong size"
    path.write_bytes(data[:-1] + b"x")
    assert dsrc.verify_file(path, row) == "checksum mismatch"
    assert dsrc.verify_file(path, row, "size") == "ok"
    path.write_bytes(data)
    assert dsrc.verify_file(path, row) == "ok"
    row2 = dict(row, path="hafting2005/b.mat", name="b.mat")
    res = dsrc.check_dataset("hafting2005", tmp_path, rows=[row, row2], workers=2)
    assert res["status"] == "incomplete" and res["n_ok"] == 1 and res["bad_files"] == ["hafting2005/b.mat"]
    assert res["problems"] == {"hafting2005/b.mat": "missing"} and res["bytes"] == len(data)


def test_data_check_script(tmp_path, monkeypatch, capsys):
    """data_check writes data_check.json; without data the check fails and the exit code is 1."""
    monkeypatch.setenv("HEXMOD_RESULTS", str(tmp_path / "results"))
    check = load_script("data_check")
    rc = check.main(["--data", str(tmp_path / "empty"), "--quick"])
    report = json.loads((tmp_path / "results" / "data_check.json").read_text())
    assert rc == 1 and report["ok"] is False and report["checksums"] == "size"
    assert set(report["datasets"]) == set(ALL_KEYS)
    assert all(v["status"] == "missing" for v in report["datasets"].values())
    assert (tmp_path / "results" / "logs" / "data_check.json").is_file()


@pytest.mark.data
def test_data_folder_passes_size_check(data_dir):
    """The local data folder holds the six datasets with the right sizes."""
    for key in ALL_KEYS:
        res = dsrc.check_dataset(key, data_dir, "size")
        assert res["status"] == "ok", (key, res["problems"], res["extra_files"])


# --------------------------------------------------------------------------------------- download
class _Server:
    """Local HTTP server with Range support; can ignore ranges or drop the first connection.

    ``files`` is the content of ``/file.bin`` (bytes) or a mapping ``{path: bytes}``.
    """

    def __init__(self, files, honour_range=True, drop_first=False):
        self.files = {"file.bin": files} if isinstance(files, bytes) else dict(files)
        self.honour_range, self.drop_first = honour_range, drop_first
        self.ranges, self.paths = [], []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _blob(self):
                return outer.files.get(self.path.lstrip("/"))

            def do_GET(self):
                blob = self._blob()
                if blob is None:
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                outer.ranges.append(rng)
                outer.paths.append(self.path.lstrip("/"))
                start = 0
                if rng and outer.honour_range:
                    start = int(re.match(r"bytes=(\d+)-", rng).group(1))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(blob) - 1}/{len(blob)}")
                else:
                    self.send_response(200)
                body = blob[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if outer.drop_first:
                    outer.drop_first = False
                    self.wfile.write(body[: len(body) // 3])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

            def do_HEAD(self):
                blob = self._blob()
                self.send_response(200 if blob is not None else 404)
                self.send_header("Content-Length", str(len(blob or b"")))
                self.end_headers()

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        self.url = self.base + "file.bin"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        return False


@pytest.fixture
def blob():
    return _blob(300_000, 7)


def test_download_resumes_partial_file(tmp_path, blob):
    pytest.importorskip("requests")
    target = tmp_path / "d" / "file.bin"
    target.parent.mkdir()
    (target.parent / "file.bin.part").write_bytes(blob[:100_000])
    with _Server(blob) as srv:
        assert dsrc.download_file(srv.url, target, len(blob), sha256(blob), backoff=0) == "downloaded"
        assert srv.ranges == ["bytes=100000-"]
        assert dsrc.head(srv.url)[:2] == (200, len(blob))
    assert target.read_bytes() == blob and not (target.parent / "file.bin.part").exists()


def test_download_restarts_when_range_is_ignored(tmp_path, blob):
    pytest.importorskip("requests")
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(blob[:5000])
    with _Server(blob, honour_range=False) as srv:
        dsrc.download_file(srv.url, target, len(blob), sha256(blob), backoff=0)
    assert target.read_bytes() == blob


def test_download_retries_after_dropped_connection(tmp_path, blob):
    pytest.importorskip("requests")
    target = tmp_path / "file.bin"
    with _Server(blob, drop_first=True) as srv:
        dsrc.download_file(lambda: srv.url, target, len(blob), sha256(blob), md5=hashlib.md5(blob).hexdigest(),
                           backoff=0)
        assert srv.ranges[0] is None and srv.ranges[1].startswith("bytes=") and len(srv.ranges) == 2
    assert target.read_bytes() == blob


def test_download_rejects_wrong_checksum(tmp_path, blob):
    pytest.importorskip("requests")
    target = tmp_path / "file.bin"
    with _Server(blob) as srv:
        with pytest.raises(dsrc.DownloadError):
            dsrc.download_file(srv.url, target, len(blob), "0" * 64, retries=1, backoff=0)
    assert not target.exists() and not (tmp_path / "file.bin.part").exists()


def _row(key, path, role, data, source="", archive="", md5=True):
    return {"dataset": key, "path": path, "name": path.rsplit("/", 1)[-1], "role": role, "bytes": len(data),
            "sha256": sha256(data), "md5": hashlib.md5(data).hexdigest() if md5 else "", "source": source,
            "archive": archive}


def test_download_script_end_to_end(tmp_path, monkeypatch, capsys):
    """data_download.py fetches files and an archive from a local server, extracts and verifies."""
    pytest.importorskip("requests")
    files = {"a.mat": _blob(40_000, 1), "b.mat": _blob(20_000, 2)}
    members = {"Arch/rec/x.mat": _blob(30_000, 3), "Arch/rec/y.mat": b"y" * 5000}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
        z.writestr("Arch/other.txt", b"not used")
    files["arch.zip"] = buf.getvalue()
    manifests = {
        "hafting2005": [_row("hafting2005", f"hafting2005/{n}", "file", files[n], source=n)
                        for n in ("a.mat", "b.mat")],
        "brun2008": [_row("brun2008", "brun2008/archive/arch.zip", "archive", files["arch.zip"], source="arch.zip")]
        + [_row("brun2008", f"brun2008/{n}", "member", d, source=n, archive="brun2008/archive/arch.zip", md5=False)
           for n, d in members.items()],
    }
    dl = load_script("data_download")
    dest = tmp_path / "data"
    with _Server(files) as srv:
        monkeypatch.setattr(dsrc, "read_manifest", lambda key, path=None: [dict(r) for r in manifests[key]])
        monkeypatch.setattr(dsrc, "resolve_url", lambda key, source, session=None, timeout=30.0: srv.base + source)
        args = ["--datasets", "hafting2005,brun2008", "--dest", str(dest), "--retries", "0"]
        assert dl.main(args) == 0
        assert sorted(srv.paths) == ["a.mat", "arch.zip", "b.mat"]
        for n in ("a.mat", "b.mat"):
            assert (dest / "hafting2005" / n).read_bytes() == files[n]
        for n, d in members.items():
            assert (dest / "brun2008" / n).read_bytes() == d
        assert not (dest / "brun2008" / "Arch" / "other.txt").exists()
        assert dl.main(args) == 0 and len(srv.paths) == 3          # nothing downloaded again
        (dest / "hafting2005" / "a.mat").unlink()
        (dest / "hafting2005" / "b.mat").write_bytes(b"damaged")
        assert dl.main(args + ["--only", "a.mat"]) == 0 and srv.paths[3:] == ["a.mat"]
        assert dl.main(args) == 0 and srv.paths[4:] == ["b.mat"]   # the damaged file is replaced
        assert dl.main(args + ["--delete-archives"]) == 0
        assert not (dest / "brun2008" / "archive" / "arch.zip").exists()
        assert dl.main(args) == 0 and len(srv.paths) == 5          # members valid: no archive needed
    out = capsys.readouterr().out
    assert "2 files extracted and verified" in out
