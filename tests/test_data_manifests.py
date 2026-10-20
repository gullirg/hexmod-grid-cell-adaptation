"""Tests of the data manifests, the archive reader, the download list and the data check.

No test downloads anything. Split zip archives are made with the ``zip`` command when it is
available.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import struct
import subprocess
import zipfile
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
