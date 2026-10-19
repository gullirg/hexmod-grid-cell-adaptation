"""Tests of the session and unit manifests, the small rat inputs and ``hexmod.preprocessing.labels``.

The manifests fix every label and the canonical order of all recordings, so that no step depends
on the order in which a file system lists the raw files.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hexmod import data_sources
from hexmod import datasets as ds
from hexmod.preprocessing import labels

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "inputs"

# sessions and units produced by preprocessing, per dataset
N_SESSIONS = {"sargolini2006": 13, "brun2008": 2, "olafsdottir2016": 24, "vollan2025": 27, "hafting2005": 3,
              "barry2012": 57}
N_UNITS = {"sargolini2006": 109, "brun2008": 5, "olafsdottir2016": 406, "vollan2025": 5751, "hafting2005": 23,
           "barry2012": 321}
#: datasets distributed as one HDF5 file that holds every session
HDF5_DATASETS = ("barry2012", "olafsdottir2016")
TABLE1_ORDER = ["brun2008", "barry2012", "olafsdottir2016", "hafting2005", "sargolini2006", "vollan2025"]
# Table 1: animals, sessions, cells of the published cohort
TABLE1 = {"brun2008": (1, 2, 5), "barry2012": (7, 57, 118), "olafsdottir2016": (6, 24, 61), "hafting2005": (2, 3, 23),
          "sargolini2006": (1, 2, 7), "vollan2025": (15, 24, 3724)}
UNIT_KEY_PATTERNS = {
    "vollan2025": r"mec:[1-9][0-9]*",
    "brun2008": r"T[0-9]+C[0-9]+",
    "hafting2005": r"[A-Za-z0-9_]+\.mat:[A-Za-z0-9_]+",
    "sargolini2006": r"[0-9]+-[0-9+]+_T[0-9]+C[0-9]+\.mat",
    "barry2012": r"unit_[0-9]{3}",
    "olafsdottir2016": r"unit_[0-9]{3}",
}


@pytest.fixture(scope="module")
def sessions():
    return labels.read_sessions()


@pytest.fixture(scope="module")
def units():
    return labels.read_units()


# ------------------------------------------------------------------------------------ sessions.csv
def test_session_manifest_layout(sessions):
    assert list(sessions.columns) == labels.SESSION_COLUMNS
    assert sessions["order"].tolist() == list(range(len(sessions)))
    assert sessions["session_id"].is_unique
    raw = pd.read_csv(labels.sessions_path(), keep_default_na=False)
    assert raw["order"].tolist() == list(range(len(raw))), "rows must be stored in canonical order"


def test_session_labels(sessions):
    for r in sessions.itertuples():
        assert ds.key_from_label(r.animal_id) == r.dataset
        assert re.fullmatch(r"[A-Z][0-9]+", r.animal_id)
        assert re.fullmatch(re.escape(r.animal_id) + r"_[0-9]+", r.session_id)


def test_datasets_in_canonical_order(sessions, units):
    for df in (sessions, units):
        blocks = list(dict.fromkeys(df["dataset"]))
        assert blocks == ds.CANONICAL_ORDER
        rank = df["dataset"].map(ds.canonical_rank).to_numpy()
        assert np.all(np.diff(rank) >= 0), "each dataset must be one contiguous block"


def test_animals_and_sessions_contiguous(sessions, units):
    for col, df in (("animal_id", sessions), ("session_id", units), ("animal_id", units)):
        runs = (df[col] != df[col].shift()).sum()
        assert runs == df[col].nunique(), f"{col} values are not contiguous"


def test_session_counts(sessions):
    assert sessions.groupby("dataset").size().to_dict() == N_SESSIONS
    assert len(sessions) == 126


def test_cohort_sessions_match_table1(sessions):
    coh = sessions[sessions["in_cohort"]]
    assert len(coh) == 112 and coh["animal_id"].nunique() == 32
    for key, (n_animals, n_sessions, _) in TABLE1.items():
        part = coh[coh["dataset"] == key]
        assert (part["animal_id"].nunique(), len(part)) == (n_animals, n_sessions), key
    # the HDF5 files hold only sessions with cohort cells
    assert sessions.loc[sessions["dataset"].isin(HDF5_DATASETS), "in_cohort"].all()


def test_raw_sources(sessions):
    for r in sessions.itertuples():
        assert r.raw_path, r.session_id
        for part in r.raw_path.split(";"):
            suffix = ".h5" if r.dataset in HDF5_DATASETS else ".mat"
            assert part.endswith(suffix) and not part.startswith("/") and "\\" not in part and ".." not in part
    for key in HDF5_DATASETS:
        (row,) = [r for r in data_sources.read_manifest(key) if r["role"] == "file"]
        assert set(sessions.loc[sessions["dataset"] == key, "raw_path"]) == {row["name"]}, key
    vol = sessions[sessions["dataset"] == "vollan2025"]
    for r in vol.itertuples():
        m = re.fullmatch(r"sharing_v4/navigation/of/([0-9]+)_([0-9]+)\.mat", r.raw_path)
        assert m and r.session_id.endswith("_" + m.group(2)), r.raw_path
    files = sessions.loc[~sessions["dataset"].isin(HDF5_DATASETS), "raw_path"].str.split(";").explode()
    assert files.is_unique


def test_raw_paths_listed_in_file_manifests(sessions, units):
    """Every raw file named by the manifests is in the dataset's checksum list."""
    for key in ds.CANONICAL_ORDER:
        listing = INPUTS / "manifests" / f"files_{key}.csv"
        known = set(pd.read_csv(listing, keep_default_na=False)["path"])
        files = [p for r in sessions[sessions["dataset"] == key].itertuples() for p in r.raw_path.split(";")]
        if key == "sargolini2006":
            rows = sessions[sessions["dataset"] == key].itertuples()
            folder = {r.session_id: r.raw_path.rsplit("/", 1)[0] for r in rows}
            files += [folder[u.session_id] + "/" + u.raw_unit_key for u in units[units["dataset"] == key].itertuples()]
        missing = [f for f in files if f"{key}/{f}" not in known]
        assert not missing, (key, missing[:5])


# --------------------------------------------------------------------------------------- units.csv
def test_unit_manifest_layout(units, sessions):
    assert list(units.columns) == labels.UNIT_COLUMNS
    assert units["order"].tolist() == list(range(len(units)))
    assert not units.duplicated(["session_id", "cell_id"]).any()
    assert units["unique_cell_id"].is_unique
    assert len(units) == 6615 and units.groupby("dataset").size().to_dict() == N_UNITS
    # units follow the session order, and n_units agrees
    sess_rank = dict(zip(sessions["session_id"], sessions["order"]))
    assert np.all(np.diff(units["session_id"].map(sess_rank).to_numpy()) >= 0)
    assert units.groupby("session_id").size().reindex(sessions["session_id"]).tolist() == sessions["n_units"].tolist()
    per_session = units.drop_duplicates("session_id").set_index("session_id")
    s = sessions.set_index("session_id")
    assert (per_session["animal_id"] == s.loc[per_session.index, "animal_id"]).all()
    assert (per_session["dataset"] == s.loc[per_session.index, "dataset"]).all()


def test_unit_labels_and_raw_keys(units):
    for u in units.itertuples():
        assert u.unique_cell_id.startswith(f"{u.animal_id}_{u.session_id}_"), u.unique_cell_id
        assert re.fullmatch(UNIT_KEY_PATTERNS[u.dataset], u.raw_unit_key), (u.dataset, u.raw_unit_key)
        if u.dataset == "vollan2025":
            assert u.raw_unit_key == f"mec:{u.cell_id}" and u.unique_cell_id.endswith(f"_cell{u.cell_id}")
        elif u.dataset == "brun2008":
            assert u.raw_unit_key.endswith(f"C{u.cell_id}")
        elif u.dataset == "barry2012":
            assert u.raw_unit_key == f"unit_{u.cell_id:03d}" and u.unique_cell_id.endswith(f"_T0_C{u.cell_id}")
        elif u.dataset == "olafsdottir2016":
            assert u.raw_unit_key == f"unit_{u.cell_id - 1:03d}" and u.unique_cell_id.endswith(f"_Cell{u.cell_id}")
        elif u.dataset == "sargolini2006":
            assert u.raw_unit_key.startswith(_sargolini_trial(u.raw_unit_key))
    sar = units[units["dataset"] == "sargolini2006"]
    assert sorted(sar["cell_id"]) == list(range(1, len(sar) + 1)), "Sargolini cells are numbered over the whole dataset"


def _sargolini_trial(name: str) -> str:
    return name.split("_")[0]


def test_sargolini_units_belong_to_their_session(sessions, units):
    rows = sessions[sessions["dataset"] == "sargolini2006"].itertuples()
    pos = {r.session_id: r.raw_path.rsplit("/", 1)[1].split("_")[0] for r in rows}
    for u in units[units["dataset"] == "sargolini2006"].itertuples():
        assert _sargolini_trial(u.raw_unit_key) == pos[u.session_id]


# ------------------------------------------------------------------------------------ labels module
def _toy_data(units_df, sessions_df, key):
    """Nested dict of one dataset in a scrambled order (no arrays needed)."""
    rng = np.random.default_rng(0)
    data = {}
    srows = sessions_df[sessions_df["dataset"] == key]
    for r in srows.iloc[rng.permutation(len(srows))].itertuples():
        u = units_df[units_df["session_id"] == r.session_id]
        u = u.iloc[rng.permutation(len(u))]
        spikes = {int(c): np.array([float(c)]) for c in u["cell_id"]}
        data.setdefault(r.animal_id, {})[r.session_id] = {
            "spike_times": spikes,
            "unique_cell_ids": dict(zip(u["cell_id"].astype(int), u["unique_cell_id"])),
            "session_id": r.session_id,
        }
    return data


def _flatten(data):
    return [(a, s, list(sd["spike_times"])) for a, ss in data.items() for s, sd in ss.items()]


def test_canonical_order_restores_manifest_order(sessions, units):
    for key in ("sargolini2006", "hafting2005", "brun2008"):
        data = _toy_data(units, sessions, key)
        out = labels.canonical_order(data, key)
        exp_sessions = sessions.loc[sessions["dataset"] == key, "session_id"].tolist()
        assert [s for _, s, _ in _flatten(out)] == exp_sessions
        for a, s, cells in _flatten(out):
            assert cells == units.loc[units["session_id"] == s, "cell_id"].tolist()
            assert list(out[a][s]["unique_cell_ids"]) == cells


def test_canonical_order_refuses_mismatches(sessions, units):
    data = _toy_data(units, sessions, "hafting2005")
    a = next(iter(data))
    s = next(iter(data[a]))
    bad = {k: dict(v) for k, v in data.items()}
    bad[a] = dict(bad[a])
    bad[a][s] = dict(data[a][s], spike_times={**data[a][s]["spike_times"], 999: np.zeros(1)},
                     unique_cell_ids={**data[a][s]["unique_cell_ids"], 999: "x"})
    with pytest.raises(labels.ManifestError):
        labels.canonical_order(bad, "hafting2005")
    missing = {k: dict(v) for k, v in data.items()}
    del missing[a][s]
    with pytest.raises(labels.ManifestError):
        labels.canonical_order(missing, "hafting2005")
    relabelled = {k: dict(v) for k, v in data.items()}
    relabelled[a][s] = dict(data[a][s], unique_cell_ids={c: "wrong" for c in data[a][s]["unique_cell_ids"]})
    with pytest.raises(labels.ManifestError):
        labels.canonical_order(relabelled, "hafting2005")


def test_merge_datasets_uses_canonical_dataset_order(sessions, units):
    parts = {key: _toy_data(units, sessions, key) for key in ("hafting2005", "brun2008", "sargolini2006")}
    merged = labels.merge_datasets(parts)
    prefixes = list(dict.fromkeys(a[0] for a in merged))
    assert prefixes == ["S", "B", "H"]


def test_lookups(sessions):
    row = labels.session_for_raw_path("brun2008", "Brun2008-v3.0.0/square/recordings/11468_07040601+02.mat")
    assert row["session_id"] == "B1_2"
    h11 = sessions[sessions["session_id"] == "H1_1"].iloc[0]
    for part in h11.raw_path.split(";"):
        assert labels.session_for_raw_path("hafting2005", part)["session_id"] == "H1_1"
    u = labels.unit_for_raw_key("vollan2025", "V10_1", "mec:341")
    assert (u["cell_id"], u["unique_cell_id"]) == (341, "V10_V10_1_cell341")
    with pytest.raises(labels.ManifestError):
        labels.session_for_raw_path("vollan2025", "sharing_v4/navigation/of/00000_1.mat")
    assert labels.read_sessions("brun2008")["dataset"].eq("brun2008").all()
    assert len(labels.read_units(["brun2008"])) == 5


def test_unit_order_and_sorting(units):
    sub = units.sample(n=200, random_state=1)
    out = labels.sort_canonical(sub)
    assert out["order"].tolist() == sorted(sub["order"].tolist())
    assert labels.is_canonical(out) and not labels.is_canonical(sub.iloc[::-1])
    with pytest.raises(labels.ManifestError):
        labels.unit_order(["V10_1"], [10 ** 6])


# ------------------------------------------------------------------------------------------ inputs/rat
def test_module_labels(sessions):
    bins = labels.read_module_bins()
    cohort_animals = sessions.loc[sessions["in_cohort"], "animal_id"].unique().tolist()
    assert list(bins) == cohort_animals
    for a, e in bins.items():
        assert e[0] == 0 and np.all(np.diff(e) > 0), a
    # numpy.digitize semantics: lower edge included, upper edge excluded, outside = 0
    e = bins["B1"]
    np.testing.assert_array_equal(e, [0, 80, 110, 160])
    got = labels.assign_modules(["B1"] * 6, [10.0, 79.999, 80.0, 109.9, 110.0, 160.0])
    np.testing.assert_array_equal(got, [1, 1, 2, 2, 3, 0])
    assert labels.assign_modules(["V10"], [100.0])[0] == 3


def test_example_cells(units):
    ex = labels.read_example_cells()
    known = set(zip(units["session_id"], units["cell_id"]))
    assert all((s, c) in known for s, c in zip(ex["session_id"], ex["cell_id"]))
    for r in ex.itertuples():
        assert ds.key_from_label(r.session_id) == r.dataset and r.session_id.startswith(r.animal_id + "_")
    assert list(ex.columns) == labels.EXAMPLE_COLUMNS
    assert ex.groupby("panel").size().to_dict() == {"Fig2A": 5, "Fig3A": 1, "Fig3B": 1, "Fig4A": 2, "Fig4B": 2,
                                                    "SuppFig2A": 1, "SuppFig3A": 12}
    assert not ex.duplicated(["panel", "position"]).any()
    fig4 = ex[ex["panel"] == "Fig4A"]
    assert fig4[["t_start_s", "t_end_s"]].drop_duplicates().values.tolist() == [[126.0, 138.0]]
    fig3 = ex[ex["panel"] == "Fig3A"].iloc[0]
    assert (fig3.session_id, fig3.cell_id, fig3.t_start_s, fig3.t_end_s) == ("B1_2", 2, 143.0, 155.0)


def test_table1_text():
    t = pd.read_csv(INPUTS / "rat" / "table1_text.csv", keep_default_na=False)
    assert list(t.columns) == ["table_row", "dataset", "environment", "grid_scale_text"]
    assert t["dataset"].tolist() == TABLE1_ORDER
    assert t["table_row"].tolist() == list(range(1, 7))
    assert (t["environment"] != "").all() and t["grid_scale_text"].str.fullmatch(r"~[0-9]+-[0-9]+ cm").all()


def test_timescale_fit_settings():
    fs = json.loads((INPUTS / "rat" / "timescale_fit_settings.json").read_text(encoding="utf-8"))
    assert len(fs["single"]["p0"]) == 3 and len(fs["double"]["p0"]) == 5
    for k in ("single", "double"):
        lo, hi = fs[k]["bounds"]
        assert len(lo) == len(hi) == len(fs[k]["p0"]) and fs[k]["method"] == "trf"


# ------------------------------------------------------------------------ no file-system listing order
LISTING_FUNCTIONS = {("os", "listdir"), ("os", "scandir"), ("os", "walk"), ("glob", "glob"), ("glob", "iglob")}
LISTING_METHODS = {"glob", "rglob", "iterdir"}


def listing_calls_without_sorted(source: str):
    """Return (line, call) for directory listings whose result is not passed straight to sorted()."""
    tree = ast.parse(source)
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f, name = node.func, None
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name) and (f.value.id, f.attr) in LISTING_FUNCTIONS:
                name = f"{f.value.id}.{f.attr}"
            elif f.attr in LISTING_METHODS:
                name = f".{f.attr}"
        elif isinstance(f, ast.Name) and f.id in {"listdir", "scandir", "walk", "glob", "iglob"}:
            name = f.id
        if name is None:
            continue
        p = parent.get(node)
        if not (isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "sorted"):
            bad.append((node.lineno, name))
    return bad


def test_listing_check_works():
    assert listing_calls_without_sorted("import os\nx = os.listdir(d)\n")
    assert listing_calls_without_sorted("from pathlib import Path\nx = list(Path(d).glob('*.mat'))\n")
    assert not listing_calls_without_sorted("import glob\nx = sorted(glob.glob(p))\n")


def test_preprocessing_does_not_depend_on_listing_order():
    files = sorted((REPO / "src" / "hexmod" / "preprocessing").glob("*.py"))
    step = REPO / "scripts" / "step01_preprocess.py"
    if step.is_file():
        files.append(step)
    bad = [f"{p.relative_to(REPO)}:{ln}: {name}"
           for p in files for ln, name in listing_calls_without_sorted(p.read_text(encoding="utf-8"))]
    assert not bad, "raw files must be taken from the manifests, not from a directory listing:\n" + "\n".join(bad)


# ------------------------------------------------------------------------------- reference results
@pytest.mark.frozen
def test_reference_cohort_order(frozen_path, units, sessions):
    """The published cohort table maps one to one onto the manifest and is in manifest order."""
    cells = pd.read_csv(frozen_path("rat/cohort/cells.csv"))
    key = list(zip(cells["session_id"], cells["cell_id"].astype(int)))
    pos = {(s, int(c)): i for i, (s, c) in enumerate(zip(units["session_id"], units["cell_id"]))}
    idx = [pos.get(k) for k in key]
    assert all(i is not None for i in idx), "cells missing from the manifest"
    idx = np.array(idx)
    assert len(np.unique(idx)) == len(idx)
    assert np.all(np.diff(idx) > 0), "the cohort table is not in manifest order"
    assert (units["unique_cell_id"].to_numpy()[idx] == cells["unique_cell_id"].to_numpy()).all()
    assert (units["animal_id"].to_numpy()[idx] == cells["animal_id"].to_numpy()).all()
    if cells["dataset"].isin(list(ds.DATASETS)).all():
        assert (units["dataset"].to_numpy()[idx] == cells["dataset"].to_numpy()).all()
    keys = set(units["dataset"].to_numpy()[idx])
    coh = sessions[sessions["in_cohort"] & sessions["dataset"].isin(keys)]
    assert set(cells["session_id"]) == set(coh["session_id"])
    if keys == set(ds.CANONICAL_ORDER):
        assert len(cells) == 3938 and cells["session_id"].nunique() == 112 and cells["animal_id"].nunique() == 32


@pytest.mark.frozen
def test_reference_preprocessed_order(frozen_path):
    """The merged preprocessed recordings of the published analysis are in manifest order."""
    from hexmod.io import load_cache

    data = load_cache(frozen_path("preprocessed/trajectories.pkl"))
    known = set(labels.read_sessions()["session_id"])     # the reference may hold sessions outside the manifest
    data = {a: {s: v for s, v in ss.items() if s in known} for a, ss in data.items()}
    data = {a: ss for a, ss in data.items() if ss}
    out = labels.canonical_order(data)
    assert _flatten(out) == _flatten(data)
