"""Session and unit labels, and the canonical order of all recordings.

Animal, session and unit labels (for example ``V10``, ``V10_1`` and ``V10_V10_1_cell341``), the
identifier of each unit within its session (``cell_id``) and the order in which animals, sessions
and units appear in every table are fixed by two shipped manifests. They are never taken from the
order in which a file system happens to list the raw files, which differs between machines.

``inputs/manifests/sessions.csv``
    One row per session produced by preprocessing, including sessions that later fail the cohort
    rule. Columns:

    - ``order`` (int): canonical position of the session, from 0, over all datasets;
    - ``dataset`` (str): dataset key (``hexmod.datasets``);
    - ``animal_id`` (str): animal label, dataset prefix plus number (``V10``);
    - ``session_id`` (str): session label (``V10_1``);
    - ``raw_path`` (str): raw file of the session relative to the dataset folder
      ``<data>/<dataset>/``; several files are joined by ``;`` in the order they are read
      (hafting2005). For barry2012 and olafsdottir2016 it is the dataset's HDF5 file, which
      holds the session as the group named ``session_id``;
    - ``n_units`` (int): units kept by preprocessing;
    - ``in_cohort`` (bool): the session contributes at least one grid cell to the published
      cohort (32 animals, 112 sessions, 3,938 cells).

``inputs/manifests/units.csv``
    One row per unit kept by preprocessing. Columns:

    - ``order`` (int): canonical position of the unit, from 0, over all datasets;
    - ``dataset``, ``animal_id``, ``session_id`` (str): as above;
    - ``cell_id`` (int): key of the unit in the session's ``spike_times`` dict;
    - ``unique_cell_id`` (str): unit label written by preprocessing;
    - ``raw_unit_key`` (str): where the unit comes from in the raw data (see below).

``raw_unit_key`` by dataset:

- ``vollan2025``: ``mec:<i>``, entry ``i`` (counted from 1) of ``Dsession.units.mec``; equal to
  ``cell_id``.
- ``brun2008``: ``T<tetrode>C<cell>`` from the ``units`` table of the session file; ``cell_id``
  is the cell number.
- ``hafting2005``: ``<file>:<variable>``, the spike-time variable of one of the session's files.
- ``sargolini2006``: name of the unit's spike file (``<rat>-<trial>_T<t>C<c>.mat``), in the same
  folder as the session's position file. ``cell_id`` numbers the units of the whole dataset.
- ``barry2012``, ``olafsdottir2016``: ``unit_<i>``, the unit's group in the session group of the
  HDF5 file, ``i`` (three digits, counted from 0) being its ``original_index`` attribute;
  ``cell_id`` is ``i`` (barry2012) or ``i + 1`` (olafsdottir2016).

**Canonical order.** Datasets follow ``hexmod.datasets.CANONICAL_ORDER``; within a dataset,
animals, sessions and units follow the ``order`` columns. The sessions of an animal are
contiguous, and so are the units of a session. Every per-cell table and array of the pipeline is
in this order; seeded permutation tests and seeded draws depend on it.

``inputs/rat/module_labels.csv`` holds the manual assignment of grid cells to modules by grid
scale: one row per animal and module (``animal_id``, ``module`` counted from 1,
``scale_from_cm``, ``scale_to_cm``). A cell belongs to module ``k`` when
``scale_from_cm <= scale < scale_to_cm`` for that row, as computed by ``numpy.digitize`` over the
animal's bin edges; cells outside every bin are unclassified (module 0).
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .. import config as _config
from .. import datasets as _ds

PathLike = Union[str, Path]

_BOOL = {True: True, False: False, "True": True, "False": False}

SESSION_COLUMNS = ["order", "dataset", "animal_id", "session_id", "raw_path", "n_units", "in_cohort"]
UNIT_COLUMNS = ["order", "dataset", "animal_id", "session_id", "cell_id", "unique_cell_id", "raw_unit_key"]
MODULE_COLUMNS = ["animal_id", "module", "scale_from_cm", "scale_to_cm"]
EXAMPLE_COLUMNS = ["panel", "position", "role", "dataset", "animal_id", "session_id", "cell_id", "t_start_s", "t_end_s",
                   "selection"]

__all__ = [
    "ManifestError", "SESSION_COLUMNS", "UNIT_COLUMNS", "sessions_path", "units_path", "read_sessions", "read_units",
    "sessions_of", "units_of", "session_for_raw_path", "unit_for_raw_key", "unit_order", "sort_canonical",
    "is_canonical", "canonical_order", "merge_datasets", "read_module_bins", "assign_modules", "read_example_cells",
]


class ManifestError(ValueError):
    """Raised when data do not match the shipped manifests."""


# ------------------------------------------------------------------------------------------ files
def _manifest_dir() -> Path:
    return _config.inputs_dir() / "manifests"


def sessions_path() -> Path:
    """Return the path of ``inputs/manifests/sessions.csv``."""
    return _manifest_dir() / "sessions.csv"


def units_path() -> Path:
    """Return the path of ``inputs/manifests/units.csv``."""
    return _manifest_dir() / "units.csv"


def _datasets_arg(datasets: Optional[Union[str, Iterable[str]]]) -> Optional[List[str]]:
    """Turn a dataset key or a list of keys into a list of keys."""
    if datasets is None:
        return None
    if isinstance(datasets, str):
        datasets = [datasets]
    keys = list(datasets)
    unknown = [k for k in keys if k not in _ds.DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset keys: {unknown}")
    return keys


@lru_cache(maxsize=8)
def _load(kind: str, path: str, mtime_ns: int):
    """Read and check one manifest (cached per file and modification time)."""
    import pandas as pd

    if kind == "sessions":
        df = pd.read_csv(path, dtype={"dataset": str, "animal_id": str, "session_id": str, "raw_path": str},
                         keep_default_na=False)
        if list(df.columns) != SESSION_COLUMNS:
            raise ManifestError(f"unexpected session manifest columns {list(df.columns)}")
        df["n_units"] = df["n_units"].astype(np.int64)
        df["in_cohort"] = df["in_cohort"].map(_BOOL).astype(bool)
    else:
        df = pd.read_csv(path, keep_default_na=False,
                         dtype={"dataset": str, "animal_id": str, "session_id": str, "unique_cell_id": str,
                                "raw_unit_key": str})
        if list(df.columns) != UNIT_COLUMNS:
            raise ManifestError(f"unexpected unit manifest columns {list(df.columns)}")
        df["cell_id"] = df["cell_id"].astype(np.int64)
    df["order"] = df["order"].astype(np.int64)
    return df.sort_values("order", kind="mergesort").reset_index(drop=True)


def _read(kind: str, datasets, path):
    path = Path(path or (sessions_path() if kind == "sessions" else units_path()))
    df = _load(kind, str(path.resolve()), path.stat().st_mtime_ns).copy()
    keys = _datasets_arg(datasets)
    if keys is not None:
        df = df[df["dataset"].isin(keys)].reset_index(drop=True)
    return df


def read_sessions(datasets: Optional[Union[str, Iterable[str]]] = None, path: Optional[PathLike] = None):
    """Read the session manifest.

    Parameters
    ----------
    datasets : str or iterable of str, optional
        One dataset key or several; default all datasets.
    path : str or Path, optional
        Manifest file (default ``inputs/manifests/sessions.csv``).

    Returns
    -------
    pandas.DataFrame
        Columns ``SESSION_COLUMNS`` in canonical order.
    """
    return _read("sessions", datasets, path)


def read_units(datasets: Optional[Union[str, Iterable[str]]] = None, path: Optional[PathLike] = None):
    """Read the unit manifest.

    Parameters
    ----------
    datasets : str or iterable of str, optional
        One dataset key or several; default all datasets.
    path : str or Path, optional
        Manifest file (default ``inputs/manifests/units.csv``).

    Returns
    -------
    pandas.DataFrame
        Columns ``UNIT_COLUMNS`` in canonical order.
    """
    return _read("units", datasets, path)


# ---------------------------------------------------------------------------------------- lookups
def _records(df) -> List[dict]:
    """Rows of a manifest as dicts of plain Python values (missing values as None)."""
    import pandas as pd

    out = []
    for rec in df.to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                clean[k] = None
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def sessions_of(key: str, path: Optional[PathLike] = None) -> List[dict]:
    """Return the sessions of one dataset in canonical order, one dict per manifest row."""
    return _records(read_sessions(key, path))


def units_of(key: str, path: Optional[PathLike] = None) -> "OrderedDict[str, List[dict]]":
    """Return ``{session_id: [unit rows]}`` for one dataset, sessions and units in canonical order."""
    return OrderedDict(_group_units(read_units(key, path)))


def session_for_raw_path(key: str, raw_path: str, path: Optional[PathLike] = None) -> dict:
    """Return the session row of a raw file of a dataset with one file (or two) per session.

    Parameters
    ----------
    key : str
        Dataset key.
    raw_path : str
        Raw file relative to the dataset folder, with forward slashes. For sessions read from
        several files any one of them may be given.
    path : str or Path, optional
        Session manifest.

    Returns
    -------
    dict
        The manifest row.
    """
    raw_path = str(raw_path).replace("\\", "/")
    hits = [r for r in sessions_of(key, path) if raw_path in r["raw_path"].split(";")]
    if len(hits) != 1:
        raise ManifestError(f"{key}: raw file {raw_path!r} matches {len(hits)} sessions of the manifest")
    return hits[0]


def unit_for_raw_key(key: str, session_id: str, raw_unit_key: str, path: Optional[PathLike] = None) -> dict:
    """Return the unit row of a raw unit of a session (``cell_id`` and ``unique_cell_id``)."""
    hits = [u for u in units_of(key, path).get(session_id, []) if u["raw_unit_key"] == raw_unit_key]
    if len(hits) != 1:
        raise ManifestError(f"{key}: unit {raw_unit_key!r} of {session_id} matches {len(hits)} manifest rows")
    return hits[0]


# ------------------------------------------------------------------------------- canonical order
def _unit_index(path: Optional[PathLike] = None) -> Dict[Tuple[str, int], int]:
    df = read_units(path=path)
    return {(s, int(c)): int(o) for s, c, o in zip(df["session_id"], df["cell_id"], df["order"])}


def unit_order(session_ids: Sequence[str], cell_ids: Sequence[int], path: Optional[PathLike] = None) -> np.ndarray:
    """Return the canonical position (manifest ``order``) of each (session, cell) pair.

    Parameters
    ----------
    session_ids, cell_ids : sequences of equal length
        Session labels and unit keys.
    path : str or Path, optional
        Unit manifest.

    Returns
    -------
    ndarray of int64
        Manifest order of every pair; a pair missing from the manifest raises ``ManifestError``.
    """
    index = _unit_index(path)
    out = np.empty(len(session_ids), dtype=np.int64)
    missing = []
    for i, (s, c) in enumerate(zip(session_ids, cell_ids)):
        pos = index.get((str(s), int(c)))
        if pos is None:
            missing.append((s, c))
            continue
        out[i] = pos
    if missing:
        raise ManifestError(f"{len(missing)} units are not in the manifest, first: {missing[:5]}")
    return out


def sort_canonical(df, session_col: str = "session_id", cell_col: str = "cell_id", path: Optional[PathLike] = None):
    """Return a per-cell table sorted into canonical order (index reset).

    Parameters
    ----------
    df : pandas.DataFrame
        Table with one row per unit.
    session_col, cell_col : str
        Columns holding the session label and the unit key.
    path : str or Path, optional
        Unit manifest.

    Returns
    -------
    pandas.DataFrame
    """
    pos = unit_order(df[session_col].tolist(), df[cell_col].tolist(), path)
    if len(np.unique(pos)) != len(pos):
        raise ManifestError("the table holds the same unit more than once")
    return df.iloc[np.argsort(pos, kind="mergesort")].reset_index(drop=True)


def is_canonical(df, session_col: str = "session_id", cell_col: str = "cell_id",
                 path: Optional[PathLike] = None) -> bool:
    """Return True when the rows of a per-cell table are in canonical order (strictly increasing)."""
    pos = unit_order(df[session_col].tolist(), df[cell_col].tolist(), path)
    return bool(np.all(np.diff(pos) > 0))


def _reorder_session(session: Mapping, unit_rows: List[dict], session_id: str, strict: bool) -> dict:
    """Return a copy of a session dict with every per-unit dict in manifest unit order."""
    spikes = session.get("spike_times", {})
    want = [u["cell_id"] for u in unit_rows]
    have = [int(c) for c in spikes]
    if sorted(have) != sorted(want):
        if strict:
            raise ManifestError(f"{session_id}: units {sorted(set(have) ^ set(want))[:10]} differ from the manifest")
        want = [c for c in want if c in set(have)] + [c for c in have if c not in set(want)]
    uids = session.get("unique_cell_ids")
    if strict and uids is not None:
        expected = {u["cell_id"]: u["unique_cell_id"] for u in unit_rows}
        bad = [c for c, lab in uids.items() if expected.get(int(c)) != lab]
        if bad:
            raise ManifestError(f"{session_id}: unit labels of cells {bad[:5]} differ from the manifest")
    keyset = set(spikes.keys())
    out = {}
    for name, value in session.items():
        if isinstance(value, dict) and value and set(value.keys()) == keyset:
            lookup = {int(k): k for k in value}
            out[name] = {lookup[c]: value[lookup[c]] for c in want}
        else:
            out[name] = value
    return out


def canonical_order(data: Mapping, key: Optional[str] = None, strict: bool = True,
                    sessions_file: Optional[PathLike] = None, units_file: Optional[PathLike] = None) -> dict:
    """Return preprocessed recordings with animals, sessions and units in canonical order.

    Parameters
    ----------
    data : mapping
        ``{animal_id: {session_id: session}}`` for one dataset, or for several datasets when
        ``key`` is None. Each session dict holds ``spike_times`` keyed by ``cell_id``; every
        other dict-valued entry with the same keys (``unique_cell_ids``, ``mod_id``,
        ``tetrode_ids``) is reordered in the same way.
    key : str, optional
        Dataset key; default: the datasets are inferred from the animal labels.
    strict : bool
        Raise ``ManifestError`` unless the sessions, units and unit labels equal the manifest's
        (for the datasets present). With ``strict=False`` sessions and units missing from the
        manifest are kept after the known ones.
    sessions_file, units_file : str or Path, optional
        Session and unit manifests (default: the shipped ones).

    Returns
    -------
    dict
        New nested dict (the session dicts are shallow copies; arrays are not copied).
    """
    keys = [key] if key is not None else sorted({_ds.key_from_label(a) for a in data}, key=_ds.canonical_rank)
    sessions = read_sessions(keys, sessions_file)
    units = _group_units(read_units(keys, units_file))
    present = {(a, s) for a, ss in data.items() for s in ss}
    out: Dict[str, dict] = {}
    seen = set()
    for rec in _records(sessions):
        a, s = rec["animal_id"], rec["session_id"]
        if (a, s) not in present:
            if strict:
                raise ManifestError(f"session {s} of the manifest is missing from the data")
            continue
        out.setdefault(a, {})[s] = _reorder_session(data[a][s], units.get(s, []), s, strict)
        seen.add((a, s))
    extra = sorted(present - seen)
    if extra:
        if strict:
            raise ManifestError(f"sessions not in the manifest: {[s for _, s in extra][:10]}")
        for a, s in extra:
            out.setdefault(a, {})[s] = dict(data[a][s])
    return out


def _group_units(df) -> Dict[str, List[dict]]:
    """Group unit manifest rows by session."""
    out: Dict[str, List[dict]] = {}
    for rec in _records(df):
        out.setdefault(rec["session_id"], []).append(rec)
    return out


def merge_datasets(per_dataset: Mapping[str, Mapping], strict: bool = True) -> dict:
    """Merge the recordings of several datasets into one dict in canonical order.

    Parameters
    ----------
    per_dataset : mapping
        ``{dataset key: {animal_id: {session_id: session}}}``.
    strict : bool
        Passed to ``canonical_order``.

    Returns
    -------
    dict
        ``{animal_id: {session_id: session}}`` with datasets in ``CANONICAL_ORDER``.
    """
    merged: Dict[str, dict] = {}
    for key in sorted(per_dataset, key=_ds.canonical_rank):
        part = canonical_order(per_dataset[key], key=key, strict=strict)
        clash = set(part) & set(merged)
        if clash:
            raise ManifestError(f"animal labels used by two datasets: {sorted(clash)}")
        merged.update(part)
    return merged


# ---------------------------------------------------------------------------------------- modules
def read_module_bins(path: Optional[PathLike] = None) -> Dict[str, np.ndarray]:
    """Return the grid-scale bin edges of every animal from ``inputs/rat/module_labels.csv``.

    Returns
    -------
    dict
        ``{animal_id: edges}``, edges a float64 array (cm) of length (number of modules + 1).
    """
    import pandas as pd

    df = pd.read_csv(path or (_config.inputs_dir() / "rat" / "module_labels.csv"), dtype={"animal_id": str})
    if list(df.columns) != MODULE_COLUMNS:
        raise ManifestError(f"unexpected module label columns {list(df.columns)}")
    out: Dict[str, np.ndarray] = {}
    for animal, g in df.groupby("animal_id", sort=False):
        g = g.sort_values("module")
        if list(g["module"]) != list(range(1, len(g) + 1)):
            raise ManifestError(f"{animal}: modules must be numbered 1 to {len(g)}")
        if not np.array_equal(g["scale_from_cm"].to_numpy()[1:], g["scale_to_cm"].to_numpy()[:-1]):
            raise ManifestError(f"{animal}: module bins are not contiguous")
        out[animal] = np.r_[g["scale_from_cm"].to_numpy(float)[:1], g["scale_to_cm"].to_numpy(float)]
    return out


def assign_modules(animal_ids: Sequence[str], scales_cm: Sequence[float],
                   bins: Optional[Mapping[str, np.ndarray]] = None) -> np.ndarray:
    """Assign grid cells to modules by grid scale.

    Parameters
    ----------
    animal_ids : sequence of str
        Animal label of each cell.
    scales_cm : sequence of float
        Grid scale of each cell (cm).
    bins : mapping, optional
        Output of ``read_module_bins`` (read from the shipped table by default).

    Returns
    -------
    ndarray of int64
        Module number (from 1) of each cell, 0 for cells outside every bin of their animal.
    """
    bins = read_module_bins() if bins is None else bins
    out = np.zeros(len(animal_ids), dtype=np.int64)
    for i, (a, s) in enumerate(zip(animal_ids, scales_cm)):
        edges = bins[str(a)]
        k = int(np.digitize(s, edges))
        out[i] = k if 0 < k <= len(edges) - 1 else 0
    return out


# ----------------------------------------------------------------------------------- example cells
def read_example_cells(path: Optional[PathLike] = None):
    """Read ``inputs/rat/example_cells.csv`` (the recorded cells shown as examples in the figures).

    Columns: ``panel`` (for example ``Fig3A``), ``position`` (place within the panel, from 1),
    ``role``, ``dataset``, ``animal_id``, ``session_id``, ``cell_id``, ``t_start_s`` and
    ``t_end_s`` (time window drawn, empty when the panel shows the whole session) and
    ``selection`` (how the cell was chosen). One row per panel and position.
    """
    import pandas as pd

    df = pd.read_csv(path or (_config.inputs_dir() / "rat" / "example_cells.csv"),
                     dtype={"panel": str, "role": str, "dataset": str, "animal_id": str, "session_id": str,
                            "selection": str})
    if list(df.columns) != EXAMPLE_COLUMNS:
        raise ManifestError(f"unexpected example cell columns {list(df.columns)}")
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["position"] = df["position"].astype(np.int64)
    return df
