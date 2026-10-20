"""Readers and writers for the result formats.

Formats:

- CSV: comma separated, header row, no index column, floats written with ``"%.17g"`` so that
  every value reads back exactly; read back with ``float_precision="round_trip"``.
- npy / npz: numpy arrays, never object arrays (files load with ``allow_pickle=False``).
  Lists of arrays of unequal shape ("ragged" arrays) are stored as two arrays,
  ``<name>__data`` (all items raveled and concatenated) and ``<name>__shapes`` (one row of
  int64 dimensions per item).
- JSON: UTF-8, keys sorted, numbers written with Python's shortest exact representation.
- Pickle: the preprocessed recordings of step 01 (``save_cache``, ``load_cache``). Only plain
  dicts, lists and tuples of numpy arrays, numbers and strings are written; they are read with an
  unpickler that rebuilds numpy arrays, numpy scalars and a few builtin containers and refuses
  every other global.

All writers create parent folders and replace files atomically; written files get the usual
permissions (0644 with umask 022).
"""

from __future__ import annotations

import gzip
import hashlib
import io as _io
import json
import math
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]

FLOAT_FORMAT = "%.17g"
RAGGED_DATA = "__data"
RAGGED_SHAPES = "__shapes"

__all__ = [
    "FLOAT_FORMAT", "FILE_MODE", "write_csv", "read_csv", "csv_float_roundtrip", "array_text_roundtrip",
    "write_json", "read_json", "save_npy", "load_npy", "save_npz", "load_npz", "pack_ragged", "unpack_ragged",
    "ragged_from_npz", "save_cache", "load_cache", "sha256_file", "atomic_path",
]


# ----------------------------------------------------------------------------------------- helpers
def _process_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


#: File permissions of written files: read and write for everyone, minus the process umask
#: (0644 with the usual umask 022), as for a file created with ``open``.
FILE_MODE = 0o666 & ~_process_umask()


class atomic_path:
    """Context manager yielding a temporary path that replaces ``path`` on success.

    The temporary file is created next to ``path`` and, on success, given the permissions of
    ``FILE_MODE`` and renamed onto ``path``; on failure it is removed.
    """

    def __init__(self, path: PathLike, suffix: str = ""):
        self.path = Path(path)
        self.suffix = suffix or self.path.suffix
        self.tmp: Optional[Path] = None

    def __enter__(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".tmp_" + self.path.stem + "_", suffix=self.suffix,
                                    dir=str(self.path.parent))
        os.close(fd)
        self.tmp = Path(name)
        return self.tmp

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.tmp is not None
        if exc_type is None:
            os.chmod(self.tmp, FILE_MODE)
            os.replace(self.tmp, self.path)
        else:
            try:
                self.tmp.unlink()
            except FileNotFoundError:
                pass
        return False


def sha256_file(path: PathLike, chunk: int = 1 << 22) -> str:
    """Return the sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- CSV
def write_csv(df, path: PathLike, float_format: str = FLOAT_FORMAT, **kwargs) -> Path:
    """Write a pandas DataFrame as CSV with full float precision and no index.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to write. Columns must hold scalars (no arrays).
    path : str or Path
        Output file; ``.csv.gz`` is written gzip-compressed.
    float_format : str
        printf-style float format (default ``"%.17g"``).
    **kwargs
        Passed to ``DataFrame.to_csv``.

    Returns
    -------
    Path
        The written path.
    """
    path = Path(path)
    for col in df.columns:
        if df[col].dtype == object and len(df) and isinstance(df[col].iloc[0], (np.ndarray, list, dict)):
            raise TypeError(f"column {col!r} holds arrays; store ragged values in an npz file")
    kwargs.setdefault("index", False)
    kwargs.setdefault("lineterminator", "\n")
    compressed = path.name.endswith(".gz")
    with atomic_path(path, suffix=".csv.gz" if compressed else ".csv") as tmp:
        if compressed:
            # no file name and a fixed time stamp in the gzip header: identical bytes on every run
            text = df.to_csv(None, float_format=float_format, **kwargs)
            tmp.write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))
        else:
            df.to_csv(tmp, float_format=float_format, **kwargs)
    return path


def read_csv(path: PathLike, **kwargs):
    """Read a CSV written by ``write_csv`` with exact float round trip.

    Parameters
    ----------
    path : str or Path
        CSV or ``.csv.gz`` file.
    **kwargs
        Passed to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(path, **kwargs)


def csv_float_roundtrip(values, times: int = 1) -> np.ndarray:
    """Return values after round trips through a CSV file with pandas' default float handling.

    Each round trip writes the values with pandas' default float format (the shortest
    representation that reads back exactly) and reads them with pandas' default float parser
    (``float_precision="high"``), which returns the neighbouring double for a sizeable share of
    17-digit values (one unit in the last place). Some per-cell values enter the statistics of the
    paper after such round trips (for example the grid orientation passed to the adaptation fits;
    docs/PIPELINE.md, Rounded per-cell inputs); applying the same transformation gives the
    reference values exactly.

    Parameters
    ----------
    values : array_like
        Exact values (as computed, or as read with ``read_csv``).
    times : int
        Number of successive round trips (each written and read once more).

    Returns
    -------
    ndarray
        float64 array of the same shape.
    """
    import pandas as pd

    v = np.array(values, dtype=float)
    shape = v.shape
    if v.size == 0:
        return v
    for _ in range(int(times)):
        text = "x\n" + "\n".join("" if math.isnan(x) else repr(float(x)) for x in v.ravel()) + "\n"
        v = pd.read_csv(_io.StringIO(text), skip_blank_lines=False, float_precision="high")["x"].to_numpy(
            dtype=float).reshape(shape)
    return v


_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def array_text_roundtrip(values) -> np.ndarray:
    """Return a 1-d array after it is stored as text in a CSV cell and parsed back.

    The text is numpy's default form of the array (8 significant decimals), and the numbers are
    parsed back with a regular expression. Per-field arrays enter some statistics of the paper in
    this form (the field rates passed to the adaptation fits; docs/PIPELINE.md, Rounded per-cell
    inputs).

    Parameters
    ----------
    values : array_like
        1-d float array.

    Returns
    -------
    ndarray
        float64 array of the parsed numbers.
    """
    a = np.asarray(values, dtype=float)
    with np.printoptions(precision=8, floatmode="maxprec", suppress=False, threshold=1000, linewidth=75,
                         sign="-", edgeitems=3, nanstr="nan", infstr="inf", legacy=False):
        text = str(a)
    return np.array([float(x) for x in _NUMBER.findall(text)], float)


# -------------------------------------------------------------------------------------------- JSON
def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json(obj: Any, path: PathLike, indent: int = 1, sort_keys: bool = True) -> Path:
    """Write an object as JSON (numpy scalars and arrays are converted to builtins).

    NaN and infinity are written as the non-standard tokens ``NaN`` and ``Infinity``, which
    Python's json module reads back; result files should avoid them where possible.
    """
    path = Path(path)
    text = json.dumps(_to_builtin(obj), indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    with atomic_path(path) as tmp:
        tmp.write_text(text + "\n", encoding="utf-8")
    return path


def read_json(path: PathLike) -> Any:
    """Read a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------- npy and npz
def save_npy(path: PathLike, array: np.ndarray) -> Path:
    """Save one array as ``.npy`` (object arrays are refused)."""
    array = np.asarray(array)
    if array.dtype == object:
        raise TypeError("object arrays are not allowed; use pack_ragged")
    path = Path(path)
    with atomic_path(path, suffix=".npy") as tmp:
        np.save(tmp, array, allow_pickle=False)
    return path


def load_npy(path: PathLike, mmap_mode: Optional[str] = None) -> np.ndarray:
    """Load an ``.npy`` file without pickle support."""
    return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)


def pack_ragged(items: Sequence[np.ndarray], dtype=None, ndim: Optional[int] = None):
    """Pack a list of arrays of unequal shape into ``(data, shapes)``.

    Parameters
    ----------
    items : sequence of array_like
        Arrays with the same number of dimensions (scalars count as 0-d).
    dtype : numpy dtype, optional
        dtype of ``data``; by default the common dtype of the items (float64 for an empty list).
    ndim : int, optional
        Number of dimensions; required only when ``items`` is empty.

    Returns
    -------
    data : ndarray
        1-d concatenation of all items, each raveled in C order.
    shapes : ndarray
        int64 array of shape ``(len(items), ndim)``.
    """
    arrays = [np.asarray(a) for a in items]
    if arrays:
        dims = {a.ndim for a in arrays}
        if len(dims) != 1:
            raise ValueError(f"items have different numbers of dimensions: {sorted(dims)}")
        nd = dims.pop()
        if dtype is None:
            dtype = np.result_type(*arrays)
    else:
        nd = 1 if ndim is None else ndim
        dtype = np.float64 if dtype is None else dtype
    shapes = np.array([a.shape for a in arrays], dtype=np.int64).reshape(len(arrays), nd)
    data = (np.concatenate([a.ravel() for a in arrays]).astype(dtype, copy=False)
            if arrays else np.zeros(0, dtype=dtype))
    return data, shapes


def unpack_ragged(data: np.ndarray, shapes: np.ndarray) -> List[np.ndarray]:
    """Invert ``pack_ragged``: return the list of arrays (views into ``data`` are copied)."""
    shapes = np.asarray(shapes, dtype=np.int64)
    sizes = shapes.prod(axis=1) if shapes.shape[1] else np.ones(len(shapes), dtype=np.int64)
    ends = np.cumsum(sizes)
    if len(ends) and ends[-1] != len(data):
        raise ValueError(f"shapes describe {int(ends[-1])} values but data holds {len(data)}")
    starts = ends - sizes
    return [np.array(data[s:e]).reshape(tuple(int(v) for v in shp)) for s, e, shp in zip(starts, ends, shapes)]


def save_npz(path: PathLike, arrays: Optional[Mapping[str, Any]] = None,
             ragged: Optional[Mapping[str, Sequence[np.ndarray]]] = None, compressed: bool = True) -> Path:
    """Save arrays (and ragged lists of arrays) in one ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        Output file.
    arrays : mapping, optional
        ``{name: array}``; object arrays are refused.
    ragged : mapping, optional
        ``{name: list of arrays}``, stored as ``name__data`` and ``name__shapes``.
    compressed : bool
        Use ``np.savez_compressed`` (default) or ``np.savez``.

    Returns
    -------
    Path
    """
    out: Dict[str, np.ndarray] = {}
    for name, value in (arrays or {}).items():
        a = np.asarray(value)
        if a.dtype == object:
            raise TypeError(f"array {name!r} has dtype object; pass it as ragged")
        out[name] = a
    for name, items in (ragged or {}).items():
        data, shapes = pack_ragged(items)
        out[name + RAGGED_DATA] = data
        out[name + RAGGED_SHAPES] = shapes
    path = Path(path)
    with atomic_path(path, suffix=".npz") as tmp:
        with open(tmp, "wb") as f:
            (np.savez_compressed if compressed else np.savez)(f, **out)
    return path


def load_npz(path: PathLike, unpack: bool = True) -> Dict[str, Any]:
    """Load an ``.npz`` file into a dict.

    Parameters
    ----------
    path : str or Path
        File written by ``save_npz`` (or any npz without object arrays).
    unpack : bool
        Return ragged entries as lists under their base name (default) instead of the two
        stored arrays.

    Returns
    -------
    dict
    """
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    if not unpack:
        return raw
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key.endswith(RAGGED_DATA):
            base = key[: -len(RAGGED_DATA)]
            out[base] = unpack_ragged(value, raw[base + RAGGED_SHAPES])
        elif key.endswith(RAGGED_SHAPES):
            continue
        else:
            out[key] = value
    return out


def ragged_from_npz(npz: Mapping[str, np.ndarray], name: str) -> List[np.ndarray]:
    """Return the ragged list ``name`` from a raw npz mapping (``load_npz(..., unpack=False)``)."""
    return unpack_ragged(npz[name + RAGGED_DATA], npz[name + RAGGED_SHAPES])


# ---------------------------------------------------------------------------------------- pickles
_SAFE_SCALARS = (str, bytes, int, float, bool, complex, type(None), np.generic)


def _check_plain(obj: Any, where: str = "cache") -> None:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError(f"{where}: object arrays are not allowed in caches")
        return
    if isinstance(obj, _SAFE_SCALARS):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (str, int, np.integer, tuple)):
                raise TypeError(f"{where}: dict key {k!r} of type {type(k).__name__}")
            _check_plain(v, f"{where}[{k!r}]")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_plain(v, f"{where}[{i}]")
        return
    raise TypeError(f"{where}: {type(obj).__module__}.{type(obj).__name__} is not allowed in caches")


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that accepts only the globals needed to rebuild numpy arrays and scalars.

    Any other global (for example ``builtins.eval`` or a numpy helper that runs code) is
    refused, so a modified cache file cannot execute code when it is loaded.
    """

    _ALLOWED = frozenset({
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
        ("builtins", "complex"), ("builtins", "set"), ("builtins", "frozenset"),
        ("builtins", "bytearray"), ("builtins", "slice"), ("builtins", "range"),
        ("collections", "OrderedDict"),
    })

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def save_cache(obj: Any, path: PathLike) -> Path:
    """Save plain dicts and lists of numpy arrays and scalars as a pickle (the preprocessed recordings)."""
    _check_plain(obj)
    path = Path(path)
    with atomic_path(path, suffix=".pkl") as tmp:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
    return path


def load_cache(path: PathLike) -> Any:
    """Load a cache written by ``save_cache`` (only numpy arrays, numpy scalars and builtin
    containers are accepted)."""
    with open(path, "rb") as f:
        return _RestrictedUnpickler(f).load()
