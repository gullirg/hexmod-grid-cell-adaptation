"""From raw recordings to the trajectories used by every analysis.

Preprocessing has two stages.

1. One module per dataset reads the raw files of that dataset and returns its recordings:
   :mod:`~hexmod.preprocessing.sargolini2006`, :mod:`~hexmod.preprocessing.brun2008`,
   :mod:`~hexmod.preprocessing.olafsdottir2016`, :mod:`~hexmod.preprocessing.vollan2025`,
   :mod:`~hexmod.preprocessing.hafting2005` and :mod:`~hexmod.preprocessing.barry2012`. Each module has a
   function ``preprocess(base=None, workers=1)`` that returns ``{animal_id: {session_id: session}}``
   in canonical order. Positions are in cm and times in s, each dataset on its own clock; the
   conventions of each dataset are described in its module. Barry et al. (2012) and Ólafsdóttir
   et al. (2016) are read from one HDF5 file each (:mod:`~hexmod.preprocessing.hdf5_sessions`).
2. :mod:`~hexmod.preprocessing.trajectories` merges the datasets, interpolates missing positions,
   smooths them with a 20-sample boxcar and computes running speed and movement direction.

The layout of a session dict and of the output files is given in docs/PIPELINE.md (step 01).

Which raw files are read, and in which order, comes from the file manifests
(``inputs/manifests/files_<key>.csv``, read with :func:`hexmod.data_sources.read_manifest`); animal,
session and unit labels come from the session and unit manifests (:mod:`hexmod.preprocessing.labels`).
No result depends on the order in which a file system lists a folder. The folders of Vollan et al.
(2025), Brun et al. (2008) and Sargolini et al. (2006) must hold exactly the files of the manifest:
the analysed recordings are all the files of these folders, so an extra file means that the data
differ from those analysed, and :func:`refuse_unlisted_files` stops with an error.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .. import config as _config
from .. import data_sources as _sources
from .. import datasets as _ds

__all__ = ["dataset_module", "data_base", "manifest_files", "refuse_unlisted_files", "check_file_sizes",
           "check_animal_map"]


def dataset_module(key: str):
    """Return the preprocessing module of a dataset (for example ``hexmod.preprocessing.vollan2025``)."""
    if key not in _ds.DATASETS:
        raise ValueError(f"unknown dataset key {key!r}")
    return importlib.import_module(f"{__name__}.{key}")


def data_base(base: Optional[Path] = None) -> Path:
    """Return the raw-data folder: ``base`` when given, else ``HEXMOD_DATA`` or ``<repo>/data``."""
    return Path(base) if base is not None else _config.data_dir()


def manifest_files(key: str) -> List[dict]:
    """Return the manifest rows of the files of a dataset that the analysis reads, in manifest order.

    Rows of role ``file`` and ``member`` (see :mod:`hexmod.data_sources`); archives and archive
    volumes are left out.
    """
    return _sources.final_files(_sources.read_manifest(key))


def check_file_sizes(rows: Sequence[dict], base: Optional[Path] = None) -> None:
    """Stop with an error when a manifest file is missing or has an unexpected size.

    Only sizes are compared here; ``scripts/data_check.py`` verifies the checksums.
    """
    base = data_base(base)
    bad = {}
    for row in rows:
        status = _sources.verify_file(base / row["path"], row, checksum="size")
        if status != "ok":
            bad[row["path"]] = status
    if bad:
        first = "; ".join(f"{p}: {s}" for p, s in list(bad.items())[:5])
        raise FileNotFoundError(f"{len(bad)} raw files are missing or have the wrong size ({first}); "
                                "run scripts/data_check.py")


def refuse_unlisted_files(key: str, folder: str, pattern: str = "*", base: Optional[Path] = None) -> None:
    """Stop with an error when a raw-data folder holds files that the manifest does not list.

    Parameters
    ----------
    key : str
        Dataset key.
    folder : str
        Folder relative to the data folder, for example ``"sargolini2006/subset"``.
    pattern : str
        Shell pattern of the files that the reading code would take (for example ``"*.mat"``).
    base : Path, optional
        Data folder (default ``HEXMOD_DATA`` or ``<repo>/data``).

    Raises
    ------
    hexmod.preprocessing.labels.ManifestError
        When a file matching ``pattern`` is not in the manifest (partial downloads ending in
        ``.part`` and hidden files, whose names start with a dot, such as ``.DS_Store``, are ignored).
    """
    from .labels import ManifestError

    base = data_base(base)
    listed = {row["path"] for row in _sources.read_manifest(key)}
    extra = []
    for path in sorted((base / folder).glob(pattern)):
        rel = f"{folder}/{path.name}"
        if path.is_file() and rel not in listed and not path.name.endswith(".part") and not path.name.startswith("."):
            extra.append(rel)
    if extra:
        raise ManifestError(f"{key}: {len(extra)} files in {folder}/ are not in the manifest (first: {extra[:5]}); "
                            "they would change the recordings, so remove them")


def check_animal_map(key: str, pairs: Sequence[tuple]) -> None:
    """Check that the animals named in a raw file and the manifest labels correspond one to one.

    Parameters
    ----------
    key : str
        Dataset key.
    pairs : sequence of (str, str)
        ``(animal identifier in the raw file, animal label of the manifest)`` for every session.

    Raises
    ------
    hexmod.preprocessing.labels.ManifestError
        When one raw animal has two labels or one label covers two raw animals.
    """
    from .labels import ManifestError

    forward: Dict[str, set] = {}
    backward: Dict[str, set] = {}
    for raw, label in pairs:
        forward.setdefault(str(raw), set()).add(label)
        backward.setdefault(label, set()).add(str(raw))
    bad = {r: sorted(v) for r, v in forward.items() if len(v) > 1}
    bad.update({lab: sorted(v) for lab, v in backward.items() if len(v) > 1})
    if bad:
        raise ManifestError(f"{key}: animals of the raw file and of the manifest do not correspond one to one: {bad}")
