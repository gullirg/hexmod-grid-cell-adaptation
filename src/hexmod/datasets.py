"""Metadata of the six rat datasets.

Each dataset is identified by a citation key (``vollan2025``, ``brun2008``, ``hafting2005``,
``sargolini2006``, ``barry2012``, ``olafsdottir2016``). Animal and session labels in the data
start with the dataset's one-letter prefix (for example ``V10_1`` or ``W5_3``).

``CANONICAL_ORDER`` is the order in which datasets appear in every per-cell table. Within a
dataset, sessions and units follow the shipped manifests ``inputs/manifests/sessions.csv`` and
``units.csv``. Seeded permutation tests depend on this order, so it is fixed here rather than
taken from a directory listing.

All six datasets are publicly available. Download locations are in ``hexmod.data_sources`` and
the per-file sizes and checksums in ``inputs/manifests/files_<key>.csv`` (``manifest`` below).
``raw_subpaths`` are the files or folders, relative to the data folder (``HEXMOD_DATA``), that
the analysis reads.
"""

from __future__ import annotations

from typing import Dict, List

DATASETS: Dict[str, dict] = {
    "sargolini2006": {
        "key": "sargolini2006",
        "prefix": "S",
        "label": "Sargolini et al. (2006)",
        "citation": (
            "Sargolini, F., Fyhn, M., Hafting, T., McNaughton, B. L., Witter, M. P., Moser, M.-B. and "
            "Moser, E. I. (2006). Conjunctive representation of position, direction, and velocity in "
            "entorhinal cortex. Science 312, 758-762."
        ),
        "doi": "10.1126/science.1125572",
        "data_doi": "10.11582/2017.00019",
        "repository": "NIRD Research Data Archive",
        "licence": "CC BY 4.0",
        "raw_subpaths": ["sargolini2006/subset"],
        "manifest": "inputs/manifests/files_sargolini2006.csv",
    },
    "brun2008": {
        "key": "brun2008",
        "prefix": "B",
        "label": "Brun et al. (2008)",
        "citation": (
            "Brun, V. H., Solstad, T., Kjelstrup, K. B., Fyhn, M., Witter, M. P., Moser, E. I. and "
            "Moser, M.-B. (2008). Progressive increase in grid scale from dorsal to ventral medial "
            "entorhinal cortex. Hippocampus 18, 1200-1212."
        ),
        "doi": "10.1002/hipo.20504",
        "data_doi": "10.11582/2018.00015",
        "repository": "NIRD Research Data Archive",
        "licence": "CC BY 4.0",
        "raw_subpaths": ["brun2008/Brun2008-v3.0.0/square/recordings"],
        "manifest": "inputs/manifests/files_brun2008.csv",
    },
    "olafsdottir2016": {
        "key": "olafsdottir2016",
        "prefix": "F",
        "label": "Ólafsdóttir et al. (2016)",
        "citation": (
            "Ólafsdóttir, H. F., Carpenter, F. and Barry, C. (2016). Coordinated grid and place cell "
            "replay during rest. Nature Neuroscience 19, 792-794."
        ),
        "doi": "10.1038/nn.4291",
        "data_doi": "10.5281/zenodo.23477461",
        "repository": "Zenodo",
        "licence": "CC BY 4.0",
        "raw_subpaths": ["olafsdottir2016/olafsdottir2016_open_field_sessions.h5"],
        "manifest": "inputs/manifests/files_olafsdottir2016.csv",
    },
    "vollan2025": {
        "key": "vollan2025",
        "prefix": "V",
        "label": "Vollan et al. (2025)",
        "citation": (
            "Vollan, A. Z., Gardner, R. J., Moser, M.-B. and Moser, E. I. (2025). Left-right-alternating "
            "theta sweeps in entorhinal-hippocampal maps of space. Nature 639, 995-1005."
        ),
        "doi": "10.1038/s41586-024-08527-1",
        "data_doi": "10.25493/R5FR-EDG",
        "repository": "EBRAINS",
        "licence": "CC BY-SA 4.0",
        "raw_subpaths": [
            "vollan2025/sharing_v4/navigation/of",
            "vollan2025/sharing_v4/documentation/mat_files.md",
        ],
        "manifest": "inputs/manifests/files_vollan2025.csv",
    },
    "hafting2005": {
        "key": "hafting2005",
        "prefix": "H",
        "label": "Hafting et al. (2005)",
        "citation": (
            "Hafting, T., Fyhn, M., Molden, S., Moser, M.-B. and Moser, E. I. (2005). Microstructure of a "
            "spatial map in the entorhinal cortex. Nature 436, 801-806."
        ),
        "doi": "10.1038/nature03721",
        "data_doi": "10.11582/2017.00017",
        "repository": "NIRD Research Data Archive",
        "licence": "CC BY 4.0",
        "raw_subpaths": ["hafting2005"],
        "manifest": "inputs/manifests/files_hafting2005.csv",
    },
    "barry2012": {
        "key": "barry2012",
        "prefix": "W",
        "label": "Barry et al. (2012)",
        "citation": (
            "Barry, C., Ginzberg, L. L., O'Keefe, J. and Burgess, N. (2012). Grid cell firing patterns "
            "signal environmental novelty by expansion. Proceedings of the National Academy of Sciences "
            "USA 109(43), 17687-17692."
        ),
        "doi": "10.1073/pnas.1209918109",
        "data_doi": "10.5281/zenodo.23477461",
        "repository": "Zenodo",
        "licence": "CC BY 4.0",
        "raw_subpaths": ["barry2012/barry2012_familiar_box_sessions.h5"],
        "manifest": "inputs/manifests/files_barry2012.csv",
    },
}

#: Order of datasets in every per-cell table (seeded tests and draws depend on it).
CANONICAL_ORDER: List[str] = [
    "sargolini2006",
    "brun2008",
    "olafsdottir2016",
    "vollan2025",
    "hafting2005",
    "barry2012",
]

_BY_PREFIX = {d["prefix"]: k for k, d in DATASETS.items()}


def key_from_label(label: str) -> str:
    """Return the dataset key of an animal, session or unit label such as ``"V10_1"``.

    Parameters
    ----------
    label : str
        Any label that starts with a dataset prefix letter.

    Returns
    -------
    str
        The dataset key.
    """
    try:
        return _BY_PREFIX[label[0]]
    except (KeyError, IndexError):
        raise ValueError(f"label {label!r} does not start with a known dataset prefix") from None


def label_of(key: str) -> str:
    """Return the citation label of a dataset, for example ``"Vollan et al. (2025)"``."""
    return DATASETS[key]["label"]


def canonical_rank(key: str) -> int:
    """Return the position of a dataset in ``CANONICAL_ORDER``."""
    return CANONICAL_ORDER.index(key)
