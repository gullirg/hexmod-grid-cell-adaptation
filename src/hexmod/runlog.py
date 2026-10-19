"""Run records.

Every step writes one JSON record to ``results/logs/<step>.json`` with the start and end time,
runtime, worker count, command line, software versions, the inputs it read (with size and
sha256), the outputs it wrote and free-form notes. File paths in the record start with
``results/``, ``data/`` or ``repo/``. Usage::

    with RunRecord("step04_adaptation_nonparametric", workers=n, argv=argv) as rec:
        rec.add_input(path_in)
        ...
        rec.add_output(path_out)
        rec.note("n_cells", 3938)

If the block raises, the record is written with ``"status": "failed"``, the error message and
the traceback, and the exception propagates.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__, config
from .io import sha256_file, write_json

#: Packages whose versions are recorded (nothing is imported to read them).
VERSION_PACKAGES = ("numpy", "scipy", "pandas", "scikit-image", "matplotlib", "tqdm")

#: Import names of distributions whose import name differs from the distribution name.
_IMPORT_NAMES = {"scikit-image": "skimage"}


def package_versions(packages: Sequence[str] = VERSION_PACKAGES) -> Dict[str, Optional[str]]:
    """Return the versions of the given distributions (None when absent).

    For a package that is already imported, the version of the loaded module is recorded (this
    is the code that ran, even when stale package metadata is also installed); otherwise the
    version is read from the package metadata.
    """
    from importlib import metadata

    out: Dict[str, Optional[str]] = {"python": platform.python_version(), "hexmod": __version__}
    for name in packages:
        module = sys.modules.get(_IMPORT_NAMES.get(name, name))
        version = getattr(module, "__version__", None) if module is not None else None
        if version is None:
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                version = None
        out[name] = str(version) if version is not None else None
    return out


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _relative(path: Path, bases: Sequence[Tuple[str, Path]]) -> str:
    """Path relative to the first matching base, as ``<label>/<relative path>``.

    The path is first matched as given (absolute, symbolic links kept), so that a file reached through a link
    inside a base folder (for example a data folder of links to the raw files) keeps its place under that base;
    then with links resolved. A path under no base is recorded by its file name.
    """
    for resolve in (False, True):
        target = Path(path).resolve() if resolve else Path(os.path.abspath(path))
        for label, base in bases:
            root = Path(base).resolve() if resolve else Path(os.path.abspath(base))
            try:
                return f"{label}/{target.relative_to(root).as_posix()}"
            except ValueError:
                continue
    return Path(path).name


class RunRecord:
    """Collect and write the run record of one step.

    Parameters
    ----------
    step : str
        Step name, the script stem without ``.py`` (for example ``"step02_cohort"``).
    workers : int, optional
        Worker processes used.
    results : Path, optional
        Results folder (default ``config.results_dir()``).
    hash_inputs : bool
        Compute sha256 of inputs (default True; large inputs take a few seconds).
    argv : sequence of str, optional
        Arguments the step was called with (its ``main(argv)``); default ``sys.argv``.
    """

    def __init__(self, step: str, workers: Optional[int] = None, results: Optional[Path] = None,
                 hash_inputs: bool = True, argv: Optional[Sequence[str]] = None):
        self.step = step
        self.workers = workers
        self.results = Path(results) if results is not None else config.results_dir()
        self.hash_inputs = hash_inputs
        self.inputs: List[Dict[str, Any]] = []
        self.outputs: List[Dict[str, Any]] = []
        self.notes: Dict[str, Any] = {}
        self._t0 = 0.0
        self._start = ""
        self.path = self.results / "logs" / f"{step}.json"
        self.argv = [f"{step}.py"] + [str(a) for a in argv] if argv is not None else None

    # paths are written relative to the results folder (for example "results/rat/cohort/cells.csv"),
    # the data folder or the repository
    def _bases(self):
        return [("results", self.results), ("results", config.results_dir()), ("data", config.data_dir()),
                ("repo", config.repo_root())]

    def _entry(self, path: Path, hash_it: bool) -> Dict[str, Any]:
        path = Path(path)
        entry: Dict[str, Any] = {"path": _relative(path, self._bases())}
        if path.is_file():
            entry["bytes"] = path.stat().st_size
            if hash_it:
                entry["sha256"] = sha256_file(path)
        elif path.is_dir():
            entry["kind"] = "folder"
        else:
            entry["missing"] = True
        return entry

    def add_input(self, path, hash_it: Optional[bool] = None) -> None:
        """Record an input file or folder."""
        self.inputs.append(self._entry(Path(path), self.hash_inputs if hash_it is None else hash_it))

    def add_output(self, path, hash_it: bool = True) -> None:
        """Record an output file or folder (call after it is written)."""
        self.outputs.append(self._entry(Path(path), hash_it))

    def note(self, key: str, value: Any) -> None:
        """Store a free-form value in the record."""
        self.notes[key] = value

    def __enter__(self) -> "RunRecord":
        self._t0 = time.perf_counter()
        self._start = _now()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        record = {
            "step": self.step,
            "status": "ok" if exc_type is None else "failed",
            "start": self._start,
            "end": _now(),
            "runtime_s": round(time.perf_counter() - self._t0, 3),
            "workers": self.workers,
            "argv": self.argv if self.argv is not None else _command_line(),
            "platform": {"system": platform.system(), "machine": platform.machine(),
                         "processor_count": _cpu_count()},
            "versions": package_versions(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "notes": self.notes,
        }
        if exc_type is not None:
            record["error"] = "".join(traceback.format_exception_only(exc_type, exc)).strip()
            record["traceback"] = "".join(traceback.format_exception(exc_type, exc, tb))
        write_json(record, self.path)
        return False


def _command_line() -> List[str]:
    return [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []


def _cpu_count() -> Optional[int]:
    return os.cpu_count()


def read_record(step: str) -> Dict[str, Any]:
    """Return the run record of a step, or raise FileNotFoundError."""
    from .io import read_json

    return read_json(config.results_dir() / "logs" / f"{step}.json")
