"""Ordered process-pool map.

``map_ordered(func, inputs, workers)`` applies ``func`` to every item of ``inputs`` and returns
the results in input order. With one worker it runs in the calling process; with more it uses
``concurrent.futures.ProcessPoolExecutor.map`` with one item per task (chunk size 1), which is
how the per-cell analyses were distributed. Each item is computed independently, so the
results do not depend on the number of workers or on the start method.

Rules for callers (needed for the "spawn" start method, the default on macOS):

- ``func`` must be defined at module level in an importable module (not a lambda or a
  function defined inside ``main``);
- each input item must be picklable and should carry everything the function needs
  (for example the trajectory and spikes of one cell), so workers share no state;
- scripts must call the pipeline from ``if __name__ == "__main__":``;
- numerical libraries take their thread count from environment variables when they are loaded,
  so callers outside the step scripts should call ``hexmod.config.limit_threads()`` before
  importing numpy; otherwise each worker may start as many threads as there are cores.

Progress bars (tqdm) are shown only when standard error is a terminal, so that logs of
non-interactive runs stay readable.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, Optional, Sequence

from . import config

__all__ = ["map_ordered", "default_start_method"]


def default_start_method() -> str:
    """Return the start method used for worker processes.

    ``HEXMOD_START_METHOD`` overrides the platform default ("fork" on Linux, "spawn" on macOS
    and Windows).
    """
    method = os.environ.get("HEXMOD_START_METHOD")
    if method:
        return method
    return mp.get_start_method(allow_none=False)


def _init_worker(env: dict) -> None:
    # Takes effect for libraries loaded after the worker starts (start method "spawn"); a forked
    # worker keeps the thread settings of the parent (see ``hexmod.config.limit_threads``).
    for name, value in env.items():
        os.environ.setdefault(name, value)


def _progress(iterable: Iterable, total: int, desc: str, enabled: bool) -> Iterable:
    if not enabled or not sys.stderr.isatty():
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:  # tqdm is optional
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def map_ordered(
    func: Callable,
    inputs: Sequence,
    workers: Optional[int] = None,
    chunksize: int = 1,
    start_method: Optional[str] = None,
    progress: bool = False,
    desc: str = "items",
) -> List:
    """Apply ``func`` to each input and return the results in input order.

    Parameters
    ----------
    func : callable
        Module-level function taking one input item.
    inputs : sequence
        Input items; the result list has the same length and order.
    workers : int, optional
        Number of worker processes (default ``config.workers()``). 1 runs sequentially in the
        calling process.
    chunksize : int
        Items sent to a worker at a time (default 1).
    start_method : str, optional
        "fork", "spawn" or "forkserver" (default ``default_start_method()``).
    progress : bool
        Show a progress bar when tqdm is installed and standard error is a terminal.
    desc : str
        Label of the progress bar.

    Returns
    -------
    list
        ``[func(x) for x in inputs]``.
    """
    n_workers = config.workers(workers)
    total = len(inputs)
    if n_workers == 1 or total <= 1:
        return [func(item) for item in _progress(inputs, total, desc, progress)]
    ctx = mp.get_context(start_method or default_start_method())
    results = []
    with ProcessPoolExecutor(
        max_workers=min(n_workers, total),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(config.thread_env(),),
    ) as executor:
        for result in _progress(executor.map(func, inputs, chunksize=chunksize), total, desc, progress):
            results.append(result)
    return results
