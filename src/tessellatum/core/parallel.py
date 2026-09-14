"""A shared thread pool for the pipeline's parallel stages.

OpenCV and NumPy release the GIL inside their heavy calls, so plain threads
give real parallelism for per-region work, without process start-up or
data-copying costs.
"""

from __future__ import annotations

import heapq
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def worker_count() -> int:
    """Worker threads to use: ``TESSELLATUM_THREADS`` if set, else one per CPU."""
    try:
        return max(1, int(os.environ["TESSELLATUM_THREADS"]))
    except (KeyError, ValueError):
        return os.cpu_count() or 1


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=worker_count(), thread_name_prefix="tessellatum")
        return _executor


def for_each_stripe(fn: Callable[[int, int], None], length: int) -> None:
    """Call ``fn(start, stop)`` on the shared pool for stripes covering ``range(length)``.

    Uses a few stripes per worker so uneven stripes (e.g. rows with more
    detail) still balance out. ``fn`` must release the GIL to run in parallel.
    Must not be called from inside a pool worker.
    """
    workers = worker_count()
    stripes = min(length, workers * 4)
    if workers <= 1 or stripes <= 1:
        fn(0, length)
        return
    bounds = [length * i // stripes for i in range(stripes + 1)]
    futures = [_pool().submit(fn, bounds[i], bounds[i + 1]) for i in range(stripes)]
    wait(futures)
    for future in futures:
        future.result()


def map_balanced(
    fn: Callable[[T], R], items: Sequence[T], costs: Sequence[float], min_pooled_cost: float = 0
) -> list[R]:
    """``[fn(item) for item in items]``, with the costly items spread over the shared pool.

    Items costing at least ``min_pooled_cost`` are dealt biggest-first into
    one batch per worker (each to the batch with the least total cost so far)
    and run on the pool. Cheaper items run on the calling thread meanwhile:
    for those, thread hand-off and GIL contention cost more than they save.
    Must not be called from inside a pool worker.
    """
    n = len(items)
    pooled = [i for i in range(n) if costs[i] >= min_pooled_cost]
    if worker_count() <= 1 or not pooled or n == 1:
        return [fn(item) for item in items]

    workers = min(worker_count(), len(pooled))
    batches: list[list[int]] = [[] for _ in range(workers)]
    loads = [(0.0, b) for b in range(workers)]
    for index in sorted(pooled, key=costs.__getitem__, reverse=True):
        load, b = heapq.heappop(loads)
        batches[b].append(index)
        heapq.heappush(loads, (load + costs[index], b))

    results: list = [None] * n

    def run(batch: list[int]) -> None:
        for index in batch:
            results[index] = fn(items[index])

    futures = [_pool().submit(run, batch) for batch in batches]
    try:
        for index in range(n):
            if costs[index] < min_pooled_cost:
                results[index] = fn(items[index])
    finally:
        wait(futures)
    for future in futures:
        future.result()  # re-raise anything a worker raised
    return results
