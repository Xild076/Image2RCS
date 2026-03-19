from __future__ import annotations

import os
import time
from typing import Sequence

import torch
from torch.utils.data import DataLoader


WORKER_CANDIDATES: tuple[int, ...] = (0, 1, 2, 4)


def parse_num_workers(value: str | int) -> str | int:
    if isinstance(value, int):
        if value < 0:
            raise ValueError("num_workers must be >= 0")
        return value

    text = str(value).strip().lower()
    if text == "auto":
        return "auto"

    parsed = int(text)
    if parsed < 0:
        raise ValueError("num_workers must be >= 0")
    return parsed


def configure_cpu_threads(cpu_threads: int | None) -> None:
    if cpu_threads is None or cpu_threads <= 0:
        return
    torch.set_num_threads(cpu_threads)
    try:
        interop_threads = max(1, min(4, cpu_threads // 2))
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # set_num_interop_threads can only be called once.
        pass


def _default_workers(dataset_size: int) -> int:
    if dataset_size <= 512:
        return 0
    return 2


def autotune_num_workers(
    dataset,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    max_batches: int = 8,
    candidates: Sequence[int] = WORKER_CANDIDATES,
) -> tuple[int, dict[int, float]]:
    dataset_size = len(dataset)
    if dataset_size == 0:
        return 0, {}

    cpu_count = os.cpu_count() or 1
    filtered = [c for c in candidates if 0 <= c <= cpu_count]
    if not filtered:
        return _default_workers(dataset_size), {}

    timings: dict[int, float] = {}
    for num_workers in filtered:
        try:
            persistent_workers = num_workers > 0
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            total_batches = min(max_batches, len(loader))
            if total_batches == 0:
                timings[num_workers] = float("inf")
                continue

            start = time.perf_counter()
            iterator = iter(loader)
            batch_count = 0
            while batch_count < total_batches:
                try:
                    next(iterator)
                except StopIteration:
                    break
                batch_count += 1
            elapsed = time.perf_counter() - start
            timings[num_workers] = elapsed / max(batch_count, 1)

            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            timings[num_workers] = float("inf")

    if not timings:
        return _default_workers(dataset_size), {}

    finite_timings = {k: v for k, v in timings.items() if v != float("inf")}
    if not finite_timings:
        return _default_workers(dataset_size), timings

    best_workers = min(finite_timings, key=finite_timings.get)
    return best_workers, timings


def format_worker_timings(timings: dict[int, float]) -> str:
    if not timings:
        return "n/a"
    items = []
    for num_workers, batch_time in sorted(timings.items()):
        if batch_time == float("inf"):
            items.append(f"{num_workers}:blocked")
        else:
            items.append(f"{num_workers}:{batch_time * 1000.0:.2f}ms/batch")
    return ", ".join(items)
