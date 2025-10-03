"""AsyncIO executor for the APScheduler stub."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List


class AsyncIOExecutor:
    """Very small subset of the real APScheduler executor API."""

    def __init__(self) -> None:
        self._pending_futures: List[asyncio.Future] = []

    def submit(self, coro_factory: Callable[[], Awaitable]) -> asyncio.Task:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro_factory())
        self._pending_futures.append(task)

        def _cleanup(done: asyncio.Future) -> None:
            try:
                self._pending_futures.remove(done)
            except ValueError:
                pass

        task.add_done_callback(_cleanup)
        return task
