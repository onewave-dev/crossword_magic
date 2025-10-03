"""AsyncIO scheduler for the APScheduler stub."""
from __future__ import annotations

import asyncio
import datetime as dt
import itertools
from typing import Any, Dict, Iterable, List, Optional

from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.job import Job


class AsyncIOScheduler:
    """Very small subset of :class:`apscheduler.schedulers.asyncio.AsyncIOScheduler`."""

    _id_iter = itertools.count(1)

    def __init__(
        self,
        *,
        timezone: Optional[dt.tzinfo] = None,
        executors: Optional[Dict[str, AsyncIOExecutor]] = None,
        **_: Any,
    ) -> None:
        self.timezone = timezone or dt.timezone.utc
        self.executor = executors.get("default") if executors else None
        self._jobs: Dict[str, Job] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = False

    # ------------------------------------------------------------------ helpers
    def _ensure_executor(self) -> AsyncIOExecutor:
        if self.executor is None:
            self.executor = AsyncIOExecutor()
        return self.executor

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self.timezone)

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    # ------------------------------------------------------------------ lifecycle
    def configure(
        self,
        *,
        timezone: Optional[dt.tzinfo] = None,
        executors: Optional[Dict[str, AsyncIOExecutor]] = None,
        **_: Any,
    ) -> None:
        if timezone is not None:
            self.timezone = timezone
        if executors:
            self.executor = executors.get("default", self.executor)

    def start(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        self._ensure_executor()
        self.running = True
        for job in list(self._jobs.values()):
            job.ensure_scheduled()

    def shutdown(self, *, wait: bool = True) -> None:
        self.running = False
        for job in list(self._jobs.values()):
            job.remove()
        if wait and self.executor:
            for pending in list(self.executor._pending_futures):
                pending.cancel()

    # ---------------------------------------------------------------- job mgmt
    def add_job(
        self,
        func: Any,
        *,
        trigger: str,
        args: Iterable[Any] | None = None,
        kwargs: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        name: Optional[str] = None,
        run_date: Optional[dt.datetime] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        seconds: Optional[float] = None,
        timezone: Optional[dt.tzinfo] = None,
        **_: Any,
    ) -> Job:
        job_id = id or f"job-{next(self._id_iter)}"
        job = Job(
            self,
            func,
            trigger=trigger,
            args=tuple(args or ()),
            kwargs=kwargs,
            job_id=job_id,
            name=name,
            run_date=run_date,
            start_date=start_date,
            end_date=end_date,
            seconds=seconds,
            timezone=timezone or self.timezone,
        )
        self._jobs[job.id] = job
        if self.running:
            job.ensure_scheduled()
        return job

    def get_jobs(self) -> List[Job]:
        return list(self._jobs.values())

    def remove_job(self, job_id: str) -> None:
        job = self._jobs.pop(job_id, None)
        if job:
            job.remove()

    # ---------------------------------------------------------------- utilities
    def _drop_job(self, job: Job) -> None:
        self._jobs.pop(job.id, None)

    def create_task(self, coro: asyncio.Awaitable) -> asyncio.Task:
        loop = self._loop or asyncio.get_event_loop()
        task = loop.create_task(coro)
        executor = self._ensure_executor()
        executor._pending_futures.append(task)

        def _cleanup(done: asyncio.Future) -> None:
            try:
                executor._pending_futures.remove(done)
            except ValueError:
                pass

        task.add_done_callback(_cleanup)
        return task


__all__ = ["AsyncIOScheduler"]
