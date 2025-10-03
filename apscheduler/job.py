"""Minimal job representation used by the APScheduler stub."""
from __future__ import annotations

import asyncio
import datetime as dt
import itertools
from typing import Any, Awaitable, Callable, Optional


class Job:
    """Represents a scheduled job managed by :class:`AsyncIOScheduler`."""

    _id_iter = itertools.count(1)

    def __init__(
        self,
        scheduler: "AsyncIOScheduler",
        func: Callable[..., Awaitable[Any] | Any],
        *,
        trigger: str,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        run_date: Optional[dt.datetime] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        seconds: Optional[float] = None,
        timezone: Optional[dt.tzinfo] = None,
    ) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler  # circular import guard

        if trigger not in {"date", "interval"}:
            raise ValueError(f"Unsupported trigger type: {trigger}")

        self.scheduler: AsyncIOScheduler = scheduler
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.id = job_id or f"job-{next(self._id_iter)}"
        self.name = name or self.id
        self.trigger = trigger
        self.timezone = timezone or scheduler.timezone
        self._run_date = self._normalise_datetime(run_date)
        self._start_date = self._normalise_datetime(start_date)
        self._end_date = self._normalise_datetime(end_date)
        self._interval = float(seconds) if seconds is not None else None

        self._handle: Optional[asyncio.TimerHandle] = None
        self._cancelled = False
        self._paused = False
        self._next_run_time: Optional[dt.datetime] = None

        self._prepare_initial_schedule()

    # Utilities -----------------------------------------------------------------
    def _normalise_datetime(self, value: Optional[dt.datetime]) -> Optional[dt.datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=self.timezone)
        return value.astimezone(self.timezone)

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self.timezone)

    # Scheduling ----------------------------------------------------------------
    def _prepare_initial_schedule(self) -> None:
        if self.trigger == "date":
            self._next_run_time = self._run_date or self._now()
        else:  # interval
            base = self._start_date or self._now()
            if self._interval is None:
                raise ValueError("Interval trigger requires `seconds`")
            now = self._now()
            if base <= now:
                self._next_run_time = now + dt.timedelta(seconds=self._interval)
            else:
                self._next_run_time = base

    def ensure_scheduled(self) -> None:
        if self._cancelled or self._paused:
            return
        if self._next_run_time is None:
            return
        loop = self.scheduler.loop
        if loop is None:
            return
        delay = (self._next_run_time - self._now()).total_seconds()
        if delay < 0:
            delay = 0
        self._handle = loop.call_later(delay, self._execute_once)

    def _execute_once(self) -> None:
        self._handle = None
        if self._cancelled or self._paused:
            return

        run_started = self._next_run_time or self._now()
        result = self.func(*self.args, **self.kwargs)
        if asyncio.iscoroutine(result):
            task = self.scheduler.create_task(result)
        else:
            async def _wrapped() -> Any:
                return result

            task = self.scheduler.create_task(_wrapped())

        if task is not None and hasattr(task, "add_done_callback"):
            task.add_done_callback(lambda _t: None)

        self._schedule_after_run(run_started)

    def _schedule_after_run(self, last_run: dt.datetime) -> None:
        if self.trigger == "date":
            self.remove()
            return

        if self._interval is None:
            self.remove()
            return

        next_time = last_run + dt.timedelta(seconds=self._interval)
        if self._end_date and next_time > self._end_date:
            self.remove()
            return

        self._next_run_time = next_time
        self.ensure_scheduled()

    # API mirrored by the real APScheduler job ---------------------------------
    @property
    def next_run_time(self) -> Optional[dt.datetime]:
        return None if self._cancelled else self._next_run_time

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        self._cancelled = True
        self._next_run_time = None
        self.scheduler._drop_job(self)

    def pause(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        self._paused = True

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        if self._next_run_time is None and self.trigger == "interval" and self._interval is not None:
            self._next_run_time = self._now() + dt.timedelta(seconds=self._interval)
        self.ensure_scheduled()


__all__ = ["Job"]
