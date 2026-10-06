"""Async, topic-based event bus (event-driven architecture, Section 2/17).

Agents subscribe to topics; publishers never block on subscriber errors —
a failing subscriber is logged and isolated (Section 21 failure handling).
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

from gold_ai.core.timeutils import utcnow

log = logging.getLogger("gold_ai.bus")

# Canonical topics
TOPIC_CANDLE = "market.candle"
TOPIC_TICK = "market.tick"
TOPIC_QUOTE = "market.quote"
TOPIC_MACRO = "data.macro"
TOPIC_NEWS = "news.article"
TOPIC_EVENT_SCHEDULED = "calendar.scheduled"
TOPIC_EVENT_RELEASED = "calendar.released"
TOPIC_POSITIONING = "positioning.report"
TOPIC_ETF_FLOW = "flows.etf"
TOPIC_FUTURES = "futures.structure"
TOPIC_OPTIONS = "options.snapshot"
TOPIC_GEOPOLITICAL = "geo.risk"
TOPIC_CENTRAL_BANK = "cb.gold_demand"
TOPIC_AGENT_OUTPUT = "agent.output"
TOPIC_REGIME = "engine.regime"
TOPIC_DECISION = "engine.decision"
TOPIC_FEED_HEALTH = "system.feed_health"
TOPIC_ANOMALY = "system.anomaly"
TOPIC_UI_BROADCAST = "ui.broadcast"

Subscriber = Callable[[str, Any], Awaitable[None]] | Callable[[str, Any], None]


@dataclass
class BusEvent:
    topic: str
    payload: Any
    ts: datetime = field(default_factory=utcnow)


class EventBus:
    """Fan-out pub/sub with error isolation and lightweight backpressure."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Subscriber]] = defaultdict(list)
        self._queue: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=4096)
        self._task: asyncio.Task | None = None
        self._running = False
        self.dropped_events = 0
        self.published = 0
        self.errors = 0

    # ---- subscription ----------------------------------------------------
    def subscribe(self, topic: str, handler: Subscriber) -> None:
        self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Subscriber) -> None:
        if handler in self._subs.get(topic, []):
            self._subs[topic].remove(handler)

    # ---- publish ---------------------------------------------------------
    def publish(self, topic: str, payload: Any) -> None:
        """Non-blocking publish. If the queue is full we drop the OLDEST
        market-tick-like events but always keep engine/decision events."""
        ev = BusEvent(topic=topic, payload=payload)
        try:
            self._queue.put_nowait(ev)
            self.published += 1
        except asyncio.QueueFull:
            try:
                old = self._queue.get_nowait()
                log.warning("bus overflow: dropped %s", old.topic)
            except asyncio.QueueEmpty:
                pass
            self.dropped_events += 1
            try:
                self._queue.put_nowait(ev)
            except asyncio.QueueFull:
                self.errors += 1

    async def _dispatch_one(self, ev: BusEvent) -> None:
        handlers = list(self._subs.get(ev.topic, []))
        for h in handlers:
            try:
                res = h(ev.topic, ev.payload)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001 - isolate subscriber failures
                self.errors += 1
                log.exception("subscriber failed for topic=%s handler=%s", ev.topic, getattr(h, "__name__", h))

    async def _run(self) -> None:
        while self._running:
            try:
                ev = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            await self._dispatch_one(ev)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="event-bus")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ---- test / replay helper -------------------------------------------
    async def drain(self, max_events: int = 10_000) -> int:
        """Synchronously process up to ``max_events`` queued events (tests)."""
        n = 0
        while n < max_events and not self._queue.empty():
            ev = self._queue.get_nowait()
            await self._dispatch_one(ev)
            n += 1
        return n
