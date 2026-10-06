"""Base class shared by all specialized agents (Section 5).

An agent:
- declares which bus topics it consumes and its update loop (Section 17),
- is throttled per loop so expensive recomputation never runs every tick,
- produces a normalized ``AgentOutput`` with traceable evidence,
- isolates its own failures (status FAILED) without stopping the system.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable

from gold_ai.core.enums import AgentStatus, UpdateLoop
from gold_ai.core.event_bus import EventBus, TOPIC_AGENT_OUTPUT
from gold_ai.core.models import AgentOutput, EvidenceItem
from gold_ai.core.state import MarketState
from gold_ai.core.timeutils import utcnow


class BaseAgent(ABC):
    agent_id: str = "base"
    agent_name: str = "Base Agent"
    loop: UpdateLoop = UpdateLoop.FAST
    # minimum seconds between two evaluations for this loop
    min_interval_s: float = 2.0
    # topics that trigger an evaluation attempt
    topics: tuple[str, ...] = ()
    # stale threshold: if inputs older than this, mark STALE + reduce confidence
    stale_after_s: float = 900.0
    # canonical evidence families this agent may consume (for double-counting control)
    evidence_families: tuple[str, ...] = ()

    def __init__(self, state: MarketState, bus: EventBus) -> None:
        self.state = state
        self.bus = bus
        self.log = logging.getLogger(f"gold_ai.agent.{self.agent_id}")
        self._last_run: datetime | None = None
        self._dirty = False
        self._running = False
        self.last_output: AgentOutput | None = None
        self.consecutive_errors = 0
        self.total_runs = 0
        self.total_errors = 0

    # ------------------------------------------------------------------
    def attach(self) -> None:
        for t in self.topics:
            self.bus.subscribe(t, self._on_event)

    def _on_event(self, topic: str, payload: Any) -> None:
        self._dirty = True

    async def maybe_run(self, now: datetime | None = None) -> AgentOutput | None:
        """Run only if dirty AND throttle interval elapsed (Section 17)."""
        now = now or utcnow()
        if not self._dirty:
            return None
        if self._last_run is not None:
            elapsed = (now - self._last_run).total_seconds()
            if elapsed < self.min_interval_s:
                return None
        return await self.run(now)

    async def force_run(self, now: datetime | None = None) -> AgentOutput | None:
        return await self.run(now or utcnow())

    async def run(self, now: datetime) -> AgentOutput | None:
        if self._running:
            return None
        self._running = True
        self._dirty = False
        try:
            self._last_run = now
            self.total_runs += 1
            out = await asyncio.get_event_loop().run_in_executor(None, self.analyze, now)
            if out is not None:
                out.updated_at = utcnow()
                self._apply_staleness(out, now)
                self.last_output = out
                self.state.set_agent_output(out)
                self.consecutive_errors = 0
                self.bus.publish(TOPIC_AGENT_OUTPUT, out)
            return out
        except Exception as exc:  # noqa: BLE001 - Section 21: isolate failures
            self.total_errors += 1
            self.consecutive_errors += 1
            self.log.exception("agent %s failed", self.agent_id)
            err_out = self.error_output(now, exc)
            self.state.set_agent_output(err_out)
            self.last_output = err_out
            self.bus.publish(TOPIC_AGENT_OUTPUT, err_out)
            return err_out
        finally:
            self._running = False

    # ------------------------------------------------------------------
    @abstractmethod
    def analyze(self, now: datetime) -> AgentOutput | None:
        """Synchronous analysis. Must not raise for expected data problems —
        encode them into data_quality / status instead."""

    def error_output(self, now: datetime, exc: Exception) -> AgentOutput:
        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            loop=self.loop.value,
            status=AgentStatus.FAILED.value,
            confidence=0.0,
            data_quality=0.0,
            error=f"{type(exc).__name__}: {exc}",
            horizon=self.default_horizon(),
        )

    def default_horizon(self) -> str:
        return {"FAST": "15m", "MEDIUM": "1h", "SLOW": "1D", "EVENT_DRIVEN": "4h"}[self.loop.value]

    def _apply_staleness(self, out: AgentOutput, now: datetime) -> None:
        if out.updated_at is None:
            return
        age = (now - out.updated_at).total_seconds()
        if age > self.stale_after_s and out.status == AgentStatus.ONLINE.value:
            out.status = AgentStatus.STALE.value
            out.confidence *= 0.6

    # ---- helpers ------------------------------------------------------
    def base_output(self, now: datetime) -> AgentOutput:
        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            loop=self.loop.value,
            status=AgentStatus.ONLINE.value,
            horizon=self.default_horizon(),
            updated_at=now,
        )

    def probs_from_score(self, score: float, neutrality: float = 0.15) -> tuple[float, float, float]:
        """Map a directional lean in [-1,1] to a 3-class probability triple.

        Neutrality mass is kept explicit; probabilities are never inflated.
        """
        s = max(-1.0, min(1.0, score))
        p_up = max(0.0, s) * (1.0 - neutrality)
        p_down = max(0.0, -s) * (1.0 - neutrality)
        p_neutral = 1.0 - p_up - p_down
        return p_up, p_down, p_neutral

    @staticmethod
    def ev(
        label: str,
        direction: str,
        strength: float,
        family: str,
        detail: str = "",
        source: str = "",
        tier=None,
        as_of: datetime | None = None,
    ) -> EvidenceItem:
        from gold_ai.core.enums import SourceTier

        return EvidenceItem(
            label=label,
            detail=detail,
            direction=direction,
            strength=max(0.0, min(1.0, strength)),
            evidence_id=f"{family}:{label}",
            source=source,
            tier=tier or SourceTier.TIER_2,
            as_of=as_of,
        )

    def has_recent_input(self, ts: datetime | None, now: datetime) -> bool:
        if ts is None:
            return False
        return (now - ts).total_seconds() <= self.stale_after_s
