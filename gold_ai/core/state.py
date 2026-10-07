"""Central in-memory market/data state shared by agents ("blackboard").

Thread-safe-ish for asyncio single-thread usage; every write records an
as-of timestamp so engines can reason about freshness and avoid look-ahead.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Optional

from gold_ai.core.enums import EventType, SessionName
from gold_ai.core.models import (
    AgentOutput,
    Candle,
    CentralBankGoldRecord,
    DecisionPayload,
    EtfFlowRecord,
    EconomicEventRelease,
    FuturesStructureRecord,
    GeopoliticalRiskRecord,
    MacroSeriesPoint,
    NewsArticle,
    OptionsChainSnapshot,
    PositioningReport,
    Quote,
    ScheduledEvent,
    Tick,
)
from gold_ai.core.timeutils import utcnow


class MarketState:
    """Mutable singleton-style store of the latest normalized observations."""

    def __init__(self, candle_capacity: int = 6000) -> None:
        self.symbol: str = "XAU/USD"
        self.last_price: float = 0.0
        self.prev_close_24h: float = 0.0
        self.last_tick: Optional[Tick] = None
        self.last_candle_at: Optional[datetime] = None
        self.candles: dict[str, deque[Candle]] = {
            tf: deque(maxlen=candle_capacity) for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
        }
        self.quotes: dict[str, Quote] = {}                 # watchlist symbols
        self.macro: dict[str, MacroSeriesPoint] = {}       # series_id -> latest point
        self.macro_history: dict[str, deque[MacroSeriesPoint]] = {}
        self.news: deque[NewsArticle] = deque(maxlen=500)
        self.scheduled_events: list[ScheduledEvent] = []
        self.event_releases: deque[EconomicEventRelease] = deque(maxlen=200)
        self.positioning: Optional[PositioningReport] = None
        self.etf_flows: deque[EtfFlowRecord] = deque(maxlen=120)
        self.futures: list[FuturesStructureRecord] = []
        self.options: Optional[OptionsChainSnapshot] = None
        self.geo_risks: deque[GeopoliticalRiskRecord] = deque(maxlen=100)
        self.central_bank: list[CentralBankGoldRecord] = []
        self.agent_outputs: dict[str, AgentOutput] = {}
        self.current_session: SessionName = SessionName.OFF_HOURS
        self.regime: str = "UNKNOWN"
        self.regimes_active: list[str] = []
        self.last_decision: Optional[DecisionPayload] = None
        self.decision_timeline: deque[dict[str, Any]] = deque(maxlen=500)
        self.feed_health: dict[str, dict[str, Any]] = {}
        self.updated_at: datetime = utcnow()

    # ------------------------------------------------------------------
    def add_candle(self, c: Candle) -> None:
        tf = c.timeframe if c.timeframe in self.candles else "1m"
        dq = self.candles[tf]
        # de-duplicate by open time
        if dq and dq[-1].event_at == c.event_at:
            dq[-1] = c
        else:
            dq.append(c)
        # aggregate higher timeframes from 1m candles
        if tf == "1m":
            self._aggregate_higher(c)
        self.last_price = c.close
        self.last_candle_at = c.event_at
        self.updated_at = utcnow()

    def _aggregate_higher(self, c: Candle) -> None:
        from gold_ai.core.timeutils import floor_to

        for tf, minutes in (("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("4h", 240)):
            bucket = floor_to(c.event_at, __import__("datetime").timedelta(minutes=minutes))
            dq = self.candles[tf]
            if dq and dq[-1].event_at == bucket:
                last = dq[-1]
                dq[-1] = last.model_copy(
                    update={
                        "high": max(last.high, c.high),
                        "low": min(last.low, c.low),
                        "close": c.close,
                        "volume": last.volume + c.volume,
                    }
                )
            else:
                dq.append(
                    Candle(
                        source=c.source, tier=c.tier, symbol=c.symbol, frequency=c.frequency,
                        retrieved_at=c.retrieved_at, event_at=bucket, published_at=c.published_at,
                        open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume, timeframe=tf,
                    )
                )

    def set_quote(self, q: Quote) -> None:
        self.quotes[q.symbol] = q
        self.updated_at = utcnow()

    def set_macro(self, p: MacroSeriesPoint) -> None:
        self.macro[p.series_id] = p
        hist = self.macro_history.setdefault(p.series_id, deque(maxlen=2000))
        if hist and hist[-1].effective_at >= p.effective_at:
            hist[-1] = p  # revision replaces same-time observation
        else:
            hist.append(p)
        self.updated_at = utcnow()

    def macro_series(self, series_id: str) -> list[MacroSeriesPoint]:
        return list(self.macro_history.get(series_id, []))

    def add_news(self, a: NewsArticle) -> None:
        self.news.append(a)
        self.updated_at = utcnow()

    def recent_news(self, minutes: int = 120, now: datetime | None = None) -> list[NewsArticle]:
        now = now or utcnow()
        cutoff_ts = now.timestamp() - minutes * 60
        return [a for a in self.news if a.effective_at.timestamp() >= cutoff_ts]

    def add_release(self, r: EconomicEventRelease) -> None:
        self.event_releases.append(r)
        for e in self.scheduled_events:
            if e.event_type == r.event_type and not e.released and abs((r.effective_at - e.scheduled_at).total_seconds()) < 6 * 3600:
                e.released = True
        self.updated_at = utcnow()

    def upcoming_events(self, now: datetime | None = None, hours: float = 72) -> list[ScheduledEvent]:
        now = now or utcnow()
        horizon = now.timestamp() + hours * 3600
        out = [e for e in self.scheduled_events if not e.released and now.timestamp() <= e.scheduled_at.timestamp() <= horizon]
        return sorted(out, key=lambda e: e.scheduled_at)

    def next_event(self, now: datetime | None = None) -> Optional[ScheduledEvent]:
        evs = self.upcoming_events(now=now, hours=24 * 30)
        return evs[0] if evs else None

    def releases_since(self, dt: datetime, event_type: EventType | None = None) -> list[EconomicEventRelease]:
        out = [r for r in self.event_releases if r.effective_at >= dt]
        if event_type:
            out = [r for r in out if r.event_type == event_type]
        return out

    def set_agent_output(self, o: AgentOutput) -> None:
        self.agent_outputs[o.agent_id] = o
        self.updated_at = utcnow()

    def record_decision_change(self, entry: dict[str, Any]) -> None:
        self.decision_timeline.append(entry)

    def snapshot_prices(self, timeframe: str = "1m", n: int | None = None) -> list[float]:
        dq = self.candles.get(timeframe, deque())
        closes = [c.close for c in dq]
        return closes[-n:] if n else closes

    def as_dict_summary(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "agents_online": sum(1 for a in self.agent_outputs.values() if a.status == "ONLINE"),
            "regime": self.regime,
            "session": self.current_session.value,
            "updated_at": self.updated_at.isoformat(),
        }
