"""Provider abstraction interfaces (Section 3).

The system is never hard-coded to one data provider. Every provider returns
normalized records from ``gold_ai.core.models`` with full timestamp provenance.

Real deployments plug in licensed/official endpoints here (FRED API, BLS API,
CFTC Socrata datasets, World Gold Council, exchange market-data feeds...).
API keys come ONLY from environment variables (Section 22). This repository
ships a clearly-labelled synthetic simulator so the terminal runs offline for
research/education without ever fabricating "real" source attribution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Optional

from gold_ai.core.models import (
    Candle,
    CentralBankGoldRecord,
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


class ProviderHealth:
    """Per-provider connectivity/freshness telemetry shown in the UI."""

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        self.connected = False
        self.last_success_at: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.latency_ms: float = 0.0
        self.error_count = 0

    def ok(self, latency_ms: float, now: datetime) -> None:
        self.connected = True
        self.last_success_at = now
        self.latency_ms = latency_ms
        self.last_error = None

    def fail(self, err: str) -> None:
        self.connected = False
        self.error_count += 1
        self.last_error = err


class MarketDataProvider(ABC):
    provider_id: str = "market-base"

    @abstractmethod
    async def fetch_candles(self, symbol: str, timeframe: str, since: datetime, limit: int) -> list[Candle]: ...

    @abstractmethod
    async def stream_ticks(self) -> AsyncIterator[Tick]: ...

    @abstractmethod
    async def latest_quote(self, symbol: str) -> Quote: ...


class NewsProvider(ABC):
    provider_id: str = "news-base"

    @abstractmethod
    async def fetch_recent(self, since: datetime, topics: Optional[list[str]] = None) -> list[NewsArticle]: ...

    @abstractmethod
    async def stream_new(self) -> AsyncIterator[NewsArticle]: ...


class EconomicCalendarProvider(ABC):
    provider_id: str = "calendar-base"

    @abstractmethod
    async def scheduled_events(self, start: datetime, end: datetime) -> list[ScheduledEvent]: ...

    @abstractmethod
    async def released_events(self, since: datetime) -> list[EconomicEventRelease]: ...


class MacroDataProvider(ABC):
    provider_id: str = "macro-base"

    @abstractmethod
    async def series(self, series_id: str, since: datetime) -> list[MacroSeriesPoint]: ...


class RatesProvider(ABC):
    provider_id: str = "rates-base"

    @abstractmethod
    async def treasury_yields(self, as_of: datetime) -> list[MacroSeriesPoint]: ...

    @abstractmethod
    async def real_yield_proxy(self, as_of: datetime) -> Optional[MacroSeriesPoint]: ...


class DollarIndexProvider(ABC):
    provider_id: str = "dxy-base"

    @abstractmethod
    async def dxy_quote(self) -> Quote: ...

    @abstractmethod
    async def dxy_series(self, since: datetime) -> list[MacroSeriesPoint]: ...


class FuturesProvider(ABC):
    provider_id: str = "futures-base"

    @abstractmethod
    async def structure(self, underlying: str = "gold") -> list[FuturesStructureRecord]: ...


class PositioningProvider(ABC):
    provider_id: str = "positioning-base"

    @abstractmethod
    async def latest_cot(self) -> Optional[PositioningReport]: ...


class ETFProvider(ABC):
    provider_id: str = "etf-base"

    @abstractmethod
    async def latest_flows(self) -> list[EtfFlowRecord]: ...


class GeopoliticalRiskProvider(ABC):
    provider_id: str = "geo-base"

    @abstractmethod
    async def active_risks(self, as_of: datetime) -> list[GeopoliticalRiskRecord]: ...


class CentralBankProvider(ABC):
    provider_id: str = "cb-base"

    @abstractmethod
    async def gold_demand(self, as_of: datetime) -> list[CentralBankGoldRecord]: ...


class OptionsProvider(ABC):
    provider_id: str = "options-base"

    @abstractmethod
    async def snapshot(self, underlying: str = "gold") -> Optional[OptionsChainSnapshot]: ...
