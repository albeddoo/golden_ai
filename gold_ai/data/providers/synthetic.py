"""Synthetic provider implementations (clearly labelled SIMULATED data).

These implement the provider interfaces from ``gold_ai.data.providers.base``.
Swap them for real licensed providers by changing config only — no agent code
touches a concrete provider class.

Data mode banner in the UI always shows "SIMULATED" when these are active.
"""
from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

from gold_ai.core.enums import (
    EventType,
    Frequency,
    NewsClassification,
    SourceTier,
)
from gold_ai.core.models import (
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
from gold_ai.core.timeutils import UTC, utcnow
from gold_ai.data.providers.base import (
    CentralBankProvider,
    DollarIndexProvider,
    ETFProvider,
    EconomicCalendarProvider,
    FuturesProvider,
    GeopoliticalRiskProvider,
    MacroDataProvider,
    MarketDataProvider,
    NewsProvider,
    OptionsProvider,
    PositioningProvider,
    RatesProvider,
)
from gold_ai.data.providers.synthetic_engine import SyntheticMarketEngine


class SyntheticMarketProvider(MarketDataProvider):
    provider_id = "synthetic:market"

    def __init__(self, engine: SyntheticMarketEngine, tick_hz: float = 2.0) -> None:
        self.engine = engine
        self.tick_hz = tick_hz

    async def fetch_candles(self, symbol: str, timeframe: str, since: datetime, limit: int) -> list:
        # historical bootstrap uses the same generative process (still SIMULATED)
        from gold_ai.data.providers.synthetic_engine import make_candle_from_step

        out = []
        ts = since
        step_min = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 1)
        for _ in range(limit):
            bar = self.engine.step(ts)
            out.append(make_candle_from_step(self.engine, bar, ts))
            ts = ts + timedelta(minutes=step_min)
        return out

    async def stream_ticks(self) -> AsyncIterator[Tick]:
        while True:
            await asyncio.sleep(1.0 / self.tick_hz)
            p = self.engine.price
            half_spread = max(0.02, p * 0.00004 * self.engine.vol_state)
            now = utcnow()
            yield Tick(
                source=self.provider_id, tier=SourceTier.TIER_2, symbol="XAU/USD",
                frequency=Frequency.TICK, retrieved_at=now, event_at=now,
                bid=p - half_spread, ask=p + half_spread, last=p,
            )

    async def latest_quote(self, symbol: str) -> Quote:
        return self.engine.quote(symbol, self.engine.price)


class SyntheticNewsProvider(NewsProvider):
    """Newswire-style template generator with realistic mixed quality.

    Includes Tier-3 rumor items and duplicates on purpose so the news
    intelligence agent has deduplication/unverified handling to do.
    """

    provider_id = "synthetic:news"

    TEMPLATES: list[tuple[str, str, float, SourceTier]] = [
        ("Fed officials signal comfort with current policy stance, no imminent rate change", "fed", 0.7, SourceTier.TIER_2),
        ("Dollar slips as traders pare bets on aggressive Fed tightening", "usd", 0.8, SourceTier.TIER_2),
        ("Treasury yields climb on firm services data; 10-year at multi-week high", "rates", 0.75, SourceTier.TIER_2),
        ("Inflation expectations edge higher in NY Fed survey", "inflation", 0.65, SourceTier.TIER_1),
        ("Geopolitical tensions escalate in Eastern Europe; airspace closures reported", "geo", 0.9, SourceTier.TIER_2),
        ("Ceasefire talks resume, markets weigh de-escalation scenario", "geo", 0.6, SourceTier.TIER_2),
        ("China's central bank adds to gold reserves for {n}th consecutive month", "cb_gold", 0.85, SourceTier.TIER_1),
        ("Large gold ETF sees notable weekly inflow as hedging demand rises", "flows", 0.6, SourceTier.TIER_2),
        ("Analysts debate whether recent gold rally has room above key resistance", "commentary", 0.2, SourceTier.TIER_3),
        ("Unverified report claims emergency G7 meeting on commodity sanctions", "geo", 0.5, SourceTier.TIER_3),
        ("US jobless claims rise more than expected, labor market cooling further", "labor", 0.8, SourceTier.TIER_1),
        ("Retail sales beat estimates, consumer resilience supports growth outlook", "growth", 0.7, SourceTier.TIER_1),
        ("Real yields retreat after breakevens jump on oil spike", "real_yield", 0.7, SourceTier.TIER_2),
        ("Gold miners outperform amid bullish sentiment poll", "commentary", 0.15, SourceTier.TIER_3),
        ("CPI components suggest shelter disinflation continuing", "inflation", 0.6, SourceTier.TIER_1),
    ]

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random(7)

    async def fetch_recent(self, since: datetime, topics: Optional[list[str]] = None) -> list[NewsArticle]:
        n = self.rng.randint(3, 6)
        return [self._make_article(i) for i in range(n)]

    async def stream_new(self) -> AsyncIterator[NewsArticle]:
        while True:
            await asyncio.sleep(self.rng.uniform(8, 25))
            yield self._make_article(self.rng.randint(1, 200))

    def _make_article(self, n: int) -> NewsArticle:
        headline, topic, impact, tier = self.rng.choice(self.TEMPLATES)
        now = utcnow()
        verified = not (tier == SourceTier.TIER_3 and self.rng.random() < 0.6)
        cls = NewsClassification.CONFIRMED_EVENT if impact > 0.7 and verified else (
            NewsClassification.RUMOR_UNVERIFIED if not verified else NewsClassification.MARKET_COMMENTARY
        )
        return NewsArticle(
            source=self.provider_id, tier=tier, frequency=Frequency.EVENT,
            retrieved_at=now, event_at=now, published_at=now,
            headline=headline.format(n=n), topics=[topic],
            classification=cls, verified=verified,
            potential_market_impact=impact * (0.4 if not verified else 1.0),
        )


class SyntheticCalendarProvider(EconomicCalendarProvider):
    provider_id = "synthetic:calendar"

    SCHEDULE = [
        (EventType.CPI, "US CPI m/m", 0.95, 2.0),
        (EventType.NFP, "Nonfarm Payrolls", 0.95, 18.0),
        (EventType.FOMC, "FOMC Decision", 1.0, 48.0),
        (EventType.PCE, "Core PCE price index", 0.85, 5.0),
        (EventType.GDP, "GDP q/q advance", 0.8, 2.4),
        (EventType.RETAIL_SALES, "Retail sales m/m", 0.7, 0.4),
        (EventType.ISM, "ISM Manufacturing PMI", 0.75, 49.5),
        (EventType.JOLTS, "JOLTS job openings", 0.6, 7.5),
        (EventType.CLAIMS, "Initial jobless claims", 0.5, 220),
    ]

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random(11)

    async def scheduled_events(self, start: datetime, end: datetime) -> list[ScheduledEvent]:
        out: list[ScheduledEvent] = []
        hours_span = (end - start).total_seconds() / 3600
        n = max(2, int(hours_span / 20))
        for i in range(n):
            etype, name, imp, prev = self.SCHEDULE[i % len(self.SCHEDULE)]
            when = start + timedelta(hours=self.rng.uniform(0.5, hours_span))
            out.append(ScheduledEvent(
                event_type=etype, name=name, scheduled_at=when, importance=imp,
                previous=prev, forecast=prev + self.rng.uniform(-0.5, 0.5) * abs(prev) * 0.05,
                source=self.provider_id,
            ))
        return sorted(out, key=lambda e: e.scheduled_at)

    async def released_events(self, since: datetime) -> list[EconomicEventRelease]:
        return []


class SyntheticMacroProviders:
    """Rates/DXY/Macro series come straight from the shared sim engine."""

    def __init__(self, engine: SyntheticMarketEngine) -> None:
        self.engine = engine

    def point(self, series_id: str, value: float, unit: str, freq: Frequency) -> MacroSeriesPoint:
        now = utcnow()
        return MacroSeriesPoint(
            source="synthetic:sim", tier=SourceTier.TIER_2, symbol="XAU/USD", unit=unit,
            frequency=freq, retrieved_at=now, event_at=now, series_id=series_id, value=value,
        )

    def all_points(self) -> list[MacroSeriesPoint]:
        e = self.engine
        return [
            self.point("DXY", e.dxy, "index", Frequency.SECOND),
            self.point("UST02Y", e.y2, "%", Frequency.SECOND),
            self.point("UST05Y", e.y5, "%", Frequency.SECOND),
            self.point("UST10Y", e.y10, "%", Frequency.SECOND),
            self.point("UST30Y", e.y30, "%", Frequency.SECOND),
            self.point("DFII10_REAL", e.tips10, "%", Frequency.SECOND),
        ]


class SyntheticRatesProvider(RatesProvider):
    provider_id = "synthetic:rates"

    def __init__(self, holder: SyntheticMacroProviders) -> None:
        self.holder = holder

    async def treasury_yields(self, as_of: datetime) -> list[MacroSeriesPoint]:
        return [p for p in self.holder.all_points() if p.series_id.startswith("UST")]

    async def real_yield_proxy(self, as_of: datetime) -> Optional[MacroSeriesPoint]:
        return next(p for p in self.holder.all_points() if p.series_id == "DFII10_REAL")


class SyntheticDollarIndexProvider(DollarIndexProvider):
    provider_id = "synthetic:dxy"

    def __init__(self, holder: SyntheticMacroProviders) -> None:
        self.holder = holder

    async def dxy_quote(self) -> Quote:
        p = next(x for x in self.holder.all_points() if x.series_id == "DXY")
        now = utcnow()
        return Quote(source=self.provider_id, tier=SourceTier.TIER_2, symbol="DXY",
                     frequency=Frequency.SECOND, retrieved_at=now, event_at=now, last=p.value)

    async def dxy_series(self, since: datetime) -> list[MacroSeriesPoint]:
        return [next(x for x in self.holder.all_points() if x.series_id == "DXY")]


class SyntheticPositioningProvider(PositioningProvider):
    provider_id = "synthetic:cftc"

    async def latest_cot(self) -> Optional[PositioningReport]:
        now = utcnow()
        asof = now - timedelta(days=3)  # COT is structurally lagged
        return PositioningReport(
            source=self.provider_id, tier=SourceTier.TIER_1, frequency=Frequency.WEEKLY,
            retrieved_at=now, event_at=asof, published_at=asof,
            report_asof=asof, speculative_net_long=182_500, commercial_net_long=-140_000,
            total_open=460_000, long_change_wow=4_200, short_change_wow=-1_100,
            percentile_net=0.78, lag_days=3,
        )


class SyntheticETFProvider(ETFProvider):
    provider_id = "synthetic:etf"

    async def latest_flows(self) -> list[EtfFlowRecord]:
        now = utcnow()
        rows = [
            ("GLD", "NA", 2.31, 1005.0, 871.4),
            ("IAU", "NA", 0.42, 182.0, 152.9),
            ("GLDM", "NA", 0.11, 25.0, 63.1),
            ("SGBV", "EU", -0.18, -39.0, 45.2),
            ("518880", "ASIA", 0.63, 138.0, 108.7),
        ]
        return [
            EtfFlowRecord(source=self.provider_id, tier=SourceTier.TIER_2, frequency=Frequency.DAILY,
                          retrieved_at=now, event_at=now - timedelta(hours=20), published_at=now - timedelta(hours=20),
                          fund=f, region=r, flow_tonnes=t, flow_usd_millions=u, holdings_tonnes=h)
            for f, r, t, u, h in rows
        ]


class SyntheticFuturesProvider(FuturesProvider):
    provider_id = "synthetic:futures"

    def __init__(self, engine: SyntheticMarketEngine) -> None:
        self.engine = engine

    async def structure(self, underlying: str = "gold") -> list[FuturesStructureRecord]:
        now = utcnow()
        spot = self.engine.price
        out = []
        for m, oi, vol in ((2, 285_000, 92_000), (4, 120_000, 18_000), (6, 41_000, 4_100)):
            basis = spot * 0.0009 * m  # contango proxy
            out.append(FuturesStructureRecord(
                source=self.provider_id, tier=SourceTier.TIER_1, frequency=Frequency.DAILY,
                retrieved_at=now, event_at=now, published_at=now,
                contract=f"GC{m % 12}Z26", expiry=now + timedelta(days=30 * m),
                settle=spot + basis, volume=vol, open_interest=oi,
                oi_change_1d=1_250 if m == 2 else -300, basis_vs_spot=basis,
            ))
        return out


class SyntheticOptionsProvider(OptionsProvider):
    provider_id = "synthetic:options"

    def __init__(self, engine: SyntheticMarketEngine) -> None:
        self.engine = engine

    async def snapshot(self, underlying: str = "gold") -> Optional[OptionsChainSnapshot]:
        now = utcnow()
        iv_front = 12.0 + self.engine.vol_state * 6.0
        return OptionsChainSnapshot(
            source=self.provider_id, tier=SourceTier.TIER_2, frequency=Frequency.HOUR,
            retrieved_at=now, event_at=now, published_at=now,
            implied_vol_front=iv_front,
            iv_term_structure={"1w": iv_front, "1m": iv_front * 0.95, "3m": iv_front * 0.9, "6m": iv_front * 0.88},
            put_call_ratio=0.92, skew_25d=1.8,
        )


class SyntheticGeoProvider(GeopoliticalRiskProvider):
    provider_id = "synthetic:geo"

    async def active_risks(self, as_of: datetime) -> list[GeopoliticalRiskRecord]:
        now = utcnow()
        return [
            GeopoliticalRiskRecord(source=self.provider_id, tier=SourceTier.TIER_2, frequency=Frequency.HOUR,
                                   retrieved_at=now, event_at=now, published_at=now,
                                   region="Eastern Europe", category="conflict", intensity=0.55,
                                   escalation_delta=0.12, safe_haven_relevance=0.7,
                                   summary="Escalation indicators elevated; energy-sanction risk premium."),
            GeopoliticalRiskRecord(source=self.provider_id, tier=SourceTier.TIER_3, frequency=Frequency.HOUR,
                                   retrieved_at=now, event_at=now, published_at=now,
                                   region="Middle East", category="instability", intensity=0.4,
                                   escalation_delta=-0.05, safe_haven_relevance=0.5, verified=False,
                                   summary="Unconfirmed reports of back-channel talks."),
        ]


class SyntheticCentralBankProvider(CentralBankProvider):
    provider_id = "synthetic:cb"

    async def gold_demand(self, as_of: datetime) -> list[CentralBankGoldRecord]:
        now = utcnow()
        pub = now - timedelta(days=12)
        rows = [("China", 5.2, 41.0, 4.3), ("Poland", 3.1, 22.0, 19.0),
                ("India", 1.4, 9.8, 8.9), ("Turkey", -0.6, 4.1, 27.0)]
        return [
            CentralBankGoldRecord(source=self.provider_id, tier=SourceTier.TIER_1, frequency=Frequency.MONTHLY,
                                  retrieved_at=now, event_at=pub, published_at=pub,
                                  country=c, monthly_purchase_tonnes=m, ytd_total_tonnes=y,
                                  reserve_share_pct=s, structural=True)
            for c, m, y, s in rows
        ]
