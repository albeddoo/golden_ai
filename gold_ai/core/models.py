"""Normalized data schemas returned by every provider (Section 3).

Every observation carries full timestamp provenance:
- event_time      : when the underlying market/economic fact happened
- publication_time: when the source published it (macro releases, news)
- observation_time: when the source last revised/observed the value
- retrieval_time  : when we fetched it

Freshness and reliability are computed against ``as_of`` (the "as-of" clock used
by engines and the anti-leakage validator), never against wall-clock surprises.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gold_ai.core.enums import (
    EventType,
    Frequency,
    NewsClassification,
    SourceTier,
)
from gold_ai.core.timeutils import age_seconds


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


class TimestampedRecord(BaseModel):
    """Base for all normalized records with UTC-provenance timestamps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str = Field(description="Provider id, e.g. 'synthetic', 'fred'")
    tier: SourceTier = SourceTier.TIER_3
    symbol: Optional[str] = None
    unit: Optional[str] = None
    frequency: Frequency = Frequency.MINUTE

    retrieved_at: datetime
    event_at: Optional[datetime] = None       # observation/market time
    published_at: Optional[datetime] = None   # source publication time

    @model_validator(mode="after")
    def _normalize_utc(self) -> "TimestampedRecord":
        from gold_ai.core.timeutils import to_utc

        self.retrieved_at = to_utc(self.retrieved_at)
        if self.event_at is not None:
            self.event_at = to_utc(self.event_at)
        if self.published_at is not None:
            self.published_at = to_utc(self.published_at)
        return self

    # ---- derived freshness / latency -------------------------------------
    @property
    def effective_at(self) -> datetime:
        """Best available 'information time' for as-of filtering (anti-leakage)."""
        return self.published_at or self.event_at or self.retrieved_at

    def latency_seconds(self) -> float:
        """Publication lag: retrieval minus information time."""
        return age_seconds(self.effective_at, self.retrieved_at)

    def freshness_seconds(self, now: datetime) -> float:
        """Age of the information at evaluation time ``now``."""
        return age_seconds(self.effective_at, now)

    def freshness_score(self, now: datetime, half_life_s: float = 900.0) -> float:
        """Exponential-decay freshness in [0,1]. Half-life depends on frequency."""
        halflife = half_life_s or _default_half_life(self.frequency)
        age = self.freshness_seconds(now)
        return _clamp01(math.pow(0.5, age / halflife))

    def reliability_score(self) -> float:
        """Static source-reliability prior from the tier hierarchy."""
        return {SourceTier.TIER_1: 0.97, SourceTier.TIER_2: 0.85, SourceTier.TIER_3: 0.55}[self.tier]


def _default_half_life(freq: Frequency) -> float:
    return {
        Frequency.TICK: 5.0,
        Frequency.SECOND: 15.0,
        Frequency.MINUTE: 300.0,
        Frequency.HOUR: 3600.0,
        Frequency.DAILY: 43200.0,
        Frequency.WEEKLY: 259200.0,
        Frequency.MONTHLY: 971_000.0,
        Frequency.QUARTERLY: 2_592_000.0,
        Frequency.EVENT: 7200.0,
        Frequency.STATIC: 31_536_000.0,
    }.get(freq, 900.0)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
class Candle(TimestampedRecord):
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timeframe: str = "1m"

    @model_validator(mode="after")
    def _validate_ohlc(self) -> "Candle":
        if self.high < self.low:
            raise ValueError(f"corrupted OHLC: high<low for {self.symbol} @ {self.event_at}")
        if any(v <= 0 for v in (self.open, self.high, self.low, self.close)):
            raise ValueError("non-positive price in candle")
        return self

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def range_pct(self) -> float:
        return abs(self.high - self.low) / self.close * 100.0


class Tick(TimestampedRecord):
    bid: float
    ask: float
    last: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class Quote(TimestampedRecord):
    """Instrument snapshot used by the market watchlist."""

    symbol: str
    last: float
    change_abs: float = 0.0
    change_pct: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_bps: Optional[float] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MacroSeriesPoint(TimestampedRecord):
    """A single macro/rates/FX observation (DXY, yields, real yield, ETF flows...)."""

    series_id: str
    value: float
    revision: Optional[float] = None  # delta vs previously published value

    def key(self) -> str:
        return f"{self.source}:{self.series_id}"


class EconomicEventRelease(TimestampedRecord):
    """Actual release for a scheduled economic event."""

    event_type: EventType
    event_name: str
    period: Optional[str] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None
    revised_from: Optional[float] = None
    surprise: Optional[float] = None  # actual - forecast
    surprise_pct: Optional[float] = None  # z-score style standardized surprise
    importance: float = 0.5           # 0..1
    unit: Optional[str] = None

    @model_validator(mode="after")
    def _compute_surprise(self) -> "EconomicEventRelease":
        if self.actual is not None and self.forecast is not None and self.surprise is None:
            self.surprise = self.actual - self.forecast
        return self


class ScheduledEvent(BaseModel):
    """Calendar entry (before release)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_type: EventType
    name: str
    scheduled_at: datetime
    importance: float = 0.5
    forecast: Optional[float] = None
    previous: Optional[float] = None
    source: str = "calendar"
    tier: SourceTier = SourceTier.TIER_1
    released: bool = False

    @model_validator(mode="after")
    def _normalize(self) -> "ScheduledEvent":
        from gold_ai.core.timeutils import to_utc

        self.scheduled_at = to_utc(self.scheduled_at)
        return self


class HistoricalGoldResponse(BaseModel):
    """How gold historically reacted to this event type's surprises."""

    event_type: EventType
    avg_move_bps_on_surprise: float
    hit_rate_same_direction: float
    sample_size: int
    horizon_minutes: int = 60


class NewsArticle(TimestampedRecord):
    headline: str
    body: Optional[str] = None
    url: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)

    # NLP pipeline outputs (Section 5.Q) -- filled by the news intelligence agent
    dedup_key: Optional[str] = None
    novelty: float = 0.5                 # 0 = rehash, 1 = brand-new information
    relevance_gold: float = 0.0          # 0..1
    relevance_usd: float = 0.0
    relevance_rates: float = 0.0
    sentiment: float = 0.0               # -1..+1 toward GOLD
    expected_macro_impact: float = 0.0   # 0..1 magnitude of plausible impact
    urgency: float = 0.0                 # 0..1
    potential_market_impact: float = 0.0 # 0..1 composite
    classification: NewsClassification = NewsClassification.MARKET_COMMENTARY
    verified: bool = True
    nlp_reliability: float = 0.5         # combined credibility*novelty*verification

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.body or ''}".strip()


class PositioningReport(TimestampedRecord):
    """CFTC COT-style report. Inherently LAGGED; UI must mark it periodic."""

    report_asof: datetime
    speculative_net_long: float
    commercial_net_long: float
    total_open: float = 0.0
    long_change_wow: float = 0.0
    short_change_wow: float = 0.0
    percentile_net: float = 0.5  # historical percentile of net positioning
    lag_days: int = 3
    publication_note: str = "CFTC COT (weekly, Fridays)"


class EtfFlowRecord(TimestampedRecord):
    fund: str
    region: str
    flow_tonnes: float
    flow_usd_millions: float
    holdings_tonnes: float
    tld_change_tonnes: Optional[float] = None  # trailing-day acceleration proxy


class FuturesStructureRecord(TimestampedRecord):
    contract: str
    expiry: datetime
    settle: float
    volume: float
    open_interest: float
    oi_change_1d: float = 0.0
    basis_vs_spot: Optional[float] = None  # futures - spot; NOT the same as spot XAU/USD


class OptionsChainSnapshot(TimestampedRecord):
    implied_vol_front: Optional[float] = None
    iv_term_structure: dict[str, float] = Field(default_factory=dict)  # tenor -> vol
    put_call_ratio: Optional[float] = None
    skew_25d: Optional[float] = None
    unusual_activity_note: Optional[str] = None


class GeopoliticalRiskRecord(TimestampedRecord):
    region: str
    category: str  # conflict / sanctions / instability / systemic
    intensity: float  # 0..1
    escalation_delta: float = 0.0  # change vs previous state (-1..1)
    safe_haven_relevance: float = 0.0  # estimated probability THIS matters to current regime
    verified: bool = True
    summary: str = ""


class CentralBankGoldRecord(TimestampedRecord):
    country: str
    monthly_purchase_tonnes: float
    ytd_total_tonnes: float
    reserve_share_pct: Optional[float] = None
    structural: bool = True  # slow-moving structural factor, not a short-term catalyst


# ---------------------------------------------------------------------------
# Agent outputs & engine payloads
# ---------------------------------------------------------------------------
class EvidenceItem(BaseModel):
    """One traceable piece of evidence backing an agent view."""

    label: str
    detail: str = ""
    direction: str = "NEUTRAL"  # UP/DOWN/NEUTRAL
    strength: float = 0.0       # 0..1
    evidence_id: str = ""       # canonical id for double-counting control
    source: str = ""
    tier: SourceTier = SourceTier.TIER_3
    as_of: Optional[datetime] = None


class AgentOutput(BaseModel):
    """Standardized output produced by every specialized agent."""

    agent_id: str
    agent_name: str
    loop: str = "FAST"
    status: str = "ONLINE"

    probability_up: float = 1 / 3
    probability_down: float = 1 / 3
    probability_neutral: float = 1 / 3

    confidence: float = 0.0        # 0..1 how much we trust this output
    score: float = 0.0             # -1..+1 directional lean
    signal_strength: float = 0.0   # 0..1 magnitude of lean regardless of sign

    data_quality: float = 1.0
    freshness: float = 1.0         # 0..1 decayed freshness of inputs
    source_reliability: float = 1.0
    historical_predictive_value: float = 0.5  # out-of-sample hit rate proxy
    regime_fit: float = 1.0        # how well this agent works in current regime
    horizon_fit: float = 1.0

    weight: float = 0.0            # assigned by dynamic weighting engine
    weight_reason: str = ""        # why the weight changed (explanations)

    evidence: list[EvidenceItem] = Field(default_factory=list)
    contradictions: list[EvidenceItem] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # evidence ids consumed

    horizon: str = "1h"
    updated_at: Optional[datetime] = None
    error: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_probs(self) -> "AgentOutput":
        s = self.probability_up + self.probability_down + self.probability_neutral
        if s <= 0 or math.isnan(s):
            self.probability_up = self.probability_down = self.probability_neutral = 1 / 3
        else:
            self.probability_up /= s
            self.probability_down /= s
            self.probability_neutral /= s
        return self

    @property
    def direction_label(self) -> str:
        p = {
            "UP": self.probability_up,
            "DOWN": self.probability_down,
            "NEUTRAL": self.probability_neutral,
        }
        return max(p, key=p.get)


class DecisionPayload(BaseModel):
    """Final decision emitted by the central Decision Engine (Section 12)."""

    decision_id: str
    created_at: datetime
    symbol: str = "XAU/USD"
    price: float

    direction: str = "NEUTRAL"
    probability_up: float = 1 / 3
    probability_down: float = 1 / 3
    probability_neutral: float = 1 / 3

    confidence_score: float = 0.0     # 0..100
    signal_quality: str = "C"
    regime: str = "UNKNOWN"
    regimes_active: list[str] = Field(default_factory=list)

    agent_consensus: dict[str, int] = Field(default_factory=dict)  # bullish/bearish/neutral counts
    evidence_strength: float = 0.0
    conflict_level: str = "LOW"
    conflict_notes: list[str] = Field(default_factory=list)
    data_quality: float = 1.0
    event_risk: str = "NORMAL"
    event_risk_detail: str = ""
    news_risk: str = "LOW"
    volatility_state: str = "NORMAL"

    scenario_distribution: dict[str, float] = Field(default_factory=dict)
    decision_state: str = "WAIT"
    wait_reasons: list[str] = Field(default_factory=list)

    top_bullish: list[str] = Field(default_factory=list)
    top_bearish: list[str] = Field(default_factory=list)
    top_risks: list[str] = Field(default_factory=list)

    explanation: str = ""
    paper_scenario: Optional[str] = None
    horizon: str = "1h"
    immutable: bool = True


class Scenario(BaseModel):
    kind: str
    probability: float
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    conditions_required: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    target_range: Optional[tuple[float, float]] = None
    narrative: str = ""


class PredictionRecord(BaseModel):
    """Immutable auditable prediction (Sections 18-19)."""

    prediction_id: str
    timestamp: datetime
    horizon: str
    model_version: str
    regime: str
    probability_up: float
    probability_down: float
    probability_neutral: float
    decision: str
    confidence: float
    price_at_prediction: float
    available_information_snapshot: dict[str, Any] = Field(default_factory=dict)
    realized_outcome: Optional[str] = None   # CORRECT / INCORRECT / NEUTRAL_HIT / OPEN
    realized_return_bps: Optional[float] = None
    resolved_at: Optional[datetime] = None
