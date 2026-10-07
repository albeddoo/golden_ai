"""Core enumerations for GOLD AI.

All timestamps in the system are UTC-normalized. See ``gold_ai.core.timeutils``.
"""
from __future__ import annotations

from enum import Enum


class SourceTier(str, Enum):
    """Source-quality hierarchy (Section 4 of the specification).

    Tier 3 information must NEVER independently generate a strong final decision.
    """

    TIER_1 = "TIER_1"  # Official government / exchange / primary sources
    TIER_2 = "TIER_2"  # High-quality financial news / professional market sources
    TIER_3 = "TIER_3"  # Social media sentiment / lower-confidence alternative sources


class Frequency(str, Enum):
    TICK = "TICK"
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    EVENT = "EVENT"
    STATIC = "STATIC"


class UpdateLoop(str, Enum):
    """Live agent update strategy (Section 17)."""

    FAST = "FAST"          # price, trend, momentum, price action, volatility
    MEDIUM = "MEDIUM"      # cross-asset, USD, yields, session
    SLOW = "SLOW"          # COT, ETF flows, central-bank demand, seasonality
    EVENT_DRIVEN = "EVENT_DRIVEN"  # news, macro releases, FOMC, geopolitical


class AgentStatus(str, Enum):
    ONLINE = "ONLINE"
    UPDATING = "UPDATING"
    STALE = "STALE"
    WARNING = "WARNING"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class DecisionState(str, Enum):
    """Final decision state. WAIT is a valid result, not an error (Rule 12)."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"


class Regime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NEWS_DRIVEN = "NEWS_DRIVEN"
    MACRO_DRIVEN = "MACRO_DRIVEN"
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    LIQUIDITY_STRESS = "LIQUIDITY_STRESS"
    UNKNOWN = "UNKNOWN"


class SessionName(str, Enum):
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    OFF_HOURS = "OFF_HOURS"
    CLOSED = "CLOSED"


class EventRiskState(str, Enum):
    """Event Risk Engine states (Section 7)."""

    NORMAL = "NORMAL"
    EVENT_APPROACHING = "EVENT_APPROACHING"
    EVENT_IMMINENT = "EVENT_IMMINENT"
    EVENT_RELEASED = "EVENT_RELEASED"
    POST_EVENT_REPRICING = "POST_EVENT_REPRICING"
    SHOCK = "SHOCK"


class Horizon(str, Enum):
    """Prediction horizons. Horizons are never mixed (Section 5.AB)."""

    H5M = "5m"
    H15M = "15m"
    H30M = "30m"
    H1H = "1h"
    H4H = "4h"
    H1D = "1D"

    @property
    def seconds(self) -> int:
        return _HORIZON_SECONDS[self]

    @classmethod
    def ordered(cls) -> list["Horizon"]:
        return [cls.H5M, cls.H15M, cls.H30M, cls.H1H, cls.H4H, cls.H1D]


_HORIZON_SECONDS: dict[Horizon, int] = {
    Horizon.H5M: 5 * 60,
    Horizon.H15M: 15 * 60,
    Horizon.H30M: 30 * 60,
    Horizon.H1H: 60 * 60,
    Horizon.H4H: 4 * 60 * 60,
    Horizon.H1D: 24 * 60 * 60,
}


class ScenarioKind(str, Enum):
    BASE = "BASE"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL_RANGE = "NEUTRAL_RANGE"
    SHOCK = "SHOCK"


class NewsClassification(str, Enum):
    BREAKING = "BREAKING"
    FOLLOW_UP = "FOLLOW_UP"
    OLD_INFORMATION = "OLD_INFORMATION"
    MARKET_COMMENTARY = "MARKET_COMMENTARY"
    OPINION = "OPINION"
    CONFIRMED_EVENT = "CONFIRMED_EVENT"
    RUMOR_UNVERIFIED = "RUMOR_UNVERIFIED"


class EventType(str, Enum):
    FOMC = "FOMC"
    CPI = "CPI"
    PPI = "PPI"
    PCE = "PCE"
    NFP = "NFP"
    GDP = "GDP"
    RETAIL_SALES = "RETAIL_SALES"
    ISM = "ISM"
    JOLTS = "JOLTS"
    CLAIMS = "CLAIMS"
    SPEECH = "SPEECH"
    OTHER = "OTHER"


class EventReleaseState(str, Enum):
    SCHEDULED = "SCHEDULED"
    RELEASED = "RELEASED"
    SURPRISE = "SURPRISE"
    REVISED = "REVISED"


class VolatilityRegime(str, Enum):
    COMPRESSION = "COMPRESSION"
    NORMAL = "NORMAL"
    EXPANSION = "EXPANSION"
    SHOCK = "SHOCK"


class ConfidenceBucket(str, Enum):
    """Configurable confidence buckets (Section 10)."""

    UNCERTAIN = "UNCERTAIN"     # 0-55
    WEAK = "WEAK"               # 55-65
    MODERATE = "MODERATE"       # 65-75
    STRONG = "STRONG"           # 75-85
    VERY_STRONG = "VERY_STRONG" # 85+


def bucket_for(confidence_pct: float, thresholds: tuple[float, float, float, float] = (55.0, 65.0, 75.0, 85.0)) -> ConfidenceBucket:
    """Map a 0..100 confidence value to a configurable bucket."""
    t_uncertain, t_weak, t_moderate, t_strong = thresholds
    if confidence_pct < t_uncertain:
        return ConfidenceBucket.UNCERTAIN
    if confidence_pct < t_weak:
        return ConfidenceBucket.WEAK
    if confidence_pct < t_moderate:
        return ConfidenceBucket.MODERATE
    if confidence_pct < t_strong:
        return ConfidenceBucket.STRONG
    return ConfidenceBucket.VERY_STRONG


class ConflictLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class SignalGrade(str, Enum):
    A_PLUS = "A+"
    A_MINUS = "A-"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class FeedHealth(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    OFFLINE = "OFFLINE"
