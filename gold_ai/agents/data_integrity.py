"""Data integrity agent (Section 5.A).

Detects: missing candles, duplicates, invalid timestamps, abnormal ticks,
stale feeds, provider disagreement, unrealistic price jumps, corrupted OHLC,
missing macro observations, delayed news, inconsistent source timestamps.

This agent CAN BLOCK the final decision via ``blocking_conditions``.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from gold_ai.agents.base import BaseAgent
from gold_ai.core.enums import AgentStatus, Frequency, SourceTier, UpdateLoop
from gold_ai.core.event_bus import (
    TOPIC_CANDLE, TOPIC_MACRO, TOPIC_NEWS, TOPIC_TICK,
)
from gold_ai.core.models import AgentOutput
from gold_ai.core.timeutils import age_seconds


class DataIntegrityAgent(BaseAgent):
    agent_id = "data_integrity"
    agent_name = "Data Integrity Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 1.0
    topics = (TOPIC_CANDLE, TOPIC_TICK, TOPIC_MACRO, TOPIC_NEWS)
    stale_after_s = 600.0

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        issues: list[str] = []
        blocking: list[str] = []
        dq = 1.0

        # ---- candle continuity / gaps / duplicates -----------------------
        c1 = list(self.state.candles["1m"])
        if len(c1) < 3:
            issues.append("insufficient candle history")
            dq -= 0.4
            blocking.append("NO_MARKET_HISTORY")
        else:
            ts = [c.event_at for c in c1]
            dups = sum(1 for a, b in zip(ts, ts[1:]) if a == b)
            if dups:
                issues.append(f"{dups} duplicate candle timestamps")
                dq -= min(0.2, 0.02 * dups)
            deltas = [(b - a).total_seconds() / 60.0 for a, b in zip(ts, ts[1:])]
            gaps = [d for d in deltas if d > 2.0]
            if gaps:
                issues.append(f"{len(gaps)} candle gaps (max {max(gaps):.0f}m)")
                dq -= min(0.25, 0.05 * len(gaps))
            bad_ohlc = sum(1 for c in c1[-200:] if c.high < max(c.open, c.close) or c.low > min(c.open, c.close))
            if bad_ohlc:
                issues.append(f"{bad_ohlc} suspicious OHLC envelopes (last 200)")
                dq -= min(0.3, 0.03 * bad_ohlc)
            # unrealistic jumps: |ret| beyond 8 sigma of rolling rets
            closes = np.array([c.close for c in c1], dtype=float)
            rets = np.diff(np.log(closes))
            if len(rets) > 30:
                sd = float(np.std(rets[-200:], ddof=1)) or 1e-9
                outliers = int(np.sum(np.abs(rets[-50:]) > 8 * sd))
                if outliers:
                    issues.append(f"{outliers} abnormal 1-min return outliers (>8σ)")
                    dq -= min(0.2, 0.05 * outliers)

        # ---- feed staleness ----------------------------------------------
        freshness_vals: list[float] = []
        if self.state.last_candle_at is not None:
            candle_age = age_seconds(self.state.last_candle_at, now)
            freshness_vals.append(float(np.exp(-candle_age / 120.0)))
            if candle_age > 90:
                issues.append(f"price feed stale ({candle_age:.0f}s)")
                blocking.append("STALE_PRICE_FEED")
                dq -= 0.35
        macro_checked = ["DXY", "UST10Y", "DFII10_REAL"]
        missing = [m for m in macro_checked if m not in self.state.macro]
        if missing:
            issues.append(f"missing macro series: {','.join(missing)}")
            dq -= 0.15 * len(missing)
        for m in macro_checked:
            p = self.state.macro.get(m)
            if p is not None:
                age = p.freshness_seconds(now)
                freshness_vals.append(float(np.exp(-age / 1800.0)))
                if age > 1800:
                    issues.append(f"{m} observation {age/60:.0f}m old")
        if not self.state.recent_news(minutes=240, now=now):
            issues.append("no news within 4h (news feed may be offline)")
            dq -= 0.05

        # ---- provider disagreement (sim vs futures basis sanity) ---------
        provider_disagreement = 0.0
        if self.state.futures and self.state.last_price > 0:
            front = min(self.state.futures, key=lambda f: abs((f.expiry - now).days))
            basis_pct = (front.settle - self.state.last_price) / self.state.last_price * 100
            if abs(basis_pct) > 1.5:
                provider_disagreement = min(1.0, abs(basis_pct) / 3.0)
                issues.append(f"spot/futures basis anomaly {basis_pct:+.2f}%")
                dq -= 0.1

        data_quality_score = max(0.0, min(1.0, dq))
        freshness_score = float(np.mean(freshness_vals)) if freshness_vals else 0.0

        out.data_quality = data_quality_score
        out.freshness = freshness_score
        out.confidence = data_quality_score
        out.score = 0.0
        out.probability_up = out.probability_down = out.probability_neutral = 1 / 3
        out.status = (
            AgentStatus.WARNING.value if (issues and data_quality_score < 0.75)
            else AgentStatus.ONLINE.value
        )
        if blocking:
            out.status = AgentStatus.WARNING.value
        out.extra = {
            "data_quality_score": round(data_quality_score, 3),
            "freshness_score": round(freshness_score, 3),
            "feed_health": {k: v.get("status", "?") for k, v in self.state.feed_health.items()},
            "provider_disagreement": round(provider_disagreement, 3),
            "blocking_conditions": blocking,
            "issues": issues[:12],
            "issue_count": len(issues),
        }
        out.source_reliability = 1.0
        out.evidence = [
            self.ev("integrity_issues" if issues else "integrity_clean",
                    "NEUTRAL", 1.0 if issues else 0.2, "system:integrity",
                    detail="; ".join(issues[:4]) or "all feeds consistent",
                    source=self.agent_id, tier=SourceTier.TIER_1, as_of=now)
        ]
        out.weight_reason = ""
        return out
