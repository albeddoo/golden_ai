"""News, geopolitical and cross-asset agents (Sections 5.Q, R, X).

NOTE on double-counting: these agents deliberately consume evidence families
that overlap with other agents (fx:dxy, rates:nominal, market:flow). The
evidence-correlation controller downstream penalizes the duplication — that is
by design so the dependency graph has something real to work with.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from gold_ai.agents.base import BaseAgent
from gold_ai.agents.news_nlp import NewsPipeline
from gold_ai.core.enums import AgentStatus, SourceTier, UpdateLoop
from gold_ai.core.event_bus import TOPIC_CANDLE, TOPIC_MACRO, TOPIC_NEWS, TOPIC_QUOTE
from gold_ai.core.models import AgentOutput, NewsArticle
from gold_ai.core.timeutils import age_seconds


# ---------------------------------------------------------------------------
class NewsIntelligenceAgent(BaseAgent):
    agent_id = "news_intel"
    agent_name = "News Intelligence Agent"
    loop = UpdateLoop.EVENT_DRIVEN
    min_interval_s = 5.0
    topics = (TOPIC_NEWS,)
    stale_after_s = 3600 * 4

    def __init__(self, state, bus) -> None:
        super().__init__(state, bus)
        self.pipeline = NewsPipeline()

    def on_article(self, art: NewsArticle, now: datetime) -> NewsArticle:
        processed = self.pipeline.process(art, now)
        self.state.add_news(processed)
        return processed

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        agg = self.pipeline.aggregate(now)
        score = float(np.clip(agg["score"], -1, 1))
        impact = agg["impact"]
        # Tier-3 / unverified cannot independently produce strong conviction:
        max_conf = 0.35 if all(a.tier == SourceTier.TIER_3 for a in self.pipeline.seen[-5:] or [None][:0] or []) else 0.85
        out.score = score
        out.signal_strength = abs(score) * impact
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.3)
        out.confidence = min(max_conf, 0.25 + 0.7 * impact)
        out.horizon = "1h"
        items = agg["top"]
        out.freshness = float(math.exp(-age_seconds(self.pipeline.seen[-1].effective_at, now) / 3600)) if self.pipeline.seen else 0.0
        src_tiers = {a.tier.value for a in self.pipeline.seen[-10:]}
        out.source_reliability = {"TIER_1": 0.97, "TIER_2": 0.85}.get(min(src_tiers, default="TIER_3"), 0.55)
        d = "UP" if score > 0.08 else ("DOWN" if score < -0.08 else "NEUTRAL")
        top_head = items[0]["headline"] if items else "no relevant stories in window"
        out.evidence = [self.ev("news_flow_sentiment", d, abs(score) * impact, "info:news_sentiment",
                                detail=f"{len(items)} stories; top: {top_head}", tier=SourceTier.TIER_2)]
        if agg["unverified_count"]:
            out.contradictions = [self.ev("unverified_items_present", "NEUTRAL",
                                          min(1.0, agg["unverified_count"] / 3), "info:verification",
                                          detail=f"{agg['unverified_count']} unverified/rumor items discounted")]
        out.extra = {"aggregate": agg, "breaking": agg["breaking"],
                     "method": "dedup + classification + relevance-weighted sentiment"}
        if agg["breaking"]:
            self.bus.publish("system.anomaly", {"kind": "NEWS_SHOCK", "detail": top_head})
        return out


# ---------------------------------------------------------------------------
class GeopoliticalAgent(BaseAgent):
    """Safe-haven demand estimation — NOT 'all geopolitics is bullish gold'."""

    agent_id = "geopolitical"
    agent_name = "Geopolitical / Safe-Haven Agent"
    loop = UpdateLoop.EVENT_DRIVEN
    min_interval_s = 15.0
    topics = ("geo.risk", TOPIC_NEWS)
    stale_after_s = 3600 * 12

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        risks = list(self.state.geo_risks)
        fresh = [r for r in risks if r.freshness_seconds(now) < 3600 * 24]
        if not fresh:
            out.status = AgentStatus.STALE.value
            out.confidence = 0.1
            out.error = "no geopolitical risk data in 24h window"
            return out
        score = 0.0
        notes = []
        for r in fresh:
            if not r.verified:
                w = 0.15   # unverified geo claims barely count
            else:
                w = 1.0
            # escalation raises haven demand; de-escalation reduces premium
            eff = r.escalation_delta * r.safe_haven_relevance
            contrib = eff * w * (0.5 + r.intensity)
            score += contrib
            notes.append(f"{r.region}[{r.category}] esc={r.escalation_delta:+.2f} rel={r.safe_haven_relevance:.2f}"
                         + (" UNVERIFIED" if not r.verified else ""))
        # regime mattering filter: in MACRO_DRIVEN regimes geo shocks fade faster
        if self.state.regime in ("MACRO_DRIVEN", "RISK_ON"):
            score *= 0.7
            notes.append("current regime dampens geopolitical premium")
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.35)
        verif_ratio = sum(1 for r in fresh if r.verified) / len(fresh)
        out.confidence = 0.45 + 0.35 * verif_ratio
        out.horizon = "4h"
        out.freshness = float(np.mean([r.freshness_score(now, half_life_s=43200) for r in fresh]))
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("geo_escalation_premium", d, abs(score), "risk:geopolitical",
                                detail="; ".join(notes[:4]), tier=SourceTier.TIER_2)]
        out.extra = {"risks": [{"region": r.region, "intensity": r.intensity, "esc": r.escalation_delta,
                                "verified": r.verified} for r in fresh],
                     "note": "regime-aware: geopolitical events only matter if current regime prices them"}
        return out


# ---------------------------------------------------------------------------
class CrossAssetAgent(BaseAgent):
    """Measured rolling correlations vs DXY/yields/equities/silver/oil/copper/VIX.

    Consumes overlapping evidence with USD & rates agents ON PURPOSE; the
    evidence-correlation engine will discount duplicated information.
    """

    agent_id = "cross_asset"
    agent_name = "Cross-Asset Agent"
    loop = UpdateLoop.MEDIUM
    min_interval_s = 10.0
    topics = (TOPIC_MACRO, TOPIC_QUOTE, TOPIC_CANDLE)
    stale_after_s = 1800

    PAIRS = ["DXY", "UST10Y", "DFII10_REAL"]

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        gold = np.array(self.state.snapshot_prices("5m", 240), dtype=float)
        if len(gold) < 60:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            return out
        rg = np.diff(np.log(gold))
        contribs: dict[str, float] = {}
        corrs: dict[str, float] = {}
        for sid in self.PAIRS:
            series = np.array([p.value for p in self.state.macro_series(sid)], dtype=float)
            tail = series[-len(gold):] if len(series) >= len(gold) else None
            if tail is None or len(tail) < 60:
                continue
            ro = np.diff(np.log(np.clip(tail, 1e-6, None))) if sid == "DXY" else np.diff(tail)
            if ro.std() < 1e-12:
                continue
            rho = float(np.corrcoef(rg, ro)[0, 1])
            corrs[sid] = rho
            # expected sign of relation vs observed strength
            move = float(ro[-1] / (ro.std() + 1e-9))
            exp_sign = {"DXY": -1.0, "UST10Y": -1.0, "DFII10_REAL": -1.0}[sid]
            # confirmation when observed correlation matches expected AND move agrees
            agree = 1.0 if np.sign(rho) == exp_sign or abs(rho) < 0.1 else -1.0
            contribs[sid] = float(np.clip(move * exp_sign * (0.5 + 0.5 * abs(rho)) * agree, -1, 1))

        # equity/vix context from quotes
        vix_q = self.state.quotes.get("VIX")
        spx_q = self.state.quotes.get("SPX")
        risk_note = ""
        if vix_q:
            if vix_q.last > 24:
                contribs["VIX"] = 0.25
                risk_note = f"VIX {vix_q.last:.1f} elevated -> haven bid"
            elif vix_q.last < 13:
                contribs["VIX"] = -0.1
                risk_note = f"VIX {vix_q.last:.1f} complacent -> mild drag"
        score = float(np.clip(np.mean(list(contribs.values())) if contribs else 0.0, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.25)
        out.confidence = 0.55 if contribs else 0.2
        out.horizon = "1h"
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        det = ", ".join(f"{k}:ρ={corrs.get(k, float('nan')):+.2f}" for k in corrs) + ("; " + risk_note if risk_note else "")
        out.evidence = [self.ev("cross_asset_confirmation", d, abs(score), "market:cross_asset",
                                detail=det or "insufficient overlapping history")]
        out.dependencies = [f"fx:dxy:{k}" for k in corrs if k == "DXY"] + \
                           [f"rates:nominal:{k}" for k in corrs if k == "UST10Y"]
        out.extra = {"measured_correlations": {k: round(v, 3) for k, v in corrs.items()},
                     "contributions": {k: round(v, 3) for k, v in contribs.items()},
                     "note": "historical correlations treated as time-varying, not permanent"}
        return out
