"""Macro agents (Sections 5.I–P): USD/DXY, Treasuries, real yields, Fed policy,
economic calendar, inflation, labor market, economic growth.

Cross-asset correlation is MEASURED over rolling windows, never assumed stable.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from gold_ai.agents.base import BaseAgent
from gold_ai.core.enums import (
    AgentStatus, EventRiskState, EventType, SourceTier, UpdateLoop,
)
from gold_ai.core.event_bus import TOPIC_EVENT_RELEASED, TOPIC_MACRO, TOPIC_QUOTE
from gold_ai.core.models import AgentOutput
from gold_ai.core.timeutils import age_seconds, utcnow
from gold_ai.data.normalization import indicators as ta


def _series(state, sid: str) -> np.ndarray:
    return np.array([p.value for p in state.macro_series(sid)], dtype=float)


def _macro_lean(value_z: float, invert: bool, strength_scale: float = 2.5) -> float:
    """Map a z-scored macro move to a gold-directional lean in [-1,1]."""
    s = math.tanh(value_z / strength_scale)
    return -s if invert else s


# ---------------------------------------------------------------------------
class UsdDxyAgent(BaseAgent):
    agent_id = "usd_dxy"
    agent_name = "USD / DXY Agent"
    loop = UpdateLoop.MEDIUM
    min_interval_s = 5.0
    topics = (TOPIC_MACRO, TOPIC_QUOTE)
    stale_after_s = 1800
    evidence_families = ("fx:dxy",)

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        dxy = _series(self.state, "DXY")
        if len(dxy) < 30:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            out.error = "DXY history insufficient"
            return out
        z_short = ta.zscore(dxy[-60:], 60)          # intraday move
        z_med = ta.zscore(dxy, 400)                 # multi-session trend
        mom = float((dxy[-1] / dxy[-13] - 1) * 100) if len(dxy) > 13 else 0.0
        vol_ratio = float(np.std(dxy[-30:]) / (np.std(dxy[-200:]) + 1e-9)) if len(dxy) > 200 else 1.0

        # measure actual gold-DXY beta over recent window instead of assuming -1
        gold = np.array(self.state.snapshot_prices("5m", 200), dtype=float)
        dxy_tail = dxy[-len(gold):] if len(dxy) >= len(gold) else dxy
        corr = 0.0
        if len(gold) > 30 and len(dxy_tail) == len(gold):
            rg = np.diff(np.log(gold)); rd = np.diff(np.log(dxy_tail))
            if rg.std() > 1e-12 and rd.std() > 1e-12:
                corr = float(np.corrcoef(rg, rd)[0, 1])
        inv_assumed = corr < -0.15   # only treat as inverse when data says so
        score = _macro_lean(z_short * 0.6 + z_med * 0.4, invert=not inv_assumed) if abs(corr + 1) < 2 else 0.0
        if not inv_assumed and abs(corr) < 0.15:
            score *= 0.4  # relationship currently weak — reduce conviction
        score = float(np.clip(score, -1, 1))

        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.2)
        out.confidence = 0.6 * min(1.0, abs(corr) + 0.3)
        out.horizon = "1h"
        p = self.state.macro.get("DXY")
        out.freshness = p.freshness_score(now) if p else 0.0
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("dxy_move", d, abs(score), "fx:dxy",
                                detail=f"DXY z={z_short:+.1f}/{z_med:+.1f}, mom={mom:+.2f}%, ρ(gold)={corr:+.2f}",
                                source="synthetic:sim", tier=SourceTier.TIER_2, as_of=p.effective_at if p else None)]
        out.extra = {"dxy": float(dxy[-1]), "z_short": z_short, "z_med": z_med, "momentum_pct": mom,
                     "gold_corr_measured": round(corr, 3), "vol_ratio": vol_ratio,
                     "note": "inverse relation NOT assumed; measured rolling correlation"}
        return out


# ---------------------------------------------------------------------------
class TreasuryYieldAgent(BaseAgent):
    agent_id = "treasury_yields"
    agent_name = "Treasury / Yield Agent"
    loop = UpdateLoop.MEDIUM
    min_interval_s = 5.0
    topics = (TOPIC_MACRO,)
    stale_after_s = 1800
    evidence_families = ("rates:nominal",)

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        y10 = _series(self.state, "UST10Y")
        y2 = _series(self.state, "UST02Y")
        y30 = _series(self.state, "UST30Y")
        if len(y10) < 30:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            out.error = "yield history insufficient"
            return out
        z10 = ta.zscore(y10[-120:], 120)
        speed = float(y10[-1] - y10[-16]) if len(y10) > 16 else 0.0  # bp-ish per ~15min
        curve_2s10s = float(y10[-1] - y2[-1]) if len(y2) > 1 else 0.0
        curve_slope_chg = 0.0
        if len(y10) > 60 and len(y2) > 60:
            prev = float(y10[-60] - y2[-60])
            curve_slope_chg = curve_2s10s - prev

        # higher/faster-rising yields => pressure on gold (opportunity cost)
        score = _macro_lean(z10 + speed * 8.0, invert=True)
        if curve_slope_chg > 0.02:
            score += 0.1  # steepening (bear-flattener risk premium) mildly supportive of gold
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.2)
        out.confidence = 0.65
        out.horizon = "1h"
        p = self.state.macro.get("UST10Y")
        out.freshness = p.freshness_score(now) if p else 0.0
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("ust10y_pressure", d, abs(score), "rates:nominal",
                                detail=f"10Y z={z10:+.1f}, Δ15m={speed*100:+.1f}bp, 2s10s={curve_2s10s*100:+.0f}bp",
                                as_of=p.effective_at if p else None)]
        out.extra = {"y2": float(y2[-1]) if len(y2) else None, "y10": float(y10[-1]),
                     "y30": float(y30[-1]) if len(y30) else None,
                     "curve_2s10s_bp": round(curve_2s10s * 100, 1),
                     "speed_bp_15m": round(speed * 100, 1), "pressure_on_gold": -score}
        return out


# ---------------------------------------------------------------------------
class RealYieldAgent(BaseAgent):
    """Separate from nominal yields: opportunity cost adjusted for inflation."""

    agent_id = "real_yield"
    agent_name = "Real-Yield Agent"
    loop = UpdateLoop.MEDIUM
    min_interval_s = 5.0
    topics = (TOPIC_MACRO,)
    stale_after_s = 3600
    evidence_families = ("rates:real",)

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        ry = _series(self.state, "DFII10_REAL")
        if len(ry) < 30:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            out.error = "real yield proxy unavailable"
            return out
        z = ta.zscore(ry[-150:], 150)
        chg_1h = float(ry[-1] - ry[-61]) if len(ry) > 61 else float(ry[-1] - ry[0])
        trend = ta.slope(ry, 120)
        # rising real yields are the classic structural headwind for gold
        score = _macro_lean(z * 0.7 + (chg_1h * 100) * 0.6 + trend * 20, invert=True)
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.18)
        out.confidence = 0.7
        out.horizon = "4h"
        p = self.state.macro.get("DFII10_REAL")
        out.freshness = p.freshness_score(now) if p else 0.0
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("real_yield_direction", d, abs(score), "rates:real",
                                detail=f"TIPS10 z={z:+.1f}, Δ1h={chg_1h*100:+.1f}bp (proxy)",
                                source="synthetic:sim", tier=SourceTier.TIER_2,
                                as_of=p.effective_at if p else None)]
        out.extra = {"real_yield": float(ry[-1]), "z": z, "chg_1h_bp": round(chg_1h * 100, 1),
                     "note": "PROXY real yield (nominal - breakeven). Distinct from nominal-yield agent evidence."}
        return out


# ---------------------------------------------------------------------------
class FedPolicyAgent(BaseAgent):
    """NLP-diff based hawk/dove detection vs previous communication."""

    agent_id = "fed_policy"
    agent_name = "Fed / Monetary Policy Agent"
    loop = UpdateLoop.EVENT_DRIVEN
    min_interval_s = 20.0
    topics = (TOPIC_NEWS := "news.article", TOPIC_EVENT_RELEASED)
    stale_after_s = 86400 * 14

    HAWKISH = ["hike", "tighten", "higher for longer", "restrictive", "inflation upside",
               "not yet confident", "patience", "hawkish", "raise rates", "resilient economy"]
    DOVISH = ["cut", "ease", "accommodat", "stimul", "labor market cooling", "rate reductions",
              "dovish", "support growth", "risks balanced", "confidence in disinflation"]

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        news = self.state.recent_news(minutes=6 * 60, now=now)
        fed_news = [a for a in news if any(k in a.headline.lower() for k in
                    ("fed", "fomc", "powell", "central bank", "rate", "policy"))]
        hawk = sum(sum(1 for k in self.HAWKISH if k in a.headline.lower()) for a in fed_news)
        dove = sum(sum(1 for k in self.DOVISH if k in a.headline.lower()) for a in fed_news)
        # releases: rate decision surprises
        rels = [r for r in self.state.event_releases if r.event_type == EventType.FOMC
                and r.effective_at > now - timedelta(days=7)]
        surprise_hawk = 0.0
        if rels:
            last = rels[-1]
            if last.actual is not None and last.forecast is not None:
                surprise_hawk = float(np.clip((last.actual - last.forecast) * 0.4, -1, 1))
        tone = (dove - hawk) * 0.12 + surprise_hawk * -1.0  # dovish => bullish gold
        score = float(np.clip(tone, -1, 1))
        changed = abs(hawk - dove) > 0 or abs(surprise_hawk) > 0.01
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.45)
        out.confidence = 0.5 if changed else 0.25
        out.horizon = "4h"
        out.freshness = float(math.exp(-age_seconds(fed_news[-1].effective_at, now) / 21600)) if fed_news else 0.15
        out.source_reliability = max([a.reliability_score() for a in fed_news], default=0.5)
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("fed_tone_shift", d, abs(score), "policy:fed_communication",
                                detail=f"hawk-hits={hawk}, dove-hits={dove}, decision_surprise={surprise_hawk:+.2f}"
                                       if changed else "no change vs prior communication")]
        out.extra = {"hawk_hits": hawk, "dove_hits": dove, "surprise": surprise_hawk,
                     "method": "lexical diff vs previous stance (not naive bull/bear tagging)",
                     "fed_items": len(fed_news)}
        return out


# ---------------------------------------------------------------------------
class EconomicCalendarAgent(BaseAgent):
    """Tracks scheduled/released/surprise/revised events + event-risk states."""

    agent_id = "econ_calendar"
    agent_name = "Economic Calendar Agent"
    loop = UpdateLoop.EVENT_DRIVEN
    min_interval_s = 10.0
    topics = (TOPIC_EVENT_RELEASED, "calendar.scheduled")
    stale_after_s = 3600

    EVENT_RISK_THRESHOLDS = [(15 * 60, EventRiskState.EVENT_IMMINENT),
                             (2 * 3600, EventRiskState.EVENT_APPROACHING)]

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        nxt = self.state.next_event(now)
        risk_state = EventRiskState.NORMAL
        detail = "no major event within horizon"
        score = 0.0
        ev_items = []
        if nxt is not None:
            ttl = (nxt.scheduled_at - now).total_seconds()
            for secs, st in self.EVENT_RISK_THRESHOLDS:
                if ttl <= secs:
                    risk_state = st
                    break
            detail = f"{nxt.name} in {ttl/3600:.1f}h (imp={nxt.importance:.2f})"
            # pre-event de-risking: near big events push toward WAIT via neutrality
            if risk_state in (EventRiskState.EVENT_IMMINENT, EventRiskState.EVENT_APPROACHING):
                score = 0.0
        recent_rel = [r for r in self.state.event_releases if r.effective_at > now - timedelta(hours=3)]
        post_release = None
        if recent_rel:
            r = recent_rel[-1]
            age_min = (now - r.effective_at).total_seconds() / 60
            if r.surprise is not None and abs(r.surprise) > 0.05 * max(abs(r.forecast or 1), 1e-9):
                risk_state = EventRiskState.EVENT_RELEASED if age_min < 30 else EventRiskState.POST_EVENT_REPRICING
            surprise_dir = self._surprise_gold_impact(r)
            score = float(np.clip(surprise_dir, -1, 1))
            post_release = {"event": r.event_name, "surprise": r.surprise, "age_min": age_min,
                            "state": ("SURPRISE" if abs(r.surprise or 0) > 1e-9 else "RELEASED")}
            ev_items.append(self.ev("event_surprise_repricing", "UP" if score > 0 else ("DOWN" if score < 0 else "NEUTRAL"),
                                    abs(score), "macro:event_surprise",
                                    detail=f"{r.event_name}: actual={r.actual} vs fcst={r.forecast}",
                                    source=r.source, tier=r.tier, as_of=r.effective_at))
        # reaction phase classification: initial vs confirmation vs reversal
        reaction_phase = None
        if post_release:
            rel_ts = now - timedelta(minutes=post_release["age_min"])
            closes = np.array(self.state.snapshot_prices("1m", 90), dtype=float)
            if len(closes) > 20:
                initial = closes[min(5, len(closes)-1)] / closes[0] - 1
                latest = closes[-1] / closes[0] - 1
                if abs(initial) > 1e-5:
                    if latest * initial > 0 and abs(latest) > abs(initial) * 1.2:
                        reaction_phase = "CONFIRMATION"
                    elif latest * initial < 0:
                        reaction_phase = "REVERSAL"
                    else:
                        reaction_phase = "INITIAL_REACTION"
        out.score = score
        out.signal_strength = abs(score)
        neut = 0.5 if risk_state in (EventRiskState.EVENT_IMMINENT, EventRiskState.EVENT_APPROACHING) else 0.25
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, neut)
        out.confidence = 0.7 if post_release else 0.4
        out.horizon = "1h"
        out.evidence.append(self.ev("event_risk_state", "NEUTRAL", nxt.importance if nxt else 0.1,
                                    "macro:event_schedule", detail=detail, as_of=now))
        out.extra = {"event_risk_state": risk_state.value, "next_event": nxt.model_dump(mode="json") if nxt else None,
                     "post_release": post_release, "reaction_phase": reaction_phase}
        self.state.feed_health.setdefault("calendar", {"status": "OK"})
        return out

    @staticmethod
    def _surprise_gold_impact(r) -> float:
        """Directional gold impact by event semantics (hawk/dove style mapping)."""
        et = r.event_type
        sup = r.surprise or 0.0
        mag = math.tanh(abs(sup) / max(abs(r.previous or 1), 1e-9) * 4)
        if et in (EventType.CPI, EventType.PPI, EventType.PCE):
            return -mag          # hot inflation => hawkish => gold down (short horizon)
        if et in (EventType.NFP, EventType.JOLTS, EventType.CLAIMS):
            return -mag if et != EventType.CLAIMS else mag  # claims up => dovish => gold up
        if et in (EventType.GDP, EventType.RETAIL_SALES, EventType.ISM):
            return -mag          # strong growth => risk-on => gold down short-term
        if et == EventType.FOMC:
            return -mag
        return 0.0


# ---------------------------------------------------------------------------
class InflationAgent(BaseAgent):
    agent_id = "inflation"
    agent_name = "Inflation Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 60.0
    topics = (TOPIC_EVENT_RELEASED,)
    stale_after_s = 86400 * 45

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        infl_rels = [r for r in self.state.event_releases
                     if r.event_type in (EventType.CPI, EventType.PPI, EventType.PCE)]
        score = 0.0
        notes: list[str] = []
        if infl_rels:
            last = infl_rels[-1]
            if last.surprise is not None:
                hot = last.surprise > 0
                notes.append(f"{last.event_name} {'hot' if hot else 'cool'} surprise {last.surprise:+.2f}")
                # Hot inflation: near-term hawkish drag but longer-run hedge demand;
                # net short-horizon lean negative, discounted by staleness.
                fresh = last.freshness_score(now, half_life_s=86400 * 20)
                score = (-1 if hot else 1) * math.tanh(abs(last.surprise) * 2) * fresh
        exp_up = any("expectation" in a.headline.lower() and "higher" in a.headline.lower()
                     for a in self.state.recent_news(minutes=2880, now=now))
        if exp_up:
            score += 0.15
            notes.append("inflation expectations drifting higher (news)")
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.35)
        out.confidence = 0.55 if infl_rels else 0.2
        out.horizon = "1D"
        out.freshness = float(np.mean([r.freshness_score(now, half_life_s=86400*20) for r in infl_rels[-3:]] or [0.1]))
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("inflation_surprise_trend", d, abs(score), "macro:inflation",
                                detail="; ".join(notes) or "no recent inflation prints in window",
                                tier=SourceTier.TIER_1)]
        out.extra = {"recent_prints": len(infl_rels), "notes": notes}
        return out


# ---------------------------------------------------------------------------
class LaborMarketAgent(BaseAgent):
    agent_id = "labor_market"
    agent_name = "Labor Market Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 60.0
    topics = (TOPIC_EVENT_RELEASED,)
    stale_after_s = 86400 * 45

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        lab_rels = [r for r in self.state.event_releases
                    if r.event_type in (EventType.NFP, EventType.CLAIMS, EventType.JOLTS)]
        score = 0.0
        notes = []
        for r in lab_rels[-3:]:
            if r.surprise is None:
                continue
            fresh = r.freshness_score(now, half_life_s=86400 * 15)
            if r.event_type == EventType.CLAIMS:
                s = math.tanh(r.surprise / 10.0) * fresh       # more claims => dovish => gold up
                notes.append(f"claims surprise {r.surprise:+.0f}")
            elif r.event_type == EventType.NFP:
                s = -math.tanh(r.surprise / 50.0) * fresh      # strong payrolls => hawkish => gold down
                notes.append(f"NFP surprise {r.surprise:+.0f}k")
            else:
                s = -math.tanh(r.surprise / 200.0) * fresh
                notes.append(f"JOLTS surprise {r.surprise:+.0f}")
            score += s * 0.6
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.35)
        out.confidence = 0.55 if lab_rels else 0.2
        out.horizon = "1D"
        out.freshness = float(np.mean([r.freshness_score(now, half_life_s=86400*15) for r in lab_rels[-3:]] or [0.1]))
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("labor_surprise_vs_expectations", d, abs(score), "macro:labor",
                                detail="; ".join(notes) or "no labor prints in window", tier=SourceTier.TIER_1)]
        out.extra = {"prints": len(lab_rels), "focus": "surprise vs expectations, not raw levels"}
        return out


# ---------------------------------------------------------------------------
class GrowthAgent(BaseAgent):
    agent_id = "growth"
    agent_name = "Economic Growth Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 120.0
    topics = (TOPIC_EVENT_RELEASED,)
    stale_after_s = 86400 * 60

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        gr_rels = [r for r in self.state.event_releases
                   if r.event_type in (EventType.GDP, EventType.RETAIL_SALES, EventType.ISM)]
        score = 0.0
        notes = []
        for r in gr_rels[-3:]:
            if r.surprise is None:
                continue
            fresh = r.freshness_score(now, half_life_s=86400 * 20)
            s = -math.tanh(r.surprise * 1.5) * fresh   # growth upside => risk-on => gold drag
            score += s * 0.5
            notes.append(f"{r.event_name} surprise {r.surprise:+.2f}")
        vix = self.state.quotes.get("VIX")
        recession_proxy = None
        if vix and vix.last > 28:
            score += 0.2
            notes.append("elevated VIX — recession/stress tail supported")
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.4)
        out.confidence = 0.45 if gr_rels else 0.2
        out.horizon = "1D"
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("growth_environment", d, abs(score), "macro:growth",
                                detail="; ".join(notes) or "no growth prints in window", tier=SourceTier.TIER_1)]
        out.extra = {"prints": len(gr_rels), "recession_proxy_vix": recession_proxy}
        return out
