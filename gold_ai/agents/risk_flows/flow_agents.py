"""Structural flow agents (Sections 5.S–W, Y): central-bank gold demand, ETF
flows, CFTC positioning, futures structure, options/implied vol, seasonality.

All of these are SLOW / periodic factors. The UI must mark them as lagged; the
weighting engine discounts them for short horizons via ``horizon_fit``.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from gold_ai.agents.base import BaseAgent
from gold_ai.core.enums import AgentStatus, SourceTier, UpdateLoop
from gold_ai.core.models import AgentOutput


class CentralBankDemandAgent(BaseAgent):
    agent_id = "central_bank"
    agent_name = "Central Bank Gold Demand Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 300.0
    topics = ("cb.gold_demand",)
    stale_after_s = 86400 * 75

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        rows = list(self.state.central_bank)
        if not rows:
            out.status = AgentStatus.STALE.value
            out.confidence = 0.05
            out.error = "no central bank purchase data"
            return out
        net_tonnes = sum(r.monthly_purchase_tonnes for r in rows)
        structural = all(r.structural for r in rows)
        # structural factor: small persistent lean, never a short-term catalyst
        score = float(np.clip(net_tonnes / 12.0, -1, 1)) * 0.35
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.5)
        out.confidence = 0.6 if structural else 0.3
        out.horizon = "1D"
        out.freshness = float(np.mean([r.freshness_score(now, half_life_s=86400 * 45) for r in rows]))
        out.source_reliability = float(np.mean([r.reliability_score() for r in rows]))
        d = "UP" if score > 0.02 else "NEUTRAL"
        out.evidence = [self.ev("official_sector_demand", d, abs(score), "flow:central_bank",
                                detail=f"net monthly purchases {net_tonnes:+.1f}t across {len(rows)} banks",
                                tier=SourceTier.TIER_1, as_of=max(r.published_at for r in rows))]
        out.extra = {"net_monthly_tonnes": net_tonnes, "countries": {r.country: r.monthly_purchase_tonnes for r in rows},
                     "factor_type": "SLOW STRUCTURAL — not a short-term catalyst",
                     "lag_note": "official-sector data published with multi-week delay"}
        return out


class EtfFlowAgent(BaseAgent):
    agent_id = "etf_flows"
    agent_name = "Gold ETF Flow Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 120.0
    topics = ("flows.etf",)
    stale_after_s = 86400 * 5

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        flows = list(self.state.etf_flows)
        if not flows:
            out.status = AgentStatus.STALE.value
            out.confidence = 0.05
            out.error = "no ETF flow data"
            return out
        total_t = sum(f.flow_tonnes for f in flows)
        by_region: dict[str, float] = {}
        for f in flows:
            by_region[f.region] = by_region.get(f.region, 0.0) + f.flow_tonnes
        accel = 0.0
        if len(flows) >= 6:
            recent = sum(f.flow_tonnes for f in flows[-3:])
            prior = sum(f.flow_tonnes for f in flows[-6:-3])
            accel = recent - prior
        score = float(np.clip(total_t / 3.0, -1, 1)) * 0.5 + float(np.clip(accel / 4.0, -1, 1)) * 0.2
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.4)
        out.confidence = 0.55
        out.horizon = "1D"
        out.freshness = float(np.mean([f.freshness_score(now, half_life_s=86400) for f in flows]))
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("etf_flow_direction", d, abs(score), "flow:etf",
                                detail=f"latest-day net {total_t:+.2f}t; regions {by_region}; accel {accel:+.2f}t",
                                tier=SourceTier.TIER_2)]
        out.extra = {"total_tonnes": total_t, "by_region": by_region, "acceleration_t": accel,
                     "factor_type": "MEDIUM/LONG horizon unless data supports shorter"}
        return out


class PositioningAgent(BaseAgent):
    """CFTC COT — inherently lagged. Marked periodic everywhere it appears."""

    agent_id = "positioning"
    agent_name = "CFTC Positioning Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 600.0
    topics = ("positioning.report",)
    stale_after_s = 86400 * 9

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        cot = self.state.positioning
        if cot is None:
            out.status = AgentStatus.STALE.value
            out.confidence = 0.05
            out.error = "no COT report loaded"
            return out
        age_days = (now - cot.report_asof).total_seconds() / 86400
        pct = cot.percentile_net
        notes = []
        score = 0.0
        # crowding extremes fade mean-reversion risk against the crowd
        if pct > 0.9:
            score -= (pct - 0.9) * 3.0
            notes.append(f"net-long at {pct:.0%} percentile — crowded, reversal risk")
        elif pct < 0.1:
            score += (0.1 - pct) * 3.0
            notes.append(f"shorts crowded at {pct:.0%} pct — squeeze fuel")
        else:
            chg = cot.long_change_wow - cot.short_change_wow
            score += float(np.clip(chg / 20000, -0.3, 0.3))
            notes.append(f"WoW net change {chg:+,.0f} contracts")
        score = float(np.clip(score, -1, 1)) * 0.6
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.35)
        out.confidence = 0.6 if age_days <= 5 else 0.35
        out.horizon = "1D"
        out.freshness = float(math.exp(-age_days / 7.0))
        out.source_reliability = cot.reliability_score()
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("cot_positioning", d, abs(score), "flow:positioning",
                                detail="; ".join(notes) + f" | report as-of {age_days:.1f}d ago (LAGGED)",
                                source=cot.source, tier=cot.tier, as_of=cot.report_asof)]
        out.extra = {"spec_net": cot.speculative_net_long, "comm_net": cot.commercial_net_long,
                     "percentile": pct, "lag_days": round(age_days, 1),
                     "warning": "COT IS NOT REAL-TIME — weekly, ~3-day lag",
                     "publication_note": cot.publication_note}
        return out


class FuturesStructureAgent(BaseAgent):
    agent_id = "futures_structure"
    agent_name = "Futures Market Structure Agent"
    loop = UpdateLoop.MEDIUM
    min_interval_s = 60.0
    topics = ("futures.structure",)
    stale_after_s = 86400 * 2

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        recs = list(self.state.futures)
        if not recs:
            out.status = AgentStatus.STALE.value
            out.confidence = 0.05
            out.error = "no futures data"
            return out
        front = min(recs, key=lambda r: (r.expiry - now).days)
        oi_chg_pct = front.oi_change_1d / max(front.open_interest, 1)
        price_move = 0.0
        closes = self.state.snapshot_prices("1h", 2)
        if len(closes) == 2:
            price_move = closes[1] / closes[0] - 1
        score = 0.0
        notes = []
        # classic interpretation: OI rising WITH price = new longs (bullish confirmation)
        if oi_chg_pct > 0.002 and price_move > 0:
            score += 0.3
            notes.append("OI expansion with price advance → fresh longs")
        elif oi_chg_pct > 0.002 and price_move < 0:
            score -= 0.3
            notes.append("OI expansion into falling price → new shorts")
        elif oi_chg_pct < -0.002:
            notes.append("OI contraction → position unwind, conviction low")
            score *= 0.5
        basis = front.basis_vs_spot or 0.0
        spot = self.state.last_price
        basis_pct = basis / spot * 100 if spot else 0.0
        notes.append(f"front basis {basis_pct:+.3f}% (contango-normal)")
        score = float(np.clip(score, -1, 1)) * 0.7
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.3)
        out.confidence = 0.5
        out.horizon = "4h"
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = [self.ev("futures_participation", d, abs(score), "flow:futures_oi",
                                detail="; ".join(notes), source=front.source, tier=front.tier)]
        out.extra = {"front_contract": front.contract, "open_interest": front.open_interest,
                     "oi_change_1d": front.oi_change_1d, "volume": front.volume,
                     "basis_vs_spot_pct": round(basis_pct, 4),
                     "note": "COMEX futures metrics are NOT spot XAU/USD; relationship tracked separately"}
        return out


class OptionsIvAgent(BaseAgent):
    agent_id = "options_iv"
    agent_name = "Options / Implied Volatility Agent"
    loop = UpdateLoop.MEDIUM
    min_interval_s = 120.0
    topics = ("options.snapshot",)
    stale_after_s = 86400 * 2

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        snap = self.state.options
        if snap is None or snap.implied_vol_front is None:
            out.status = AgentStatus.DISABLED.value
            out.confidence = 0.0
            out.error = "options data unavailable — used only when present"
            return out
        ts = sorted(snap.iv_term_structure.items())
        slope = (ts[0][1] - ts[-1][1]) if len(ts) >= 2 else 0.0
        pcr = snap.put_call_ratio or 1.0
        realized_ann = 0.0
        closes = np.array(self.state.snapshot_prices("1h", 200), dtype=float)
        if len(closes) > 30:
            r = np.diff(np.log(closes))
            realized_ann = float(np.std(r, ddof=1) * math.sqrt(24 * 250) * 100)
        vr = realized_ann / max(snap.implied_vol_front, 1e-9)
        score = 0.0
        notes = []
        if slope > 1.5:
            score -= 0.15
            notes.append(f"front IV {slope:.1f}pts above long-dated → near-dated hedging demand/stress")
        elif slope < -1.0:
            score += 0.05
            notes.append("normal contango vol term structure")
        if pcr > 1.2:
            score += 0.1
            notes.append(f"P/C {pcr:.2f} elevated hedging")
        elif pcr < 0.7:
            score -= 0.05
            notes.append(f"P/C {pcr:.2f} complacent")
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.5)
        out.confidence = 0.45
        out.horizon = "4h"
        out.evidence = [self.ev("options_iv_regime", "NEUTRAL", abs(slope) / 3.0, "risk:implied_vol",
                                detail="; ".join(notes) or "neutral options signals", tier=SourceTier.TIER_2)]
        out.extra = {"iv_front": snap.implied_vol_front, "term_slope": slope, "put_call_ratio": pcr,
                     "realized_ann_pct": round(realized_ann, 2), "rv_over_iv": round(vr, 2),
                     "role": "primarily a RISK/REGIME factor, weak directional input"}
        return out


class SeasonalityAgent(BaseAgent):
    agent_id = "seasonality"
    agent_name = "Seasonality Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 600.0
    topics = ()
    stale_after_s = 86400 * 30

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        # Historical stats from stored candles only (supporting factor ONLY).
        c1 = list(self.state.candles["1m"])[-8000:]
        month_ret: dict[int, list[float]] = {}
        dow_ret: dict[int, list[float]] = {}
        for c in c1:
            signed = c.range_pct * (1 if c.close >= c.open else -1)
            month_ret.setdefault(c.event_at.month, []).append(signed)
            dow_ret.setdefault(c.event_at.weekday(), []).append(signed)
        cur_m = float(np.mean(month_ret.get(now.month, [0.0])))
        cur_d = float(np.mean(dow_ret.get(now.weekday(), [0.0])))
        score = float(np.clip((cur_m * 0.6 + cur_d * 0.4) / 0.5, -1, 1)) * 0.25
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.6)
        out.confidence = 0.3   # deliberately low: supporting factor only
        out.horizon = "1D"
        out.evidence = [self.ev("seasonal_profile", "UP" if score > 0.02 else ("DOWN" if score < -0.02 else "NEUTRAL"),
                                abs(score), "stat:seasonality",
                                detail=f"month drift={cur_m:+.3f}%, dow drift={cur_d:+.3f}% (stored history only)")]
        out.extra = {"monthly_drift_pct": cur_m, "dow_drift_pct": cur_d,
                     "role": "SUPPORTING FACTOR ONLY", "sample_minutes": len(c1)}
        return out
