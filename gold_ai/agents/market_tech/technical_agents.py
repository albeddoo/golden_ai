"""Technical market agents (Sections 5.B–H): spot, trend, momentum, price
action, support/resistance/liquidity, volatility, session.

Each produces an AgentOutput with traceable evidence and canonical evidence
IDs so the double-counting controller can penalize duplicated information.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from gold_ai.agents.base import BaseAgent
from gold_ai.core.enums import AgentStatus, SessionName, SourceTier, UpdateLoop
from gold_ai.core.event_bus import TOPIC_CANDLE, TOPIC_TICK
from gold_ai.core.models import AgentOutput
from gold_ai.core.timeutils import age_seconds, fmt_age
from gold_ai.data.normalization import indicators as ta


def _closes(state, tf: str = "1m") -> np.ndarray:
    return np.array([c.close for c in state.candles[tf]], dtype=float)


def _ohlc(state, tf: str = "1m") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cs = state.candles[tf]
    return (np.array([c.open for c in cs]), np.array([c.high for c in cs]),
            np.array([c.low for c in cs]), np.array([c.close for c in cs]))


# ---------------------------------------------------------------------------
class SpotMarketAgent(BaseAgent):
    agent_id = "spot_market"
    agent_name = "Spot Market Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 1.0
    topics = (TOPIC_CANDLE, TOPIC_TICK)

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        closes = _closes(self.state)
        if len(closes) < 10:
            out.status = AgentStatus.WARNING.value
            out.data_quality = 0.3
            out.confidence = 0.0
            out.error = "insufficient candles"
            return out
        last = float(closes[-1])
        chg_5m = (last / float(closes[-6]) - 1) * 100 if len(closes) >= 6 else 0.0
        chg_30m = (last / float(closes[-31]) - 1) * 100 if len(closes) >= 31 else 0.0
        # acceleration: sign agreement between short and medium move
        accel = 1.0 if (chg_5m * chg_30m > 0 and abs(chg_5m) > 0.02) else 0.0
        spread_bps = None
        tick_rate = None
        if self.state.last_tick is not None:
            spread_bps = self.state.last_tick.spread / max(last, 1e-9) * 1e4
        recent = [t for t in [] ]  # tick rate tracked by engine service
        score = max(-1.0, min(1.0, (math.tanh(chg_30m / 0.25) * 0.7 + math.tanh(chg_5m / 0.12) * 0.3)))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.25)
        out.confidence = 0.55 if spread_bps is None or spread_bps < 3 else 0.35
        out.freshness = float(math.exp(-age_seconds(self.state.last_candle_at, now) / 120)) if self.state.last_candle_at else 0.0
        out.evidence = [
            self.ev("spot_immediate_flow", "UP" if score > 0.08 else ("DOWN" if score < -0.08 else "NEUTRAL"),
                    abs(score), "market:flow", detail=f"Δ30m {chg_30m:+.2f}%, Δ5m {chg_5m:+.2f}%",
                    source="synthetic:sim", tier=SourceTier.TIER_2, as_of=self.state.last_candle_at),
        ]
        out.extra = {"price": last, "chg_5m_pct": chg_5m, "chg_30m_pct": chg_30m,
                     "acceleration": accel, "spread_bps": spread_bps}
        return out


# ---------------------------------------------------------------------------
class TrendAgent(BaseAgent):
    agent_id = "trend"
    agent_name = "Trend Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 2.0
    topics = (TOPIC_CANDLE,)
    stale_after_s = 600

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        votes: list[tuple[str, float]] = []
        ev = []
        for tf, need in (("5m", 200), ("15m", 200), ("1h", 200), ("4h", 200)):
            closes = _closes(self.state, tf)
            if len(closes) < need // 2:
                continue
            e20, e50 = ta.ema(closes, 20), ta.ema(closes, 50)
            e100, e200 = ta.ema(closes, 100), ta.ema(closes, 200)
            last = closes[-1]
            stack = sum([last > e20[-1], e20[-1] > e50[-1], e50[-1] > e100[-1], e100[-1] > e200[-1]])
            slope = ta.slope(closes, 60)
            v = ((stack / 4.0) * 0.7 + math.tanh(slope / 0.05) * 0.3)
            votes.append((tf, float(np.clip(v, -1, 1))))
        if not votes:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            out.error = "no timeframe data"
            return out
        aligned = all(v > 0.15 for _, v in votes) or all(v < -0.15 for _, v in votes)
        score = float(np.mean([v for _, v in votes]))
        strength = min(1.0, abs(score) * (1.25 if aligned else 0.8))
        direction = "UP" if score > 0 else ("DOWN" if score < 0 else "NEUTRAL")
        supported = [tf for tf, v in votes if (v > 0.1 if score > 0 else v < -0.1)]
        out.score = score
        out.signal_strength = strength
        p_up, p_dn, p_ne = self.probs_from_score(score, neutrality=0.12 if aligned else 0.3)
        out.probability_up, out.probability_down, out.probability_neutral = p_up, p_dn, p_ne
        out.confidence = 0.5 + 0.3 * (len(supported) / max(1, len(votes))) + (0.15 if aligned else 0.0)
        out.horizon = "1h"
        out.evidence = [self.ev("trend_alignment_" + tf, direction, abs(v), "market:trend_structure",
                                detail=f"{tf} EMA stack/slope", source="synthetic:sim",
                                as_of=self.state.last_candle_at)
                        for tf, v in votes]
        hh_hl = any(tf == "1h" and v > 0.3 for tf, v in votes)
        out.extra = {"votes": {tf: round(v, 3) for tf, v in votes}, "aligned": aligned,
                     "timeframes_supported": supported, "trend_strength": round(strength, 3)}
        return out


# ---------------------------------------------------------------------------
class MomentumAgent(BaseAgent):
    agent_id = "momentum"
    agent_name = "Momentum Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 2.0
    topics = (TOPIC_CANDLE,)

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        closes = _closes(self.state, "5m")
        if len(closes) < 80:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            return out
        r = ta.rsi(closes, 14)[-1]
        macd_line, sig, hist = ta.macd(closes)
        roc = ta.roc(closes, 12)[-1]
        hist_slope = float(hist[-1] - hist[-6]) if len(hist) > 6 else 0.0
        div = ta.momentum_divergence(closes, r, window=80)

        # Context-aware scoring: strong RSI in a trend is continuation, not sell.
        trend_up = closes[-1] > ta.ema(closes, 50)[-1]
        score = 0.0
        notes: list[str] = []
        score += math.tanh((r - 50) / 18) * 0.45
        score += math.tanh(hist_slope / (np.std(hist[-60:]) + 1e-9) * 0.6) * 0.3
        score += math.tanh(roc / 0.35) * 0.25
        if div == "BEARISH_DIV" and r > 62:
            score -= 0.35
            notes.append("bearish divergence at extension")
        elif div == "BULLISH_DIV" and r < 38:
            score += 0.35
            notes.append("bullish divergence at washout")
        # overextension guard: extreme RSI without trend backing decays conviction
        if abs(r - 50) > 32 and not trend_up and score > 0:
            score *= 0.5
            notes.append("overextended vs trend")
        exhaustion = abs(r - 50) > 30 and abs(hist_slope) < 1e-4
        if exhaustion:
            score *= 0.7
            notes.append("momentum exhaustion signs")

        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.2)
        out.confidence = 0.55 if not notes else 0.45
        out.horizon = "30m"
        d = "UP" if score > 0.1 else ("DOWN" if score < -0.1 else "NEUTRAL")
        out.evidence = [self.ev("rsi_macd_roc_context", d, abs(score), "market:momentum",
                                detail=f"RSI={r:.0f}, ROC={roc:+.2f}% | " + "; ".join(notes),
                                as_of=self.state.last_candle_at)]
        out.extra = {"rsi": float(r), "macd_hist_slope": hist_slope, "roc": float(roc),
                     "divergence": div, "exhaustion": exhaustion, "notes": notes}
        return out


# ---------------------------------------------------------------------------
class PriceActionAgent(BaseAgent):
    agent_id = "price_action"
    agent_name = "Price Action Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 3.0
    topics = (TOPIC_CANDLE,)

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        o, h, l, c = _ohlc(self.state, "15m")
        if len(c) < 60:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            return out
        hh, ll = ta.find_swings(h, l, lookback=3)
        score = 0.0
        pattern = "NONE"
        res = ta.market_structure(h, l)
        # breakout attempt: close beyond prior swing high
        if hh and c[-1] > h[hh[-1]]:
            body_ratio = abs(c[-1] - o[-1]) / max(h[-1] - l[-1], 1e-9)
            if body_ratio > 0.6:
                score += 0.5
                pattern = "BREAKOUT_CONFIRM"
            else:
                score += 0.15
                pattern = "BREAKOUT_WEAK_BODY"
        # false breakout: pierced swing high then closed back below
        if hh and h[-1] > h[hh[-1]] and c[-1] < h[hh[-1]]:
            score -= 0.45
            pattern = "FALSE_BREAKOUT"
        # rejection wick
        rng = max(h[-1] - l[-1], 1e-9)
        lower_wick = (min(o[-1], c[-1]) - l[-1]) / rng
        upper_wick = (h[-1] - max(o[-1], c[-1])) / rng
        if lower_wick > 0.6 and abs(c[-1] - c[-5]) / c[-1] < 0.002:
            score += 0.3
            pattern = "REJECTION_BOUNCE"
        if upper_wick > 0.6 and abs(c[-1] - c[-5]) / c[-1] < 0.002:
            score -= 0.3
            pattern = "REJECTION_FADE"
        # consolidation detection
        bb_mid, bb_up, bb_low = ta.bollinger(c, 20, 2.0)
        width = (bb_up[-1] - bb_low[-1]) / bb_mid[-1]
        consolidating = width < np.nanmedian((bb_up[-40:] - bb_low[-40:]) / bb_mid[-40:]) * 0.75
        if consolidating:
            score *= 0.4
            pattern = "CONSOLIDATION" if pattern == "NONE" else pattern
        if res["pattern"] == "HH_HL_UPTREND":
            score += 0.2
        elif res["pattern"] == "LH_LL_DOWNTREND":
            score -= 0.2
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        neut = 0.35 if consolidating else 0.2
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, neut)
        out.confidence = 0.5 if pattern != "NONE" else 0.3
        out.horizon = "1h"
        d = "UP" if score > 0.1 else ("DOWN" if score < -0.1 else "NEUTRAL")
        out.evidence = [self.ev(f"pa_{pattern.lower()}", d, abs(score), "market:price_action",
                                detail=f"struct={res['pattern']}, bb_width={width*100:.2f}%",
                                as_of=self.state.last_candle_at)]
        out.extra = {"pattern": pattern, "structure": res["pattern"], "consolidation": consolidating}
        return out


# ---------------------------------------------------------------------------
class LiquidityZonesAgent(BaseAgent):
    """Support/resistance zones with confidence — avoids meaningless lines."""

    agent_id = "liquidity_zones"
    agent_name = "S/R & Liquidity Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 5.0
    topics = (TOPIC_CANDLE,)

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        o, h, l, c = _ohlc(self.state, "1h")
        if len(c) < 80:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            return out
        price = float(self.state.last_price)
        # cluster swing levels within 0.35% bands; require >=2 touches to matter
        hh, ll = ta.find_swings(h, l, lookback=2)
        levels = sorted([float(h[i]) for i in hh] + [float(l[i]) for i in ll])
        zones: list[dict] = []
        used = set()
        for i, lv in enumerate(levels):
            if i in used:
                continue
            group = [lv]
            used.add(i)
            for j in range(i + 1, len(levels)):
                if j in used:
                    continue
                if abs(levels[j] - lv) / lv < 0.0035:
                    group.append(levels[j]); used.add(j)
            if len(group) >= 2:
                zc = float(np.mean(group))
                zones.append({"level": zc, "touches": len(group),
                              "kind": "RESISTANCE" if zc > price else "SUPPORT",
                              "confidence": min(1.0, 0.4 + 0.15 * len(group))})
        near_res = [z for z in zones if z["kind"] == "RESISTANCE" and 0 < (z["level"] - price) / price < 0.006]
        near_sup = [z for z in zones if z["kind"] == "SUPPORT" and 0 < (price - z["level"]) / price < 0.006]
        score = 0.0
        notes = []
        if near_res:
            strongest = max(near_res, key=lambda z: z["confidence"])
            score -= 0.35 * strongest["confidence"]
            notes.append(f"major resistance {strongest['level']:.0f} ({strongest['touches']} touches)")
        if near_sup:
            strongest = max(near_sup, key=lambda z: z["confidence"])
            score += 0.3 * strongest["confidence"]
            notes.append(f"support zone {strongest['level']:.0f} holding")
        score = float(np.clip(score, -1, 1))
        out.score = score
        out.signal_strength = abs(score)
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.3)
        out.confidence = 0.5 if zones else 0.25
        out.horizon = "4h"
        d = "UP" if score > 0.05 else ("DOWN" if score < -0.05 else "NEUTRAL")
        out.evidence = ([self.ev("sr_zone_position", d, abs(score), "market:sr_zones",
                                 detail="; ".join(notes) or "no high-confidence zones nearby",
                                 as_of=self.state.last_candle_at)] if zones else [])
        out.extra = {"zones": zones[:8], "near_resistance": near_res[:2], "near_support": near_sup[:2]}
        return out


# ---------------------------------------------------------------------------
class VolatilityAgent(BaseAgent):
    agent_id = "volatility"
    agent_name = "Volatility Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 2.0
    topics = (TOPIC_CANDLE,)

    def analyze(self, now: datetime) -> AgentOutput:
        from gold_ai.core.enums import VolatilityRegime

        out = self.base_output(now)
        o, h, l, c = _ohlc(self.state, "5m")
        if len(c) < 60:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            return out
        a = ta.atr(h, l, c, 14)
        atr_pct = float(a[-1] / c[-1] * 100)
        atr_hist = a[-200:]
        atr_z = ta.zscore(np.array([x / y for x, y in zip(a[-200:], c[-200:])], dtype=float) * 100, 120)
        rv = ta.realized_vol(c, 60)
        rv_now = float(np.nanmean(rv[-10:])) if len(rv) else 0.0
        rv_base = float(np.nanmedian(rv[-200:])) if len(rv) > 30 else rv_now + 1e-9
        expansion = rv_now / max(rv_base, 1e-9)

        if atr_z > 2.5 or expansion > 2.5:
            regime = VolatilityRegime.SHOCK
        elif expansion > 1.4:
            regime = VolatilityRegime.EXPANSION
        elif expansion < 0.65:
            regime = VolatilityRegime.COMPRESSION
        else:
            regime = VolatilityRegime.NORMAL

        iv = None
        if self.state.options and self.state.options.implied_vol_front:
            iv = float(self.state.options.implied_vol_front)
        shock_prob = float(np.clip(0.5 + atr_z / 6.0, 0.02, 0.95)) if regime == VolatilityRegime.SHOCK else \
            float(np.clip(0.12 + max(0.0, expansion - 1) * 0.25, 0.02, 0.6))

        # Volatility itself is not directional; slight bearish lean on shocks
        # (risk-parity de-leveraging proxy) but keep it near-neutral.
        score = -0.1 if regime == VolatilityRegime.SHOCK else 0.0
        out.score = score
        out.signal_strength = 0.0
        out.probability_up, out.probability_down, out.probability_neutral = self.probs_from_score(score, 0.5)
        out.confidence = 0.7
        out.horizon = "1h"
        out.evidence = [self.ev("vol_regime_state", "NEUTRAL", abs(atr_z) / 3.0, "market:volatility",
                                detail=f"ATR%={atr_pct:.3f}, z={atr_z:+.1f}, expansion={expansion:.2f}x")]
        out.extra = {"volatility_regime": regime.value, "volatility_score": round(float(np.clip(atr_z / 3, -1, 1)), 3),
                     "shock_probability": round(shock_prob, 3), "atr_pct": atr_pct,
                     "realized_vol": rv_now, "expansion": expansion, "iv_front": iv}
        return out


# ---------------------------------------------------------------------------
class SessionAgent(BaseAgent):
    agent_id = "session"
    agent_name = "Session / Time Agent"
    loop = UpdateLoop.MEDIUM
    min_interval_s = 30.0
    topics = (TOPIC_CANDLE,)
    stale_after_s = 3600

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        h = now.hour + now.minute / 60.0
        if 7 <= h < 12:
            sess = SessionName.LONDON
        elif 12 <= h < 16:
            sess = SessionName.LONDON_NY_OVERLAP
        elif 16 <= h < 21:
            sess = SessionName.NEW_YORK
        elif 21 <= h or h < 2:
            sess = SessionName.ASIAN
        elif 2 <= h < 7:
            sess = SessionName.ASIAN
        else:
            sess = SessionName.OFF_HOURS
        self.state.current_session = sess

        # Historical statistics by session from our own candle history
        closes = _closes(self.state, "1m")
        buckets: dict[str, list[float]] = {}
        for c in list(self.state.candles["1m"])[-3000:]:
            hh = c.event_at.hour + c.event_at.minute / 60
            key = ("overlap" if 12 <= hh < 16 else "london" if 7 <= hh < 12 else
                   "ny" if 16 <= hh < 21 else "asia")
            buckets.setdefault(key, []).append(abs(c.range_pct))
        cur_key = {"LONDON": "london", "LONDON_NY_OVERLAP": "overlap", "NEW_YORK": "ny", "ASIAN": "asia"}.get(sess.value, "asia")
        stats = {k: (float(np.mean(v)) if v else 0.0) for k, v in buckets.items()}
        rel_activity = stats.get(cur_key, 0.0) / max(0.01, sum(stats.values()) / max(1, len(stats)))
        dow = now.weekday()
        dow_note = "weekend thin liquidity risk" if dow >= 5 else ""

        score = 0.0  # session alone is not directional
        out.score = score
        out.confidence = 0.4
        out.horizon = "4h"
        out.evidence = [self.ev("session_activity_profile", "NEUTRAL", min(1.0, rel_activity / 2), "market:session",
                                detail=f"{sess.value} avg|range|={stats.get(cur_key, 0):.3f}% {dow_note}")]
        out.extra = {"session": sess.value, "session_stats_pct_avg_range": stats,
                     "relative_activity": round(rel_activity, 2), "day_of_week": dow,
                     "note": "historical behavior by session computed from stored candles"}
        return out
