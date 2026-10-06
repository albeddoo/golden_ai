"""Market regime agent (Section 5.AA) — central agent.

Multi-label regime classification; the output feeds the dynamic weighting
engine directly. Regimes are NOT mutually exclusive: e.g. HIGH_VOLATILITY +
MACRO_DRIVEN + TRENDING_BULL can co-exist with stated primary regime.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from gold_ai.agents.base import BaseAgent
from gold_ai.core.enums import AgentStatus, EventRiskState, Regime, UpdateLoop
from gold_ai.core.event_bus import TOPIC_AGENT_OUTPUT, TOPIC_ANOMALY, TOPIC_CANDLE
from gold_ai.core.models import AgentOutput
from gold_ai.data.normalization import indicators as ta


class MarketRegimeAgent(BaseAgent):
    agent_id = "regime"
    agent_name = "Market Regime Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 3.0
    topics = (TOPIC_CANDLE, TOPIC_AGENT_OUTPUT, TOPIC_ANOMALY)
    stale_after_s = 600

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        closes = np.array(self.state.snapshot_prices("5m", 400), dtype=float)
        scores: dict[str, float] = {}
        notes: list[str] = []
        if len(closes) < 80:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.1
            out.regime_extra = {}
            self._publish(Regime.UNKNOWN, [Regime.UNKNOWN], now)
            return out

        # --- trend strength via ADX-like measure on 1h -------------------
        h1 = np.array(self.state.snapshot_prices("1h", 200), dtype=float)
        adx = self._adx_like(h1) if len(h1) > 40 else 0.0
        slope_fast = ta.slope(closes, 80)
        ema_spread = (ta.ema(closes, 20)[-1] / ta.ema(closes, 100)[-1] - 1) * 100

        trending = adx > 22 and abs(slope_fast) > 0.015
        scores[Regime.TRENDING_BULL.value] = 1.0 if trending and slope_fast > 0 else max(0.0, (abs(slope_fast) / 0.05) * (adx / 40)) * (1 if slope_fast > 0 else 0)
        scores[Regime.TRENDING_BEAR.value] = scores[Regime.TRENDING_BULL.value] if slope_fast >= 0 else 0.0
        if trending and slope_fast < 0:
            scores[Regime.TRENDING_BEAR.value] = 1.0
            scores[Regime.TRENDING_BULL.value] = 0.0

        # --- volatility regimes ------------------------------------------
        vol_out = self.state.agent_outputs.get("volatility")
        vol_regime = vol_out.extra.get("volatility_regime", "NORMAL") if vol_out else "NORMAL"
        scores[Regime.HIGH_VOLATILITY.value] = 1.0 if vol_regime in ("SHOCK", "EXPANSION") else 0.3
        scores[Regime.LOW_VOLATILITY.value] = 1.0 if vol_regime == "COMPRESSION" else 0.2

        # --- range / mean reversion: low ADX + bollinger squeeze ---------
        bb_mid, bb_up, bb_low = ta.bollinger(closes, 20, 2.0)
        width = float(np.nanmean((bb_up[-10:] - bb_low[-10:]) / bb_mid[-10:]))
        is_range = adx < 18 and abs(slope_fast) < 0.01
        scores[Regime.RANGE.value] = 0.9 if is_range else 0.15
        mr_score = 0.0
        if len(closes) > 100:
            r = np.diff(np.log(closes))
            autocorr = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r.std() > 1e-12 else 0.0
            mr_score = max(0.0, -autocorr * 2.0)
        scores[Regime.MEAN_REVERSION.value] = min(1.0, mr_score + (0.3 if is_range else 0.0))

        # --- breakout: price beyond recent consolidation box -------------
        box_hi = float(np.max(closes[-100:-5])) if len(closes) > 105 else float(np.max(closes))
        box_lo = float(np.min(closes[-100:-5])) if len(closes) > 105 else float(np.min(closes))
        last = float(closes[-1])
        brk = (last > box_hi) or (last < box_lo)
        body_ratio = None
        c15 = list(self.state.candles["15m"])[-3:]
        if c15:
            cc = c15[-1]
            body_ratio = abs(cc.close - cc.open) / max(cc.high - cc.low, 1e-9)
        scores[Regime.BREAKOUT.value] = 0.85 if (brk and (body_ratio is None or body_ratio > 0.5)) else 0.1

        # --- news driven ----------------------------------------------------
        news_out = self.state.agent_outputs.get("news_intel")
        if news_out:
            imp = news_out.extra.get("aggregate", {}).get("impact", 0.0)
            breaking = news_out.extra.get("breaking", False)
            scores[Regime.NEWS_DRIVEN.value] = 0.9 if breaking else min(1.0, imp * 0.8)
        else:
            scores[Regime.NEWS_DRIVEN.value] = 0.05

        # --- macro driven: calendar state + rates agent activity ----------
        cal = self.state.agent_outputs.get("econ_calendar")
        ers = cal.extra.get("event_risk_state", "NORMAL") if cal else "NORMAL"
        rates_mag = 0.0
        for aid in ("usd_dxy", "treasury_yields", "real_yield"):
            ao = self.state.agent_outputs.get(aid)
            if ao:
                rates_mag = max(rates_mag, abs(ao.score) * ao.signal_strength)
        scores[Regime.MACRO_DRIVEN.value] = min(1.0, (0.7 if ers != EventRiskState.NORMAL.value else 0.1) + rates_mag)

        # --- risk on/off from cross asset & geo ----------------------------
        ca = self.state.agent_outputs.get("cross_asset")
        geo = self.state.agent_outputs.get("geopolitical")
        vix_q = self.state.quotes.get("VIX")
        risk_off = 0.0
        if vix_q:
            risk_off += min(1.0, max(0.0, (vix_q.last - 14) / 20))
        if geo:
            risk_off += max(0.0, geo.score) * 0.5
        scores[Regime.RISK_OFF.value] = min(1.0, risk_off)
        scores[Regime.RISK_ON.value] = min(1.0, max(0.0, 0.5 - risk_off / 2) + (-(ca.score) * 0.4 if ca and ca.score < 0 else 0.0))

        # --- liquidity stress: volume collapse + wide spreads --------------
        liq_stress = 0.0
        spot = self.state.agent_outputs.get("spot_market")
        if spot and spot.extra.get("spread_bps") and spot.extra["spread_bps"] > 6:
            liq_stress += 0.5
        vols_series = [c.volume for c in list(self.state.candles["1m"])[-240:]]
        if len(vols_series) > 60:
            ratio = float(np.mean(vols_series[-10:])) / (float(np.mean(vols_series)) + 1e-9)
            if ratio < 0.35:
                liq_stress += 0.4
        scores[Regime.LIQUIDITY_STRESS.value] = min(1.0, liq_stress)

        active = [Regime(k) for k, v in scores.items() if v >= 0.55 and k not in (Regime.UNKNOWN.value,)]
        if not active:
            active = [Regime(max(scores, key=scores.get))]
        primary = max(active, key=lambda r: scores[r.value])
        if all(v < 0.3 for v in scores.values()):
            primary = Regime.UNKNOWN
            active = [Regime.UNKNOWN]

        self._publish(primary, active, now)
        out.score = 0.0
        out.confidence = float(np.clip(np.mean([scores[r.value] for r in active]), 0.2, 0.95))
        out.probability_up = out.probability_down = out.probability_neutral = 1 / 3
        out.horizon = "1h"
        out.evidence = [self.ev(f"regime_{r.value}", "NEUTRAL", scores[r.value], "system:regime",
                                detail=f"score={scores[r.value]:.2f}") for r in active]
        out.extra = {"primary_regime": primary.value, "active_regimes": [r.value for r in active],
                     "regime_scores": {k: round(v, 3) for k, v in scores.items()},
                     "adx_like": round(adx, 1), "slope_pct_per_bar": round(slope_fast, 4),
                     "event_risk_state": ers}
        return out

    # ------------------------------------------------------------------
    def _publish(self, primary: Regime, active: list[Regime], now: datetime) -> None:
        self.state.regime = primary.value
        self.state.regimes_active = [r.value for r in active]
        self.bus.publish("engine.regime", {"primary": primary.value,
                                           "active": [r.value for r in active], "ts": now.isoformat()})

    @staticmethod
    def _adx_like(closes: np.ndarray, period: int = 14) -> float:
        """Simplified directional-movement index on closes (proxy, documented)."""
        if len(closes) < period + 5:
            return 0.0
        up = np.clip(np.diff(closes), 0, None)
        dn = -np.clip(np.diff(closes), None, 0)
        rng = np.abs(np.diff(closes)) + 1e-9
        dx = 100 * np.abs(up - dn) / rng
        dx = dx[-4 * period:]
        return float(np.mean(dx[-period:])) if len(dx) >= period else 0.0
