"""Dynamic weighting engine (Section 6) + freshness/reliability engine (5.AG).

Weights are NEVER permanently fixed. Each agent's weight depends on:
current regime, horizon, data freshness, historical performance, calibration
quality, source reliability, current event state, and evidence independence.
Every weight change carries a human-readable reason (explanations required).
"""
from __future__ import annotations

import math
from datetime import datetime

from gold_ai.core.enums import EventRiskState, Horizon, Regime
from gold_ai.core.models import AgentOutput


class WeightingEngine:
    # base prior weights per agent — starting point only, always modulated
    BASE_WEIGHTS: dict[str, float] = {
        "trend": 0.10, "momentum": 0.07, "price_action": 0.06, "spot_market": 0.04,
        "liquidity_zones": 0.05, "volatility": 0.03, "session": 0.02,
        "usd_dxy": 0.08, "treasury_yields": 0.08, "real_yield": 0.08,
        "fed_policy": 0.05, "econ_calendar": 0.05, "inflation": 0.04,
        "labor_market": 0.04, "growth": 0.03,
        "news_intel": 0.07, "geopolitical": 0.05, "cross_asset": 0.05,
        "central_bank": 0.03, "etf_flows": 0.03, "positioning": 0.03,
        "futures_structure": 0.02, "options_iv": 0.02, "seasonality": 0.01,
        "ml_ensemble": 0.08,
    }

    # regime multipliers: how much MORE/LESS important an agent family is
    REGIME_MULT: dict[str, dict[str, float]] = {
        Regime.TRENDING_BULL.value: {"trend": 1.6, "momentum": 1.4, "price_action": 1.2,
                                     "news_intel": 0.7, "geopolitical": 0.7},
        Regime.TRENDING_BEAR.value: {"trend": 1.6, "momentum": 1.4, "price_action": 1.2,
                                     "news_intel": 0.7, "geopolitical": 0.7},
        Regime.RANGE.value: {"mean_rev_like": 1.2, "liquidity_zones": 1.5, "price_action": 1.3,
                             "trend": 0.5, "momentum": 0.6, "seasonality": 1.2},
        Regime.BREAKOUT.value: {"price_action": 1.5, "liquidity_zones": 1.3, "volume_proxy": 1.2,
                                "trend": 1.2},
        Regime.MEAN_REVERSION.value: {"liquidity_zones": 1.4, "trend": 0.5, "momentum": 0.5},
        Regime.HIGH_VOLATILITY.value: {"volatility": 1.5, "options_iv": 1.4, "anomaly": 1.3,
                                       "trend": 0.7, "momentum": 0.7},
        Regime.LOW_VOLATILITY.value: {"breakout_watch": 1.2, "volatility": 1.2},
        Regime.NEWS_DRIVEN.value: {"news_intel": 2.0, "geopolitical": 1.5, "trend": 0.6,
                                   "momentum": 0.6, "price_action": 0.6},
        Regime.MACRO_DRIVEN.value: {"usd_dxy": 1.7, "treasury_yields": 1.7, "real_yield": 1.7,
                                   "fed_policy": 1.6, "econ_calendar": 1.5, "inflation": 1.4,
                                   "labor_market": 1.3, "trend": 0.7},
        Regime.RISK_OFF.value: {"geopolitical": 1.5, "cross_asset": 1.3, "volatility": 1.2},
        Regime.RISK_ON.value: {"cross_asset": 1.2, "geopolitical": 0.6},
        Regime.LIQUIDITY_STRESS.value: {"futures_structure": 1.4, "spot_market": 1.3,
                                        "volatility": 1.5, "trend": 0.5},
        Regime.UNKNOWN.value: {},
    }

    # horizon fit per loop type: slow factors barely matter at 5m
    HORIZON_FIT: dict[str, dict[str, float]] = {
        "FAST":   {"5m": 1.0, "15m": 1.0, "30m": 1.0, "1h": 0.9, "4h": 0.7, "1D": 0.5},
        "MEDIUM": {"5m": 0.5, "15m": 0.7, "30m": 0.9, "1h": 1.0, "4h": 1.0, "1D": 0.9},
        "SLOW":   {"5m": 0.1, "15m": 0.2, "30m": 0.3, "1h": 0.5, "4h": 0.8, "1D": 1.0},
        "EVENT_DRIVEN": {"5m": 0.8, "15m": 1.0, "30m": 1.0, "1h": 1.0, "4h": 0.9, "1D": 0.6},
    }

    EVENT_BOOST_AGENTS = ("usd_dxy", "treasury_yields", "real_yield", "fed_policy",
                          "econ_calendar", "inflation", "labor_market")

    def compute(self, outputs: list[AgentOutput], regimes_active: list[str],
                horizon: Horizon, event_state: str, dup_report: dict,
                now: datetime) -> tuple[dict[str, float], dict[str, str]]:
        weights: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for o in outputs:
            if o.agent_id in ("data_integrity", "regime", "conflict_detector", "anomaly"):
                continue  # meta agents gate/block; they don't vote directionally
            base = self.BASE_WEIGHTS.get(o.agent_id, 0.02)
            parts = [f"base {base:.2f}"]
            w = base

            # regime modulation
            mults: list[str] = []
            for rg in regimes_active[:3]:
                m = self.REGIME_MULT.get(rg, {}).get(o.agent_id)
                if m:
                    w *= m
                    mults.append(f"{rg}×{m}")
            if mults:
                parts.append("regime " + ",".join(mults))

            # event-state boost: near major events macro/rates/calendar matter more
            if event_state in (EventRiskState.EVENT_APPROACHING.value, EventRiskState.EVENT_IMMINENT.value) \
                    and o.agent_id in self.EVENT_BOOST_AGENTS:
                w *= 1.4
                parts.append("event approaching ×1.4")

            # horizon fit
            hf = self.HORIZON_FIT.get(o.loop, {}).get(horizon.value, 0.7)
            o.horizon_fit = hf
            w *= hf
            if abs(hf - 1.0) > 0.05:
                parts.append(f"horizon({horizon.value}) ×{hf}")

            # freshness & reliability engines (5.AG): old factor ≠ fresh catalyst
            fresh = max(0.05, min(1.0, o.freshness))
            w *= 0.35 + 0.65 * fresh
            if fresh < 0.6:
                parts.append(f"freshness ×{0.35 + 0.65 * fresh:.2f}")
            w *= 0.4 + 0.6 * max(0.05, min(1.0, o.source_reliability))
            # historical predictive value (out-of-sample proxy from calibration store)
            hpv = max(0.1, min(1.0, o.historical_predictive_value))
            w *= 0.5 + hpv
            parts.append(f"histPV {hpv:.2f}")
            # calibration quality handled upstream via confidence already
            w *= 0.5 + 0.5 * max(0.0, min(1.0, o.confidence))

            # status penalties
            if o.status == "STALE":
                w *= 0.5
                parts.append("STALE ×0.5")
            elif o.status in ("WARNING", "FAILED", "DISABLED"):
                w *= 0.15
                parts.append(f"{o.status} ×0.15")

            # evidence independence (double-counting control)
            haircut, reason = _dup_haircut(o, dup_report)
            w *= haircut
            if reason:
                parts.append(reason)

            # Tier-3 sources can never independently drive strong decisions
            if o.source_reliability <= 0.6 and w > 0.05:
                w = min(w, 0.05)
                parts.append("capped: low-tier source ≤0.05")

            weights[o.agent_id] = w
            reasons[o.agent_id] = "; ".join(parts)

        # normalize to sum 1 over participating agents
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        return weights, reasons


def _dup_haircut(o: AgentOutput, dup_report: dict) -> tuple[float, str]:
    if not o.evidence:
        return 1.0, ""
    shared = dup_report.get("shared_families", {})
    n_shared = 0
    fams = set()
    for e in o.evidence:
        fam = ":".join(e.evidence_id.split(":")[:2])
        if len(shared.get(fam, [])) > 1:
            n_shared += 1
            fams.add(fam)
    if not n_shared:
        return 1.0, ""
    frac = n_shared / max(1, len(o.evidence))
    haircut = 1.0 - 0.5 * frac
    return haircut, f"independence ×{haircut:.2f} ({', '.join(sorted(fams))} shared with other agents)"
