"""Scenario engine (Section 5.AH) + event-risk engine (Section 7).

Never outputs a single deterministic future. Produces BASE / BULLISH / BEARISH /
NEUTRAL-RANGE / SHOCK scenarios with probability, supporting & contradicting
evidence, required conditions and invalidation conditions.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from gold_ai.core.enums import EventRiskState, Regime
from gold_ai.core.models import AgentOutput, Scenario


class ScenarioEngine:
    def generate(self, probs: dict[str, float], outputs: dict[str, AgentOutput],
                 weights: dict[str, float], regimes: list[str], price: float,
                 vol_state: str, event_state: str, conflict_score: float,
                 now: datetime) -> list[Scenario]:
        p_up = probs["up"]; p_dn = probs["down"]; p_ne = probs["neutral"]
        vol_out = outputs.get("volatility")
        shock_p = float(vol_out.extra.get("shock_probability", 0.1)) if vol_out else 0.1
        if event_state in (EventRiskState.EVENT_IMMINENT.value,):
            shock_p = min(0.45, shock_p + 0.2)
        # renormalize so probabilities sum to 1 after carving out SHOCK
        shock = float(np.clip(shock_p * 0.6, 0.02, 0.35))
        scale = (1.0 - shock) / max(1e-9, p_up + p_dn + p_ne)
        base_p = max(p_up, p_dn, p_ne) * scale
        others = {"up": p_up * scale, "down": p_dn * scale, "neut": p_ne * scale}
        bull_p = others["up"] if base_p != others["neut"] else others["up"]
        # explicit scenario mapping
        s_base = self._base(base_p, outputs, weights, price, now)
        s_bull = self._bull(others["up"], outputs, weights, price, now)
        s_bear = self._bear(others["down"], outputs, weights, price, now)
        s_neut = self._neutral(others["neut"], outputs, weights, price, now)
        s_shock = self._shock(shock, outputs, price, vol_state, event_state, now)
        scens = [s_base, s_bull, s_bear, s_neut, s_shock]
        tot = sum(s.probability for s in scens) or 1.0
        for s in scens:
            s.probability = round(s.probability / tot, 3)
        return scens

    # ------------------------------------------------------------------
    @staticmethod
    def _top_drivers(outputs: dict[str, AgentOutput], weights: dict[str, float], direction: str,
                     k: int = 3) -> list[str]:
        want = 1 if direction == "UP" else -1
        items = []
        for o in outputs.values():
            if o.agent_id in ("data_integrity", "regime", "conflict_detector", "anomaly",
                              "calibration", "model_drift"):
                continue
            contrib = o.score * weights.get(o.agent_id, 0.0)
            if contrib * want > 0.004:
                label = o.evidence[0].label.replace("_", " ") if o.evidence else o.agent_name
                items.append((abs(contrib), f"{o.agent_name}: {label}"))
        items.sort(reverse=True)
        return [t for _, t in items[:k]]

    @staticmethod
    def _atr_price_range(outputs: dict[str, AgentOutput], price: float, hours: float) -> tuple[float, float]:
        vol = outputs.get("volatility")
        atr_pct = float(vol.extra.get("atr_pct", 0.08)) if vol else 0.08
        sigma = atr_pct / 100 * math.sqrt(max(1.0, hours * 12)) * price
        return price - 1.28 * sigma, price + 1.28 * sigma

    def _base(self, p: float, outputs, weights, price, now) -> Scenario:
        lo, hi = self._atr_price_range(outputs, price, 4)
        return Scenario(
            kind="BASE", probability=p,
            supporting_evidence=self._top_drivers(outputs, weights, "UP" if p >= 0 else "DOWN", 3),
            contradicting_evidence=[],
            conditions_required=["no new macro surprise", "current regime persists"],
            invalidation_conditions=["event surprise beyond forecast band", "volatility shock"],
            target_range=(round(lo, 1), round(hi, 1)),
            narrative="Most likely continuation of current evidence-weighted state over 1–4h.",
        )

    def _bull(self, p, outputs, weights, price, now) -> Scenario:
        lo, hi = self._atr_price_range(outputs, price, 4)
        drivers = self._top_drivers(outputs, weights, "UP")
        contra = self._top_drivers(outputs, weights, "DOWN")
        return Scenario(
            kind="BULLISH", probability=p,
            supporting_evidence=drivers or ["technical strength"],
            contradicting_evidence=contra,
            conditions_required=["real yields fall or USD weakens further",
                                 "breakout above nearest resistance holds on close"],
            invalidation_conditions=[f"price closes below {lo:.0f}", "hot inflation surprise",
                                     "Fed hawkish repricing"],
            target_range=(round(price, 1), round(hi * 1.002, 1)),
            narrative="Upside scenario conditional on macro tailwinds persisting.",
        )

    def _bear(self, p, outputs, weights, price, now) -> Scenario:
        lo, hi = self._atr_price_range(outputs, price, 4)
        drivers = self._top_drivers(outputs, weights, "DOWN")
        contra = self._top_drivers(outputs, weights, "UP")
        return Scenario(
            kind="BEARISH", probability=p,
            supporting_evidence=drivers or ["macro headwinds"],
            contradicting_evidence=contra,
            conditions_required=["real yields keep rising / USD strength persists",
                                 "loss of nearby support zone"],
            invalidation_conditions=[f"price reclaims {hi:.0f}", "dovish Fed surprise",
                                     "geopolitical escalation shock"],
            target_range=(round(lo * 0.998, 1), round(price, 1)),
            narrative="Downside scenario driven by opportunity-cost and momentum pressure.",
        )

    def _neutral(self, p, outputs, weights, price, now) -> Scenario:
        mid_lo, mid_hi = price * 0.997, price * 1.003
        reg_ok = any(r in (Regime.RANGE.value, Regime.MEAN_REVERSION.value, Regime.LOW_VOLATILITY.value)
                     for r in outputs.get("regime", AgentOutput(agent_id="regime", agent_name="r")).extra.get("active_regimes", []))
        return Scenario(
            kind="NEUTRAL_RANGE", probability=p,
            supporting_evidence=["compression/consolidation structure" if reg_ok else "balanced evidence"],
            contradicting_evidence=["strong one-sided catalyst would break range"],
            conditions_required=["no major event surprise", "volatility stays compressed"],
            invalidation_conditions=[f"range break ±0.5%", "news shock"],
            target_range=(round(mid_lo, 1), round(mid_hi, 1)),
            narrative="Range/mean-reversion: two-way risk until structure breaks.",
        )

    def _shock(self, p, outputs, price, vol_state, event_state, now) -> Scenario:
        wide = (price * 0.975, price * 1.03)
        return Scenario(
            kind="SHOCK", probability=p,
            supporting_evidence=[f"vol state={vol_state}", f"event state={event_state}"],
            contradicting_evidence=["compressed realized vol"] if vol_state == "COMPRESSION" else [],
            conditions_required=["unpriced headline or data miss/gap beyond 2σ"],
            invalidation_conditions=["quiet tape through event window"],
            target_range=(round(wide[0], 1), round(wide[1], 1)),
            narrative="Tail scenario: gap/rank-run moves where normal weighting logic fails; "
                      "system is designed to WAIT in this state.",
        )


# ---------------------------------------------------------------------------
class EventRiskEngine:
    """Dedicated event-risk layer (Section 7)."""

    IMMINENT_S = 15 * 60
    APPROACHING_S = 2 * 3600
    REPRICING_S = 90 * 60

    def evaluate(self, state, now: datetime) -> dict:
        nxt = state.next_event(now)
        ttl = None
        name = None
        importance = 0.0
        if nxt is not None:
            ttl = (nxt.scheduled_at - now).total_seconds()
            name = nxt.name
            importance = nxt.importance

        recent_rel = [r for r in state.event_releases if (now - r.effective_at) < timedelta(hours=6)]
        last_rel = recent_rel[-1] if recent_rel else None
        rel_age = (now - last_rel.effective_at).total_seconds() if last_rel else None

        state_name = EventRiskState.NORMAL
        detail = "no major event within horizon"
        if last_rel is not None and rel_age is not None:
            if rel_age < 30 * 60:
                state_name = EventRiskState.EVENT_RELEASED
                detail = f"{last_rel.event_name} released {rel_age/60:.0f}m ago (surprise={last_rel.surprise})"
            elif rel_age < self.REPRICING_S:
                state_name = EventRiskState.POST_EVENT_REPRICING
                detail = f"repricing after {last_rel.event_name}"
        if nxt is not None and ttl is not None and not last_rel:
            if ttl <= self.IMMINENT_S:
                state_name = EventRiskState.EVENT_IMMINENT
                detail = f"{name} in {ttl/60:.0f} min"
            elif ttl <= self.APPROACHING_S:
                state_name = EventRiskState.EVENT_APPROACHING
                detail = f"{name} in {ttl/3600:.1f} h"
        # anomaly-driven shock override
        geo = state.agent_outputs.get("geopolitical")
        news = state.agent_outputs.get("news_intel")
        anom = state.agent_outputs.get("anomaly")
        if anom and any(a.get("kind") in ("PRICE_SHOCK", "NEWS_SHOCK", "VOLATILITY_SHOCK")
                        for a in anom.extra.get("anomalies", [])):
            state_name = EventRiskState.SHOCK
            detail = "market/news shock detected"
        risk_level = self._risk_level(state_name, importance, rel_age)
        phase = None
        if state_name in (EventRiskState.EVENT_RELEASED, EventRiskState.POST_EVENT_REPRICING) and last_rel:
            phase = self._reaction_phase(state, now, last_rel.effective_at)
        return {"state": state_name.value, "detail": detail, "next_event": name,
                "ttl_seconds": ttl, "importance": importance, "risk_level": risk_level,
                "reaction_phase": phase,
                "note": "first move after release is NOT assumed final (initial/confirm/reversal tracked)"}

    @staticmethod
    def _risk_level(state_name: EventRiskState, importance: float, rel_age: float | None) -> str:
        if state_name == EventRiskState.SHOCK:
            return "EXTREME"
        if state_name == EventRiskState.EVENT_IMMINENT:
            return "HIGH" if importance > 0.7 else "MEDIUM"
        if state_name == EventRiskState.EVENT_APPROACHING:
            return "ELEVATED" if importance > 0.7 else "MODERATE"
        if state_name in (EventRiskState.EVENT_RELEASED, EventRiskState.POST_EVENT_REPRICING):
            return "HIGH" if (rel_age or 999) < 30 * 60 else "ELEVATED"
        return "NORMAL"

    @staticmethod
    def _reaction_phase(state, now: datetime, released_at: datetime) -> str:
        """Classify initial reaction vs confirmation vs reversal post-release."""
        closes = np.array(state.snapshot_prices("1m", 60), dtype=float)
        idx0 = None
        for i, c in enumerate(list(state.candles["1m"])[-60:]):
            if c.event_at >= released_at:
                idx0 = i
                break
        if idx0 is None or len(closes) - idx0 < 6:
            return "INITIAL_REACTION"
        seg = closes[idx0:]
        initial = seg[min(4, len(seg) - 1)] / seg[0] - 1
        latest = seg[-1] / seg[0] - 1
        if abs(initial) < 1e-5:
            return "FLAT_INITIAL"
        if latest * initial > 0 and abs(latest) > abs(initial) * 1.15:
            return "CONFIRMATION"
        if latest * initial < 0 and abs(latest) > abs(initial) * 0.5:
            return "REVERSAL"
        return "CONSOLIDATING_AFTER_RELEASE"
