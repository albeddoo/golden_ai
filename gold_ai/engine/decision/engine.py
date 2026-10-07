"""Central Decision Engine (Sections 11, 12, 13).

Receives all validated agent outputs → applies double-counting control, dynamic
weights, conflict penalty, data-integrity gating and event-risk gating → emits
an immutable DecisionPayload with full explanation, or a WAIT state.

WAIT is a first-class, successful output — never an error, never forced.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np

from gold_ai.core.enums import (
    AgentStatus, ConflictLevel, DecisionState, EventRiskState, Horizon,
)
from gold_ai.core.models import AgentOutput, DecisionPayload, Scenario
from gold_ai.engine.evidence.conflict import EvidenceCorrelationController
from gold_ai.engine.scenario.event_risk import ScenarioEngine


class DecisionEngine:
    # configurable & empirically validated thresholds (Section 10/11)
    MIN_CONFIDENCE_FOR_DIRECTION = 55.0     # below this → WAIT
    MAX_CONFLICT_FOR_DIRECTION = {ConflictLevel.LOW.value, ConflictLevel.MEDIUM.value}
    MIN_DATA_QUALITY = 0.60
    MIN_INDEPENDENT_EVIDENCE = 0.35         # independent-mass / naive-mass floor ratio proxy
    BLOCKING_EVENT_STATES = {EventRiskState.EVENT_IMMINENT.value, EventRiskState.SHOCK.value}
    HIGH_EVENT_BLOCK_RISK = {"EXTREME"}

    CONFIDENCE_BANDS = [(85, "VERY STRONG"), (75, "STRONG"), (65, "MODERATE"), (55, "WEAK"), (0, "UNCERTAIN")]

    def __init__(self) -> None:
        self.dup_controller = EvidenceCorrelationController()
        self.scenario_engine = ScenarioEngine()

    # ------------------------------------------------------------------
    def decide(
        self,
        outputs: dict[str, AgentOutput],
        weights: dict[str, float],
        weight_reasons: dict[str, str],
        regimes_active: list[str],
        event_risk: dict,
        conflict: dict,
        price: float,
        horizon: Horizon,
        now: datetime,
        calibration_ece: float | None = None,
    ) -> DecisionPayload:
        participating = [o for o in outputs.values() if o.agent_id in weights]
        dup_report = self.dup_controller.independence_report(participating)
        # apply independence haircuts to raw weights then renormalize
        adj_w: dict[str, float] = {}
        reasons_extra: dict[str, str] = {}
        for o in participating:
            haircut, reason = self.dup_controller.adjusted_weight(o, dup_report)
            adj_w[o.agent_id] = weights[o.agent_id] * haircut
            if reason:
                reasons_extra[o.agent_id] = reason
        tot = sum(adj_w.values()) or 1.0
        adj_w = {k: v / tot for k, v in adj_w.items()}

        # aggregate probabilities (weighted mean of calibrated-ish agent probs)
        pu = sum(o.probability_up * adj_w.get(o.agent_id, 0) for o in participating)
        pd_ = sum(o.probability_down * adj_w.get(o.agent_id, 0) for o in participating)
        pn = sum(o.probability_neutral * adj_w.get(o.agent_id, 0) for o in participating)
        s = pu + pd_ + pn
        pu, pd_, pn = pu / s, pd_ / s, pn / s

        # evidence strength: mass of directional, independent evidence
        ev_strength = float(min(1.0, dup_report["independent_evidence_mass"] / 6.0))

        # consensus counts
        bull = sum(1 for o in participating if o.score > 0.15 and adj_w.get(o.agent_id, 0) > 0.01)
        bear = sum(1 for o in participating if o.score < -0.15 and adj_w.get(o.agent_id, 0) > 0.01)
        neut = len(participating) - bull - bear

        di = outputs.get("data_integrity")
        data_quality = di.data_quality if di else 0.5
        blocking = list(di.extra.get("blocking_conditions", [])) if di else ["DATA_INTEGRITY_MISSING"]
        vol_out = outputs.get("volatility")
        vol_state = vol_out.extra.get("volatility_regime", "NORMAL") if vol_out else "UNKNOWN"
        news_out = outputs.get("news_intel")
        news_risk = ("HIGH" if news_out and news_out.extra.get("breaking") else
                     "MEDIUM" if news_out and news_out.extra.get("aggregate", {}).get("impact", 0) > 0.4 else "LOW")

        # ---- confidence composition (distinct from probability!) ---------
        base_conf = 0.5 * ev_strength + 0.3 * data_quality + 0.2 * min(1.0, len(participating) / 18)
        conf_mult = 1.0
        clevel = conflict["level"]
        if clevel == ConflictLevel.HIGH.value:
            conf_mult *= 0.72
        elif clevel == ConflictLevel.EXTREME.value:
            conf_mult *= 0.5
        er_state = event_risk["state"]
        if er_state == EventRiskState.EVENT_APPROACHING.value:
            conf_mult *= 0.85
        elif er_state in (EventRiskState.EVENT_RELEASED.value, EventRiskState.POST_EVENT_REPRICING.value):
            conf_mult *= 0.9
        if vol_state in ("SHOCK",):
            conf_mult *= 0.75
        if calibration_ece is not None and calibration_ece > 0.08:
            conf_mult *= max(0.6, 1.0 - calibration_ece)   # poorly calibrated → distrust
        stale_n = sum(1 for o in participating if o.status in (AgentStatus.STALE.value, AgentStatus.FAILED.value,
                                                                AgentStatus.WARNING.value))
        conf_mult *= max(0.55, 1.0 - 0.03 * stale_n)
        margin = abs(pu - pd_)
        conf_score = float(np.clip(100 * (0.35 + 0.65 * base_conf) * conf_mult * (0.6 + 0.4 * min(1.0, margin * 3)), 0, 100))

        # ---- direction ---------------------------------------------------
        if pu >= pd_ and pu >= pn:
            direction = "UP"
        elif pd_ >= pu and pd_ >= pn:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        # ---- abstention gates (Section 11) --------------------------------
        wait_reasons: list[str] = []
        if blocking:
            wait_reasons += [f"blocked by data integrity: {b}" for b in blocking]
        if data_quality < self.MIN_DATA_QUALITY:
            wait_reasons.append(f"data quality {data_quality:.2f} < {self.MIN_DATA_QUALITY}")
        if clevel not in self.MAX_CONFLICT_FOR_DIRECTION:
            wait_reasons.append(f"agent conflict {clevel} too high")
        if er_state in self.BLOCKING_EVENT_STATES:
            wait_reasons.append(f"event risk state {er_state}")
        if event_risk.get("risk_level") in self.HIGH_EVENT_BLOCK_RISK:
            wait_reasons.append("extreme event risk")
        if vol_state == "SHOCK":
            wait_reasons.append("volatility shock — normal weighting invalid")
        if conf_score < self.MIN_CONFIDENCE_FOR_DIRECTION:
            wait_reasons.append(f"confidence {conf_score:.0f}/100 below threshold {self.MIN_CONFIDENCE_FOR_DIRECTION:.0f}")
        if dup_report["duplication_ratio"] > 0.55:
            wait_reasons.append(f"evidence duplication {dup_report['duplication_ratio']:.0%} — insufficient independent information")
        if len(participating) < 8:
            wait_reasons.append(f"only {len(participating)} agents participating")
        if direction == "NEUTRAL" and pn < 0.45:
            wait_reasons.append("directional margin too small to be meaningful")

        decision_state = DecisionState.WAIT.value if wait_reasons else {
            "UP": DecisionState.BULLISH.value, "DOWN": DecisionState.BEARISH.value,
            "NEUTRAL": DecisionState.NEUTRAL.value}[direction]

        # ---- signal quality letter grade ----------------------------------
        band = next(lbl for th, lbl in self.CONFIDENCE_BANDS if conf_score >= th)
        quality = {"VERY STRONG": "A+", "STRONG": "A-", "MODERATE": "B", "WEAK": "C", "UNCERTAIN": "D"}[band]
        if wait_reasons:
            quality = "—"

        # ---- scenarios -----------------------------------------------------
        scenarios = self.scenario_engine.generate(
            {"up": pu, "down": pd_, "neutral": pn}, outputs, adj_w, regimes_active,
            price, vol_state, er_state, conflict["score_conflict"], now)
        scen_dist = {s.kind: s.probability for s in scenarios}

        top_bull = sorted(((o.score * adj_w.get(o.agent_id, 0), f"{o.agent_name}: "
                           f"{(o.evidence[0].label.replace('_', ' ') if o.evidence else 'lean')}")
                          for o in participating if o.score > 0), reverse=True)[:5]
        top_bear = sorted(((o.score * adj_w.get(o.agent_id, 0), f"{o.agent_name}: "
                           f"{(o.evidence[0].label.replace('_', ' ') if o.evidence else 'lean')}")
                          for o in participating if o.score < 0))[:5]
        risks = []
        if er_state != EventRiskState.NORMAL.value:
            risks.append(f"Event risk: {event_risk['detail']}")
        if vol_state in ("EXPANSION", "SHOCK"):
            risks.append(f"Volatility {vol_state}")
        if news_risk != "LOW":
            risks.append(f"News flow {news_risk}")
        if conflict["notes"]:
            risks.append(f"Conflict: {conflict['notes'][0]}")
        if data_quality < 0.85:
            risks.append(f"Data quality degraded ({data_quality:.0%})")

        paper = None
        if decision_state != DecisionState.WAIT.value:
            paper = {"BULLISH": "Paper Scenario: Bullish", "BEARISH": "Paper Scenario: Bearish",
                     "NEUTRAL": "Paper Scenario: Wait"}[decision_state]

        expl = self._explain(direction, conf_score, decision_state, bull, bear, neut,
                             regimes_active, conflict, event_risk, top_bull, top_bear, wait_reasons)

        payload = DecisionPayload(
            decision_id=uuid.uuid4().hex[:12],
            created_at=now, price=price,
            direction=direction,
            probability_up=round(pu, 4), probability_down=round(pd_, 4), probability_neutral=round(pn, 4),
            confidence_score=round(conf_score, 1), signal_quality=quality,
            regime=(regimes_active[0] if regimes_active else "UNKNOWN"),
            regimes_active=regimes_active,
            agent_consensus={"bullish": bull, "bearish": bear, "neutral": neut},
            evidence_strength=round(ev_strength, 3),
            conflict_level=clevel, conflict_notes=conflict["notes"],
            data_quality=round(data_quality, 3),
            event_risk=event_risk.get("risk_level", "NORMAL"),
            event_risk_detail=event_risk.get("detail", ""),
            news_risk=news_risk, volatility_state=vol_state,
            scenario_distribution=scen_dist,
            decision_state=decision_state, wait_reasons=wait_reasons,
            top_bullish=[t for _, t in top_bull], top_bearish=[t for _, t in top_bear],
            top_risks=risks[:5],
            explanation=expl, paper_scenario=paper, horizon=horizon.value,
        )
        # attach weight bookkeeping for the inspector UI
        for aid, w in adj_w.items():
            o = outputs.get(aid)
            if o is not None:
                o.weight = w
                parts = [weight_reasons.get(aid, "")]
                if aid in reasons_extra:
                    parts.append(reasons_extra[aid])
                o.weight_reason = " | ".join(p for p in parts if p)
        return payload

    # ------------------------------------------------------------------
    @staticmethod
    def _explain(direction, conf, dstate, bull, bear, neut, regimes, conflict, event_risk,
                 top_bull, top_bear, wait_reasons) -> str:
        lines = [
            f"FINAL VIEW — Direction: {direction} | State: {dstate} | Confidence: {conf:.0f}/100",
            f"Regime: {'/'.join(regimes[:3]) or 'UNKNOWN'}",
            f"Agent consensus: {bull} bullish / {bear} bearish / {neut} neutral",
        ]
        if top_bull:
            lines.append("Top positive drivers: " + "; ".join(t.split(": ")[-1] for _, t in top_bull[:4]))
        if top_bear:
            lines.append("Top negative drivers: " + "; ".join(t.split(": ")[-1] for _, t in top_bear[:4]))
        if conflict["notes"]:
            lines.append("Main conflict: " + conflict["notes"][0])
        if event_risk.get("detail"):
            lines.append(f"Event context: {event_risk['detail']} [{event_risk.get('risk_level')}]")
        if wait_reasons:
            lines.append("ABSTENTION (WAIT) because: " + "; ".join(wait_reasons[:4]))
            lines.append("WAIT is a valid system output — evidence does not support a directional call.")
        else:
            lines.append("Decision issued with traceable evidence; probabilities are weighted agent "
                         "consensus, NOT a single model's opinion.")
        return "\n".join(lines)
