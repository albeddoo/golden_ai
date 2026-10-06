"""Conflict detector (Section 5.AE) and evidence-correlation / double-counting
control (Section 5.AF).

These are engine-layer validators, not directional agents — but they expose the
same AgentOutput surface so the UI's agent monitor can display them.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from gold_ai.core.enums import ConflictLevel, UpdateLoop
from gold_ai.core.models import AgentOutput, EvidenceItem


# ---------------------------------------------------------------------------
class EvidenceCorrelationController:
    """Builds an evidence dependency graph and penalizes duplicated evidence.

    Every evidence item carries a canonical ``evidence_id`` of the form
    ``family:label``. Two items sharing the same *family* are treated as
    correlated observations of the same underlying information (e.g. DXY agent,
    USD-surprise logic inside cross-asset agent, and a rates agent quoting DXY
    all reduce to family ``fx:dxy``). When computing independent evidence mass
    we take max(strength) per family instead of summing — so the same fact is
    never counted three times.
    """

    # families that map to the same underlying information cluster
    CLUSTERS = {
        "fx:dxy": "usd_complex",
        "macro:rates_move": "rates_complex",
        "rates:nominal": "rates_complex",
        "rates:real": "rates_complex",
        "market:flow": "price_action_cluster",
        "market:trend_structure": "price_action_cluster",
        "market:momentum": "price_action_cluster",
        "market:price_action": "price_action_cluster",
        "market:sr_zones": "price_action_cluster",
        "market:cross_asset": None,   # composite; partially overlaps both clusters below
    }
    OVERLAP_PARTIAL = {"market:cross_asset": ("usd_complex", "rates_complex")}

    def independence_report(self, outputs: list[AgentOutput]) -> dict:
        fam_mass: dict[str, float] = defaultdict(float)
        fam_agents: dict[str, set[str]] = defaultdict(set)
        for o in outputs:
            for e in list(o.evidence) + list(o.contradictions):
                fam = e.evidence_id.split(":")[0] + ":" + e.evidence_id.split(":")[1] if ":" in e.evidence_id else e.evidence_id
                fam_mass[fam] += e.strength
                fam_agents[fam].add(o.agent_id)
        # naive sum vs independent (max-per-cluster with partial overlap haircut)
        naive = sum(fam_mass.values())
        cluster_mass: dict[str, float] = {}
        for fam, mass in fam_mass.items():
            cl = self.CLUSTERS.get(fam, fam)
            if cl is None:
                # partial-overlap composite: split its mass across overlapped
                # clusters with 50% attribution each (documented assumption)
                for target in self.OVERLAP_PARTIAL.get(fam, (fam,)):
                    cluster_mass[target] = cluster_mass.get(target, 0.0) + mass * (0.5 if len(self.OVERLAP_PARTIAL.get(fam, ())) > 1 else 1.0)
            else:
                cluster_mass[cl] = cluster_mass.get(cl, 0.0) + mass
        independent = sum(cluster_mass.values())
        dup_ratio = 1.0 - (independent / naive if naive > 1e-9 else 1.0)
        offenders = {f: sorted(a) for f, a in fam_agents.items() if len(a) > 1}
        return {
            "naive_evidence_mass": round(naive, 3),
            "independent_evidence_mass": round(independent, 3),
            "duplication_ratio": round(max(0.0, min(1.0, dup_ratio)), 3),
            "shared_families": offenders,
            "cluster_mass": {k: round(v, 3) for k, v in cluster_mass.items()},
        }

    def adjusted_weight(self, o: AgentOutput, report: dict) -> tuple[float, str]:
        """Haircut an agent's raw weight by how much of ITS evidence duplicates
        stronger evidence already contributed by other agents."""
        if not o.evidence:
            return 1.0, ""
        shared_fams = []
        for e in o.evidence:
            fam = e.evidence_id.split(":")[:2]
            fam_s = ":".join(fam) if len(fam) >= 2 else e.evidence_id
            agents = report["shared_families"].get(fam_s, [])
            if len(agents) > 1:
                shared_fams.append(fam_s)
        if not shared_fams:
            return 1.0, ""
        frac = len(shared_fams) / max(1, len(o.evidence))
        haircut = 1.0 - 0.5 * frac   # up to 50% penalty when ALL evidence duplicated
        reason = (f"weight ×{haircut:.2f}: {frac:.0%} of evidence duplicates other agents "
                  f"({', '.join(sorted(set(shared_fams)))}) — double-counting control")
        return haircut, reason


# ---------------------------------------------------------------------------
class ConflictDetector:
    """Compares validated agent outputs; high conflict must reduce confidence."""

    def detect(self, outputs: list[AgentOutput], weights: dict[str, float]) -> dict:
        active = [o for o in outputs if o.status in ("ONLINE", "UPDATING", "STALE") and weights.get(o.agent_id, 0) > 0.01]
        if len(active) < 3:
            return {"level": ConflictLevel.HIGH.value, "score_conflict": 1.0,
                    "notes": ["fewer than 3 weighted agents available"], "pairs": []}
        scores = np.array([o.score for o in active])
        wts = np.array([weights[o.agent_id] for o in active])
        wts = wts / wts.sum()
        net = float(np.dot(scores, wts))
        dispersion = float(np.sqrt(np.dot(wts, (scores - net) ** 2)))
        bullish = [o for o in active if o.score > 0.15]
        bearish = [o for o in active if o.score < -0.15]
        bull_w = sum(weights[o.agent_id] for o in bullish)
        bear_w = sum(weights[o.agent_id] for o in bearish)
        total_w = max(1e-9, bull_w + bear_w)
        head_to_head = min(bull_w, bear_w) / total_w   # 0.5 = perfect standoff
        score_conflict = float(np.clip(0.6 * head_to_head * 2 + 0.4 * min(1.0, dispersion * 1.8), 0, 1))

        notes: list[str] = []
        pairs: list[dict] = []
        # pairwise strong disagreements among heavily-weighted agents
        ranked = sorted(active, key=lambda o: -weights[o.agent_id])[:10]
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                a, b = ranked[i], ranked[j]
                if a.score * b < -0.35 and abs(a.score) > 0.25 and abs(b.score) > 0.25:
                    pairs.append({"a": a.agent_name, "b": b.agent_name,
                                  "detail": f"{a.score:+.2f} vs {b.score:+.2f}"})
        # evidence conflict: same family cited opposite directions by two agents
        fam_dir: dict[str, set[str]] = defaultdict(set)
        for o in active:
            for e in o.evidence:
                fam = ":".join(e.evidence_id.split(":")[:2])
                if e.direction in ("UP", "DOWN"):
                    fam_dir[fam].add(e.direction)
        ev_conflicts = [f for f, d in fam_dir.items() if len(d) > 1]
        if ev_conflicts:
            score_conflict = min(1.0, score_conflict + 0.1 * len(ev_conflicts))
            notes.append(f"evidence families internally contradictory: {ev_conflicts[:3]}")
        # timeframe conflict: fast-loop vs slow-loop sign disagreement
        fast = [o for o in active if o.loop == UpdateLoop.FAST.value]
        slow = [o for o in active if o.loop == UpdateLoop.SLOW.value]
        if fast and slow:
            fs = float(np.mean([o.score for o in fast]))
            ss = float(np.mean([o.score for o in slow]))
            if fs * ss < -0.05:
                notes.append(f"timeframe conflict: fast loop {fs:+.2f} vs slow loop {ss:+.2f}")
                score_conflict = min(1.0, score_conflict + 0.12)
        # regime conflict: NEWS_DRIVEN/MACRO_DRIVEN while technicals unanimous
        regimes = {o.agent_id: o for o in active}
        reg = regimes.get("regime")
        tech_ids = ("trend", "momentum", "price_action")
        tech_scores = [regimes[t].score for t in tech_ids if t in regimes]
        if reg and "NEWS_DRIVEN" in reg.extra.get("active_regimes", []) and tech_scores:
            if all(t > 0.2 for t in tech_scores) or all(t < -0.2 for t in tech_scores):
                notes.append("regime conflict: price driven by news but technicals unanimous — "
                             "news evidence outranks chart evidence currently")
                score_conflict = min(1.0, score_conflict + 0.08)

        level = (ConflictLevel.LOW if score_conflict < 0.3 else
                 ConflictLevel.MEDIUM if score_conflict < 0.55 else
                 ConflictLevel.HIGH if score_conflict < 0.75 else ConflictLevel.EXTREME)
        if pairs:
            top = max(pairs, key=lambda p: abs(float(p["detail"].split(" vs ")[0].replace("+", ""))))
            notes.insert(0, f"main conflict: {top['a']} vs {top['b']} ({top['detail']})")
        return {"level": level.value, "score_conflict": round(score_conflict, 3),
                "net_directional_score": round(net, 3), "dispersion": round(dispersion, 3),
                "head_to_head_weight": round(head_to_head, 3),
                "bullish_agents": len(bullish), "bearish_agents": len(bearish),
                "neutral_agents": len(active) - len(bullish) - len(bearish),
                "notes": notes[:6], "pairs": pairs[:6]}

    def to_agent_output(self, result: dict, now: datetime) -> AgentOutput:
        out = AgentOutput(agent_id="conflict_detector", agent_name="Conflict Detector",
                          loop=UpdateLoop.FAST.value, status="ONLINE", horizon="1h")
        sc = result["score_conflict"]
        out.confidence = 0.9
        out.signal_strength = sc
        out.probability_up = out.probability_down = out.probability_neutral = 1 / 3
        out.evidence = [EvidenceItem(
            label="conflict_" + result["level"], direction="NEUTRAL", strength=sc,
            evidence_id="system:conflict", detail="; ".join(result["notes"])[:200])]
        out.extra = result
        out.updated_at = now
        return out
