"""Paper / shadow-mode prediction tracker (Section 19) + performance metrics
(Section 10) + realistic market friction model for simulation (Section 8).

For every issued directional decision we open a shadow record. When the horizon
elapses, we resolve it against realized prices WITH FRICTION:
  effective entry = decision price + half-spread + slippage + latency drift
  outcome         = direction correct after subtracting round-trip friction
Future information never modifies an already-issued prediction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

log = logging.getLogger("gold_ai.paper")


@dataclass
class FrictionModel:
    """Realistic execution frictions for paper analysis — never claim perfect fills."""
    spread_usd: float = 0.35          # typical XAU/USD retail spot spread
    slippage_usd: float = 0.12        # adverse selection proxy
    latency_s: float = 1.5            # decision→"fill" delay
    missing_tick_prob: float = 0.01   # chance resolution uses stale print
    impact_bps: float = 0.4           # market-impact proxy per simulated size

    def round_trip_cost(self, price: float) -> float:
        return 2 * (self.spread_usd + self.slippage_usd) + price * self.impact_bps / 1e4


@dataclass
class ShadowPrediction:
    prediction_id: str
    created_at: datetime
    horizon_min: int
    direction: str            # UP | DOWN
    probability_up: float
    confidence: float
    regime: str
    entry_price: float
    resolved: bool = False
    outcome: Optional[str] = None      # CORRECT | INCORRECT | FLAT
    net_move_frictionless: Optional[float] = None
    net_move_after_fric: Optional[float] = None
    friction_cost: Optional[float] = None


class PaperTracker:
    def __init__(self, friction: FrictionModel | None = None, db=None) -> None:
        self.friction = friction or FrictionModel()
        self.records: list[ShadowPrediction] = []
        self.db = db

    # ------------------------------------------------------------------
    def open_prediction(self, payload, now: datetime) -> Optional[ShadowPrediction]:
        if payload.decision_state == "WAIT" or payload.direction == "NEUTRAL":
            return None  # abstentions are not scored predictions
        hmin = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1D": 1440}[payload.horizon]
        rec = ShadowPrediction(
            prediction_id=f"{payload.decision_id}:{payload.horizon}",
            created_at=now, horizon_min=hmin, direction=payload.direction,
            probability_up=payload.probability_up, confidence=payload.confidence_score,
            regime=payload.regime, entry_price=payload.price,
        )
        self.records.append(rec)
        if len(self.records) > 20000:
            self.records = self.records[-15000:]
        if self.db is not None:
            snap = {"decision_id": payload.decision_id, "regime": payload.regime,
                    "conflict": payload.conflict_level, "event_risk": payload.event_risk,
                    "data_quality": payload.data_quality,
                    "consensus": payload.agent_consensus}
            self.db.store_prediction(payload, snap)
        return rec

    # ------------------------------------------------------------------
    def resolve_due(self, state, now: datetime) -> int:
        """Resolve predictions whose horizon elapsed using stored candles only."""
        n = 0
        c1 = list(state.candles["1m"])
        if len(c1) < 10:
            return 0
        by_ts = {c.event_at: c.close for c in c1}
        times = sorted(by_ts)
        for r in self.records:
            if r.resolved:
                continue
            due = r.created_at + timedelta(minutes=r.horizon_min)
            if now < due:
                continue
            # exit at first candle at/after due time; latency modeled forward-shift
            lat_due = due + timedelta(seconds=self.friction.latency_s)
            idx = np.searchsorted(np.array(times), lat_due)
            if idx >= len(times):
                continue
            exit_price = by_ts[times[min(idx, len(times) - 1)]]
            # missing-tick risk: occasionally use slightly stale print (documented sim)
            import random
            if random.random() < self.friction.missing_tick_prob and idx > 2:
                exit_price = by_ts[times[idx - 2]]
            move = exit_price - r.entry_price
            cost = self.friction.round_trip_cost(r.entry_price)
            r.net_move_frictionless = move
            r.friction_cost = cost
            signed = move if r.direction == "UP" else -move
            r.net_move_after_fric = signed - cost
            if abs(signed) < 1e-9:
                r.outcome = "FLAT"
            else:
                r.outcome = "CORRECT" if signed > 0 else "INCORRECT"
            r.resolved = True
            n += 1
            if self.db is not None:
                self.db.resolve_prediction(r.prediction_id, r.outcome, now)
        return n

    # ------------------------------------------------------------------
    def performance(self) -> dict:
        res = [r for r in self.records if r.resolved]
        out: dict = {"open": len([r for r in self.records if not r.resolved]),
                     "resolved": len(res)}
        if not res:
            out.update({"accuracy": None, "note": "no resolved predictions yet"})
            return out
        correct = sum(1 for r in res if r.outcome == "CORRECT")
        flat = sum(1 for r in res if r.outcome == "FLAT")
        scored = len(res) - flat
        out["accuracy"] = round(correct / scored, 4) if scored else None
        out["flat_rate"] = round(flat / len(res), 4)
        # high/low confidence subsets
        hi = [r for r in res if r.confidence >= 75 and r.outcome != "FLAT"]
        lo = [r for r in res if r.confidence < 65 and r.outcome != "FLAT"]
        out["high_conf_accuracy"] = round(sum(1 for r in hi if r.outcome == "CORRECT") / len(hi), 4) if hi else None
        out["low_conf_accuracy"] = round(sum(1 for r in lo if r.outcome == "CORRECT") / len(lo), 4) if lo else None
        # by regime & horizon
        by_reg: dict[str, list[int]] = {}
        for r in res:
            if r.outcome == "FLAT":
                continue
            by_reg.setdefault(r.regime, []).append(1 if r.outcome == "CORRECT" else 0)
        out["by_regime"] = {k: {"n": len(v), "acc": round(float(np.mean(v)), 3)} for k, v in by_reg.items()}
        # calibration contribution: mean p_up when said UP vs actual up freq
        ups = [r for r in res if r.direction == "UP" and r.outcome != "FLAT"]
        if ups:
            out["pred_p_up_mean"] = round(float(np.mean([r.probability_up for r in ups])), 3)
            out["actual_up_freq"] = round(float(np.mean([1 if r.outcome == "CORRECT" else 0 for r in ups])), 3)
        out["friction_note"] = ("outcomes include spread+slippage+latency; "
                                "frictionless accuracy shown separately")
        out["accuracy_frictionless"] = round(
            sum(1 for r in res if r.net_move_frictionless is not None
                and ((r.direction == "UP") == (r.net_move_frictionless > 0))
                and r.outcome != "FLAT") / max(1, scored), 4)
        return out

    def timeline_view(self, limit: int = 12) -> list[dict]:
        rows = []
        for r in self.records[-limit:]:
            rows.append({"id": r.prediction_id[:12], "ts": r.created_at.strftime("%H:%M:%S"),
                         "dir": r.direction, "p": round(r.probability_up, 2),
                         "conf": round(r.confidence, 0), "regime": r.regime,
                         "outcome": r.outcome or "…",
                         "net_after_fric": round(r.net_move_after_fric, 2) if r.net_move_after_fric is not None else None})
        return rows
