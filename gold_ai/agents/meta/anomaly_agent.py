"""Anomaly agent (Section 5.Z).

Detects: price shock, volatility shock, volume anomaly, cross-asset divergence,
news shock, feed anomaly. Publishes TOPIC_ANOMALY so the regime/weighting layer
and event-risk engine can react immediately.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta

import numpy as np

from gold_ai.agents.base import BaseAgent
from gold_ai.core.enums import AgentStatus, UpdateLoop
from gold_ai.core.event_bus import TOPIC_ANOMALY, TOPIC_CANDLE, TOPIC_NEWS, TOPIC_QUOTE
from gold_ai.core.models import AgentOutput


class AnomalyAgent(BaseAgent):
    agent_id = "anomaly"
    agent_name = "Anomaly Agent"
    loop = UpdateLoop.FAST
    min_interval_s = 1.0
    topics = (TOPIC_CANDLE, TOPIC_QUOTE, TOPIC_NEWS)
    stale_after_s = 600

    def __init__(self, state, bus) -> None:
        super().__init__(state, bus)
        self._ret_hist: deque[float] = deque(maxlen=1200)
        self._vol_hist: deque[float] = deque(maxlen=1200)
        self._last_news_ts: datetime | None = None

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        anomalies: list[dict] = []
        c1 = list(self.state.candles["1m"])[-400:]
        if len(c1) >= 30:
            rets = [math.log(c1[i].close / c1[i - 1].close) for i in range(1, len(c1))]
            mu, sd = float(np.mean(rets)), float(np.std(rets, ddof=1)) or 1e-9
            z_last = (rets[-1] - mu) / sd
            if abs(z_last) > 5:
                anomalies.append({"kind": "PRICE_SHOCK", "z": round(z_last, 1),
                                  "move_pct": round((math.exp(rets[-1]) - 1) * 100, 3)})
            vols = [c.high - c.low for c in c1]
            vmu, vsd = float(np.mean(vols)), float(np.std(vols, ddof=1)) or 1e-9
            zv = (vols[-1] - vmu) / vsd
            if zv > 4:
                anomalies.append({"kind": "VOLATILITY_SHOCK", "z": round(zv, 1)})
            volumes = [c.volume for c in c1]
            vv = volumes[-1] / (float(np.mean(volumes[-120:-1])) + 1e-9)
            if vv > 4.5:
                anomalies.append({"kind": "VOLUME_ANOMALY", "ratio": round(vv, 1)})
            # cross-asset divergence: gold up while real yields AND dxy up strongly
            ry = self.state.macro.get("DFII10_REAL")
            dxy = self.state.macro.get("DXY")
            if ry and dxy and len(c1) >= 31:
                g30 = c1[-1].close / c1[-31].close - 1
                r30 = ry.value / (self.state.macro_series("DFII10_REAL")[-31].value if len(self.state.macro_series("DFII10_REAL")) >= 31 else ry.value) - 1
                d30 = dxy.value / (self.state.macro_series("DXY")[-31].value if len(self.state.macro_series("DXY")) >= 31 else dxy.value) - 1
                if g30 > 0.0015 and r30 > 0.0005 and d30 > 0.0008:
                    anomalies.append({"kind": "CROSS_ASSET_DIVERGENCE",
                                      "detail": f"gold {g30*100:+.2f}% vs rising DXY {d30*100:+.2f}% & real yields {r30*100:+.2f}%"})
            # feed anomaly: zero-range candles burst
            flat = sum(1 for c in c1[-10:] if c.high == c.low)
            if flat >= 8:
                anomalies.append({"kind": "FEED_ANOMALY", "detail": f"{flat}/10 zero-range candles"})
        # news shock tracked by dedicated topic
        recent = self.state.recent_news(minutes=10, now=now)
        shocks = [a for a in recent if a.classification.value == "BREAKING" and a.potential_market_impact > 0.5]
        if shocks and (self._last_news_ts is None or shocks[-1].effective_at > self._last_news_ts):
            self._last_news_ts = shocks[-1].effective_at
            anomalies.append({"kind": "NEWS_SHOCK", "detail": shocks[-1].headline[:90]})

        severity = 0.0
        for a in anomalies:
            z = abs(a.get("z", 3.0))
            severity += min(1.0, (z - 3) / 4 + 0.35)
        severity = min(1.0, severity)
        out.score = 0.0
        out.signal_strength = severity
        out.probability_up = out.probability_down = 0.5 - severity * 0.15
        out.probability_neutral = 1 - (out.probability_up + out.probability_down)
        out.confidence = 0.7
        out.status = AgentStatus.WARNING.value if anomalies else AgentStatus.ONLINE.value
        out.horizon = "15m"
        kinds = {a["kind"] for a in anomalies}
        out.evidence = [self.ev("anomaly_" + k.lower(), "NEUTRAL", severity, "system:anomaly",
                                detail=str([x for x in anomalies if x["kind"] == k][0])[:160])
                        for k in kinds] or [self.ev("no_anomalies", "NEUTRAL", 0.05, "system:anomaly")]
        out.extra = {"anomalies": anomalies, "severity": round(severity, 3)}
        for a in anomalies:
            self.bus.publish(TOPIC_ANOMALY, {"agent": self.agent_id, **a})
        return out
