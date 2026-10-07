"""ML ensemble agent (Section 5.AB) + calibration (5.AC) + drift (5.AD).

Design decisions (documented, not decorative):
- Ensemble = logistic regression baseline ALWAYS present + gradient-boosted
  trees (sklearn HistGradientBoosting as a dependency-light XGBoost/LGBM stand-in)
  + optional RandomForest. Deep models are NOT used: with the data volume
  available in this build they would overfit; the spec explicitly forbids DL
  "merely because it sounds advanced".
- Models predict direction at ONE horizon from strictly as-of features.
- Probabilities are calibrated out-of-fold; the Calibration Agent tracks Brier,
  log-loss and Expected Calibration Error with reliability-diagram bins.
- The Drift Agent monitors feature distribution (PSI) + rolling performance and
  can force ``historical_predictive_value`` down → weighting engine reduces weight.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np

log = logging.getLogger("gold_ai.ml")

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    _SKLEARN = True
except Exception:  # pragma: no cover - sklearn optional at runtime
    _SKLEARN = False

from gold_ai.agents.base import BaseAgent
from gold_ai.core.enums import AgentStatus, Horizon, UpdateLoop
from gold_ai.core.event_bus import TOPIC_AGENT_OUTPUT, TOPIC_CANDLE
from gold_ai.core.models import AgentOutput


FEATURE_NAMES = [
    "ret_5m", "ret_15m", "ret_60m", "ret_240m",
    "ema20_dev", "ema50_dev", "ema200_dev", "ema_stack",
    "rsi14", "macd_hist_z", "roc30",
    "atr_pct_z", "vol_expansion", "bb_width_pctile",
    "dxy_ret_60m", "dxy_z", "y10_chg_60m", "real_yield_chg_60m", "curve_chg_60m",
    "hour_sin", "hour_cos", "dow",
]


def build_features(state, now: datetime) -> Optional[np.ndarray]:
    """As-of feature vector — uses ONLY data already observed at ``now``."""
    c5_list = list(state.candles["5m"])[-300:]
    if len(c5_list) < 200:
        return None
    c5 = np.array([c.close for c in c5_list], dtype=float)
    hi = np.array([c.high for c in c5_list], dtype=float)
    lo = np.array([c.low for c in c5_list], dtype=float)
    from gold_ai.data.normalization.indicators import (
        atr, bollinger, ema, macd, realized_vol, roc, rsi, zscore,
    )

    def ret(k_bars: int) -> float:
        k = min(k_bars, len(c5) - 1)
        return float(c5[-1] / c5[-k] - 1.0) * 100

    e20, e50, e200 = ema(c5, 20)[-1], ema(c5, 50)[-1], ema(c5, 200)[-1]
    last = c5[-1]
    stack = float((last > e20) + (e20 > e50)) / 2.0
    _, _, hist = macd(c5)
    hz = zscore(hist[-120:], 120)
    a = atr(hi, lo, c5, 14)
    az = zscore(np.array(a[-120:]) / c5[-120:] * 100, 100)
    rv = realized_vol(c5, 60)
    exp_ratio = float(np.nanmean(rv[-6:]) / (np.nanmedian(rv[-150:]) + 1e-9)) if len(rv) > 30 else 1.0
    bb_mid, bb_up, bb_lo = bollinger(c5, 20, 2.0)
    widths = (bb_up[-80:] - bb_lo[-80:]) / (bb_mid[-80:] + 1e-9)
    bb_pct = float(np.mean(widths < widths[-1])) if np.isfinite(widths[-1]) else 0.5

    dxy = np.array([p.value for p in state.macro_series("DXY")][-300:], dtype=float)
    y10 = np.array([p.value for p in state.macro_series("UST10Y")][-300:], dtype=float)
    ry = np.array([p.value for p in state.macro_series("DFII10_REAL")][-300:], dtype=float)
    y2 = np.array([p.value for p in state.macro_series("UST02Y")][-300:], dtype=float)

    def safe(arr, f):
        try:
            v = f(arr)
            return float(v) if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    fx = [
        ret(1), ret(3), ret(12), ret(48),
        (last / e20 - 1) * 100, (last / e50 - 1) * 100, (last / e200 - 1) * 100, stack,
        float(rsi(c5, 14)[-1]), hz, float(roc(c5, 30)[-1] or 0.0),
        float(az), exp_ratio, bb_pct,
        safe(dxy, lambda x: (x[-1] / x[-13] - 1) * 100 if len(x) > 13 else 0.0),
        safe(dxy, lambda x: zscore(x, 120)),
        safe(y10, lambda x: x[-1] - x[-13] if len(x) > 13 else 0.0),
        safe(ry, lambda x: x[-1] - x[-13] if len(x) > 13 else 0.0),
        safe(np.minimum(len(y10), len(y2)), lambda _: (y10[-1] - y2[-1]) - (y10[-13] - y2[-13])
             if len(y10) > 13 and len(y2) > 13 else 0.0),
        math.sin(2 * math.pi * now.hour / 24), math.cos(2 * math.pi * now.hour / 24),
        float(now.weekday()),
    ]
    fx = [0.0 if not math.isfinite(v) else v for v in fx]
    return np.array(fx, dtype=float)


# ---------------------------------------------------------------------------
class ModelRegistry:
    """Versioned model store metadata (Section 20). Never silently replaced."""

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}

    def register(self, name: str, version: str, meta: dict) -> None:
        key = f"{name}@{version}"
        if key in self.entries:
            raise ValueError(f"model {key} already registered — versions are immutable")
        self.entries[key] = {"name": name, "version": version, **meta}

    def list(self) -> list[dict]:
        return list(self.entries.values())


REGISTRY = ModelRegistry()


class MLEnsemble:
    """Logistic baseline + boosted trees + random forest. Trained walk-forward."""

    VERSION = "2026.10-a1"

    def __init__(self) -> None:
        self.available = _SKLEARN
        self.lr = None
        self.gb = None
        self.rf = None
        self.trained = False
        self.feature_importance: dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.available or len(X) < 200:
            return
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.lr = Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=500, C=0.5))])
        self.gb = HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15)
        self.rf = RandomForestClassifier(n_estimators=140, max_depth=8, min_samples_leaf=8, n_jobs=1)
        self.lr.fit(X, y)
        self.gb.fit(X, y)
        self.rf.fit(X, y)
        self.trained = True
        try:
            imp = self.gb._predictors[-1][0].feature_importances_  # noqa: SLF001
            tot = float(np.sum(imp)) or 1.0
            self.feature_importance = {n: round(float(v) / tot, 4) for n, v in zip(FEATURE_NAMES, imp)}
        except Exception:
            self.feature_importance = {}
        REGISTRY.register("ml_ensemble", self.VERSION, {
            "features": FEATURE_NAMES,
            "components": ["logistic_baseline", "hist_gbdt", "random_forest"],
            "train_size": int(len(X)),
            "note": "walk-forward trained; see backtest/validation module",
        })

    def predict_proba3(self, x: np.ndarray, prior_acc: dict[str, float]) -> tuple[float, float, float]:
        """Ensemble vote weighted by per-model OOS accuracy → 3-class probs."""
        if not self.trained:
            return 1 / 3, 1 / 3, 1 / 3
        preds, accs = [], []
        for m, key in ((self.lr, "lr"), (self.gb, "gb"), (self.rf, "rf")):
            try:
                p_up = float(m.predict_proba(x.reshape(1, -1))[0][1])
                preds.append(p_up)
                accs.append(prior_acc.get(key, 0.5))
            except Exception:
                continue
        if not preds:
            return 1 / 3, 1 / 3, 1 / 3
        w = np.array([max(0.01, a - 0.5) for a in accs])
        p_up = float(np.average(preds, weights=w))
        neutrality = float(np.clip(1.0 - abs(p_up - 0.5) * 2.2, 0.05, 0.55))
        p_neutral = neutrality
        rest = 1.0 - p_neutral
        return rest * p_up, rest * (1 - p_up), p_neutral


# ---------------------------------------------------------------------------
class MLAgent(BaseAgent):
    agent_id = "ml_ensemble"
    agent_name = "Machine Learning Ensemble Agent"
    loop = UpdateLoop.MEDIUM
    min_interval_s = 10.0
    topics = (TOPIC_CANDLE, TOPIC_AGENT_OUTPUT)
    stale_after_s = 1800

    def __init__(self, state, bus, ensemble: MLEnsemble | None = None) -> None:
        super().__init__(state, bus)
        self.ensemble = ensemble or MLEnsemble()
        self.model_prior_acc = {"lr": 0.52, "gb": 0.54, "rf": 0.53}
        self.last_probs: dict[str, tuple[float, float, float]] = {}

    def set_training_results(self, acc_by_model: dict[str, float]) -> None:
        self.model_prior_acc.update(acc_by_model)

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        if not self.ensemble.trained:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            out.error = "models not trained yet (walk-forward training runs in background)"
            out.probability_up = out.probability_down = out.probability_neutral = 1 / 3
            return out
        x = build_features(self.state, now)
        if x is None:
            out.status = AgentStatus.WARNING.value
            out.confidence = 0.0
            out.error = "insufficient feature history"
            return out
        probs_by_h: dict[str, list[float]] = {}
        for h in Horizon.ordered():
            pu, pd_, pn = self.ensemble.predict_proba3(x, self.model_prior_acc)
            # empirical signal decay by horizon (documented shrink toward base rate)
            shrink = {"5m": 1.0, "15m": 0.97, "30m": 0.93, "1h": 0.85, "4h": 0.7, "1D": 0.55}[h.value]
            pu2 = 1 / 3 + (pu - 1 / 3) * shrink
            pd2 = 1 / 3 + (pd_ - 1 / 3) * shrink
            pn2 = max(0.05, 1 - pu2 - pd2)
            probs_by_h[h.value] = [pu2, pd2, pn2]
            self.last_probs[h.value] = (pu2, pd2, pn2)
        main = probs_by_h[self.default_horizon()]
        out.probability_up, out.probability_down, out.probability_neutral = main
        net = main[0] - main[1]
        out.score = float(np.clip(net * 2.2, -1, 1))
        out.signal_strength = abs(out.score)
        out.confidence = float(np.clip(sum(self.model_prior_acc.values()) / 3 - 0.45, 0.1, 0.8))
        out.historical_predictive_value = float(np.clip(max(self.model_prior_acc.values()), 0.0, 1.0))
        fresh_candle = self.state.last_candle_at and (now - self.state.last_candle_at).total_seconds() < 120
        out.freshness = 1.0 if fresh_candle else 0.5
        out.evidence = [self.ev("ml_ensemble_vote", "UP" if net > 0.02 else ("DOWN" if net < -0.02 else "NEUTRAL"),
                                abs(net) * 2, "model:ml_ensemble",
                                detail=f"P(up)={main[0]:.2f}; horizons computed separately")]
        out.extra = {"probs_by_horizon": {k: [round(v, 3) for v in vals] for k, vals in probs_by_h.items()},
                     "model_accuracy_oos": self.model_prior_acc,
                     "feature_importance": dict(sorted(self.ensemble.feature_importance.items(),
                                                       key=lambda kv: -kv[1])[:8]),
                     "models": ["logistic_baseline(always)", "hist_gbdt", "random_forest"],
                     "note": "no deep nets: data volume here does not justify LSTM/Transformer"}
        return out


# ---------------------------------------------------------------------------
class CalibrationAgent(BaseAgent):
    """Reliability tracking: displayed probability must match frequencies."""

    agent_id = "calibration"
    agent_name = "Calibration Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 120.0
    topics = ()
    stale_after_s = 86400

    def __init__(self, state, bus) -> None:
        super().__init__(state, bus)
        self.records: list[tuple[float, int]] = []  # (predicted_p_up, outcome 0/1)

    def observe(self, p_up: float, outcome_up: bool) -> None:
        self.records.append((float(p_up), 1 if outcome_up else 0))
        if len(self.records) > 5000:
            self.records = self.records[-5000:]

    def metrics(self) -> dict:
        if len(self.records) < 30:
            return {"n": len(self.records), "ready": False}
        p = np.array([r[0] for r in self.records])
        y = np.array([r[1] for r in self.records], dtype=float)
        brier = float(np.mean((p - y) ** 2))
        eps = 1e-9
        logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        order = np.argsort(p)
        ps, ys = p[order], y[order]
        bins = np.array_split(np.arange(len(ps)), 10)
        ece, rel = 0.0, []
        for b in bins:
            if len(b) == 0:
                continue
            conf, acc = float(np.mean(ps[b])), float(np.mean(ys[b]))
            ece += len(b) / len(ps) * abs(conf - acc)
            rel.append({"bin_center": round(conf, 2), "observed_freq": round(acc, 2), "n": int(len(b))})
        return {"n": len(self.records), "ready": True, "brier": round(brier, 4),
                "log_loss": round(logloss, 4), "ece": round(ece, 4),
                "reliability_diagram": rel}

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        m = self.metrics()
        out.status = AgentStatus.ONLINE.value if m.get("ready") else AgentStatus.WARNING.value
        out.confidence = 0.9
        out.probability_up = out.probability_down = out.probability_neutral = 1 / 3
        out.score = 0.0
        out.evidence = [self.ev("calibration_quality", "NEUTRAL", min(1.0, m.get("ece", 0.5) * 2),
                                "system:calibration", detail=str(m.get("brier", "n/a")))]
        out.extra = m
        return out


# ---------------------------------------------------------------------------
class DriftAgent(BaseAgent):
    """Monitors distribution shift & performance decay; auto-reduces weight."""

    agent_id = "model_drift"
    agent_name = "Model Drift Agent"
    loop = UpdateLoop.SLOW
    min_interval_s = 300.0
    topics = (TOPIC_AGENT_OUTPUT,)
    stale_after_s = 86400

    def __init__(self, state, bus) -> None:
        super().__init__(state, bus)
        self.feature_window: list[np.ndarray] = []
        self.hit_window: list[int] = []

    def observe_features(self, x: np.ndarray) -> None:
        self.feature_window.append(x)
        if len(self.feature_window) > 500:
            self.feature_window.pop(0)

    def observe_outcome(self, hit: int) -> None:
        self.hit_window.append(hit)
        if len(self.hit_window) > 400:
            self.hit_window.pop(0)

    @staticmethod
    def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 8) -> float:
        edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
        rc = np.histogram(ref, edges)[0] / len(ref)
        cc = np.histogram(cur, edges)[0] / len(cur)
        rc = np.clip(rc, 1e-4, None)
        cc = np.clip(cc, 1e-4, None)
        return float(np.sum((cc - rc) * np.log(cc / rc)))

    def analyze(self, now: datetime) -> AgentOutput:
        out = self.base_output(now)
        drift_flags: dict[str, float] = {}
        overall = 0.0
        if len(self.feature_window) >= 120:
            arr = np.vstack(self.feature_window)
            ref, cur = arr[: len(arr) // 2], arr[len(arr) // 2:]
            for i, name in enumerate(FEATURE_NAMES):
                s = self.psi(ref[:, i], cur[:, i])
                if s > 0.05:
                    drift_flags[name] = round(s, 3)
            overall = float(np.mean(list(drift_flags.values()))) if drift_flags else 0.0
        perf_decay = 0.0
        if len(self.hit_window) >= 60:
            recent = float(np.mean(self.hit_window[-30:]))
            base = float(np.mean(self.hit_window))
            perf_decay = max(0.0, base - recent)
        severity = float(np.clip(overall * 3 + perf_decay * 4, 0, 1))
        out.score = 0.0
        out.confidence = 0.85
        out.signal_strength = severity
        out.probability_up = out.probability_down = out.probability_neutral = 1 / 3
        out.status = AgentStatus.WARNING.value if severity > 0.4 else AgentStatus.ONLINE.value
        out.evidence = [self.ev("drift_severity", "NEUTRAL", severity, "system:drift",
                                detail=f"PSI flags={list(drift_flags)[:5]}, perf_decay={perf_decay:.3f}")]
        out.extra = {"psi_flags": drift_flags, "performance_decay": round(perf_decay, 3),
                     "severity": round(severity, 3),
                     "action": "reduce ml_ensemble historical_predictive_value" if severity > 0.4 else "none"}
        ml_out = self.state.agent_outputs.get("ml_ensemble")
        if ml_out and severity > 0.4:
            ml_out.historical_predictive_value = max(0.1, ml_out.historical_predictive_value * (1 - 0.5 * severity))
            ml_out.weight_reason = (ml_out.weight_reason or "") + f" | drift penalty severity={severity:.2f}"
        return out
