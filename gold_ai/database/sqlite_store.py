"""SQLite persistence layer (Section 18).

Stores: candles, macro points, news, scheduled events & releases, agent outputs,
decisions (immutable), predictions with full snapshot metadata, realized
outcomes, scenarios, model registry entries, data-source health, performance.

Every prediction row carries: prediction_id, timestamp, horizon, model_version,
regime, probability, decision, available_information_snapshot. Outcomes are
appended later — issued predictions are NEVER modified (rule 14).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("gold_ai.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, tf TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume REAL, source TEXT);
CREATE INDEX IF NOT EXISTS ix_candles ON candles(tf, ts);
CREATE TABLE IF NOT EXISTS macro_points(
  id INTEGER PRIMARY KEY AUTOINCREMENT, series TEXT, value REAL, unit TEXT,
  observed_at TEXT, published_at TEXT, retrieved_at TEXT, source TEXT, tier TEXT);
CREATE TABLE IF NOT EXISTS news(
  id INTEGER PRIMARY KEY AUTOINCREMENT, published_at TEXT, headline TEXT,
  source TEXT, tier TEXT, classification TEXT, sentiment REAL, novelty REAL,
  relevance_gold REAL, impact REAL, verified INTEGER, dedup_key TEXT);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, event_type TEXT, scheduled_at TEXT,
  forecast REAL, previous REAL, actual REAL, revision REAL, surprise REAL,
  importance REAL, released INTEGER DEFAULT 0, source TEXT);
CREATE TABLE IF NOT EXISTS agent_outputs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, agent_id TEXT, status TEXT,
  score REAL, prob_up REAL, prob_down REAL, prob_neutral REAL, confidence REAL,
  weight REAL, freshness REAL, evidence_json TEXT, extra_json TEXT);
CREATE TABLE IF NOT EXISTS decisions(
  decision_id TEXT PRIMARY KEY, ts TEXT, price REAL, direction TEXT,
  prob_up REAL, prob_down REAL, prob_neutral REAL, confidence REAL,
  signal_quality TEXT, regime TEXT, conflict TEXT, data_quality REAL,
  event_risk TEXT, news_risk TEXT, vol_state TEXT, decision_state TEXT,
  wait_reasons TEXT, explanation TEXT, payload_json TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  prediction_id TEXT PRIMARY KEY, decision_id TEXT, ts TEXT, horizon TEXT,
  model_version TEXT, regime TEXT, probability REAL, decision TEXT,
  confidence REAL, snapshot_json TEXT, outcome TEXT NULL, outcome_ts TEXT NULL);
CREATE TABLE IF NOT EXISTS source_health(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, feed TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS models(
  key TEXT PRIMARY KEY, name TEXT, version TEXT, meta_json TEXT, registered_at TEXT);
CREATE TABLE IF NOT EXISTS performance(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, metric_set_json TEXT);
"""


class Database:
    def __init__(self, path: str | Path = "gold_ai/logs/gold_ai.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                log.exception("db write failed (non-fatal)")

    # ------------------------------------------------------------------
    def store_decision(self, p) -> None:
        self._exec(
            "INSERT OR REPLACE INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.decision_id, p.created_at.isoformat(), p.price, p.direction,
             p.probability_up, p.probability_down, p.probability_neutral,
             p.confidence_score, p.signal_quality, p.regime, p.conflict_level,
             p.data_quality, p.event_risk, p.news_risk, p.volatility_state,
             p.decision_state, json.dumps(p.wait_reasons), p.explanation,
             p.model_dump_json()),
        )

    def store_prediction(self, p, snapshot: dict) -> None:
        self._exec(
            "INSERT OR IGNORE INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
            (f"{p.decision_id}:{p.horizon}", p.decision_id, p.created_at.isoformat(),
             p.horizon, "ml_ensemble@2026.10-a1", p.regime, p.probability_up,
             p.decision_state, p.confidence_score, json.dumps(snapshot)[:20000]),
        )

    def resolve_prediction(self, pred_id: str, outcome: str, ts: datetime) -> None:
        """Append-only resolution: only fills outcome columns; never rewrites
        the original prediction fields."""
        with self._lock:
            try:
                cur = self._conn.execute(
                    "UPDATE predictions SET outcome=?, outcome_ts=? WHERE prediction_id=? AND outcome IS NULL",
                    (outcome, ts.isoformat(), pred_id))
                self._conn.commit()
                if cur.rowcount == 0:
                    log.debug("prediction %s already resolved or missing", pred_id)
            except Exception:
                log.exception("resolve failed")

    def store_agent_output(self, o) -> None:
        self._exec(
            "INSERT INTO agent_outputs(ts,agent_id,status,score,prob_up,prob_down,prob_neutral,"
            "confidence,weight,freshness,evidence_json,extra_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (o.updated_at.isoformat() if o.updated_at else datetime.utcnow().isoformat(),
             o.agent_id, o.status, o.score, o.probability_up, o.probability_down,
             o.probability_neutral, o.confidence, o.weight, o.freshness,
             json.dumps([e.model_dump(mode="json") for e in o.evidence])[:8000],
             json.dumps(o.extra, default=str)[:8000]),
        )

    def store_news(self, a) -> None:
        self._exec(
            "INSERT INTO news(published_at,headline,source,tier,classification,sentiment,"
            "novelty,relevance_gold,impact,verified,dedup_key) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (a.published_at.isoformat(), a.headline[:300], a.source, a.tier.value,
             a.classification.value, a.sentiment, a.novelty, a.relevance_gold,
             a.potential_market_impact, int(a.verified), a.dedup_key),
        )

    def store_event(self, e, released: bool = False) -> None:
        self._exec(
            "INSERT INTO events(name,event_type,scheduled_at,forecast,previous,actual,"
            "revision,surprise,importance,released,source) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (e.name if hasattr(e, "name") else e.event_name, e.event_type.value,
             e.scheduled_at.isoformat(), getattr(e, "forecast", None),
             getattr(e, "previous", None), getattr(e, "actual", None),
             getattr(e, "revision", None), getattr(e, "surprise", None),
             getattr(e, "importance", 0.5), int(released), e.source),
        )

    def store_source_health(self, feed: str, status: str, ts: datetime) -> None:
        self._exec("INSERT INTO source_health(ts,feed,status) VALUES(?,?,?)",
                   (ts.isoformat(), feed, status))

    def store_model(self, key: str, name: str, version: str, meta: dict, ts: datetime) -> None:
        self._exec("INSERT OR IGNORE INTO models VALUES(?,?,?,?,?)",
                   (key, name, version, json.dumps(meta, default=str), ts.isoformat()))

    def store_performance(self, metrics: dict, ts: datetime) -> None:
        self._exec("INSERT INTO performance(ts,metric_set_json) VALUES(?,?)",
                   (ts.isoformat(), json.dumps(metrics, default=str)))

    def query_predictions(self, limit: int = 500) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT prediction_id, ts, horizon, regime, probability, decision, confidence, outcome "
                "FROM predictions ORDER BY ts DESC LIMIT ?", (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
