"""System orchestrator: wires providers → bus → agents → engines on separate
update loops (Section 17), with failure isolation and feed-health tracking.

Loops (agents are throttled individually by their own min_interval_s):
- market stream task : ticks + 1m candle production (FAST producers)
- FAST loop   0.5s   : spot/trend/momentum/PA/zones/volatility/anomaly/regime/data-integrity
- MEDIUM loop 2s     : usd/rates/real-yield/session/cross-asset/futures/options/ML
- SLOW loop   30s    : COT/ETF/CB/seasonality/inflation/labor/growth/calibration/drift
- EVENT loop  1s     : news/calendar/Fed/geo/news-intel (topic-triggered, dirty-flagged)
- decision cadence 3s: recompute final analysis; timeline entry only on material change

Expensive models are NOT recomputed every tick — the event-driven dirty/throttle
design in BaseAgent.maybe_run enforces that.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from gold_ai.agents.base import BaseAgent
from gold_ai.agents.data_integrity import DataIntegrityAgent
from gold_ai.agents.market_tech.technical_agents import (
    LiquidityZonesAgent, MomentumAgent, PriceActionAgent, SessionAgent,
    SpotMarketAgent, TrendAgent, VolatilityAgent,
)
from gold_ai.agents.macro.macro_agents import (
    EconomicCalendarAgent, FedPolicyAgent, GrowthAgent, InflationAgent,
    LaborMarketAgent, RealYieldAgent, TreasuryYieldAgent, UsdDxyAgent,
)
from gold_ai.agents.meta.anomaly_agent import AnomalyAgent
from gold_ai.agents.meta.ml_agents import (
    CalibrationAgent, DriftAgent, MLAgent, MLEnsemble, build_features,
)
from gold_ai.agents.meta.regime_agent import MarketRegimeAgent
from gold_ai.agents.risk_flows.flow_agents import (
    CentralBankDemandAgent, EtfFlowAgent, FuturesStructureAgent,
    OptionsIvAgent, PositioningAgent, SeasonalityAgent,
)
from gold_ai.agents.risk_flows.news_geo_agents import (
    CrossAssetAgent, GeopoliticalAgent, NewsIntelligenceAgent,
)
from gold_ai.core.enums import EventRiskState, Frequency, Horizon, SourceTier, UpdateLoop
from gold_ai.core.event_bus import TOPIC_CANDLE, TOPIC_MACRO, TOPIC_NEWS, TOPIC_QUOTE, EventBus
from gold_ai.core.models import DecisionPayload, Tick
from gold_ai.core.state import MarketState
from gold_ai.core.timeutils import utcnow
from gold_ai.data.providers.synthetic import (
    SyntheticCalendarProvider, SyntheticCentralBankProvider, SyntheticETFProvider,
    SyntheticFuturesProvider, SyntheticGeoProvider, SyntheticMacroProviders,
    SyntheticMarketProvider, SyntheticNewsProvider, SyntheticOptionsProvider,
    SyntheticPositioningProvider,
)
from gold_ai.data.providers.synthetic_engine import SyntheticMarketEngine, make_candle_from_step
from gold_ai.engine.decision.engine import DecisionEngine
from gold_ai.engine.evidence.conflict import ConflictDetector
from gold_ai.engine.scenario.event_risk import EventRiskEngine
from gold_ai.engine.weighting.dynamic_weights import WeightingEngine

log = logging.getLogger("gold_ai.orchestrator")


class Orchestrator:
    def __init__(self, seed: int = 42, history_minutes: int = 3000, speed: float = 60.0,
                 engine: Optional[SyntheticMarketEngine] = None) -> None:
        self.engine = engine or SyntheticMarketEngine(seed=seed)
        self.bus = EventBus()
        self.state = MarketState()
        self.speed = speed  # simulated minutes advanced per real second (research accelerator)

        # ---- providers (swap via config for real licensed feeds) ---------
        self.market_provider = SyntheticMarketProvider(self.engine)
        self.macro_holder = SyntheticMacroProviders(self.engine)
        self.news_provider = SyntheticNewsProvider()
        self.calendar_provider = SyntheticCalendarProvider()
        self.positioning_provider = SyntheticPositioningProvider()
        self.etf_provider = SyntheticETFProvider()
        self.futures_provider = SyntheticFuturesProvider(self.engine)
        self.options_provider = SyntheticOptionsProvider(self.engine)
        self.geo_provider = SyntheticGeoProvider()
        self.cb_provider = SyntheticCentralBankProvider()

        # ---- agents --------------------------------------------------------
        self.news_agent = NewsIntelligenceAgent(self.state, self.bus)
        self.ml_agent = MLAgent(self.state, self.bus, MLEnsemble())
        self.calibration_agent = CalibrationAgent(self.state, self.bus)
        self.drift_agent = DriftAgent(self.state, self.bus)
        agents: list[BaseAgent] = [
            DataIntegrityAgent(self.state, self.bus),
            SpotMarketAgent(self.state, self.bus),
            TrendAgent(self.state, self.bus),
            MomentumAgent(self.state, self.bus),
            PriceActionAgent(self.state, self.bus),
            LiquidityZonesAgent(self.state, self.bus),
            VolatilityAgent(self.state, self.bus),
            SessionAgent(self.state, self.bus),
            UsdDxyAgent(self.state, self.bus),
            TreasuryYieldAgent(self.state, self.bus),
            RealYieldAgent(self.state, self.bus),
            FedPolicyAgent(self.state, self.bus),
            EconomicCalendarAgent(self.state, self.bus),
            InflationAgent(self.state, self.bus),
            LaborMarketAgent(self.state, self.bus),
            GrowthAgent(self.state, self.bus),
            self.news_agent,
            GeopoliticalAgent(self.state, self.bus),
            CentralBankDemandAgent(self.state, self.bus),
            EtfFlowAgent(self.state, self.bus),
            PositioningAgent(self.state, self.bus),
            FuturesStructureAgent(self.state, self.bus),
            OptionsIvAgent(self.state, self.bus),
            CrossAssetAgent(self.state, self.bus),
            SeasonalityAgent(self.state, self.bus),
            AnomalyAgent(self.state, self.bus),
            MarketRegimeAgent(self.state, self.bus),
            self.ml_agent,
            self.calibration_agent,
            self.drift_agent,
        ]
        self.agents: dict[str, BaseAgent] = {a.agent_id: a for a in agents}
        for a in agents:
            a.attach()

        # ---- engines -------------------------------------------------------
        self.weighting = WeightingEngine()
        self.conflict_detector = ConflictDetector()
        self.event_risk = EventRiskEngine()
        self.decision_engine = DecisionEngine()
        self.horizon = Horizon.H1
        self.last_decision: Optional[DecisionPayload] = None
        self._decision_lock = asyncio.Lock()
        self._sig_state = ""
        self._sig_prob = 0.0

        self.history_minutes = history_minutes
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self.prediction_cb = None  # wired by paper tracker / DB writer

    # ------------------------------------------------------------------
    async def bootstrap_history(self) -> None:
        """Load SIMULATED history + slow periodic datasets once."""
        now = utcnow()
        start = now - timedelta(minutes=self.history_minutes)
        candles = await self.market_provider.fetch_candles("XAU/USD", "1m", start, self.history_minutes)
        for c in candles:
            self.state.add_candle(c)
        if candles:
            self.state.prev_close_24h = candles[max(0, len(candles) - 1440)].close
        for p in self.macro_holder.all_points():
            self.state.set_macro(p)
        self.state.scheduled_events = await self.calendar_provider.scheduled_events(
            now, now + timedelta(hours=96))
        self.state.positioning = await self.positioning_provider.latest_cot()
        self.state.etf_flows.extend(await self.etf_provider.latest_flows())
        self.state.futures = await self.futures_provider.structure()
        self.state.options = await self.options_provider.snapshot()
        self.state.geo_risks.extend(await self.geo_provider.active_risks(now))
        self.state.central_bank = await self.cb_provider.gold_demand(now)
        for art in await self.news_provider.fetch_recent(now):
            self.news_agent.on_article(art, now)
        log.info("bootstrap done: %d simulated 1m candles", len(candles))

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.bus.start()
        await self.bootstrap_history()
        self._tasks = [
            asyncio.create_task(self._market_stream(), name="market-stream"),
            asyncio.create_task(self._loop(UpdateLoop.FAST, 0.5), name="fast-loop"),
            asyncio.create_task(self._loop(UpdateLoop.MEDIUM, 2.0), name="medium-loop"),
            asyncio.create_task(self._loop(UpdateLoop.SLOW, 30.0), name="slow-loop"),
            asyncio.create_task(self._loop(UpdateLoop.EVENT_DRIVEN, 1.0), name="event-loop"),
            asyncio.create_task(self._macro_refresh(2.0), name="macro-refresh"),
            asyncio.create_task(self._news_stream(), name="news-stream"),
            asyncio.create_task(self._decision_cadence(3.0), name="decision-cadence"),
            asyncio.create_task(self._ml_training(), name="ml-train"),
        ]

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.bus.stop()

    # ------------------------------------------------------------------
    async def _market_stream(self) -> None:
        """Advance sim clock at ``speed``×, emit ticks + completed 1m candles."""
        try:
            while self._running:
                await asyncio.sleep(0.5)
                steps = max(1, int(self.speed * 0.5))  # sim-minutes per wall interval
                bar = None
                ts = utcnow()
                for i in range(steps):
                    bar = self.engine.step(ts)
                if bar is not None:
                    candle = make_candle_from_step(self.engine, bar, ts)
                    self.state.add_candle(candle)
                    self.bus.publish(TOPIC_CANDLE, candle)
                half = max(0.02, self.engine.price * 0.00004 * self.engine.vol_state)
                t = utcnow()
                tk = Tick(source="synthetic:sim", tier=SourceTier.TIER_2, symbol="XAU/USD",
                          frequency=Frequency.TICK, retrieved_at=t, event_at=t,
                          bid=self.engine.price - half, ask=self.engine.price + half,
                          last=self.engine.price)
                self.state.last_tick = tk
                self.state.last_price = self.engine.price
                self.bus.publish(TOPIC_QUOTE, tk)
                self.state.feed_health["market"] = {"status": "OK", "at": t.isoformat()}
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("market stream crashed — marking feed unhealthy")
            self.state.feed_health["market"] = {"status": "FAILED", "at": utcnow().isoformat()}

    async def _macro_refresh(self, interval_s: float) -> None:
        try:
            while self._running:
                await asyncio.sleep(interval_s)
                for p in self.macro_holder.all_points():
                    self.state.set_macro(p)
                    self.bus.publish(TOPIC_MACRO, p)
                for q in self.engine.watchlist_quotes():
                    self.state.set_quote(q)
                    self.bus.publish(TOPIC_QUOTE, q)
                self.state.feed_health["macro"] = {"status": "OK", "at": utcnow().isoformat()}
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("macro refresh crashed")
            self.state.feed_health["macro"] = {"status": "STALE", "at": utcnow().isoformat()}

    async def _news_stream(self) -> None:
        try:
            stream = self.news_provider.stream_new()
            while self._running:
                art = await stream.__anext__()
                processed = self.news_agent.on_article(art, utcnow())
                self.bus.publish(TOPIC_NEWS, processed)
                self.state.feed_health["news"] = {"status": "OK", "at": utcnow().isoformat()}
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("news stream crashed — NEWS OFFLINE")
            self.state.feed_health["news"] = {"status": "OFFLINE", "at": utcnow().isoformat()}

    async def _loop(self, loop: UpdateLoop, period_s: float) -> None:
        try:
            while self._running:
                await asyncio.sleep(period_s)
                now = utcnow()
                for agent in self.agents.values():
                    if agent.loop == loop:
                        try:
                            await agent.maybe_run(now)
                        except Exception:
                            log.exception("agent %s orchestration error", agent.agent_id)
        except asyncio.CancelledError:
            raise

    async def _ml_training(self) -> None:
        """Background walk-forward-ish training on as-of features from the
        simulated history (anti-leakage: labels strictly after features)."""
        try:
            while self._running and not self.ml_agent.ensemble.trained:
                await asyncio.sleep(5)
                await asyncio.get_event_loop().run_in_executor(None, self._train_now)
        except asyncio.CancelledError:
            raise

    def _train_now(self) -> None:
        from gold_ai.backtest.walk_forward import train_walk_forward
        res = train_walk_forward(self.state, self.ml_agent.ensemble)
        if res.get("trained"):
            self.ml_agent.set_training_results(res["accuracy_by_model"])
            log.info("ML ensemble trained: %s", res["accuracy_by_model"])

    # ------------------------------------------------------------------
    async def recompute_decision(self, force: bool = False) -> Optional[DecisionPayload]:
        async with self._decision_lock:
            now = utcnow()
            outputs = dict(self.state.agent_outputs)
            if not outputs:
                return None
            participating = list(outputs.values())
            conflict = self.conflict_detector.detect(participating, {})
            er_state = outputs.get("econ_calendar").extra.get("event_risk_state", "NORMAL") \
                if outputs.get("econ_calendar") else "NORMAL"
            weights_pre, reasons = self.weighting.compute(
                participating, self.state.regimes_active, self.horizon, er_state, now)
            er = self.event_risk.evaluate(self.state, now)
            cal = outputs.get("calibration")
            ece = cal.extra.get("ece") if cal and cal.extra.get("ready") else None
            payload = self.decision_engine.decide(
                outputs, weights_pre, reasons, self.state.regimes_active, er, conflict,
                self.state.last_price, self.horizon, now, calibration_ece=ece)
            self.last_decision = payload
            self.state.last_decision = payload
            sig = f"{payload.decision_state}:{payload.direction}"
            material = force or sig != self._sig_state or abs(payload.probability_up - self._sig_prob) > 0.03
            if material:
                entry = {
                    "ts": now.isoformat(), "decision": payload.decision_state,
                    "direction": payload.direction, "probability_up": payload.probability_up,
                    "probability_down": payload.probability_down,
                    "confidence": payload.confidence_score, "regime": payload.regime,
                    "conflict": payload.conflict_level, "trigger": self._detect_trigger(),
                }
                self.state.record_decision_change(entry)
                self._sig_state, self._sig_prob = sig, payload.probability_up
                if self.prediction_cb:
                    try:
                        self.prediction_cb(payload, now)
                    except Exception:
                        log.exception("prediction callback failed")
            self.bus.publish("engine.decision", payload)
            return payload

    def _detect_trigger(self) -> str:
        best, best_delta = "", 0.0
        for aid, o in self.state.agent_outputs.items():
            if o.updated_at and (utcnow() - o.updated_at).total_seconds() < 8 and abs(o.score) > best_delta:
                best, best_delta = aid, abs(o.score)
        return best or "periodic_recompute"

    async def _decision_cadence(self, period_s: float) -> None:
        try:
            while self._running:
                await asyncio.sleep(period_s)
                await self.recompute_decision()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("decision cadence crashed")
