"""Synthetic multi-asset market simulator (clearly labelled MOCK data).

Purpose: let the whole terminal run offline for research/education/backtesting
WITHOUT pretending it is a real feed. Every record is tagged
``source='synthetic:*'`` and tier TIER_2 with an explicit note that this is
SIMULATED data. The generator produces internally consistent XAU/USD plus
correlated DXY / Treasury yields / silver / oil / copper / equities / VIX so
cross-asset and macro agents have meaningful input to work with.

Modeling notes
--------------
* Gold log-price follows a regime-switching drift + GARCH-like volatility.
* Real-yield proxy drives gold drift negatively (opportunity cost), DXY drift
  negatively but with time-varying correlation (the relationship is NOT assumed
  permanently stable anywhere downstream — agents measure it).
* Intraday session effects raise volatility in London/NY overlap.
"""
from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timedelta

import numpy as np

from gold_ai.core.enums import Frequency, SourceTier
from gold_ai.core.models import Candle, Quote, Tick
from gold_ai.core.timeutils import UTC, floor_to


class SyntheticMarketEngine:
    """Single correlated state process shared by all synthetic quotes."""

    def __init__(self, seed: int = 42, base_price: float = 4325.0) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.t = 0
        self.price = base_price
        self.base_price = base_price
        # latent factors
        self.drift = 0.0
        self.vol_state = 1.0
        self.dxy = 104.2
        self.y2 = 4.35
        self.y5 = 4.28
        self.y10 = 4.42
        self.y30 = 4.60
        self.tips10 = 2.10  # 10Y TIPS real yield proxy
        self.silver = 51.2
        self.oil = 71.4
        self.copper = 4.12
        self.spx = 6420.0
        self.vix = 15.8
        self.eur = 1.085
        self.jpy = 152.3
        # GARCH-ish persistence
        self._g_vol = 1.0
        self._regime_until = 0
        self._regime_drift = 0.0
        self.day_open = base_price

    # ------------------------------------------------------------------
    def _session_factor(self, dt: datetime) -> tuple[float, float]:
        h = dt.hour + dt.minute / 60.0
        # vol multiplier & liquidity by UTC hour
        if 7 <= h < 12:      # London
            return 1.15, 0.9
        if 12 <= h < 16:     # LDN/NY overlap
            return 1.5, 1.0
        if 16 <= h < 21:     # New York
            return 1.2, 0.85
        if 0 <= h < 6:       # Asia
            return 0.7, 0.5
        return 0.6, 0.35

    def step(self, dt: datetime) -> dict[str, float]:
        """Advance one minute of simulated time; return factor snapshot."""
        self.t += 1
        sess_vol, _liq = self._session_factor(dt)

        # occasionally switch drift regime (~ every 3-10 hours)
        if self.t > self._regime_until:
            self._regime_until = self.t + self.rng.randint(180, 600)
            self._regime_drift = self.rng.choice([0.6, -0.5, 0.25, -0.2, 0.0]) * 1e-4

        # GARCH-like vol
        shock = float(self.np_rng.normal(0, 1))
        self._g_vol = math.sqrt(max(0.05, 0.85 * self._g_vol**2 + 0.15 * shock**2))
        base_sigma = 0.00035 * sess_vol * self._g_vol
        self.vol_state = sess_vol * self._g_vol

        # macro co-moves -------------------------------------------------
        dy_shock = float(self.np_rng.normal(0, 0.0004))
        self.dxy *= math.exp(dy_shock)
        d_y10 = float(self.np_rng.normal(0, 0.006)) + (-dy_shock * 18.0)  # USD up => yields nudged up
        self.y10 = max(1.0, self.y10 + d_y10)
        self.y2 = max(0.5, self.y2 + d_y10 * 0.55 + float(self.np_rng.normal(0, 0.003)))
        self.y5 = max(0.8, (self.y2 + self.y10) / 2 + float(self.np_rng.normal(0, 0.002)))
        self.y30 = max(1.5, self.y10 + 0.18 + float(self.np_rng.normal(0, 0.002)))
        # breakeven drifts slowly; real yield = nominal - breakeven
        self._breakeven = getattr(self, "_breakeven", 2.32) + float(self.np_rng.normal(0, 0.0008))
        self.tips10 = self.y10 - self._breakeven

        # gold: drift + negative real-yield sensitivity + negative DXY beta + noise
        real_yield_pressure = -12.0 * (self.tips10 - 2.10) * 1e-4
        dxy_pressure = -6.0 * ((self.dxy / 104.2) - 1.0) * 1e-4
        mu = self._regime_drift + real_yield_pressure + dxy_pressure
        ret = mu + base_sigma * float(self.np_rng.normal(0, 1))
        prev = self.price
        self.price = max(500.0, self.price * math.exp(ret))

        # other assets loosely coupled
        self.silver *= math.exp(ret * 1.15 + float(self.np_rng.normal(0, 0.0004)))
        self.oil *= math.exp(float(self.np_rng.normal(0.0, 0.0006)) - dy_shock * 0.6)
        self.copper *= math.exp(float(self.np_rng.normal(0.0, 0.0005)) - dy_shock * 0.8)
        self.spx *= math.exp(float(self.np_rng.normal(0.00002, 0.0005)) + dy_shock * 0.4)
        self.vix = max(9.0, self.vix * math.exp(-float(self.np_rng.normal(0.0, 0.006)) - ret * 40))
        self.eur *= math.exp(-dy_shock * 0.9)
        self.jpy *= math.exp(-dy_shock * 0.5)

        o = prev
        c = self.price
        hi = max(o, c) * (1 + abs(float(self.np_rng.normal(0, base_sigma * 0.4))))
        lo = min(o, c) * (1 - abs(float(self.np_rng.normal(0, base_sigma * 0.4))))
        vol = float(self.np_rng.lognormal(mean=math.log(900), sigma=0.4)) * sess_vol
        return {
            "open": o, "high": hi, "low": lo, "close": c, "volume": vol,
            "ret": ret, "sigma": base_sigma,
        }

    def quote(self, sym: str, val: float, prev_val: float | None = None) -> Quote:
        now = datetime.now(tz=UTC)
        chg = 0.0 if prev_val is None else (val / prev_val - 1) * 100
        return Quote(
            source="synthetic:sim", tier=SourceTier.TIER_2, symbol=sym, frequency=Frequency.SECOND,
            retrieved_at=now, event_at=now, last=val, change_pct=chg,
            bid=val * 0.99995, ask=val * 1.00005, spread_bps=1.0,
            extra={"data_mode": "SIMULATED"},
        )

    def watchlist_quotes(self) -> list[Quote]:
        now = datetime.now(tz=UTC)
        out = []
        for sym, val, prev_key in (
            ("DXY", self.dxy, "dxy"), ("US2Y", self.y2, "y2"), ("US5Y", self.y5, "y5"),
            ("US10Y", self.y10, "y10"), ("US30Y", self.y30, "y30"), ("US10Y_REAL", self.tips10, "tips10"),
            ("XAG/USD", self.silver, "silver"), ("WTI", self.oil, "oil"), ("COPPER", self.copper, "copper"),
            ("SPX", self.spx, "spx"), ("VIX", self.vix, "vix"), ("EURUSD", self.eur, "eur"), ("USDJPY", self.jpy, "jpy"),
        ):
            q = Quote(
                source="synthetic:sim", tier=SourceTier.TIER_2, symbol=sym, frequency=Frequency.SECOND,
                retrieved_at=now, event_at=now, last=val, change_pct=float(self.np_rng.normal(0, 0.15)),
                extra={"data_mode": "SIMULATED"},
            )
            out.append(q)
        return out


def make_candle_from_step(engine: SyntheticMarketEngine, bar: dict, ts: datetime) -> Candle:
    return Candle(
        source="synthetic:sim", tier=SourceTier.TIER_2, symbol="XAU/USD",
        frequency=Frequency.MINUTE, retrieved_at=datetime.now(tz=UTC), event_at=ts, published_at=ts,
        open=bar["open"], high=bar["high"], low=bar["low"], close=bar["close"],
        volume=bar["volume"], timeframe="1m",
    )
