"""Vectorized technical-indicator library used by the market/technical agents.

All functions operate on numpy arrays of closes/highs/lows/volumes aligned to
the same candle sequence. No look-ahead: value at index i uses data <= i only.
"""
from __future__ import annotations

import numpy as np


def ema(values: np.ndarray, span: int) -> np.ndarray:
    if len(values) == 0:
        return values
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def sma(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    csum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    return (csum[window:] - csum[:-window]) / window


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(values, prepend=values[0])
    gain = np.clip(deltas, 0, None)
    loss = -np.clip(deltas, None, 0)
    avg_gain = ema(gain, period)
    avg_loss = ema(loss, period)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss > 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def macd(values: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = ema(values, fast) - ema(values, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def roc(values: np.ndarray, period: int = 10) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if len(values) > period:
        out[period:] = (values[period:] / values[:-period] - 1.0) * 100.0
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return ema(tr, period)


def realized_vol(closes: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling std of log returns, annualization-free (per-bar vol)."""
    if len(closes) < 2:
        return np.array([])
    rets = np.diff(np.log(closes))
    n = len(rets)
    out = np.full(n, np.nan)
    if n >= window:
        roll = np.lib.stride_tricks.sliding_window_view(rets, window)
        out[window - 1:] = roll.std(axis=1, ddof=1)
    elif n > 5:
        out[5:] = np.array([rets[max(0, i - window):i + 1].std(ddof=1) for i in range(5, n)])
    return out


def bollinger(values: np.ndarray, window: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(values, window)
    if len(values) < window:
        std = np.full_like(values, np.nan)
    else:
        roll = np.lib.stride_tricks.sliding_window_view(values.astype(float), window)
        std = np.full(len(values), np.nan)
        std[window - 1:] = roll.std(axis=1, ddof=1)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def slope(values: np.ndarray, window: int = 20) -> float:
    """OLS slope over last ``window`` points, normalized by mean price (%/bar)."""
    v = values[-window:]
    if len(v) < 3:
        return 0.0
    x = np.arange(len(v), dtype=float)
    b = np.polyfit(x, v, 1)[0]
    return float(b / np.mean(v) * 100.0)


def find_swings(high: np.ndarray, low: np.ndarray, lookback: int = 3) -> tuple[list[int], list[int]]:
    """Fractal swing highs/lows: point higher/lower than `lookback` neighbors."""
    hh, ll = [], []
    n = len(high)
    for i in range(lookback, n - lookback):
        if high[i] == max(high[i - lookback:i + lookback + 1]):
            hh.append(i)
        if low[i] == min(low[i - lookback:i + lookback + 1]):
            ll.append(i)
    return hh, ll


def market_structure(high: np.ndarray, low: np.ndarray, lookback: int = 3) -> dict[str, object]:
    """HH/HL/LH/LL classification from the last four swings."""
    hh, ll = find_swings(high, low, lookback)
    res: dict[str, object] = {"pattern": "INSUFFICIENT_DATA", "hh_hl": False, "lh_ll": False,
                              "last_highs": hh[-3:], "last_lows": ll[-3:]}
    if len(hh) >= 2 and len(ll) >= 2:
        h_up = high[hh[-1]] > high[hh[-2]]
        l_up = low[ll[-1]] > low[ll[-2]]
        if h_up and l_up:
            res["pattern"] = "HH_HL_UPTREND"
            res["hh_hl"] = True
        elif (not h_up) and (not l_up):
            res["pattern"] = "LH_LL_DOWNTREND"
            res["lh_ll"] = True
        else:
            res["pattern"] = "MIXED_STRUCTURE"
    return res


def momentum_divergence(values: np.ndarray, osc: np.ndarray, window: int = 60) -> str:
    """Detect simple regular divergences over the last ``window`` bars.

    Returns 'BULLISH_DIV', 'BEARISH_DIV' or 'NONE'. Uses swing lows/highs of
    price vs extrema of the oscillator — context-aware, not rule-of-70s.
    """
    if len(values) < window or len(osc) < window:
        return "NONE"
    v = values[-window:]
    o = osc[-window:]
    # bearish: higher price high but lower oscillator high
    i_ph1 = int(np.argmax(v[: window // 2]))
    i_ph2 = int(np.argmax(v[window // 2:])) + window // 2
    if v[i_ph2] > v[i_ph1] * 1.0005 and o[i_ph2] < o[i_ph1] - 1.0:
        return "BEARISH_DIV"
    i_pl1 = int(np.argmin(v[: window // 2]))
    i_pl2 = int(np.argmin(v[window // 2:])) + window // 2
    if v[i_pl2] < v[i_pl1] * 0.9995 and o[i_pl2] > o[i_pl1] + 1.0:
        return "BULLISH_DIV"
    return "NONE"


def zscore(values: np.ndarray, window: int = 120) -> float:
    v = np.asarray(values, dtype=float)
    if len(v) < 8:
        return 0.0
    tail = v[-window:] if len(v) >= window else v
    mu, sd = float(np.mean(tail)), float(np.std(tail, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float((v[-1] - mu) / sd)
