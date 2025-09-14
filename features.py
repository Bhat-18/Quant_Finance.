import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "adj close" in out.columns and "close" not in out.columns:
        out["close"] = pd.to_numeric(out["adj close"], errors="coerce")
    for c in ("open","high","low","close","volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_index().dropna(how="all")

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_ohlcv(df)
    if "close" not in out.columns:
        raise ValueError("No 'close' column after normalization.")
    close = out["close"]

    out["ret_1"] = close.pct_change()
    out["log_ret_1"] = np.log1p(out["ret_1"])

    out["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    out["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    out["ema_10"] = EMAIndicator(close, window=10).ema_indicator()
    out["ema_20"] = EMAIndicator(close, window=20).ema_indicator()

    out["rsi_14"] = RSIIndicator(close, window=14).rsi()
    if {"high","low"}.issubset(out.columns):
        stoch = StochasticOscillator(
            high=out["high"], low=out["low"], close=close,
            window=14, smooth_window=3
        )
        out["stoch_k"] = stoch.stoch()
        out["stoch_d"] = stoch.stoch_signal()
    else:
        out["stoch_k"] = np.nan
        out["stoch_d"] = np.nan

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"]  = bb.bollinger_lband()
    denom = (out["bb_high"] - out["bb_low"]).replace(0, np.nan)
    out["bb_pct"]  = (close - out["bb_low"]) / denom

    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["close_sma10"] = close / out["sma_10"] - 1
    out["close_sma20"] = close / out["sma_20"] - 1

    return out.dropna().copy()

def build_latest_window(df_feat: pd.DataFrame, feature_cols, lookback: int) -> np.ndarray:
    arr = df_feat[feature_cols].tail(lookback).values.astype(np.float32)
    if arr.shape[0] < lookback:
        raise ValueError(f"Not enough rows ({arr.shape[0]}) for lookback {lookback}.")
    return arr[None, :, :]

def apply_saved_scaler(X_seq: np.ndarray, scaler_state: dict) -> np.ndarray:
    flat = X_seq.reshape(X_seq.shape[0], -1)
    mean = np.array(scaler_state["mean"])
    scale = np.array(scaler_state["scale"])
    flat_s = (flat - mean) / (scale + 1e-12)
    return flat_s.reshape(X_seq.shape)