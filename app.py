import os
import json
import traceback
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow import keras

from features import (
    add_technical_indicators,
    build_latest_window,
    apply_saved_scaler,
)

# ---------------------------------------------------------------------
# Config via env vars
# ---------------------------------------------------------------------
ARTIFACT_DIR   = os.getenv("ARTIFACT_DIR", "./artifacts")
MODEL_PATH     = os.path.join(ARTIFACT_DIR, "model_tf_cnn1d.keras")
SCALER_PATH    = os.path.join(ARTIFACT_DIR, "scaler.json")
FEATURES_PATH  = os.path.join(ARTIFACT_DIR, "features.csv")

LOOKBACK       = int(os.getenv("LOOKBACK", "60"))
THRESHOLD      = float(os.getenv("THRESHOLD", "0.55"))
PRED_HORIZON   = int(os.getenv("PRED_HORIZON", "1"))
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "CBA.AX")
YF_START       = os.getenv("YF_START", "2010-01-01")

# Optional Alpha Vantage key for ASX fallback
ALPHA_KEY      = os.getenv("ALPHAVANTAGE_API_KEY", "")

# ---------------------------------------------------------------------
# Load artifacts on boot
# ---------------------------------------------------------------------
with open(SCALER_PATH) as f:
    SCALER_STATE = json.load(f)

FEATURES = pd.read_csv(FEATURES_PATH, header=None)[0].tolist()
MODEL = keras.models.load_model(MODEL_PATH)

app = FastAPI(title="ASX Real-time Signal API", version="1.0.0")
logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------
# Data fetchers: yfinance -> Alpha Vantage (CSV) -> Stooq (CSV)
# ---------------------------------------------------------------------
def _alpha_vantage_csv(ticker: str, start="2010-01-01") -> pd.DataFrame:
    """
    Fetch daily adjusted data from Alpha Vantage as CSV (no extra deps).
    Works well for ASX tickers like 'BHP.AX'. Needs ALPHAVANTAGE_API_KEY.
    """
    if not ALPHA_KEY:
        return pd.DataFrame()
    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={ticker}"
        f"&apikey={ALPHA_KEY}"
        "&datatype=csv&outputsize=full"
    )
    try:
        df = pd.read_csv(url)
        if "timestamp" not in df.columns:
            return pd.DataFrame()
        df = df.rename(columns=str.lower).rename(columns={"timestamp": "date", "adjusted_close": "adj close"})
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        # use adjusted close as close (similar to yfinance auto_adjust=True)
        if "adj close" in df.columns and "close" not in df.columns:
            df["close"] = df["adj close"]
        df = df.sort_index()
        return df.loc[df.index >= pd.to_datetime(start)]
    except Exception:
        return pd.DataFrame()


def _stooq_csv(ticker: str, start="2010-01-01") -> pd.DataFrame:
    """
    Fetch from Stooq CSV. Coverage for US is good; ASX coverage is limited,
    but this provides a last-resort fallback without API keys.
    """
    sym = ticker.lower()
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        df = pd.read_csv(url)
        if "Date" not in df.columns:
            return pd.DataFrame()
        df = df.rename(columns=str.lower).rename(columns={"date": "date"})
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df.loc[df.index >= pd.to_datetime(start)]
    except Exception:
        return pd.DataFrame()


def fetch_history(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    """
    Try yfinance -> Alpha Vantage -> Stooq and return standardized OHLCV.
    """
    # 1) yfinance (may be blocked on some networks)
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        df = df.rename(columns=str.lower).dropna(how="all")
        if not df.empty:
            return df
    except Exception:
        pass

    # 2) Alpha Vantage CSV (requires free API key)
    df = _alpha_vantage_csv(ticker, start=start)
    if not df.empty:
        return df

    # 3) Stooq CSV
    df = _stooq_csv(ticker, start=start)
    if not df.empty:
        return df

    # Nothing worked
    return pd.DataFrame()


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class PredictResponse(BaseModel):
    ticker: str
    asof: str
    p_up: float
    threshold: float
    go_long: bool
    lookback: int
    horizon: int
    n_features: int


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": len(FEATURES),
        "lookback": LOOKBACK,
        "alpha_vantage_key_set": bool(ALPHA_KEY),
    }


@app.get("/predict", response_model=PredictResponse)
def predict(
    ticker: str = Query(DEFAULT_TICKER, description="Ticker (e.g., CBA.AX, BHP.AX, AAPL)"),
    threshold: Optional[float] = Query(None, description="Override decision threshold"),
    lookback: Optional[int] = Query(None, description="Override lookback window"),
    verbose: int = Query(0, description="Set to 1 to return error details on failure"),
):
    try:
        th = threshold or THRESHOLD
        lb = lookback or LOOKBACK

        # Fetch data with fallbacks
        df = fetch_history(ticker, start=YF_START)
        if df.empty:
            msg = (
                f"No data returned for {ticker}. "
                "If you're on a restricted network, set ALPHAVANTAGE_API_KEY and try an ASX ticker like BHP.AX."
            )
            if verbose:
                return JSONResponse(status_code=500, content={"error": msg})
            raise HTTPException(status_code=500, detail=msg)

        # Feature engineering
        df_feat = add_technical_indicators(df)

        # Ensure all training features exist now
        have = set(df_feat.columns)
        need = list(FEATURES)
        missing = [f for f in need if f not in have]
        if missing:
            msg = (
                f"Missing features at inference: {missing[:15]} (showing up to 15). "
                "Make sure artifacts/features.csv matches the features created by features.py."
            )
            if verbose:
                return JSONResponse(status_code=500, content={"error": msg})
            raise HTTPException(status_code=500, detail=msg)

        # Build window, scale, predict
        X = build_latest_window(df_feat, FEATURES, lb)
        Xs = apply_saved_scaler(X, SCALER_STATE)
        p_up = float(MODEL.predict(Xs, verbose=0).ravel()[0])
        go_long = bool(p_up >= th)
        asof = df_feat.index[-1].strftime("%Y-%m-%d")

        return PredictResponse(
            ticker=ticker,
            asof=asof,
            p_up=round(p_up, 6),
            threshold=th,
            go_long=go_long,
            lookback=lb,
            horizon=PRED_HORIZON,
            n_features=len(FEATURES),
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        if verbose:
            return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/columns")
def debug_columns(ticker: str = Query(DEFAULT_TICKER)):
    """
    Compare the columns produced now by features.py vs. the training-time features list.
    """
    df = fetch_history(ticker, start=YF_START)
    if df.empty:
        return {"error": f"No data for {ticker} via any source."}
    df_feat = add_technical_indicators(df)
    return {
        "n_df_feat_cols": len(df_feat.columns),
        "df_feat_cols_sample": list(df_feat.columns)[:30],
        "n_artifact_features": len(FEATURES),
        "features_from_artifacts_sample": FEATURES[:30],
    }

@app.get("/history")
def history(
    ticker: str = Query(DEFAULT_TICKER),
    days: int = Query(365, ge=30, le=5000, description="How many latest days to return")
):
    """
    Return recent daily OHLCV for plotting in Streamlit.
    Uses the same fetch_history() with yfinance/AlphaVantage/Stooq fallback.
    """
    df = fetch_history(ticker, start=YF_START)
    if df.empty:
        return {"error": f"No data for {ticker}"}
    out = df.tail(days).copy()
    out = out.reset_index().rename(columns={"index": "date"})
    # keep a compact payload
    cols = [c for c in ["date","open","high","low","close","volume","adj close"] if c in out.columns]
    out[cols[0]] = out[cols[0]].astype(str)  # date -> str for JSON
    return {"ticker": ticker, "rows": out[cols].to_dict(orient="records")}


@app.get("/")
def root():
    return {
        "msg": "ASX Real-time Signal API is running 🚀",
        "try": ["/health", f"/predict?ticker={DEFAULT_TICKER}"],
        "notes": "If yfinance is blocked on your network, set ALPHAVANTAGE_API_KEY and try an ASX ticker like BHP.AX.",
    }