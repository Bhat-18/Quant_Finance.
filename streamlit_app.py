import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# ---- Config ----
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8080")

st.set_page_config(page_title="ASX Signal Dashboard", layout="wide")
st.title("📈 ASX Real-time Signal Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base URL", API_BASE)
    ticker = st.text_input("Ticker", "BHP.AX")
    lookback = st.number_input("Lookback", min_value=20, max_value=200, value=60, step=5)
    threshold = st.number_input("Decision threshold", min_value=0.5, max_value=0.9, value=0.55, step=0.01, format="%.2f")
    days = st.number_input("Chart window (days)", min_value=60, max_value=2000, value=365, step=30)
    go_btn = st.button("🚀 Predict")

# Health check
try:
    resp = requests.get(f"{api_base}/health", timeout=10)
    resp.raise_for_status()
    health = resp.json()
    ok = (health.get("status") == "ok")
except Exception as e:
    ok = False
    health = {}
    st.error(f"Cannot reach API at {api_base}. Error: {e}")

if ok:
    st.success("API healthy ✅")
else:
    st.warning("API not healthy")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model loaded", str(health.get("model_loaded", False)))
c2.metric("#Features", health.get("n_features", "—"))
c3.metric("Lookback (default)", health.get("lookback", "—"))
c4.metric("AlphaVantage key", "set" if health.get("alpha_vantage_key_set") else "not set")

# Price chart
hist = requests.get(f"{api_base}/history", params={"ticker": ticker, "days": days}, timeout=30).json()
if "error" in hist:
    st.warning(f"History error: {hist['error']}")
    df_hist = pd.DataFrame()
else:
    df_hist = pd.DataFrame(hist["rows"])
    if not df_hist.empty:
        fig = px.line(df_hist, x="date", y="close", title=f"{ticker} — Close (last {days} days)")
        st.plotly_chart(fig, use_container_width=True)

# Predict
if go_btn:
    try:
        resp = requests.get(
            f"{api_base}/predict",
            params={"ticker": ticker, "lookback": lookback, "threshold": threshold},
            timeout=30,
        )
        if resp.status_code == 200:
            pred = resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ticker", pred["ticker"])
            c2.metric("As of", pred["asof"])
            c3.metric("P(up)", f"{pred['p_up']:.3f}")
            c4.metric("Go long?", "YES ✅" if pred["go_long"] else "NO ⛔")
            st.caption(f"n_features={pred['n_features']}, lookback={pred['lookback']}, threshold={pred['threshold']}")
        else:
            st.error(f"Prediction failed: {resp.status_code} — {resp.text}")
    except Exception as e:
        st.error(f"Prediction error: {e}")