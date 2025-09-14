# Quant_Finance.
Real-time ASX stock signal prediction API and dashboard built with FastAPI, TensorFlow, and Streamlit. Fetches live market data (AlphaVantage/Yahoo Finance), applies feature engineering with technical indicators, and serves ML-powered predictions via a REST API and interactive dashboard.

# ASX Real-time Signal Prediction

**Short Summary**  
This project is a complete **end-to-end machine learning pipeline** for **real-time ASX stock prediction**.  
It fetches live stock data, applies **feature engineering with technical indicators**, runs a trained **TensorFlow CNN model**, and serves predictions through a **FastAPI backend** and an interactive **Streamlit dashboard**.

---

## 🚀 Features
-  Fetches live stock data (Yahoo Finance, Alpha Vantage, Stooq fallback)  
-  Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, log returns  
-  Pre-trained **TensorFlow/Keras CNN model** for price movement prediction  
-  REST API with **FastAPI** (`/health`, `/predict`, `/history`)  
-  Interactive **Streamlit dashboard** with Plotly charts  
-  Modular structure: API, features, artifacts, and frontend separated  

---

##  Tech Stack
- **Backend**: FastAPI, Uvicorn  
- **Frontend**: Streamlit, Plotly  
- **ML/DL**: TensorFlow, Keras, scikit-learn  
- **Data**: Yahoo Finance, Alpha Vantage, Stooq  
- **Python Libraries**: Pandas, NumPy, Requests  

asx-realtime/
│── app.py              # FastAPI backend (API endpoints)
│── features.py         # Feature engineering & preprocessing
│── streamlit_app.py    # Streamlit frontend dashboard
│── requirements.txt    # Project dependencies
│── artifacts/          # Saved ML model, config, scaler
│── Untitled-1.ipynb    # Training notebook (TensorFlow CNN)

API Endpoints
	•	GET /health → API & model status
	•	GET /predict?ticker=BHP.AX → Predict next-day signal for ticker
	•	GET /history?ticker=BHP.AX&days=365 → Get historical OHLCV data for plotting

Example Workflow
	1.	User enters ticker (e.g., BHP.AX) in the Streamlit dashboard.
	2.	Dashboard requests /history and /predict from the FastAPI backend.
	3.	API fetches stock data (via Yahoo/AlphaVantage/Stooq), applies features, and runs ML model.
	4.	Dashboard displays price chart, P(up) probability, and Go Long / No Go signal.

⸻
Developed by Bhanuprakash Bhat

Built with FastAPI + TensorFlow + Streamlit for real-time financial signal prediction.
