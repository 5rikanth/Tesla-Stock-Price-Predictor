import os
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Tesla Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

WINDOW_SIZE  = 60
DEFAULT_CSV  = "TSLA.csv"   # Auto-loaded if found in the same folder as app.py

# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():
    rnn  = load_model("tesla_simplernn_model.keras")
    gru  = load_model("tesla_gru_model.keras")
    lstm = load_model("tesla_lstm_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return rnn, gru, lstm, scaler

rnn_model, gru_model, lstm_model, scaler = load_models()

# ==================================================
# TITLE
# ==================================================

st.title("📈 Tesla Stock Price Predictor")
st.markdown("""
Deep Learning based stock forecasting using **SimpleRNN**, **GRU**, and **LSTM**.
""")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("⚙️ Settings")

# Only show uploader if TSLA.csv is NOT found automatically
if os.path.exists(DEFAULT_CSV):
    st.sidebar.success(f"✅ Auto-loaded: {DEFAULT_CSV}")
    uploaded_file = None   # Not needed — we use the local file
else:
    st.sidebar.info("TSLA.csv not found in app folder. Please upload it.")
    uploaded_file = st.sidebar.file_uploader("Upload TSLA.csv", type="csv")

horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    [1, 5, 10],
    index=0
)

model_choice = st.sidebar.radio(
    "Model to Display",
    ["All Three", "LSTM", "GRU", "SimpleRNN"]
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    return df

try:
    if os.path.exists(DEFAULT_CSV):
        df = load_csv(DEFAULT_CSV)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
    else:
        st.info("👈 Place TSLA.csv in the same folder as app.py, or upload it from the sidebar.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ==================================================
# DATASET OVERVIEW
# ==================================================

latest_close = float(df["Close"].iloc[-1])
prev_close   = float(df["Close"].iloc[-2])
change_pct   = ((latest_close - prev_close) / prev_close) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Latest Close",  f"${latest_close:.2f}", f"{change_pct:+.2f}%")
with col2:
    st.metric("All-Time High", f"${df['Close'].max():.2f}")
with col3:
    st.metric("All-Time Low",  f"${df['Close'].min():.2f}")
with col4:
    st.metric("Total Records", f"{len(df):,}")

# ==================================================
# HISTORICAL CHART
# ==================================================

st.subheader("📈 Historical Closing Price")

df["MA_30"] = df["Close"].rolling(30).mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df["Close"], linewidth=1.2, color="#2196F3", label="Close")
ax.plot(df.index, df["MA_30"], linewidth=1.5, color="#FF9800",
        linestyle="--", label="30-Day MA")
ax.set_ylabel("Price ($)")
ax.set_title("Tesla Closing Price with 30-Day Moving Average")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ==================================================
# PREPARE INPUT SEQUENCE
# ==================================================

close_prices = np.array(df["Close"]).reshape(-1, 1)
scaled_data  = scaler.transform(close_prices)
last_seq     = scaled_data[-WINDOW_SIZE:, 0]

# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_n_days(model, last_sequence, n_days, scaler):
    preds = []
    seq   = last_sequence.copy()
    for _ in range(n_days):
        val = model.predict(seq.reshape(1, len(seq), 1), verbose=0)[0, 0]
        preds.append(val)
        seq = np.append(seq[1:], val)
    return scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

# ==================================================
# RUN PREDICTIONS
# ==================================================

st.subheader(f"🔮 {horizon}-Day Ahead Predictions")

models_to_run = {
    "SimpleRNN": rnn_model,
    "GRU":       gru_model,
    "LSTM":      lstm_model,
}

if model_choice != "All Three":
    models_to_run = {model_choice: models_to_run[model_choice]}

all_preds = {}
for name, model in models_to_run.items():
    all_preds[name] = predict_n_days(model, last_seq, horizon, scaler)

# Metric cards
cols = st.columns(len(all_preds))
for i, (name, preds) in enumerate(all_preds.items()):
    with cols[i]:
        st.metric(
            f"{name} — Day 1",
            f"${preds[0]:.2f}",
            f"{((preds[0] - latest_close) / latest_close * 100):+.2f}% vs today"
        )

# Prediction tables
st.markdown("#### Predicted Prices by Day")
table_cols = st.columns(len(all_preds))
for i, (name, preds) in enumerate(all_preds.items()):
    with table_cols[i]:
        st.markdown(f"**{name}**")
        pred_df = pd.DataFrame({
            "Day": [f"Day {d+1}" for d in range(horizon)],
            "Predicted Price ($)": np.round(preds, 2)
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ==================================================
# FORECAST CHART
# ==================================================

st.subheader("📊 Forecast Trend")

colors = {"SimpleRNN": "#FF5722", "GRU": "#9C27B0", "LSTM": "#2196F3"}
days   = np.arange(1, horizon + 1)

fig, ax = plt.subplots(figsize=(10, 4))
for name, preds in all_preds.items():
    ax.plot(days, preds, marker="o", linewidth=2,
            label=name, color=colors.get(name, "gray"))
ax.axhline(latest_close, color="black", linestyle="--",
           linewidth=1, label=f"Today's Close (${latest_close:.2f})")
ax.set_xlabel("Days Ahead")
ax.set_ylabel("Predicted Price ($)")
ax.set_title(f"Tesla {horizon}-Day Price Forecast")
ax.set_xticks(days)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ==================================================
# COMPARISON SUMMARY (all 3 only)
# ==================================================

if model_choice == "All Three":
    st.subheader("🏆 Model Comparison Summary")
    summary_rows = []
    for name, preds in all_preds.items():
        summary_rows.append({
            "Model":                    name,
            "Day 1 Price ($)":          round(preds[0], 2),
            f"Day {horizon} Price ($)": round(preds[-1], 2),
            "Change from Today ($)":    round(preds[-1] - latest_close, 2),
            "Change from Today (%)":    round((preds[-1] - latest_close) / latest_close * 100, 2),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.info(
        "💡 Check the notebook's evaluation section for the full RMSE comparison "
        "across all three models on the held-out test set."
    )

# ==================================================
# RAW DATA
# ==================================================

with st.expander("📄 View Raw Dataset (last 20 rows)"):
    st.dataframe(df.tail(20), use_container_width=True)

# ==================================================
# MODEL INFO
# ==================================================

with st.expander("📚 About the Models"):
    st.markdown("""
**SimpleRNN** — Basic recurrent unit. Struggles with long sequences (vanishing gradient).

**GRU** — Uses reset and update gates. Lighter than LSTM, often comparable accuracy.

**LSTM** — Three gates + cell state. Best at long-range dependencies. Tuned with GridSearchCV.

**Input:** Last 60 trading days of normalized closing prices.  
**Output:** Predicted closing price for the next N days.
""")