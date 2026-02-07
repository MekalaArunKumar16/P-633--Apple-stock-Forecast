# =====================================================
# IMPORTS
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import os


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Apple Stock Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("<h2>📈 Apple Stock Forecast Dashboard</h2>", unsafe_allow_html=True)


# =====================================================
# BASE DIRECTORY (FOR DEPLOYMENT)
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================
# CONSTANTS (SAFE PATHS)
# =====================================================

DATA_PATH = os.path.join(BASE_DIR, "data", "AAPL.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.pkl")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "model", "predictions.pkl")

PRICE_COL = "Close"
WINDOW_SIZE = 60


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown(
    """
    <style>

    .metric-card {
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 5px;
        border: 1px solid #262730;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    .metric-title {
        font-size: 13px;
        color: #E0E0E0;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 22px;
        font-weight: 700;
        margin: 2px 0;
    }

    .metric-sub {
        font-size: 12px;
        opacity: 0.9;
    }

    .green {
        background: linear-gradient(135deg,#0f5132,#198754);
        color: white;
    }

    .red {
        background: linear-gradient(135deg,#842029,#dc3545);
        color: white;
    }

    .blue {
        background: linear-gradient(135deg,#084298,#0d6efd);
        color: white;
    }

    .gold {
        background: linear-gradient(135deg,#664d03,#ffc107);
        color: black;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Settings")

forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        st.error("❌ data/AAPL.csv not found in repository.")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    df = df.dropna(subset=[PRICE_COL])

    return df


df = load_data()


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    files = {
        "Model": MODEL_PATH,
        "Scaler": SCALER_PATH,
        "Metrics": METRICS_PATH,
        "Predictions": PREDICTIONS_PATH
    }

    for name, path in files.items():

        if not os.path.exists(path):
            st.error(f"❌ {name} file missing. Upload model folder.")
            st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)

    with open(PREDICTIONS_PATH, "rb") as f:
        preds = pickle.load(f)

    return model, scaler, metrics, preds


model, scaler, metrics, preds = load_model()


# =====================================================
# LSTM FORECAST
# =====================================================

def lstm_forecast(df, days):

    scaled = scaler.transform(df[[PRICE_COL]])

    seq = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)

    future = []

    for _ in range(days):

        pred = model.predict(seq, verbose=0)[0][0]

        future.append(pred)

        seq = np.append(
            seq[0, 1:, :],
            [[pred]],
            axis=0
        ).reshape(1, WINDOW_SIZE, 1)

    future = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    ).flatten()

    dates = pd.date_range(
        df["Date"].iloc[-1] + timedelta(days=1),
        periods=days
    )

    return pd.DataFrame({
        "Date": dates,
        "Forecast": future
    })


forecast_df = lstm_forecast(df, forecast_days)


# =====================================================
# KPI CALCULATIONS
# =====================================================

ath = df[PRICE_COL].max()
ath_date = df.loc[df[PRICE_COL].idxmax(), "Date"]

atl = df[PRICE_COL].min()
atl_date = df.loc[df[PRICE_COL].idxmin(), "Date"]

last = df[PRICE_COL].iloc[-1]
prev = df[PRICE_COL].iloc[-2]

change = last - prev
pct = (change / prev) * 100

f_high = forecast_df["Forecast"].max()
f_low = forecast_df["Forecast"].min()


# =====================================================
# KPI COMPONENT
# =====================================================

def kpi(col, title, value, sub, color):

    with col:

        st.markdown(
            f"""
            <div class="metric-card {color}">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(["📊 Dashboard", "💡 Market Insight"])


# =====================================================
# DASHBOARD TAB
# =====================================================

with tab1:

    st.markdown("### 📊 Stock Summary")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    kpi(k1, "ATH", f"${ath:.2f}", ath_date.strftime("%d %b %Y"), "gold")
    kpi(k2, "ATL", f"${atl:.2f}", atl_date.strftime("%d %b %Y"), "blue")

    kpi(
        k3,
        "Latest Price",
        f"${last:.2f}",
        f"{pct:+.2f}%",
        "green" if pct >= 0 else "red"
    )

    kpi(
        k4,
        "Daily Change",
        f"{change:+.2f}",
        "Today",
        "green" if change >= 0 else "red"
    )

    kpi(k5, "Forecast High", f"${f_high:.2f}", "Next Period", "green")
    kpi(k6, "Forecast Low", f"${f_low:.2f}", "Next Period", "red")


    # ============================
    # PRICE CHART
    # ============================

    st.markdown("### 📈 Price Analysis")

    c1, c2 = st.columns(2)

    with c1:

        fig1 = go.Figure()

        fig1.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="Price"
        ))

        fig1.update_layout(
            title="Historical Closing Price",
            template="plotly_dark"
        )

        st.plotly_chart(fig1, use_container_width=True)


    with c2:

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast"],
            name="Forecast",
            line=dict(width=3)
        ))

        fig2.update_layout(
            title="Future Forecast",
            template="plotly_dark"
        )

        st.plotly_chart(fig2, use_container_width=True)


    # ============================
    # DATA TABLE
    # ============================

    st.markdown("### 📋 Data")

    with st.expander("View Data"):

        t1, t2 = st.columns(2)

        with t1:
            st.dataframe(df.tail(30), use_container_width=True)

        with t2:
            st.dataframe(forecast_df.round(2), use_container_width=True)


    # ============================
    # MODEL PERFORMANCE
    # ============================

    st.markdown("### 🧪 Model Performance")

    results = pd.DataFrame({

        "Model": ["ARIMA", "SARIMA", "LSTM"],

        "RMSE": [
            metrics["arima"]["rmse"],
            metrics["sarima"]["rmse"],
            metrics["lstm"]["rmse"]
        ],

        "MAE": [
            metrics["arima"]["mae"],
            metrics["sarima"]["mae"],
            metrics["lstm"]["mae"]
        ],

        "MAPE": [
            metrics["arima"]["mape"],
            metrics["sarima"]["mape"],
            metrics["lstm"]["mape"]
        ]
    })


    fig_metrics = px.bar(
        results.melt("Model"),
        x="Model",
        y="value",
        color="variable",
        barmode="group",
        template="plotly_dark"
    )


    train_size = int(len(df) * 0.8)
    test_df = df.iloc[train_size:]


    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(
        x=test_df["Date"],
        y=test_df[PRICE_COL],
        name="Actual"
    ))


    for name, values in preds.items():

        if name.lower() == "arimax":
            continue

        fig_compare.add_trace(go.Scatter(
            x=test_df["Date"].iloc[-len(values):],
            y=np.array(values).flatten(),
            name=name.upper()
        ))


    fig_compare.update_layout(
        title="Prediction Comparison",
        template="plotly_dark"
    )


    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(fig_metrics, use_container_width=True)

    with c2:
        st.plotly_chart(fig_compare, use_container_width=True)


    best = results.iloc[:, 1:].mean(axis=1).idxmin()

    st.success(f"🏆 Best Model: {results.loc[best, 'Model']}")


# =====================================================
# MARKET INSIGHT TAB
# =====================================================

with tab2:

    st.markdown("### 💡 Should You Buy This Stock?")


    close = df[PRICE_COL]


    last_30 = close.tail(30)

    trend_pct = ((last_30.iloc[-1] - last_30.iloc[0]) / last_30.iloc[0]) * 100


    forecast_pct = (
        (forecast_df["Forecast"].iloc[-1] -
         forecast_df["Forecast"].iloc[0]) /
        forecast_df["Forecast"].iloc[0]
    ) * 100


    risk = close.pct_change().std()


    if trend_pct > 2 and forecast_pct > 3:

        decision = "✅ BUY"
        message = "Strong upward momentum"

    elif trend_pct > 0 and forecast_pct > 0:

        decision = "⚠️ HOLD"
        message = "Moderate stability"

    else:

        decision = "❌ AVOID"
        message = "Weak trend detected"


    st.markdown("### 📢 Our Advice")


    if decision == "✅ BUY":
        st.success(f"{decision}\n\n{message}")

    elif decision == "⚠️ HOLD":
        st.warning(f"{decision}\n\n{message}")

    else:
        st.error(f"{decision}\n\n{message}")


    c1, c2, c3 = st.columns(3)

    c1.metric("Last 30 Days Trend", f"{trend_pct:.2f}%")
    c2.metric("Forecast Trend", f"{forecast_pct:.2f}%")
    c3.metric("Risk (Volatility)", f"{risk:.2%}")


    st.markdown("### 👉 What You Should Do")


    if decision == "✅ BUY":

        st.success("""
        ✔ Consider entry  
        ✔ Set stop-loss  
        ✔ Monitor closely
        """)

    elif decision == "⚠️ HOLD":

        st.warning("""
        ✔ Wait for confirmation  
        ✔ Track news  
        ✔ Buy on dips
        """)

    else:

        st.error("""
        ✔ Avoid entry  
        ✔ Protect capital  
        ✔ Find alternatives
        """)
