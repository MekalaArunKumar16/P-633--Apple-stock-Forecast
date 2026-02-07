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


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Apple Stock Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 Apple Stock Price Forecast Dashboard")


# =====================================================
# CONSTANTS (MATCH NOTEBOOK)
# =====================================================

DATA_PATH = "../data/AAPL.csv"

MODEL_PATH = "../model/model.pkl"
SCALER_PATH = "../model/scaler.pkl"
METRICS_PATH = "../model/metrics.pkl"
PREDICTIONS_PATH = "../model/predictions.pkl"

PRICE_COL = "Adj Close"
SEQ_LEN = 30


# =====================================================
# SESSION STATE
# =====================================================

if "section" not in st.session_state:
    st.session_state.section = "History"


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

menu = {
    "History": "📊 History",
    "Business": "💡 Business Insights",
    "Evaluation": "🧪 Model Evaluation"
}

for key, label in menu.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.section = key


forecast_days = st.sidebar.slider(
    "Forecast Days (LSTM)",
    7, 120, 30
)

section = st.session_state.section


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    df = df.dropna(subset=[PRICE_COL])

    return df


df = load_data()

if df.empty:
    st.error("Dataset not found.")
    st.stop()


# =====================================================
# LOAD MODEL + SCALER
# =====================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_model()


# =====================================================
# LSTM FORECAST (MATCH NOTEBOOK LOGIC)
# =====================================================

def lstm_forecast(df, days):

    last_vals = df[PRICE_COL].values[-SEQ_LEN:]

    scaled = scaler.transform(last_vals.reshape(-1, 1))

    seq = scaled.reshape(SEQ_LEN, 1)

    preds = []

    for _ in range(days):

        pred = model.predict(
            seq.reshape(1, SEQ_LEN, 1),
            verbose=0
        )

        preds.append(pred[0][0])

        seq = np.vstack([
            seq[1:],
            [[pred[0][0]]]
        ])


    preds = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()


    # Light smoothing
    preds = pd.Series(preds).rolling(3, min_periods=1).mean().values


    dates = [
        df["Date"].iloc[-1] + timedelta(days=i + 1)
        for i in range(days)
    ]


    return pd.DataFrame({
        "Date": dates,
        "Forecast Price": preds
    })


# =====================================================
# PLOTS
# =====================================================

def plot_history(df):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df[PRICE_COL],
        name="History"
    ))

    fig.update_layout(
        title="Historical Prices",
        hovermode="x unified"
    )

    return fig


def plot_forecast(df):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Forecast Price"],
        name="LSTM Forecast"
    ))

    fig.update_layout(
        title="LSTM Forecast",
        hovermode="x unified"
    )

    return fig


# =====================================================
# METRIC CARDS
# =====================================================

def small_card(col, title, value, subtitle):

    with col:
        st.caption(title)
        st.markdown(f"### {value}")
        st.caption(subtitle)


# =====================================================
# PRECOMPUTE
# =====================================================

forecast_df = lstm_forecast(df, forecast_days)

fig_history = plot_history(df)
fig_forecast = plot_forecast(forecast_df)


# =====================================================
# HISTORY TAB
# =====================================================

if section == "History":

    st.subheader("📊 Price Charts")

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(fig_history, use_container_width=True)

    with c2:
        st.plotly_chart(fig_forecast, use_container_width=True)


    t1, t2 = st.columns(2)

    with t1:

        st.subheader("Last 30 Days")

        st.dataframe(
            df.tail(30),
            use_container_width=True,
            height=400
        )

    with t2:

        st.subheader("Forecast Data")

        table = forecast_df.copy()
        table["Forecast Price"] = table["Forecast Price"].round(2)

        st.dataframe(
            table,
            use_container_width=True,
            height=400
        )


# =====================================================
# BUSINESS INSIGHTS
# =====================================================

if section == "Business":

    st.subheader("📊 Key Records")

    ath = df[PRICE_COL].max()
    ath_date = df.loc[df[PRICE_COL].idxmax(), "Date"]

    atl = df[PRICE_COL].min()
    atl_date = df.loc[df[PRICE_COL].idxmin(), "Date"]

    last = df[PRICE_COL].iloc[-1]
    prev = df[PRICE_COL].iloc[-2]

    change = last - prev
    pct = (change / prev) * 100

    f_high = forecast_df["Forecast Price"].max()
    f_low = forecast_df["Forecast Price"].min()


    cols = st.columns(6)

    small_card(cols[0], "All-Time High", f"${ath:.2f}", ath_date.strftime("%d %b %Y"))
    small_card(cols[1], "All-Time Low", f"${atl:.2f}", atl_date.strftime("%d %b %Y"))
    small_card(cols[2], "Latest Price", f"${last:.2f}", f"{pct:+.2f}%")
    small_card(cols[3], "Daily Change", f"{change:+.2f}", "Last Close")
    small_card(cols[4], "Forecast High", f"${f_high:.2f}", "Next Period")
    small_card(cols[5], "Forecast Low", f"${f_low:.2f}", "Next Period")


    st.markdown("""
    ### 📌 Observations

    • Strong long-term growth  
    • Short-term volatility  
    • LSTM performs best in near future  
    • Long horizon accuracy reduces  
    • Forecasts are indicative only  
    """)


# =====================================================
# MODEL EVALUATION
# =====================================================

if section == "Evaluation":

    # Load metrics
    with open(METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)


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


    long_df = results.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Value"
    )


    fig_metrics = px.bar(
        long_df,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Comparison"
    )


    # Load predictions
    with open(PREDICTIONS_PATH, "rb") as f:
        preds = pickle.load(f)


    train_size = int(len(df) * 0.8)
    test_df = df.iloc[train_size:]


    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(
        x=test_df["Date"],
        y=test_df[PRICE_COL],
        name="Actual"
    ))


    for name, values in preds.items():

        fig_compare.add_trace(go.Scatter(
            x=test_df["Date"].iloc[-len(values):],
            y=np.array(values).flatten(),
            name=name.upper()
        ))


    fig_compare.update_layout(
        title="Actual vs Predictions",
        hovermode="x unified"
    )


    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(fig_metrics, use_container_width=True)

    with c2:
        st.plotly_chart(fig_compare, use_container_width=True)


    best = results.iloc[:, 1:].mean(axis=1).idxmin()

    st.success(f"🏆 Best Model: {results.loc[best, 'Model']}")
