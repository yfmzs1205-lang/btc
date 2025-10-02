"""
Streamlit application for predicting short-term cryptocurrency price trends.

This app uses the public Binance API to retrieve recent 1-minute candlestick
data for selected symbols (currently BTCUSDT and ETHUSDT).  It computes
standard technical indicators (moving averages, exponential moving averages,
RSI and MACD), constructs a binary classification target indicating
whether the price declines after a chosen horizon (10 or 30 minutes),
and trains a random forest classifier on the fly.  Predictions for
the latest data point are displayed along with a simple buy/sell/hold
recommendation and evaluation metrics on a validation split.

Note: This project is for educational purposes only.  The predictions
produced by the model are not guaranteed to be accurate or profitable,
and should not be used for real trading decisions without further
validation and risk assessment.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data_utils import fetch_klines, prepare_target
from utils.indicators import add_indicators
from utils.model import train_classifier, predict_latest, simulate_trades


@st.cache_data(show_spinner=False)
def load_data(symbol: str, limit: int = 600) -> pd.DataFrame:
    """Load recent candlestick data from Binance.

    This function is cached by Streamlit to avoid repeated network
    requests when the same symbol is chosen.
    """
    try:
        df = fetch_klines(symbol=symbol, interval="1m", limit=limit)
    except Exception as exc:
        st.error(f"Error fetching data from Binance: {exc}")
        raise
    return df


def main() -> None:
    st.set_page_config(page_title="Binance Trend Predictor", layout="wide")
    st.title("📈 Binance 10/30‑Minute Trend Predictor (Education Only)")
    st.markdown(
        "This tool fetches recent minute‑level data from Binance and trains a simple "
        "machine learning model on the fly to forecast whether the price of Bitcoin "
        "(BTC) or Ethereum (ETH) is likely to decline after a selected time horizon. "
        "It is intended for learning and demonstration purposes only."
    )

    # Sidebar controls
    st.sidebar.header("Configuration")
    symbol_choice = st.sidebar.selectbox(
        "Select trading pair", ["BTCUSDT", "ETHUSDT"], index=0
    )
    horizon_choice = st.sidebar.radio(
        "Prediction horizon",
        ["10 minutes", "30 minutes"],
        index=0,
    )
    horizon = 10 if horizon_choice.startswith("10") else 30

    # Fetch and prepare data
    with st.spinner("Fetching data and computing indicators..."):
        df_raw = load_data(symbol_choice)
        df = add_indicators(df_raw)
        df = prepare_target(df, horizon=horizon)

    # Feature columns to use
    feature_cols = [
        "close",
        "ma5",
        "ma10",
        "ma20",
        "ema12",
        "ema26",
        "rsi",
        "macd",
        "macd_signal",
    ]

    # Train model
    with st.spinner("Training model and evaluating performance..."):
        model, metrics = train_classifier(df, feature_cols=feature_cols, target_col="target")

    # Backtest simulation
    # To help users gauge the potential effectiveness of the model we run a simple simulated
    # trading strategy on the historical data.  For each data point (excluding the last
    # `horizon` rows) the model prediction is used to decide whether to enter a long or
    # short position.  The position is closed after `horizon` minutes and the profit is
    # calculated as the difference between entry and exit prices (without transaction fees).
    # This simulation does not execute real trades and is intended only for educational
    # verification of the prediction logic.
    with st.spinner("Running simulated trades to evaluate strategy performance..."):
        sim_results = simulate_trades(df, model, feature_cols=feature_cols, horizon=horizon)

    # Prediction for the latest row
    latest_row = df.iloc[-1]
    pred, prob = predict_latest(model, latest_row, feature_cols=feature_cols)
    confidence = prob if pred == 1 else 1 - prob

    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")

    outcome_str = "⬇️ Price likely to fall" if pred == 1 else "⬆️ Price likely to rise"
    st.subheader("Prediction for the next {}".format(horizon_choice))
    st.write(
        f"**{outcome_str}** with confidence {confidence*100:.2f}%. "
        "The model suggests that the {symbol_choice} price in {horizon_choice} "
        "will { 'decline' if pred == 1 else 'increase or stay the same' }."
    )

    # Simple buy/sell/hold suggestion
    suggestion = ""
    if pred == 1:
        if confidence > 0.6:
            suggestion = "👎 Suggestion: Consider selling or avoiding a long position."
        else:
            suggestion = "🤔 Suggestion: Uncertain decline; exercise caution or hold."
    else:
        if confidence > 0.6:
            suggestion = "👍 Suggestion: Consider buying or holding a long position."
        else:
            suggestion = "🤔 Suggestion: Uncertain rise; exercise caution or hold."
    st.warning(suggestion)

    # Plot price and moving averages
    st.subheader("Recent Price Chart (last 200 minutes)")
    df_plot = df.tail(200).copy().reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["ma5"],
            mode="lines",
            name="MA5",
            line=dict(color="#ff7f0e", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["ma10"],
            mode="lines",
            name="MA10",
            line=dict(color="#2ca02c", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["ma20"],
            mode="lines",
            name="MA20",
            line=dict(color="#d62728", dash="dashdot"),
        )
    )
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display backtest results
    st.subheader("Simulated Trading Performance")
    st.markdown(
        "The following metrics are derived from a simple backtest that applies the model's "
        "predictions to historical data. A 'long' trade is entered if the model predicts the "
        "price will rise (predicted class 0) and closed after the selected horizon. A 'short' "
        "trade is entered if the model predicts a decline (predicted class 1). Profits are "
        "calculated without transaction costs. These results are for illustration only and do "
        "not guarantee future performance."
    )
    col4, col5, col6 = st.columns(3)
    col4.metric("Total Profit", f"{sim_results['total_profit']:.4f} USDT")
    col5.metric("Avg Profit/Trade", f"{sim_results['avg_profit']:.4f} USDT")
    col6.metric("Win Rate", f"{sim_results['win_rate']*100:.2f}%")

    # Display raw data toggle
    with st.expander("Show raw data"):
        st.dataframe(df.tail(30))


if __name__ == "__main__":
    main()