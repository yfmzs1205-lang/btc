# binance_prediction_site/app.py
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pickle
import time

from utils.data_utils import fetch_klines, prepare_target
from utils.model import (
    train_classifier,
    init_incremental_model,
    partial_fit_step,
    predict_proba_safe,
)

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Binance 10/30-Minute Trend Predictor (Education Only)", layout="wide")
st.title("📈 Binance 10/30-Minute Trend Predictor (Education Only)")
st.caption(
    "Demo only. Fetches recent 1-minute klines and trains a lightweight model. "
    "Includes paper trading. No real trading. Educational use only."
)

# =========================
# Session state
# =========================
def init_state():
    ss = st.session_state
    ss.setdefault("balance", 100.0)           # 初始本金
    ss.setdefault("open_orders", [])          # 未结算订单
    ss.setdefault("order_history", [])        # 已结算订单
    ss.setdefault("order_id", 1)
    ss.setdefault("latest_price", None)
    ss.setdefault("latest_ts", None)
    ss.setdefault("df_cache", None)           # 数据缓存（避免免费层频繁请求）
    # 增量学习模型与训练进度
    ss.setdefault("inc_model", None)
    ss.setdefault("last_train_index", 0)

init_state()

# =========================
# Simple technical indicators
# =========================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma_14"] = out["close"].rolling(14).mean()
    out["rsi_14"] = rsi(out["close"], 14)
    m, s = macd(out["close"])
    out["macd"] = m
    out["macd_signal"] = s
    return out

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Configuration")
    symbol_choice = st.selectbox("Select trading pair", ["BTCUSDT", "ETHUSDT"])
    horizon_choice = st.radio("Prediction horizon", ["10 minutes", "30 minutes"])
    horizon = 10 if horizon_choice.startswith("10") else 30
    use_indicators = st.checkbox("Add technical indicators (SMA / RSI / MACD)", value=True)
    if st.button("🔄 Refresh latest price & retrain"):
        st.session_state.df_cache = None   # 强制重拉数据（并触发重训）

# =========================
# Data
# =========================
def get_data(symbol: str, limit: int = 600) -> pd.DataFrame:
    if st.session_state.df_cache is None:
        df = fetch_klines(symbol=symbol, interval="1m", limit=limit)
        st.session_state.df_cache = df
    return st.session_state.df_cache.copy()

df_raw = get_data(symbol_choice, limit=600)

# Latest price
if not df_raw.empty:
    last_row = df_raw.iloc[-1]
    st.session_state.latest_price = float(last_row["close"])
    st.session_state.latest_ts = pd.to_datetime(last_row["timestamp"]).tz_convert("UTC")
    c1, c2 = st.columns(2)
    c1.metric("Latest price", f"{st.session_state.latest_price:.2f} {symbol_choice[-4:]}")
    c2.write(f"UTC time: **{st.session_state.latest_ts}**")

# =========================
# Features & label
# =========================
df_work = add_indicators(df_raw) if use_indicators else df_raw.copy()
df_label = prepare_target(df_work, horizon=horizon, target_col="target")

# 基础特征
feature_cols = ["close"]
df_label["ret_1"] = df_label["close"].pct_change().fillna(0.0)
feature_cols.append("ret_1")
if use_indicators:
    for c in ["sma_14", "rsi_14", "macd", "macd_signal"]:
        if c in df_label.columns:
            df_label[c] = df_label[c].fillna(method="ffill")
            feature_cols.append(c)
df_label[feature_cols] = df_label[feature_cols].astype(float)

# =========================
# Training (batch + optional incremental)
# =========================
st.markdown("### Training")
tc1, tc2, tc3 = st.columns([1,1,2])
use_incremental = tc1.checkbox("Enable incremental learning (partial_fit)", value=True)
save_btn = tc2.button("💾 Save model")
load_file = tc3.file_uploader("Load a saved model (.pkl)", type=["pkl"], label_visibility="collapsed")

# Load uploaded model
if load_file is not None:
    try:
        st.session_state.inc_model = pickle.load(load_file)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Load failed: {e}")

# Init incremental model if enabled
if use_incremental and st.session_state.inc_model is None:
    st.session_state.inc_model = init_incremental_model()
    st.session_state.last_train_index = 0

# Baseline batch model for reference metrics
with st.spinner("Training baseline model (batch) and evaluating..."):
    batch_model, metrics = train_classifier(df_label, feature_cols=feature_cols, target_col="target")

# Incremental update on new chunk
if use_incremental:
    start_idx = st.session_state.get("last_train_index", 0)
    end_idx = len(df_label) - 2   # 留最后一条做推理
    if end_idx > start_idx:
        X_chunk = df_label[feature_cols].iloc[start_idx:end_idx].values
        y_chunk = df_label["target"].iloc[start_idx:end_idx].values.astype(int)
        partial_fit_step(st.session_state.inc_model, X_chunk, y_chunk)
        st.session_state.last_train_index = end_idx

    # Simple rolling evaluation
    X_eval = df_label[feature_cols].iloc[-200:].values
    y_eval = df_label["target"].iloc[-200:].values.astype(int)
    p = predict_proba_safe(st.session_state.inc_model, X_eval)[:, 1]
    y_pred = (p >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
    }
    current_model = st.session_state.inc_model
else:
    current_model = batch_model

# Save model
if save_btn:
    try:
        buf = pickle.dumps(current_model)
        st.download_button("Download model.pkl", data=buf, file_name="model.pkl", mime="application/octet-stream")
    except Exception as e:
        st.error(f"Save failed: {e}")

# Show metrics
m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
m2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
m3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")

# =========================
# Inference
# =========================
def predict_direction_latest(model, df: pd.DataFrame) -> tuple[int, float]:
    row = pd.DataFrame([df[feature_cols].iloc[-1].values], columns=feature_cols)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
    else:
        proba = predict_proba_safe(model, row.values)[0]
    prob_up = float(proba[1])
    pred = int(prob_up >= 0.5)
    conf = float(max(prob_up, 1 - prob_up))
    return pred, conf

pred, conf = predict_direction_latest(current_model, df_label)
st.subheader(f"Prediction for the next {10 if horizon==10 else 30} minutes")
st.info(
    f"📌 Price likely to **{'rise' if pred==1 else 'fall'}** with confidence **{conf*100:.2f}%** "
    f"for `{symbol_choice}` in `{horizon}m` horizon."
)
st.warning("💡 Consider avoiding a long position." if pred == 0 else "💡 Consider avoiding a short position.")

# =========================
# Recent price chart
# =========================
plot_df = df_raw.tail(200)
fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["close"], mode="lines", name="Close"))
fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
st.subheader("Recent Price Chart (last 200 minutes)")
st.plotly_chart(fig, use_container_width=True)

# =========================
# Paper trading
# =========================
st.subheader("💸 Paper Trading (Simulation)")
st.caption("Initial balance 100 USDT. Min order 5U, max 125U. Payout +80% on correct direction; "
           "lose stake on wrong direction. Orders settle after selected horizon.")

def settle_orders(current_index: int, df: pd.DataFrame):
    to_close = []
    for od in st.session_state.open_orders:
        if current_index >= od["settle_index"]:
            entry_close = df.iloc[od["entry_index"]]["close"]
            settle_close = df.iloc[od["settle_index"]]["close"]
            win = (settle_close > entry_close) if od["side"] == "LONG" else (settle_close < entry_close)
            if win:
                pnl = round(od["amount"] * 0.8, 2)  # +80%
                st.session_state.balance += pnl
            else:
                pnl = round(-od["amount"], 2)       # -100%
                st.session_state.balance += pnl
            od["pnl"] = pnl
            od["exit_price"] = float(settle_close)
            od["exit_time"] = str(df.iloc[od["settle_index"]]["timestamp"]) if "timestamp" in df.columns else "N/A"
            od["status"] = "WIN" if win else "LOSS"
            st.session_state.order_history.append(od)
            to_close.append(od)
    for od in to_close:
        st.session_state.open_orders.remove(od)

cur_idx = len(df_label) - 1
if cur_idx >= 0:
    settle_orders(cur_idx, df_label)

cA, cB, cC, cD = st.columns([1, 1, 1, 2])
side = cA.radio("Direction", ["LONG", "SHORT"], horizontal=True)
amount = cB.number_input("Order amount (USDT)", min_value=5, max_value=125, step=5, value=10)
cC.metric("Balance (USDT)", f"{st.session_state.balance:.2f}")
place = cD.button("✅ Place order now")

def place_order(side: str, amount: float):
    if amount < 5 or amount > 125:
        st.error("Order size must be between 5 and 125 USDT.")
        return
    if amount > st.session_state.balance:
        st.error("Insufficient balance.")
        return
    entry_index = len(df_label) - 1
    order = {
        "id": st.session_state.order_id,
        "symbol": symbol_choice,
        "side": side,
        "amount": float(amount),
        "entry_price": float(df_label.iloc[entry_index]["close"]),
        "entry_time": str(df_label.iloc[entry_index]["timestamp"]) if "timestamp" in df_label.columns else "N/A",
        "entry_index": entry_index,
        "settle_index": entry_index + horizon,
        "horizon_min": horizon,
        "status": "OPEN",
    }
    st.session_state.balance -= amount           # 锁定本金
    st.session_state.open_orders.append(order)
    st.session_state.order_id += 1
    st.success(f"Order #{order['id']} placed: {side} {amount} USDT @ {order['entry_price']:.2f}")

if place:
    place_order(side, amount)

def _fmt_table(rows):
    if not rows:
        return pd.DataFrame([], columns=["id","symbol","side","amount","entry_price","entry_time","status"])
    return pd.DataFrame(rows)

st.markdown("**Open orders**")
st.dataframe(_fmt_table(st.session_state.open_orders), use_container_width=True, height=190)

st.markdown("**Order history**")
hist_df = _fmt_table(st.session_state.order_history)
if not hist_df.empty:
    cols = ["id","symbol","side","amount","entry_price","exit_price","pnl","entry_time","exit_time","status"]
    for c in cols:
        if c not in hist_df.columns:
            hist_df[c] = np.nan
    hist_df = hist_df[cols]
st.dataframe(hist_df, use_container_width=True, height=220)

st.caption("Note: Model retrains on each refresh. Incremental learning will keep adapting during the session. "
           "Use Save/Load to persist a model snapshot.")
