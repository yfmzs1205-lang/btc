# binance_prediction_site/app.py
from __future__ import annotations
import time
import pandas as pd
import numpy as np
import streamlit as st
from utils.data_utils import fetch_klines, prepare_target
from utils.model import train_classifier

# -----------------------------
# Streamlit 基础设置
# -----------------------------
st.set_page_config(page_title="Binance 10/30-Minute Trend Predictor (Education Only)",
                   layout="wide")

st.title("📈 Binance 10/30-Minute Trend Predictor (Education Only)")
st.caption(
    "This tool fetches recent minute-level data from Binance and trains a simple machine learning model "
    "on the fly to forecast whether the price of Bitcoin (BTC) or Ethereum (ETH) is likely to decline after "
    "a selected time horizon. It is intended for learning and demonstration purposes only."
)

# -----------------------------
# Session State：资金与订单
# -----------------------------
def init_state():
    if "balance" not in st.session_state:
        st.session_state.balance = 100.0   # 初始本金 100 USDT
    if "open_orders" not in st.session_state:
        st.session_state.open_orders = []   # 未结算订单
    if "order_history" not in st.session_state:
        st.session_state.order_history = [] # 已结算订单
    if "order_id" not in st.session_state:
        st.session_state.order_id = 1
    if "latest_price" not in st.session_state:
        st.session_state.latest_price = None
        st.session_state.latest_ts = None
    if "df_cache" not in st.session_state:
        st.session_state.df_cache = None

init_state()

# -----------------------------
# 工具：技术指标（轻量实现，避免依赖冲突）
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(period).mean()
    roll_down = pd.Series(loss).rolling(period).mean()
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

# -----------------------------
# 左侧参数区
# -----------------------------
with st.sidebar:
    st.header("Configuration")
    symbol_choice = st.selectbox("Select trading pair", ["BTCUSDT", "ETHUSDT"])
    horizon_choice = st.radio("Prediction horizon", ["10 minutes", "30 minutes"])
    horizon = 10 if horizon_choice.startswith("10") else 30
    use_indicators = st.checkbox("Add technical indicators (SMA/RSI/MACD) to improve accuracy", value=True)

    # 实时刷新按钮
    if st.button("🔄 Refresh latest price & retrain"):
        st.session_state.df_cache = None  # 强制重新取数

# -----------------------------
# 取数 + 实时价格
# -----------------------------
def get_data(symbol: str, limit: int = 600) -> pd.DataFrame:
    # 使用缓存，避免免费层频繁请求
    if st.session_state.df_cache is None:
        df = fetch_klines(symbol=symbol, interval="1m", limit=limit)
        st.session_state.df_cache = df
    return st.session_state.df_cache.copy()

df_raw = get_data(symbol_choice, limit=600)

# 最新价格展示
if not df_raw.empty:
    last_row = df_raw.iloc[-1]
    st.session_state.latest_price = float(last_row["close"])
    st.session_state.latest_ts = pd.to_datetime(last_row["timestamp"]).tz_convert("UTC")
    colp1, colp2 = st.columns(2)
    colp1.metric("Latest price", f"{st.session_state.latest_price:.2f} {symbol_choice[-4:]}")
    colp2.write(f"UTC time: **{st.session_state.latest_ts}**")

# -----------------------------
# 训练集构造 + 训练
# -----------------------------
# 添加指标（可选）
df_work = add_indicators(df_raw) if use_indicators else df_raw.copy()

# 准备标签
df_label = prepare_target(df_work, horizon=horizon, target_col="target")

# 简单特征列：价格变化 +（可选）指标
feature_cols = ["close"]
# 加入衍生变化率
df_label["ret_1"] = df_label["close"].pct_change().fillna(0)
feature_cols.append("ret_1")

if use_indicators:
    for c in ["sma_14", "rsi_14", "macd", "macd_signal"]:
        if c in df_label.columns:
            df_label[c] = df_label[c].fillna(method="ffill")
            feature_cols.append(c)

# 安全过滤数值
df_label[feature_cols] = df_label[feature_cols].astype(float)

# 训练与评估
with st.spinner("Training model and evaluating performance..."):
    model, metrics = train_classifier(df_label, feature_cols=feature_cols, target_col="target")

# 展示指标
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
c2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
c3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")

# -----------------------------
# 推理：未来方向 & 置信度
# -----------------------------
def predict_direction_latest(model, df: pd.DataFrame) -> tuple[int, float]:
    row = pd.DataFrame([df[feature_cols].iloc[-1].values], columns=feature_cols)
    prob = model.predict_proba(row)[0, 1]  # 预测“上涨”的概率
    pred = int(prob >= 0.5)
    conf = float(max(prob, 1 - prob))
    return pred, conf

pred, conf = predict_direction_latest(model, df_label)
msg_dir = "increase or stay the same" if pred == 1 else "decline"
st.subheader(f"Prediction for the next {horizon} minutes")
st.info(
    f"📌 Price likely to **{('rise' if pred==1 else 'fall')}** with confidence **{conf*100:.2f}%**. "
    f"The model suggests that the `{symbol_choice}` price in `{horizon_choice}` will "
    f"`{'increase or stay the same' if pred==1 else 'decline'}`."
)

st.warning("💡 Suggestion: Consider selling or avoiding a long position." if pred == 0
           else "💡 Suggestion: Consider avoiding a short position.")

# -----------------------------
# 最近价格图
# -----------------------------
import plotly.graph_objects as go
plot_df = df_raw.tail(200)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot_df["timestamp"], y=plot_df["close"],
    mode="lines", name="Close Price"
))
fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
st.subheader("Recent Price Chart (last 200 minutes)")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 模拟交易
# -----------------------------
st.subheader("💸 Paper Trading (Simulation)")
st.caption("Initial balance: 100 USDT. Order size: min 5U, max 125U. Payout: 80% on correct direction; loss = stake on incorrect.")

# 结算到期订单：按照 horizon 的步数结算（基于 1m K线）
def settle_orders(current_index: int, df: pd.DataFrame):
    to_close = []
    for od in st.session_state.open_orders:
        if current_index >= od["settle_index"]:
            entry_close = df.iloc[od["entry_index"]]["close"]
            settle_close = df.iloc[od["settle_index"]]["close"]
            # 方向判断
            win = (settle_close > entry_close) if od["side"] == "LONG" else (settle_close < entry_close)
            if win:
                pnl = round(od["amount"] * 0.8, 2)  # 80% 赔率
                st.session_state.balance += pnl
            else:
                pnl = round(-od["amount"], 2)
                st.session_state.balance += pnl  # 直接扣除下注额（负数）
            od["pnl"] = pnl
            od["exit_price"] = float(settle_close)
            od["status"] = "WIN" if win else "LOSS"
            od["exit_time"] = str(df.iloc[od["settle_index"]]["timestamp"])
            st.session_state.order_history.append(od)
            to_close.append(od)

    # 从未结列表移除
    for od in to_close:
        st.session_state.open_orders.remove(od)

# 当前索引（最后一根K线）
cur_idx = len(df_label) - 1
if cur_idx >= 0:
    # 结算可能到期的订单
    settle_orders(cur_idx, df_label)

# 下单控件
colA, colB, colC, colD = st.columns([1, 1, 1, 2])
side = colA.radio("Direction", ["LONG", "SHORT"], horizontal=True)
amount = colB.number_input("Order amount (USDT)", min_value=5, max_value=125, step=5, value=10)
colC.metric("Balance (USDT)", f"{st.session_state.balance:.2f}")
place = colD.button("✅ Place order now")

def place_order(side: str, amount: float):
    # 资金检查
    if amount < 5 or amount > 125:
        st.error("Order size must be between 5 and 125 USDT.")
        return
    if amount > st.session_state.balance:
        st.error("Insufficient balance.")
        return
    # 记录订单
    entry_index = len(df_label) - 1
    settle_index = entry_index + horizon
    order = {
        "id": st.session_state.order_id,
        "symbol": symbol_choice,
        "side": side,
        "amount": float(amount),
        "entry_price": float(df_label.iloc[entry_index]["close"]),
        "entry_time": str(df_label.iloc[entry_index]["timestamp"])
                        if "timestamp" in df_label.columns else "N/A",
        "entry_index": entry_index,
        "settle_index": settle_index,
        "horizon_min": horizon,
        "status": "OPEN",
    }
    # 预先扣除成本（把本金锁定）
    st.session_state.balance -= amount
    st.session_state.open_orders.append(order)
    st.session_state.order_id += 1
    st.success(f"Order #{order['id']} placed: {side} {amount} USDT at {order['entry_price']:.2f}")

if place:
    place_order(side, amount)

# 展示订单表
def _fmt_table(rows):
    if not rows:
        return pd.DataFrame([], columns=["id","symbol","side","amount","entry_price",
                                         "entry_time","status"])
    return pd.DataFrame(rows)

st.markdown("**Open orders**")
st.dataframe(_fmt_table(st.session_state.open_orders), use_container_width=True, height=190)

st.markdown("**Order history**")
hist_df = _fmt_table(st.session_state.order_history)
if not hist_df.empty:
    cols = ["id","symbol","side","amount","entry_price","exit_price","pnl",
            "entry_time","exit_time","status"]
    for c in cols:
        if c not in hist_df.columns:
            hist_df[c] = np.nan
    hist_df = hist_df[cols]
st.dataframe(hist_df, use_container_width=True, height=220)

# 说明
st.caption(
    "Notes: This is a paper-trading simulator. Orders settle after the selected horizon "
    "using close-to-close direction. Payout is +80% of stake on correct direction; "
    "otherwise you lose the stake. The model is retrained each refresh on the latest data."
)
