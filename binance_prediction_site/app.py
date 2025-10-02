# binance_prediction_site/app.py
from __future__ import annotations
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
    ss.setdefault("balance", 100.0)            # 初始本金
    ss.setdefault("open_orders", [])           # 未结算订单
    ss.setdefault("order_history", [])         # 已结算订单
    ss.setdefault("order_id", 1)
    ss.setdefault("latest_price", None)
    ss.setdefault("latest_ts", None)
    ss.setdefault("df_cache", None)            # 数据缓存（避免免费层频繁请求）
    ss.setdefault("inc_model", None)           # 增量学习模型
    ss.setdefault("last_train_index", 0)       # 已增量训练到的索引
    ss.setdefault("best_thr", 0.5)             # 调优得到的决策阈值
    ss.setdefault("_last_auto_ts", 0.0)        # 自动刷新时间戳

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

    st.divider()
    st.subheader("Label & decision")
    # 标签噪声阈值：只有涨幅 > τ 才算上涨
    label_tau_bp = st.slider("Label threshold τ (basis points)", 0, 50, 10, step=1,
                             help="Only count as 'up' if future return > τ. 10bp = 0.1%.")
    label_tau = label_tau_bp / 10000.0

    # 观望阈值：概率在 [1-a, a] 区间内视为低置信，建议观望
    abstain_threshold = st.slider("No-trade band (a)", 0.50, 0.70, 0.55, 0.01,
                                  help="Require probability ≥ a to long, or ≤ (1-a) to short.")

    # 强信号提醒阈值
    alert_threshold = st.slider("Strong-signal alert threshold", 0.60, 0.90, 0.75, 0.01,
                                help="P(up) ≥ t ⇒ STRONG LONG; P(up) ≤ 1-t ⇒ STRONG SHORT")

    st.divider()
    st.subheader("Refresh")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_sec = st.slider("Interval (seconds)", 5, 120, 30, 5)
    if st.button("🔄 Refresh now"):
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
    st.session_state.latest_ts = pd.to_datetime(last_row["timestamp"])
    c1, c2 = st.columns(2)
    c1.metric("Latest price", f"{st.session_state.latest_price:.2f} {symbol_choice[-4:]}")
    c2.write(f"UTC time: **{st.session_state.latest_ts}**")

# =========================
# Features & label
# =========================
df_work = add_indicators(df_raw) if use_indicators else df_raw.copy()
df_label = prepare_target(df_work, horizon=horizon, target_col="target")

# —— 用阈值 τ 重写 target（抑制微小噪声）
ret = df_work["close"].shift(-horizon) / df_work["close"] - 1.0
df_label["target"] = (ret > label_tau).astype(int)

# 基础与新增特征
feature_cols = ["close"]
df_label["ret_1"] = df_label["close"].pct_change().fillna(0.0)
df_label["ret_3"] = df_label["close"].pct_change(3).fillna(0.0)
df_label["std_15"] = df_label["close"].rolling(15).std()
pos_num = (df_label["close"] - df_label["close"].rolling(20).min())
pos_den = (df_label["close"].rolling(20).max() - df_label["close"].rolling(20).min() + 1e-9)
df_label["pos_20"] = (pos_num / pos_den).clip(0, 1)

feature_cols += ["ret_1", "ret_3", "std_15", "pos_20"]

if use_indicators:
    for c in ["sma_14", "rsi_14", "macd", "macd_signal"]:
        if c in df_label.columns:
            df_label[c] = df_label[c].astype(float)
            feature_cols.append(c)

# =========================
# Training (batch + optional incremental) —— 含 NaN/Inf 清洗 & 阈值调优
# =========================
st.markdown("### Training")
tc1, tc2, tc3 = st.columns([1, 1, 2])
use_incremental = tc1.checkbox("Enable incremental learning (partial_fit)", value=True)
save_btn = tc2.button("💾 Save model")
load_file = tc3.file_uploader("Load a saved model (.pkl)", type=["pkl"], label_visibility="collapsed")

# 载入用户上传的模型
if load_file is not None:
    try:
        st.session_state.inc_model = pickle.load(load_file)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Load failed: {e}")

# —— 清洗：前后填充 + 去除残留缺失/无穷
use_cols = feature_cols + ["target"]
tmp = df_label[use_cols].replace([np.inf, -np.inf], np.nan)
tmp = tmp.fillna(method="ffill").fillna(method="bfill").dropna()
train_df = df_label.loc[tmp.index].copy()
train_df[use_cols] = tmp.astype(float)

if len(train_df) < 80:
    st.error("Not enough clean samples to train. Try disabling indicators or wait for more data.")
    st.stop()

# 初始化增量模型
if use_incremental and st.session_state.inc_model is None:
    st.session_state.inc_model = init_incremental_model()
    st.session_state.last_train_index = 0

# 批量基准模型（仅供参考指标）
with st.spinner("Training baseline model (batch) and evaluating..."):
    batch_model, base_metrics = train_classifier(train_df, feature_cols=feature_cols, target_col="target")

# 增量更新：仅对“新样本”partial_fit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
if use_incremental:
    start_idx = int(st.session_state.get("last_train_index", 0))
    end_idx = len(train_df) - 2  # 留最后一条用于推理
    if end_idx > start_idx:
        X_chunk = train_df[feature_cols].iloc[start_idx:end_idx].values
        y_chunk = train_df["target"].iloc[start_idx:end_idx].values.astype(int)
        partial_fit_step(st.session_state.inc_model, X_chunk, y_chunk)
        st.session_state.last_train_index = end_idx

    # 简单滚动评估 + 阈值调优（F1）
    X_eval = train_df[feature_cols].iloc[-400:].values
    y_eval = train_df["target"].iloc[-400:].values.astype(int)
    p = predict_proba_safe(st.session_state.inc_model, X_eval)[:, 1]

    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.45, 0.65, 41):
        y_hat = (p >= thr).astype(int)
        f1 = f1_score(y_eval, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    st.session_state.best_thr = best_thr

    y_pred = (p >= best_thr).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": best_f1,
    }
    current_model = st.session_state.inc_model
else:
    X_eval = train_df[feature_cols].iloc[-400:].values
    y_eval = train_df["target"].iloc[-400:].values.astype(int)
    p = predict_proba_safe(batch_model, X_eval)[:, 1]
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.45, 0.65, 41):
        y_hat = (p >= thr).astype(int)
        f1 = f1_score(y_eval, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    st.session_state.best_thr = best_thr
    y_pred = (p >= best_thr).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": best_f1,
    }
    current_model = batch_model

# 保存模型
if save_btn:
    try:
        buf = pickle.dumps(current_model)
        st.download_button("Download model.pkl", data=buf, file_name="model.pkl", mime="application/octet-stream")
    except Exception as e:
        st.error(f"Save failed: {e}")

# 展示指标
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
m2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
m3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
m4.metric("F1 (tuned)", f"{metrics.get('f1', 0)*100:.2f}%")
st.caption(f"Auto-tuned threshold: **{st.session_state.best_thr:.3f}** | Label τ = **{label_tau_bp} bp** | No-trade band ≥ **{abstain_threshold:.2f}** | Alert ≥ **{alert_threshold:.2f}**")

# =========================
# Inference + Alerts
# =========================
def predict_direction_latest(model, df: pd.DataFrame, thr: float) -> tuple[int, float, float]:
    row = pd.DataFrame([df[feature_cols].iloc[-1].values], columns=feature_cols)
    proba = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else predict_proba_safe(model, row.values)[0]
    prob_up = float(proba[1])
    pred = int(prob_up >= thr)
    # 置信度：与阈值的距离（0~0.5），再映射到 0~1 便于直觉
    conf = float(2 * abs(prob_up - thr))
    return pred, conf, prob_up

pred, conf, prob = predict_direction_latest(current_model, train_df, st.session_state.best_thr)
st.subheader(f"Prediction for the next {horizon} minutes")
st.info(
    f"📌 P(up) = **{prob*100:.2f}%** | threshold = **{st.session_state.best_thr:.3f}** → "
    f"**{'rise' if pred==1 else 'fall'}** (confidence {conf*100:.1f}%)."
)

# 观望与强信号提醒
abstain_low, abstain_high = (1 - abstain_threshold), abstain_threshold
should_abstain = (prob > abstain_low) and (prob < abstain_high)

# 强信号：高于 alert_threshold 或低于 1-alert_threshold
strong_long = prob >= alert_threshold
strong_short = prob <= (1 - alert_threshold)

if should_abstain:
    st.warning("⚠️ Low confidence: probability is in the no-trade band. Consider skipping this round.")
else:
    st.success("✅ Confidence passes no-trade band.")

# 🔔 强信号提醒（视觉 Toast + 高亮）
try:
    if strong_long:
        st.toast(f"🔔 STRONG LONG signal for {symbol_choice}: P(up)={prob*100:.2f}%", icon="✅")
        st.success("🟢 Strong LONG signal detected.")
    elif strong_short:
        st.toast(f"🔔 STRONG SHORT signal for {symbol_choice}: P(up)={prob*100:.2f}%", icon="⚠️")
        st.error("🔴 Strong SHORT signal detected.")
except Exception:
    # 旧版 Streamlit 没有 toast 就用普通提示
    if strong_long:
        st.success(f"🔔 STRONG LONG: P(up)={prob*100:.2f}%")
    elif strong_short:
        st.error(f"🔔 STRONG SHORT: P(up)={prob*100:.2f}%")

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
            od["exit_time"] = str(df.iloc[od["settle_index"]].get("timestamp", "N/A"))
            od["status"] = "WIN" if win else "LOSS"
            st.session_state.order_history.append(od)
            to_close.append(od)
    for od in to_close:
        st.session_state.open_orders.remove(od)

cur_idx = len(train_df) - 1
if cur_idx >= 0:
    settle_orders(cur_idx, train_df)

cA, cB, cC, cD = st.columns([1, 1, 1, 2])
side = cA.radio("Direction", ["LONG", "SHORT"], horizontal=True)
amount = cB.number_input("Order amount (USDT)", min_value=5, max_value=125, step=5, value=10)
cC.metric("Balance (USDT)", f"{st.session_state.balance:.2f}")
place = cD.button("✅ Place order now")

def place_order(side: str, amount: float):
    # 观望带：禁止下单
    if should_abstain:
        st.error("Probability in no-trade band. Order blocked to avoid low-confidence trades.")
        return
    if amount < 5 or amount > 125:
        st.error("Order size must be between 5 and 125 USDT.")
        return
    if amount > st.session_state.balance:
        st.error("Insufficient balance.")
        return
    entry_index = len(train_df) - 1
    order = {
        "id": st.session_state.order_id,
        "symbol": symbol_choice,
        "side": side,
        "amount": float(amount),
        "entry_price": float(train_df.iloc[entry_index]["close"]),
        "entry_time": str(train_df.iloc[entry_index].get("timestamp", "N/A")),
        "entry_index": entry_index,
        "settle_index": entry_index + horizon,   # 1m K线 ⇒ horizon 分钟后
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

st.caption(
    "Auto-refresh updates data and retrains. Strong-signal alerts trigger when P(up) crosses alert threshold. "
    "Incremental learning adapts during the session. Use Save/Load to persist a model snapshot."
)

# =========================
# Auto-refresh loop (simple)
# =========================
if auto_refresh:
    # 控制刷新频率，避免过于频繁
    now = time.time()
    if now - st.session_state.get("_last_auto_ts", 0) >= refresh_sec:
        # 清除缓存 → 重新拉取与训练
        st.session_state.df_cache = None
        st.session_state._last_auto_ts = now
        # 轻量延时，避免连环触发
        time.sleep(0.2)
        st.experimental_rerun()
