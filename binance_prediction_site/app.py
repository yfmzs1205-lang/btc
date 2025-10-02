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
    train_classifier,          # 批量 RF
    init_incremental_model,    # 增量 _IncModel(Scaler+SGD)
    partial_fit_step,
    predict_proba_safe,
)

# ---------------------- #
# Page & Session
# ---------------------- #
st.set_page_config(page_title="Binance MTF Trend Predictor (Education Only)", layout="wide")
st.title("📈 Binance Multi-Timeframe Trend Predictor (Education Only)")
st.caption("Educational demo only. Uses 1m klines (+3m/5m features), incremental learning, walk-forward eval, "
           "probability calibration, and soft-voting ensemble. Includes paper trading. No real trading.")

def _init_state():
    ss = st.session_state
    ss.setdefault("balance", 100.0)
    ss.setdefault("open_orders", [])
    ss.setdefault("order_history", [])
    ss.setdefault("order_id", 1)
    ss.setdefault("latest_price", None)
    ss.setdefault("latest_ts", None)
    ss.setdefault("df_cache_1m", None)
    ss.setdefault("df_cache_3m", None)
    ss.setdefault("df_cache_5m", None)
    ss.setdefault("inc_model", None)
    ss.setdefault("last_train_index", 0)
    ss.setdefault("_n_features", None)
    ss.setdefault("best_thr", 0.5)
    ss.setdefault("_last_auto_ts", 0.0)
_init_state()

# ---------------------- #
# Sidebar
# ---------------------- #
with st.sidebar:
    st.header("Configuration")
    symbol = st.selectbox("Trading pair", ["BTCUSDT", "ETHUSDT"])
    horizon = 10 if st.radio("Prediction horizon", ["10 minutes", "30 minutes"]).startswith("10") else 30
    use_ind = st.checkbox("Add classic indicators (SMA/RSI/MACD)", value=True)

    st.divider()
    st.subheader("Label & data")
    label_basis = st.selectbox("Label basis", ["close", "mid_price ( (H+L)/2 )"])
    tau_bp = st.slider("Label threshold τ (basis points)", 0, 50, 10, 1,
                       help="Only count as UP if future return > τ. 10bp=0.1%")
    tau = tau_bp / 10000.0
    drop_neutral = st.checkbox("Drop 'neutral' samples (|ret|≤τ) from training", value=True)
    wins_clip_bp = st.slider("Winsorize returns (bp)", 0, 100, 20, 5,
                             help="Clip returns into [-x, x] bp to suppress spikes. 0 to disable.")

    st.divider()
    st.subheader("Validation")
    walk_forward = st.checkbox("Enable walk-forward validation", value=True)
    tx_cost_bp = st.slider("Transaction cost / slippage (bp)", 0, 20, 2, 1)
    calibrate = st.checkbox("Calibrate probabilities (auto sigmoid/isotonic)", value=True)

    st.divider()
    st.subheader("Online learning")
    use_inc = st.checkbox("Enable incremental learning (partial_fit)", value=True)
    inc_window = st.slider("Incremental window N (most recent samples)", 500, 5000, 2000, 100)

    st.divider()
    st.subheader("Decision & alerts")
    abstain_a = st.slider("No-trade band a", 0.50, 0.70, 0.55, 0.01)
    alert_thr = st.slider("Strong-signal alert threshold", 0.60, 0.90, 0.75, 0.01)
    ens_w_rf = st.slider("Ensemble weight: RF", 0.0, 1.0, 0.5, 0.05)
    ens_w_lin = 1.0 - ens_w_rf

    st.divider()
    st.subheader("Refresh")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_sec = st.slider("Interval (seconds)", 5, 120, 30, 5)
    if st.button("🔄 Refresh now"):
        st.session_state.df_cache_1m = None
        st.session_state.df_cache_3m = None
        st.session_state.df_cache_5m = None

# ---------------------- #
# Helpers
# ---------------------- #
def get_df_cached(symbol: str, interval: str, limit: int = 600) -> pd.DataFrame:
    key = f"df_cache_{interval}"
    if st.session_state.get(key) is None:
        st.session_state[key] = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    return st.session_state[key].copy()

def add_classic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # SMA/RSI/MACD
    x["sma_14"] = x["close"].rolling(14).mean()
    delta = x["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    rs = pd.Series(up, index=x.index).rolling(14).mean() / (pd.Series(down, index=x.index).rolling(14).mean() + 1e-9)
    x["rsi_14"] = 100 - 100/(1+rs)
    ema_fast = x["close"].ewm(span=12, adjust=False).mean()
    ema_slow = x["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    x["macd"] = macd_line
    x["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean()
    return x

def add_rich_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # returns
    x["ret_1"] = x["close"].pct_change().fillna(0.0)
    x["ret_3"] = x["close"].pct_change(3).fillna(0.0)
    x["ret_5"] = x["close"].pct_change(5).fillna(0.0)
    # volatility
    x["std_5"] = x["close"].rolling(5).std()
    x["std_15"] = x["close"].rolling(15).std()
    # ATR(14)
    tr = np.maximum(x["high"]-x["low"], np.maximum((x["high"]-x["close"].shift()).abs(),
                                                   (x["low"]-x["close"].shift()).abs()))
    x["atr_14"] = pd.Series(tr).rolling(14).mean()
    # volume features
    vol_ma = x["volume"].rolling(20).mean()
    x["vol_z"] = ((x["volume"] - vol_ma) / (vol_ma + 1e-9)).fillna(0.0)
    x["ret1_volz"] = x["ret_1"] * x["vol_z"]
    # position in range
    roll_min = x["close"].rolling(20).min()
    roll_max = x["close"].rolling(20).max()
    x["pos_20"] = ((x["close"] - roll_min) / (roll_max - roll_min + 1e-9)).clip(0, 1)
    # Bollinger
    ma20 = x["close"].rolling(20).mean()
    sd20 = x["close"].rolling(20).std()
    x["boll_z"] = (x["close"] - ma20) / (sd20 + 1e-9)
    x["boll_bw"] = (2*sd20) / (ma20 + 1e-9)
    # KDJ
    ll9 = x["low"].rolling(9).min()
    hh9 = x["high"].rolling(9).max()
    rsv = (x["close"] - ll9) / (hh9 - ll9 + 1e-9) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    x["kdj_k"] = k
    x["kdj_d"] = d
    x["kdj_j"] = 3*k - 2*d
    # EMA cross
    e12 = x["close"].ewm(span=12, adjust=False).mean()
    e26 = x["close"].ewm(span=26, adjust=False).mean()
    x["ema_fast_gt"] = (e12 > e26).astype(int)
    x["ema_dist"] = (e12 - e26) / (x["close"] + 1e-9)
    # OBV
    x["obv"] = (np.sign(x["close"].diff().fillna(0.0)) * x["volume"]).cumsum().fillna(0.0)
    # Candle body/wick
    x["body"] = (x["close"] - x["open"]).abs()
    x["upper_wick"] = (x["high"] - x[["close","open"]].max(axis=1)).clip(lower=0)
    x["lower_wick"] = (x[["close","open"]].min(axis=1) - x["low"]).clip(lower=0)
    x["body_ratio"] = x["body"] / (x["high"] - x["low"] + 1e-9)
    # time factors
    ts = pd.to_datetime(x["timestamp"])
    x["min_of_hour"] = ts.dt.minute.astype(int)
    x["is_weekend"] = ts.dt.weekday.isin([5,6]).astype(int)
    return x

def align_merge_mtf(base_1m: pd.DataFrame, df3: pd.DataFrame, df5: pd.DataFrame) -> pd.DataFrame:
    """对齐 3m/5m 到 1m 时间戳（向前填充对齐）。"""
    out = base_1m.copy()
    for add, prefix in [(df3, "3m"), (df5, "5m")]:
        y = add[["timestamp","close","volume"]].copy()
        y = y.set_index(pd.to_datetime(y["timestamp"]))[["close","volume"]]
        x = out.set_index(pd.to_datetime(out["timestamp"]))
        z = y.reindex(x.index, method="ffill")
        out[f"close_{prefix}"] = z["close"].values
        out[f"ret1_{prefix}"] = out[f"close_{prefix}"].pct_change().fillna(0.0)
        out[f"vol_{prefix}"] = z["volume"].values
    return out

def winsorize_ret(s: pd.Series, bp: int) -> pd.Series:
    if bp <= 0: 
        return s
    r = bp / 10000.0
    return s.clip(-r, r)

# ---------------------- #
# Data: 1m + 3m + 5m
# ---------------------- #
df1m = get_df_cached(symbol, "1m", 1000)  # 多拿点样本
df3m = get_df_cached(symbol, "3m",  1000)
df5m = get_df_cached(symbol, "5m",  1000)

# 最新价展示
if not df1m.empty:
    last = df1m.iloc[-1]
    st.session_state.latest_price = float(last["close"])
    st.session_state.latest_ts = pd.to_datetime(last["timestamp"])
    a,b = st.columns(2)
    a.metric("Latest price", f"{st.session_state.latest_price:.2f} {symbol[-4:]}")
    b.write(f"UTC time: **{st.session_state.latest_ts}**")

# 多周期合并 + 指标
df_work = align_merge_mtf(df1m, df3m, df5m)
if use_ind:
    df_work = add_classic_indicators(df_work)
df_feat = add_rich_features(df_work)

# ---------------------- #
# Labeling
# ---------------------- #
basis_col = "close" if label_basis.startswith("close") else None
future_ref = None
if basis_col is None:
    # mid price
    mid = (df_work["high"] + df_work["low"]) / 2.0
    future_ref = mid.shift(-horizon)
    cur_ref = mid
else:
    future_ref = df_work["close"].shift(-horizon)
    cur_ref = df_work["close"]

raw_ret = future_ref/cur_ref - 1.0
raw_ret = winsorize_ret(raw_ret, wins_clip_bp)
target_bin = (raw_ret > tau).astype(int)  # 二分类标签（先粗定义）

# “三分类思路”：中性区间 |ret|≤τ 作为 neutral；默认过滤掉 neutral
neutral_mask = raw_ret.abs() <= tau
if drop_neutral:
    keep_idx = (~neutral_mask).to_numpy().nonzero()[0]
else:
    keep_idx = np.arange(len(df_feat))

# ---------------------- #
# Feature table
# ---------------------- #
feature_cols = [
    # price & returns
    "close","ret_1","ret_3","ret_5","std_5","std_15","atr_14",
    # volume & interactions
    "volume","vol_z","ret1_volz",
    # range/position & bands
    "pos_20","boll_z","boll_bw",
    # KDJ / EMA cross / OBV
    "kdj_k","kdj_d","kdj_j","ema_fast_gt","ema_dist","obv",
    # candle + time
    "body_ratio","upper_wick","lower_wick","min_of_hour","is_weekend",
    # multi-timeframe
    "close_3m","ret1_3m","vol_3m","close_5m","ret1_5m","vol_5m",
]
# 保证存在（早期窗口可能缺失）
feature_cols = [c for c in feature_cols if c in df_feat.columns]
df_feat[feature_cols] = df_feat[feature_cols].astype(float)

# 清洗并构造训练表
df_all = df_feat.copy()
df_all["target"] = target_bin
use_cols = feature_cols + ["target"]
tmp = df_all[use_cols].replace([np.inf,-np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").dropna()

# 过滤中性样本（如选）
tmp = tmp.iloc[keep_idx] if drop_neutral else tmp

train_df = df_all.loc[tmp.index].copy()
train_df[use_cols] = tmp.astype(float)

if len(train_df) < 200:
    st.error("Samples after cleaning are too few. Reduce τ / disable some options or wait for more data.")
    st.stop()

# 特征维度变化 → 重置增量模型
n_feats = len(feature_cols)
if st.session_state._n_features is None or st.session_state._n_features != n_feats:
    st.session_state._n_features = n_feats
    if st.session_state.inc_model is not None:
        st.warning("Feature dimension changed. Reset incremental model to avoid mismatch.")
    st.session_state.inc_model = None
    st.session_state.last_train_index = 0

# ---------------------- #
# Batch model (RF) + (optional) calibration & walk-forward
# ---------------------- #
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

with st.spinner("Training batch model (RandomForest) and evaluating..."):
    rf_model, base_metrics = train_classifier(
        train_df, feature_cols=feature_cols, target_col="target"
    )  # 内含 class_weight='balanced'

# 概率校准（对 RF 做校准更稳一些）
if calibrate:
    try:
        # 留最近 20% 做校准
        cut = int(len(train_df)*0.8)
        Xc, yc = train_df[feature_cols].iloc[:cut].values, train_df["target"].iloc[:cut].values.astype(int)
        Xcv, ycv = train_df[feature_cols].iloc[cut:].values, train_df["target"].iloc[cut:].values.astype(int)
        # 如果样本较少，用 sigmoid；否则 isotonic
        method = "isotonic" if len(Xcv) >= 1000 else "sigmoid"
        cal = CalibratedClassifierCV(rf_model, method=method, cv="prefit")
        cal.fit(Xcv, ycv)
        rf_calibrated = cal
    except Exception:
        rf_calibrated = rf_model
else:
    rf_calibrated = rf_model

# Walk-forward（轻量）：分 5 段滚动，统计平均分数与最佳阈值（按净收益分数）
def wf_eval(df: pd.DataFrame, k_splits: int = 5, cost: float = 0.0):
    n = len(df)
    seg = n // (k_splits + 1)
    scores, best_thr = [], 0.5
    best_gain = -1e9
    for i in range(1, k_splits+1):
        train_end = seg * i
        valid_end = seg * (i+1)
        tr = df.iloc[:train_end]; va = df.iloc[train_end:valid_end]
        if len(va) < 50: continue
        # 重新训练一个 RF（简易）
        m, _ = train_classifier(tr, feature_cols, "target")
        if calibrate:
            try:
                method = "isotonic" if len(tr) >= 1000 else "sigmoid"
                cal = CalibratedClassifierCV(m, method=method, cv=3)
                cal.fit(tr[feature_cols].values, tr["target"].values.astype(int))
                m = cal
            except Exception:
                pass
        p = predict_proba_safe(m, va[feature_cols].values)[:,1]
        # 阈值扫描（考虑交易成本）：净收益分数 = (命中率-失误率) - 成本
        thr_local, gain_local = 0.5, -1e9
        for thr in np.linspace(0.45, 0.65, 41):
            y_hat = (p >= thr).astype(int)
            hit = (y_hat == va["target"].values).mean()
            miss = 1 - hit
            # 简单把每次交易扣一份 cost（双边合并为一次）
            # 交易触发比例 ≈ 有效预测比例
            trade_ratio = max(1e-9, ( (p>=thr).mean() + (p<=1-thr).mean() )/2 )
            gain = (hit - miss) - trade_ratio * cost
            if gain > gain_local:
                gain_local, thr_local = gain, thr
        scores.append(gain_local)
        if gain_local > best_gain:
            best_gain, best_thr = gain_local, thr_local
    return (np.mean(scores) if scores else 0.0), best_thr

wf_score, thr_wf = wf_eval(train_df, k_splits=5, cost=tx_cost_bp/10000.0) if walk_forward else (0.0, 0.5)

# ---------------------- #
# Incremental learning (windowed)
# ---------------------- #
if use_inc and st.session_state.inc_model is None:
    st.session_state.inc_model = init_incremental_model()
    st.session_state.last_train_index = 0

current_model = rf_calibrated  # 先用 RF；后面做集成
lin_proba = None

if use_inc:
    # 用最近 inc_window 做增量（窗口式遗忘旧 regime）
    start_idx = max(0, len(train_df) - inc_window)
    inc_slice = train_df.iloc[start_idx:].copy()
    X_inc = inc_slice[feature_cols].values
    y_inc = inc_slice["target"].values.astype(int)
    # 分块增量，避免一次太大
    chunk = max(200, inc_window//10)
    for i in range(0, len(inc_slice)-1, chunk):
        Xc = inc_slice[feature_cols].iloc[i:i+chunk].values
        yc = inc_slice["target"].iloc[i:i+chunk].values.astype(int)
        partial_fit_step(st.session_state.inc_model, Xc, yc)

    # 用增量模型算一份概率
    lin_proba = predict_proba_safe(st.session_state.inc_model, train_df[feature_cols].values)[:,1]

# ---------------------- #
# Ensemble (RF + Linear)
# ---------------------- #
def ensemble_proba(p_rf: np.ndarray, p_lin: np.ndarray|None, w_rf: float, w_lin: float) -> np.ndarray:
    if p_lin is None:
        return p_rf
    p = w_rf*p_rf + w_lin*p_lin
    return np.clip(p, 1e-6, 1-1e-6)

p_rf_full = predict_proba_safe(rf_calibrated, train_df[feature_cols].values)[:,1]
p_full = ensemble_proba(p_rf_full, lin_proba, ens_w_rf, ens_w_lin)

# 阈值调优：综合 walk-forward 推荐与本地评估（考虑交易成本）
best_thr, best_f1 = (thr_wf if walk_forward else 0.5), -1.0
X_eval = train_df[feature_cols].iloc[-400:].values
y_eval = train_df["target"].iloc[-400:].values.astype(int)
p_eval = p_full[-400:]
for thr in np.linspace(0.45, 0.65, 41):
    y_hat = (p_eval >= thr).astype(int)
    f1 = f1_score(y_eval, y_hat, zero_division=0)
    # 简单把交易成本惩罚到分数上（可换收益函数）
    trade_ratio = max(1e-9, ((p_eval>=thr).mean() + (p_eval<=1-thr).mean())/2)
    score = f1 - (tx_cost_bp/10000.0)*trade_ratio
    if score > best_f1:
        best_f1, best_thr = float(score), float(thr)

st.session_state.best_thr = best_thr

y_pred = (p_eval >= best_thr).astype(int)
metrics = {
    "accuracy": float(accuracy_score(y_eval, y_pred)),
    "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
    "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
    "f1_adj": best_f1,
    "wf_score": wf_score,
}

# 展示指标
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
c3.metric("Recall", f"{metrics['recall']*100:.2f}%")
c4.metric("F1*(cost)", f"{metrics['f1_adj']*100:.2f}%")
c5.metric("WF gain", f"{metrics['wf_score']:.4f}")
st.caption(f"best_thr={best_thr:.3f} | τ={tau_bp}bp | cost={tx_cost_bp}bp | ensemble RF:{ens_w_rf:.2f}/LIN:{ens_w_lin:.2f}")

# ---------------------- #
# Inference + Alerts
# ---------------------- #
def predict_latest_prob(p_series: np.ndarray) -> float:
    return float(p_series[-1])

prob_up = predict_latest_prob(p_full)
pred = int(prob_up >= best_thr)
conf = 2*abs(prob_up - best_thr)

st.subheader(f"Prediction for next {horizon} minutes")
st.info(f"P(up)={prob_up*100:.2f}% @ thr={best_thr:.3f} → **{'RISE' if pred==1 else 'FALL'}** "
        f"(confidence {conf*100:.1f}%)")

ab_low, ab_high = 1 - abstain_a, abstain_a
no_trade = (ab_low < prob_up < ab_high)
strong_long = prob_up >= alert_thr
strong_short = prob_up <= (1 - alert_thr)

if no_trade:
    st.warning("⚠️ In no-trade band → consider SKIP")
else:
    st.success("✅ Confidence passes no-trade band")

try:
    if strong_long:
        st.toast(f"🔔 STRONG LONG {symbol}: P(up)={prob_up*100:.2f}%", icon="✅")
        st.success("🟢 Strong LONG signal")
    elif strong_short:
        st.toast(f"🔔 STRONG SHORT {symbol}: P(up)={prob_up*100:.2f}%", icon="⚠️")
        st.error("🔴 Strong SHORT signal")
except Exception:
    if strong_long:
        st.success(f"🔔 STRONG LONG: {prob_up*100:.2f}%")
    elif strong_short:
        st.error(f"🔔 STRONG SHORT: {prob_up*100:.2f}%")

# ---------------------- #
# Recent chart
# ---------------------- #
fig = go.Figure()
tail = df1m.tail(200)
fig.add_trace(go.Scatter(x=pd.to_datetime(tail["timestamp"]), y=tail["close"], mode="lines", name="Close"))
fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
st.subheader("Recent Price Chart (last 200 minutes)")
st.plotly_chart(fig, use_container_width=True)

# ---------------------- #
# Paper trading (unchanged core)
# ---------------------- #
st.subheader("💸 Paper Trading (Simulation)")
st.caption("Initial balance 100 USDT. Min 5U, max 125U. +80% on correct direction; lose stake on wrong direction.")

def settle_orders(cur_idx: int, df: pd.DataFrame):
    close = df["close"].values
    ts = df["timestamp"].values
    to_close = []
    for od in st.session_state.open_orders:
        if cur_idx >= od["settle_index"]:
            entry_p = close[od["entry_index"]]
            exit_p  = close[od["settle_index"]]
            win = (exit_p > entry_p) if od["side"]=="LONG" else (exit_p < entry_p)
            pnl = round(od["amount"]*0.8,2) if win else round(-od["amount"],2)
            st.session_state.balance += pnl
            od.update({
                "pnl": pnl,
                "exit_price": float(exit_p),
                "exit_time": str(ts[od["settle_index"]]),
                "status": "WIN" if win else "LOSS"
            })
            st.session_state.order_history.append(od)
            to_close.append(od)
    for od in to_close:
        st.session_state.open_orders.remove(od)

cur_idx = len(train_df) - 1
if cur_idx >= 0:
    settle_orders(cur_idx, train_df)

A,B,C,D = st.columns([1,1,1,2])
side = A.radio("Direction", ["LONG","SHORT"], horizontal=True)
amount = B.number_input("Order amount (USDT)", 5, 125, 10, 5)
C.metric("Balance (USDT)", f"{st.session_state.balance:.2f}")
place = D.button("✅ Place order now")

def place_order():
    if no_trade:
        st.error("Inside no-trade band, order blocked.")
        return
    if amount > st.session_state.balance:
        st.error("Insufficient balance.")
        return
    idx = len(train_df) - 1
    od = {
        "id": st.session_state.order_id,
        "symbol": symbol,
        "side": side,
        "amount": float(amount),
        "entry_price": float(train_df.iloc[idx]["close"]),
        "entry_time": str(train_df.iloc[idx]["timestamp"]),
        "entry_index": idx,
        "settle_index": idx + horizon,
        "horizon_min": horizon,
        "status": "OPEN",
    }
    st.session_state.balance -= amount
    st.session_state.open_orders.append(od)
    st.session_state.order_id += 1
    st.success(f"Order #{od['id']} placed: {side} {amount}U @ {od['entry_price']:.2f}")

if place: place_order()

def _as_table(rows):
    if not rows: return pd.DataFrame([], columns=["id","symbol","side","amount","entry_price","entry_time","status"])
    return pd.DataFrame(rows)

st.markdown("**Open orders**")
st.dataframe(_as_table(st.session_state.open_orders), use_container_width=True, height=190)

st.markdown("**Order history**")
hist = _as_table(st.session_state.order_history)
if not hist.empty:
    cols = ["id","symbol","side","amount","entry_price","exit_price","pnl","entry_time","exit_time","status"]
    for c in cols:
        if c not in hist.columns: hist[c]=np.nan
    hist = hist[cols]
st.dataframe(hist, use_container_width=True, height=220)

st.caption("Walk-forward selects a robust threshold (cost-aware). Incremental model adapts on a sliding window.")

# ---------------------- #
# Auto-refresh
# ---------------------- #
if auto_refresh:
    now = time.time()
    if now - st.session_state.get("_last_auto_ts", 0) >= refresh_sec:
        st.session_state.df_cache_1m = None
        st.session_state.df_cache_3m = None
        st.session_state.df_cache_5m = None
        st.session_state._last_auto_ts = now
        time.sleep(0.2)
        st.rerun()
