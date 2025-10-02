import pandas as pd
import requests
from typing import List

# 依次尝试这些 Binance 接口（主站 + 备选 + 公共镜像）
BINANCE_BASES: List[str] = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api-gcp.binance.com",
    # 官方公开镜像
    "https://data-api.binance.vision",
]

# 一些环境不带 UA 会被拦，统一加一个
_REQ_HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 600) -> pd.DataFrame:
    """
    拉取K线，自动在多个域名之间切换；返回包含 open/high/low/close/volume/timestamp 等列的 DataFrame。
    """
    last_err = None
    for base in BINANCE_BASES:
        url = f"{base}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            r = requests.get(url, timeout=10, headers=_REQ_HEADERS)
            if r.status_code == 200:
                k = r.json()
                cols = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
                ]
                df = pd.DataFrame(k, columns=cols)

                # 数值列转型
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                # ✅ 新增用于画图/展示的统一时间列
                # Binance 返回的 open_time 单位是毫秒
                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

                # 按时间排序，避免乱序
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df
            else:
                # 451/429/5xx → 尝试下一个域名
                last_err = RuntimeError(f"{r.status_code} from {base}")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All Binance endpoints failed. Last error: {last_err}")

def prepare_target(df: pd.DataFrame, horizon: int = 10, target_col: str = "target") -> pd.DataFrame:
    """
    生成二分类目标 target：未来 horizon 步收盘价是否高于当前收盘价。
    新增列：future_close, target；丢弃尾部缺失行。
    """
    out = df.copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["future_close"] = out["close"].shift(-horizon)
    # target: 1=上涨, 0=不涨
    out[target_col] = (out["future_close"] > out["close"]).astype(int)
    out = out.dropna(subset=["future_close"])
    return out
