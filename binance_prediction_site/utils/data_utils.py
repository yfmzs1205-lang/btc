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
    # 官方公开镜像，常用于被墙/合规限制场景
    "https://data-api.binance.vision",
]

_REQ_HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 600) -> pd.DataFrame:
    """
    拉取K线，自动在多个域名之间切换；返回包含 open/high/low/close/volume 等列的 DataFrame。
    """
    last_err = None
    for base in BINANCE_BASES:
        url = f"{base}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            r = requests.get(url, timeout=10, headers=_REQ_HEADERS)
            if r.status_code == 200:
                k = r.json()
                cols = [
                    "open_time","open","high","low","close","volume",
                    "close_time","qav","num_trades","taker_base","taker_quote","ignore"
                ]
                df = pd.DataFrame(k, columns=cols)
                for c in ("open","high","low","close","volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
            else:
                last_err = RuntimeError(f"{r.status_code} from {base}")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All Binance endpoints failed. Last error: {last_err}")

def prepare_target(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """
    根据未来收盘价与当前收盘价的比较，生成二分类目标:
      y = 1 (上涨) / 0 (下跌或持平)
    会新增两列：future_close, y；并丢弃因 shift 产生的尾部缺失行。
    返回：带 y 的 DataFrame（后续特征工程从这里继续）。
    """
    out = df.copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["future_close"] = out["close"].shift(-horizon)
    out["y"] = (out["future_close"] > out["close"]).astype("Int64")
    out = out.dropna(subset=["future_close", "y"])
    return out
