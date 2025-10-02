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

# 统一的请求头（有些环境没有 UA 会被拦）
_REQ_HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_klines(symbol="BTCUSDT", interval="1m", limit=600) -> pd.DataFrame:
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
                # 只转你真正用到的列，防止类型错误
                for c in ("open","high","low","close","volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
            else:
                # 451/4xx/5xx 都尝试下一个域名
                last_err = RuntimeError(f"{r.status_code} from {base}")
        except Exception as e:
            last_err = e
            continue
    # 全部失败再抛错
    raise RuntimeError(f"All Binance endpoints failed. Last error: {last_err}")
