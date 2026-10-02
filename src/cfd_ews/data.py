from __future__ import annotations

from dataclasses import dataclass
import os
import pandas as pd
import requests


@dataclass(frozen=True)
class APIConfig:
    alpha_vantage_key: str | None = os.getenv("ALPHA_VANTAGE_API_KEY")


class AlphaVantageClient:
    """Optional market-data adapter.

    The system remains usable in demo mode without credentials. Production
    studies should persist raw responses with retrieval timestamps.
    """

    base_url = "https://www.alphavantage.co/query"

    def __init__(self, config: APIConfig | None = None):
        self.config = config or APIConfig()

    def daily_adjusted(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        if not self.config.alpha_vantage_key:
            raise RuntimeError("ALPHA_VANTAGE_API_KEY is required for live market data.")
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.config.alpha_vantage_key,
        }
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        series = payload.get("Time Series (Daily)")
        if not series:
            raise ValueError(f"No daily data returned for {symbol}: {payload}")
        out = pd.DataFrame.from_dict(series, orient="index")
        out.index = pd.to_datetime(out.index)
        out = out.rename(columns=lambda c: c.split(". ", 1)[-1])
        return out.apply(pd.to_numeric, errors="coerce").sort_index()


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_snapshot(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
