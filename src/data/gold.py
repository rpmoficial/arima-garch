"""Utilities for downloading and preparing gold price data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - runtime safeguard
    raise ImportError(
        "yfinance is required to download gold price data. Install it via `pip install yfinance`."
    ) from exc


@dataclass(frozen=True)
class GoldPriceConfig:
    """Configuration for downloading gold prices."""

    symbol: str = "GC=F"
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    interval: str = "1d"


def load_gold_prices(config: GoldPriceConfig | None = None) -> pd.Series:
    """Download daily gold prices from Yahoo Finance.

    Parameters
    ----------
    config:
        Optional configuration containing download parameters. When omitted,
        the function downloads the full history available for the continuous
        gold futures contract (``GC=F``).

    Returns
    -------
    pandas.Series
        Adjusted close prices indexed by ``DatetimeIndex``.
    """
    cfg = config or GoldPriceConfig()

    data = yf.download(
        cfg.symbol,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError("No data was returned from Yahoo Finance. Check the symbol or date range.")

    return data["Adj Close"].rename("gold_price")


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""
    cleaned = prices.dropna().astype("float64")
    if cleaned.empty:
        raise ValueError("Price series only contains missing values.")

    log_prices = np.log(cleaned)
    log_returns = log_prices.diff().dropna().rename("log_return")
    return log_returns
