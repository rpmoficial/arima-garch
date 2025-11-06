"""Time-series modelling utilities for ARIMA-GARCH style models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA


@dataclass(frozen=True)
class ArimaGarchResult:
    """Container for ARIMA-GARCH style results."""

    arima_result: object
    volatility_result: object

    @property
    def residuals(self) -> pd.Series:
        return pd.Series(
            self.arima_result.resid,
            index=getattr(self.arima_result.model.data, "row_labels", None),
        )


def fit_arima(returns: pd.Series, order: Tuple[int, int, int]) -> object:
    """Fit an ARIMA model to the provided return series."""
    model = ARIMA(returns, order=order)
    return model.fit()


def fit_garch(residuals: pd.Series, order: Tuple[int, int], dist: str = "normal") -> object:
    """Fit a symmetric GARCH model to ARIMA residuals."""
    resid = residuals.dropna() * 100  # scale to percentage
    garch = arch_model(resid, mean="Zero", vol="GARCH", p=order[0], q=order[1], dist=dist)
    return garch.fit(disp="off")


def fit_gjrgarch(residuals: pd.Series, order: Tuple[int, int, int], dist: str = "normal") -> object:
    """Fit a GJR-GARCH (threshold GARCH) model to ARIMA residuals."""
    p, o, q = order
    resid = residuals.dropna() * 100
    gjr = arch_model(resid, mean="Zero", vol="GARCH", p=p, o=o, q=q, dist=dist)
    return gjr.fit(disp="off")


def fit_arima_garch(
    returns: pd.Series,
    arima_order: Tuple[int, int, int] = (1, 0, 1),
    garch_order: Tuple[int, int] = (1, 1),
    dist: str = "normal",
) -> ArimaGarchResult:
    """Fit an ARIMA model with GARCH volatility."""
    arima_res = fit_arima(returns, arima_order)
    garch_res = fit_garch(arima_res.resid, garch_order, dist=dist)
    return ArimaGarchResult(arima_res, garch_res)


def fit_arima_gjrgarch(
    returns: pd.Series,
    arima_order: Tuple[int, int, int] = (1, 0, 1),
    gjr_order: Tuple[int, int, int] = (1, 1, 1),
    dist: str = "normal",
) -> ArimaGarchResult:
    """Fit an ARIMA model with GJR-GARCH volatility."""
    arima_res = fit_arima(returns, arima_order)
    gjr_res = fit_gjrgarch(arima_res.resid, gjr_order, dist=dist)
    return ArimaGarchResult(arima_res, gjr_res)


def forecast_volatility(
    result: ArimaGarchResult,
    horizon: int = 30,
    retransform: bool = True,
) -> pd.DataFrame:
    """Generate mean and volatility forecasts from a fitted ARIMA-GARCH model."""
    arima_forecast = result.arima_result.get_forecast(steps=horizon)
    mean_forecast = arima_forecast.predicted_mean

    vol_forecast = result.volatility_result.forecast(horizon=horizon)
    cond_var = vol_forecast.variance.iloc[-1]

    if retransform:
        cond_std = np.sqrt(cond_var) / 100
    else:
        cond_std = np.sqrt(cond_var)

    cond_std.index = mean_forecast.index

    df = pd.DataFrame({
        "mean": mean_forecast,
        "volatility": cond_std,
    })
    return df
