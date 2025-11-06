"""Command line entry point for fitting ARIMA-GARCH models to gold prices."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .data.gold import GoldPriceConfig, compute_log_returns, load_gold_prices
from .models.arima_garch import (
    ArimaGarchResult,
    fit_arima_garch,
    fit_arima_gjrgarch,
    forecast_volatility,
)


def parse_order(value: str, length: int) -> tuple[int, ...]:
    parts = tuple(int(x) for x in value.split(","))
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"Expected {length} comma-separated integers, got {value!r}")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--symbol", type=str, default="GC=F", help="Yahoo Finance ticker symbol")
    parser.add_argument("--arima", type=lambda s: parse_order(s, 3), default=(1, 0, 1))
    parser.add_argument("--garch", type=lambda s: parse_order(s, 2), default=(1, 1))
    parser.add_argument("--gjr", type=lambda s: parse_order(s, 3), default=(1, 1, 1))
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory to store CSV files")
    return parser


def maybe_parse_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(name: str, result: ArimaGarchResult, horizon: int, output_dir: Path) -> None:
    forecast = forecast_volatility(result, horizon=horizon)
    forecast.to_csv(output_dir / f"{name}_forecast.csv", index_label="date")

    arima_summary = result.arima_result.summary().as_text()
    (output_dir / f"{name}_arima.txt").write_text(arima_summary)

    vol_summary = result.volatility_result.summary().as_text()
    (output_dir / f"{name}_volatility.txt").write_text(vol_summary)



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start = maybe_parse_date(args.start)
    end = maybe_parse_date(args.end)

    config = GoldPriceConfig(symbol=args.symbol, start=start, end=end)
    prices = load_gold_prices(config)
    returns = compute_log_returns(prices)

    output_dir = ensure_output_dir(args.output)

    arima_garch_result = fit_arima_garch(returns, arima_order=args.arima, garch_order=args.garch)
    save_results("arima_garch", arima_garch_result, args.horizon, output_dir)

    arima_gjr_result = fit_arima_gjrgarch(returns, arima_order=args.arima, gjr_order=args.gjr)
    save_results("arima_gjrgarch", arima_gjr_result, args.horizon, output_dir)


if __name__ == "__main__":
    main()
