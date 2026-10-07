"""
Pre-pricing batch orchestrator.

Run once per trading day (e.g. via cron) to compute and persist all inputs
required by the risk engine:

  Global (USD only):
    - Spot curve  →  spot_curve table

  Per underlying:
    - Dividend curve      →  dividend_curve table
    - IV surface + SSVI   →  vol_surface + ssvi_params tables
    - Spot scenarios      →  spot_scenarios table  (dummy for now)
    - Vol scenarios       →  vol_scenarios table   (dummy for now)

Configuration:
  FRED_API_KEY  env-var  FRED API key (https://fred.stlouisfed.org)
  TICKERS       list     underlyings to process
  DB_PATH       str      path to the DuckDB file (default: risk_engine.db at repo root)
"""

import os
from datetime import date

from api.fred import FredClient
from api.yahoo_finance import YahooClient
from db.repository import (
    DividendCurveRepository,
    ScenariosRepository,
    SpotCurveRepository,
    VolSurfaceRepository,
)
from db.schema import DEFAULT_DB_PATH, init_db
from instruments.option import ExerciseStyle
from market_data.loaders import OptionChainLoader, ParCurveLoaders
from model.volatility import SSVI

from pipeline.dividend import compute_dividend_curve
from pipeline.spot_curve import bootstrap_spot_curve
from pipeline.volatility import compute_volatility_surface

FRED_API_KEY: str = os.environ["FRED_API_KEY"]
TICKERS: dict[str, ExerciseStyle] = {
    "AAPL": ExerciseStyle.American,
    "MSFT": ExerciseStyle.American,
    "AMZN": ExerciseStyle.American,
    "^SPX": ExerciseStyle.European,
}
DB_PATH: str = os.environ.get("RISK_ENGINE_DB", DEFAULT_DB_PATH)


def run_batch(as_of: date = date.today()) -> None:
    print(f"[batch] Starting pre-pricing batch for {as_of}")

    init_db(DB_PATH)

    spot_curve_repo = SpotCurveRepository(DB_PATH)
    div_curve_repo = DividendCurveRepository(DB_PATH)
    vol_surface_repo = VolSurfaceRepository(DB_PATH)
    scenarios_repo = ScenariosRepository(DB_PATH)

    print("[batch] Computing spot curve...")
    fred_client = FredClient.from_api_key(FRED_API_KEY)
    par_curve = ParCurveLoaders(fred_client).fetch()
    spot_curve = bootstrap_spot_curve(par_curve)
    spot_curve_repo.save(as_of, spot_curve)
    print(f"[batch] Spot curve saved ({len(spot_curve.tenors)} knots)")

    for ticker, style in TICKERS.items():
        print(f"[batch] Processing {ticker}...")
        yahoo = YahooClient(ticker)
        option_chains = OptionChainLoader(yahoo, style).load()

        div_curve = compute_dividend_curve(option_chains, spot_curve, as_of)
        div_curve_repo.save(as_of, ticker, div_curve)
        print(f"[batch]   dividend curve saved ({len(div_curve.tenors)} knots)")

        iv_surface = compute_volatility_surface(
            option_chains, spot_curve, div_curve, as_of
        )
        iv_surface.calibrate(SSVI)
        vol_surface_repo.save(as_of, ticker, iv_surface, iv_surface.model)
        n_pts = sum(len(sl.points) for sl in iv_surface.slices)
        print(
            f"[batch]   vol surface saved ({len(iv_surface.slices)} slices, {n_pts} points)"
        )

    print(f"[batch] Done — all inputs stored in {DB_PATH}")


if __name__ == "__main__":
    run_batch()
