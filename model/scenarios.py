from typing import Optional

from dataclasses import dataclass
from datetime import date
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from api.yahoo_finance import YahooClient

# RISK HORIZON IS ASSUMED TO ALWAYS BE ONE DAY

N_SCENARIOS = 500
_EMA_VOL_DECAY_DAYS = 16
_HISTORY_PERIOD = "3y"


class RiskFactorType(Enum):
    Spot = auto()
    Volatility = auto()


@dataclass(frozen=True)
class Scenario:
    day: date
    ticker: str
    risk_factor_type: RiskFactorType
    reference_value: float
    realizations: np.ndarray

    volatility_ts: Optional[np.ndarray] = None
    returns_ts: Optional[np.ndarray] = None
    innovations: Optional[np.ndarray] = None


def _fetch_closing_prices(client: YahooClient) -> pd.Series:
    prices = client.get_price_history(period=_HISTORY_PERIOD)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    return prices


def _fill_gaps(prices: pd.Series) -> pd.Series:
    bday_index = pd.bdate_range(start=prices.index[0], end=prices.index[-1])
    return prices.reindex(bday_index).interpolate(method="linear").dropna()


def _compute_log_return(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def _compute_ema_volatility(
    log_returns: pd.Series, decay_days: float = _EMA_VOL_DECAY_DAYS
) -> pd.Series:
    alpha = 1.0 - np.exp(-1.0 / decay_days)
    variance = (log_returns**2).ewm(alpha=alpha, adjust=False).mean()
    return np.sqrt(variance)


def _compute_innovations(log_returns: pd.Series, volatility: pd.Series) -> pd.Series:
    return (log_returns / volatility.shift(1)).dropna()


def compute_spot_scenarios(client: YahooClient, as_of: date) -> Scenario:
    prices = _fetch_closing_prices(client)
    prices = _fill_gaps(prices)
    returns = _compute_log_return(prices)
    volatility = _compute_ema_volatility(returns)
    innovations = _compute_innovations(returns, volatility)

    if len(innovations) < N_SCENARIOS:
        raise ValueError(
            f"{client.ticker}: need {N_SCENARIOS} innovations, got {len(innovations)}"
        )

    last_innovations = innovations.iloc[-N_SCENARIOS:].to_numpy()
    last_vol = float(volatility.iloc[-1])
    last_close = float(prices.iloc[-1])

    scenario_log_returns = last_vol * last_innovations
    realizations = last_close * np.exp(scenario_log_returns)

    return Scenario(
        day=as_of,
        ticker=client.ticker,
        risk_factor_type=RiskFactorType.Spot,
        reference_value=last_close,
        realizations=realizations,
        returns_ts=returns,
        volatility_ts=volatility,
        innovations=innovations,
    )


if __name__ == "__main__":
    client = YahooClient("AAPL")
    scenario = compute_spot_scenarios(client, as_of=date.today())

    log_returns = scenario.returns_ts
    volatility = scenario.volatility_ts
    innovations = scenario.innovations
    last_close = scenario.reference_value
    last_vol = float(scenario.volatility_ts.iloc[-1])
    real = scenario.realizations

    scenario_log_returns = np.log(scenario.realizations / scenario.reference_value)

    print(f"ticker:          {scenario.ticker}")
    print(f"as of:           {scenario.day}")
    print(f"last close:      {scenario.reference_value:.4f}")
    print(f"scenarios:       {len(scenario.realizations)}")
    print(f"min realizations: {scenario.realizations.min():.4f}")
    print(f"max realizations: {scenario.realizations.max():.4f}")
    print(f"mean:            {scenario.realizations.mean():.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle(f"{client.ticker} — spot scenario pipeline", fontsize=13)

    axes[0].plot(
        log_returns.index, log_returns.values, marker=".", markersize=2, linewidth=0
    )
    axes[0].set_title("Step 2 — gap-filled log returns (business-day calendar)")
    axes[0].set_ylabel("log return")
    axes[0].axhline(0, color="black", linewidth=0.5)

    axes[1].plot(
        volatility.index,
        volatility.values * np.sqrt(252),
        marker=".",
        markersize=2,
        linewidth=0,
        label=f"annualised EMA vol (decay={_EMA_VOL_DECAY_DAYS}d)",
    )
    axes[1].set_title("Step 3 — EMA volatility, annualised")
    axes[1].set_ylabel("volatility")
    axes[1].legend()

    print(f"innovations — mean: {innovations.mean():.4f}  std: {innovations.std():.4f}")

    axes[2].plot(
        innovations.index, innovations.values, marker=".", markersize=2, linewidth=0
    )
    axes[2].set_title("Step 4 — innovations (return / σ_{t-1})")
    axes[2].set_ylabel("standardised return")
    axes[2].axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        f"{client.ticker} — scenario realizations (ref={last_close:.2f})", fontsize=13
    )

    axes2[0].plot(sorted(scenario_log_returns), marker=".", markersize=3, linewidth=0)
    axes2[0].set_title(f"Step 5 — scenario log returns (sorted)  [σ={last_vol:.4f}]")
    axes2[0].set_ylabel("log return")
    axes2[0].axhline(0, color="black", linewidth=0.5)

    axes2[1].hist(real, bins=40, edgecolor="white", linewidth=0.4)
    axes2[1].axvline(
        last_close, color="red", linestyle="--", label=f"last close {last_close:.2f}"
    )
    axes2[1].set_title("Step 5 — distribution of spot realizations")
    axes2[1].set_xlabel("spot")
    axes2[1].legend()

    plt.tight_layout()
    plt.show()

    # --- EWMA weight profile for different decay values ---
    decay_values = [4, 8, 16, 32, 64]
    lags = np.arange(0, 120)

    fig3, ax = plt.subplots(figsize=(10, 5))
    for d in decay_values:
        a = 1.0 - np.exp(-1.0 / d)
        weights = a * (1 - a) ** lags
        ax.plot(lags, weights, label=f"decay={d}d  (α={a:.3f})")

    ax.set_title("EWMA weight on past squared returns by decay parameter")
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("weight  α·(1−α)^k")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.show()
