from datetime import date
from math import exp, isnan, log
from typing import List, Optional

from instruments.option import ExerciseStyle, OptionChains, OptionQuote, OptionType
from model.curve import Curve
from model.option_pricer import OptionPricer

from pipeline.const import (
    CALENDAR_DAYS_PER_YEAR,
    MAX_YEAR_FRACTION_FOR_CALIBRATION,
    MIN_YEAR_FRACTION_FOR_CALIBRATION,
)


def compute_eu_dividend_yield(
    call_price: float, put_price: float, spot: float, strike: float, T: float, r: float
) -> float:
    """Dividend yield from put-call parity"""

    if T <= 0:
        raise ValueError(f"Time to maturity must be positive, got T={T}")

    DF = exp(-r * T)
    F = (call_price - put_price) / DF + strike
    if F <= 0:
        raise ValueError(f"Implied forward is non-positive (F={F:.4f}). ")

    return r - (1.0 / T) * log(F / spot)


def compute_am_dividend_yield(
    call_price: float, put_price: float, spot: float, strike: float, T: float, r: float
) -> float:

    tol, max_iter = 1e-6, 20

    q = compute_eu_dividend_yield(call_price, put_price, spot, strike, T, r)

    am_pricer = OptionPricer.make(ExerciseStyle.American, spot, strike, r, T)
    eu_pricer = OptionPricer.make(ExerciseStyle.European, spot, strike, r, T)

    q_new = q
    for _ in range(max_iter):
        guess_vol = eu_pricer.implied_volatility(OptionType.Call, call_price, q)
        iv_call = am_pricer.implied_volatility(
            OptionType.Call, call_price, q, guess_vol
        )
        iv_put = am_pricer.implied_volatility(OptionType.Put, put_price, q, guess_vol)

        eu_call_price = eu_pricer.price(OptionType.Call, sigma=iv_call, q=q)
        eu_put_price = eu_pricer.price(OptionType.Put, sigma=iv_put, q=q)

        q_new = compute_eu_dividend_yield(
            eu_call_price, eu_put_price, spot, strike, T, r
        )
        if abs(q_new - q) < tol:
            return q_new

        q = q_new

    return q_new


def _extract_q_from_pair(
    strike: float,
    call: object,
    put: object,
    spot: float,
    T: float,
    r: float,
    exercise_style: ExerciseStyle,
) -> Optional[float]:
    """Extract a dividend yield estimate from a single call/put pair at the same strike."""

    if not isinstance(call, OptionQuote) or not isinstance(put, OptionQuote):
        return None

    call_price, put_price = call.mid, put.mid

    if call_price / spot < 1e-3 or put_price / spot < 1e-3:
        return None

    try:
        if exercise_style == ExerciseStyle.American:
            q = compute_am_dividend_yield(call_price, put_price, spot, strike, T, r)
        else:
            q = compute_eu_dividend_yield(call_price, put_price, spot, strike, T, r)

    except Exception:
        return None

    if isnan(q):
        return None

    return q


def compute_dividend_curve(
    option_chains: OptionChains,
    spot_curve: Curve,
    as_of: date,
) -> Curve:
    spot = option_chains.spot
    tenors: List[float] = []
    values: List[float] = []

    for chain in option_chains.chains:
        T = (chain.expiry - as_of).days / CALENDAR_DAYS_PER_YEAR
        if not (
            MIN_YEAR_FRACTION_FOR_CALIBRATION < T <= MAX_YEAR_FRACTION_FOR_CALIBRATION
        ):
            continue

        clean_chain = chain.clean(as_of=as_of)
        if not clean_chain.options:
            continue

        near_atm = clean_chain.near_atm_pairs(spot, n=3)
        if not near_atm:
            continue

        r = spot_curve.value(T)

        q_estimates = [
            q
            for strike, call, put in near_atm
            if (
                q := _extract_q_from_pair(
                    strike, call, put, spot, T, r, option_chains.exercise_style
                )
            )
            is not None
        ]
        if not q_estimates:
            continue

        tenors.append(T)
        values.append(sum(q_estimates) / len(q_estimates))

    if not values:
        curve = Curve(
            tenors=[
                MIN_YEAR_FRACTION_FOR_CALIBRATION,
                MAX_YEAR_FRACTION_FOR_CALIBRATION,
            ],
            values=[0.0, 0.0],
        )
    else:
        curve = Curve(tenors, values)

    return curve
