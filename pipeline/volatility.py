from datetime import date
from math import isnan
from typing import List

from instruments.option import OptionChains, OptionType
from model.curve import Curve
from model.option_pricer import OptionPricer
from model.volatility import (
    IVPoint,
    IVSlice,
    IVSurface,
    compute_forward_moneyness,
    compute_moneyness,
)

from pipeline.const import (
    CALENDAR_DAYS_PER_YEAR,
    MAX_ATM_FWD_MONEYNESS,
    MAX_YEAR_FRACTION_FOR_CALIBRATION,
    MIN_YEAR_FRACTION_FOR_CALIBRATION,
)


def compute_volatility_surface(
    option_chains: OptionChains,
    spot_curve: Curve,
    dividend_curve: Curve,
    as_of: date,
) -> IVSurface:
    ex_style = option_chains.exercise_style
    S = option_chains.spot

    slices: List[IVSlice] = []
    for chain in option_chains.chains:
        chain = chain.clean(as_of=as_of)
        if len(chain.options) == 0:
            # TODO: Add log here
            continue

        T = (chain.expiry - as_of).days / CALENDAR_DAYS_PER_YEAR
        if not (
            MIN_YEAR_FRACTION_FOR_CALIBRATION < T <= MAX_YEAR_FRACTION_FOR_CALIBRATION
        ):
            continue

        r = spot_curve.value(T)
        q = dividend_curve.value(T)

        points: List[IVPoint] = []
        for opt in chain.options:

            market_price = opt.mid

            if market_price / S < 1e-3:
                continue

            K = opt.strike
            moneyness = compute_moneyness(S, K)
            fwd_moneyness = compute_forward_moneyness(S, K, r, q, T)

            if opt.option_type == OptionType.Call and fwd_moneyness < 0.0:
                continue
            if opt.option_type == OptionType.Put and fwd_moneyness >= 0.0:
                continue

            pricer: OptionPricer = OptionPricer.make(ex_style, S, K, r, T)
            iv = pricer.implied_volatility(opt.option_type, market_price, q)

            if iv is None or isnan(iv) or not (0.01 < iv < 5.0):
                continue

            if abs(fwd_moneyness) > 1.0:
                continue

            pricer.set_volatility(iv)
            pricer.set_dividend_yield(q)

            vega = pricer.vega(opt.option_type)

            points.append(
                IVPoint(
                    year_to_maturity=T,
                    strike=K,
                    moneyness=moneyness,
                    forward_moneyness=fwd_moneyness,
                    value=iv,
                    vega=vega,
                )
            )

        if len(points) <= 4:
            continue

        iv_slice = IVSlice(year_to_maturity=T, points=points)
        if abs(iv_slice.atm_point.forward_moneyness) > MAX_ATM_FWD_MONEYNESS:
            continue

        slices.append(iv_slice)

    return IVSurface(slices)
