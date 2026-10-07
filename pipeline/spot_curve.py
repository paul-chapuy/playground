from math import exp, log
from typing import Final, List

from model.curve import Curve, NelsonSiegel

_BILL_TENORS: Final[List[float]] = [1 / 12, 3 / 12]


def _bill_spot_rate(par_rate: float, T: float) -> float:
    """Continuously-compounded spot rate from a T-bill bond-equivalent yield (BEY)."""
    return log(1.0 + par_rate * T) / T


def _compute_spot_rate(
    dt: float, spot_tenors: List[float], spot_values: List[float], coupon: float
) -> float:
    """
    Bootstrap one spot rate from prior spot rates and the par coupon at dt.
    Assumes semi-annual coupon bonds: spot_tenors must be on a 0.5-year grid.
    """
    pv_coupons = sum(
        coupon * exp(-v * t) for t, v in zip(spot_tenors, spot_values) if t < dt
    )
    return -log((1 - pv_coupons) / (1 + coupon)) / dt


def bootstrap_spot_curve(
    par_curve: Curve, max_tenor: float = 30.0, step: float = 0.5
) -> Curve:
    """
    Bootstrap a continuously-compounded spot curve from a par yield curve.

    T-bill tenors (1M, 3M) are treated as zero-coupon bonds. All other tenors
    are bootstrapped assuming semi-annual coupon bonds (step=0.5).
    """
    fitted_par: NelsonSiegel = NelsonSiegel.make(par_curve)

    bill_spot_tenors = _BILL_TENORS
    bill_spot_values = [_bill_spot_rate(fitted_par.value(T), T) for T in _BILL_TENORS]

    bond_spot_tenors: List[float] = []
    bond_spot_values: List[float] = []

    n_steps = round(max_tenor / step)
    for i in range(1, n_steps + 1):
        dt = round(i * step, 10)
        coupon = fitted_par.value(dt) / 2  # semi-annual coupon from NS par rate
        spot_rate = _compute_spot_rate(dt, bond_spot_tenors, bond_spot_values, coupon)
        bond_spot_tenors.append(dt)
        bond_spot_values.append(spot_rate)

    return Curve(
        tenors=bill_spot_tenors + bond_spot_tenors,
        values=bill_spot_values + bond_spot_values,
        evaluation_date=par_curve.evaluation_date,
    )
