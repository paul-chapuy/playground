from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import exp, log, sqrt
from typing import List, Optional

import numpy as np
from pipeline.const import MIN_YEAR_FRACTION_FOR_CALIBRATION
from scipy.optimize import minimize

from model.curve import Curve


@dataclass
class IVPoint:
    year_to_maturity: float
    strike: float
    moneyness: float
    forward_moneyness: float
    value: float
    vega: float


@dataclass
class IVSlice:
    year_to_maturity: float
    points: List[IVPoint]

    @property
    def atm_point(self) -> IVPoint:
        return min(self.points, key=lambda p: abs(p.forward_moneyness))

    @property
    def atm_vol(self) -> float:
        return self.atm_point.value

    @property
    def moneyness_range(self) -> List[float]:
        return [p.forward_moneyness for p in self.points]

    @property
    def values(self) -> List[float]:
        return [p.value for p in self.points]


@dataclass
class IVSurface:
    slices: List[IVSlice]
    model: Optional["VolatilitySurfaceModel"] = None

    def calibrate(self, model_cls: type["VolatilitySurfaceModel"], **kwargs):
        self.model = model_cls.make(self, **kwargs)

    @property
    def min_maturity(self):
        return min([s.year_to_maturity for s in self.slices])

    @property
    def max_maturity(self):
        return max([s.year_to_maturity for s in self.slices])

    @property
    def min_fwd_moneyness(self):
        return min(p.forward_moneyness for s in self.slices for p in s.points)

    @property
    def max_fwd_moneyness(self):
        return max(p.forward_moneyness for s in self.slices for p in s.points)

    @property
    def min_strike(self) -> float:
        return min(p.strike for s in self.slices for p in s.points)

    @property
    def max_strike(self) -> float:
        return max(p.strike for s in self.slices for p in s.points)

    def maturity_range(self, n: int = 50) -> np.ndarray:
        return np.linspace(self.min_maturity, self.max_maturity, n)

    def strike_range(self, n: int = 50) -> np.ndarray:
        return np.linspace(self.min_strike, self.max_strike, n)

    def numeric_values(self) -> List[List[float]]:
        tenors, strikes, values = [], [], []
        for sl in self.slices:
            for p in sl.points:
                tenors.append(p.year_to_maturity)
                strikes.append(p.strike)
                values.append(p.value)
        return [strikes, tenors, values]


def compute_moneyness(S: float, K: float) -> float:
    return log(K / S)


def compute_forward_moneyness(
    S: float, K: float, r: float, q: float, T: float
) -> float:
    F = S * exp((r - q) * T)
    return log(K / F)


class VolatilitySurfaceModel(ABC):
    @classmethod
    @abstractmethod
    def make(
        cls,
        volatility_surface: IVSurface,
        **kwargs,
    ) -> "VolatilitySurfaceModel": ...

    @abstractmethod
    def value(self, **kwargs) -> float: ...


@dataclass(frozen=True)
class SSVI(VolatilitySurfaceModel):
    """Power-law parameterization of SSVI vol surface (Gatheral & Jacquier, 2013)"""

    variance_backbone: Curve
    gamma: float
    eta: float
    rho: float
    vega_weighted_rmse: Optional[float] = None

    def __str__(self):
        lines = [
            "SSVI Model:",
            f"  Gamma:             {self.gamma:.6f}",
            f"  Eta:               {self.eta:.6f}",
            f"  Rho:               {self.rho:.6f}",
            f"  Vega-weighted RMSE:{self.vega_weighted_rmse:.6f}",
        ]
        return "\n".join(lines)

    @classmethod
    def make(cls, volatility_surface: IVSurface, **kwargs) -> "SSVI":
        return cls._fit(volatility_surface, **kwargs)

    def value(self, moneyness: float, year_to_maturity: float) -> float:
        theta = self.variance_backbone.value(year_to_maturity)
        return self._value(
            moneyness, year_to_maturity, theta, self.gamma, self.eta, self.rho
        )

    @staticmethod
    def _value(
        moneyness: float,
        year_to_maturity: float,
        theta: float,
        gamma: float,
        eta: float,
        rho: float,
    ) -> float:
        total_variance = SSVI._value_total_variance(moneyness, theta, gamma, eta, rho)
        if total_variance < 0.0:
            raise ArithmeticError(
                f"SSVI total variance is negative ({total_variance:.4e}) at "
                f"moneyness={moneyness:.4f}, theta={theta:.4f}. "
                "This indicates a calendar spread arbitrage in the calibrated surface."
            )
        return sqrt(total_variance / year_to_maturity)

    @staticmethod
    def _value_total_variance(
        moneyness: float,
        theta: float,
        gamma: float,
        eta: float,
        rho: float,
    ) -> float:
        p = SSVI._phi(theta, gamma, eta)
        return (
            0.5
            * theta
            * (
                1.0
                + rho * p * moneyness
                + sqrt((p * moneyness + rho) ** 2 + (1.0 - rho**2))
            )
        )

    @staticmethod
    def _phi(theta: float, gamma: float, eta: float) -> float:
        return eta / (pow(theta, gamma) * pow((1 + theta), (1 - gamma)))

    @staticmethod
    def _cost(
        volatility_surface: IVSurface,
        variance_backbone: Curve,
        parameters: np.ndarray,
    ) -> float:
        weighted_error = 0.0
        total_weight = 0.0

        gamma, eta, rho = parameters
        if SSVI._is_infeasible(parameters, variance_backbone):
            return 1e9
        for slice in volatility_surface.slices:
            if slice.year_to_maturity < MIN_YEAR_FRACTION_FOR_CALIBRATION:
                continue
            theta = variance_backbone.value(slice.year_to_maturity)
            for p in slice.points:
                model_iv = SSVI._value(
                    p.forward_moneyness,
                    p.year_to_maturity,
                    theta,
                    gamma,
                    eta,
                    rho,
                )
                weighted_error += p.vega * (model_iv - p.value) ** 2
                total_weight += p.vega

        return sqrt(weighted_error / total_weight)

    @staticmethod
    def _fit(
        volatility_surface: IVSurface,
        parameter_guess: Optional[List[float]] = None,
    ) -> "SSVI":
        if parameter_guess is None:
            parameter_guess = [0.5, 0.8, -0.5]

        slices = sorted(volatility_surface.slices, key=lambda s: s.year_to_maturity)

        tenors = [s.year_to_maturity for s in slices]
        values = list(
            np.maximum.accumulate([s.year_to_maturity * s.atm_vol**2 for s in slices])
        )
        variance_backbone = Curve(tenors=tenors, values=values)

        def cost_function(x):
            return SSVI._cost(volatility_surface, variance_backbone, x)

        starting_points = [
            parameter_guess,
            [0.3, 1.0, -0.3],
            [0.4, 0.5, 0.0],
            [0.2, 1.5, 0.3],
        ]
        best_result = None
        for x0 in starting_points:
            res = minimize(
                cost_function,
                x0=x0,
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10_000},
            )
            if best_result is None or res.fun < best_result.fun:
                best_result = res

        if best_result is None or not np.isfinite(best_result.fun):
            raise RuntimeError("SSVI optimization failed: no feasible solution found")

        return SSVI(
            variance_backbone=variance_backbone,
            gamma=best_result.x[0],
            eta=best_result.x[1],
            rho=best_result.x[2],
            vega_weighted_rmse=SSVI._cost(
                volatility_surface, variance_backbone, best_result.x
            ),
        )

    @staticmethod
    def _is_infeasible(parameters: np.ndarray, variance_backbone: Curve) -> bool:
        gamma, eta, rho = parameters

        if not (0 < gamma <= 0.5):
            return True
        if eta <= 0:
            return True
        if not (abs(rho) <= 1):
            return True
        if (2 - eta * (1 + abs(rho))) < 0:
            return True

        # Check no-calendar-spread-arbitrage condition across a dense grid of
        # interpolated theta values, not just at backbone knots, since the
        # piecewise-linear backbone can violate the condition between knots.
        n_samples = max(100, len(variance_backbone.tenors) * 20)
        sample_tenors = np.linspace(
            variance_backbone.tenors[0], variance_backbone.tenors[-1], n_samples
        )
        for t in sample_tenors:
            theta = variance_backbone.value(t)
            phi = SSVI._phi(theta, gamma, eta)
            lhs = theta * phi * (1 + abs(rho))
            if not (0 < lhs < 4):
                return True

        return False
