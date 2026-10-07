from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from math import exp
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


class CurveModel(ABC):

    @classmethod
    @abstractmethod
    def make(cls, curve: "Curve", **kwargs) -> "CurveModel": ...

    @abstractmethod
    def value(self, dt: float) -> float: ...


class PiecewiseLinear(CurveModel):
    """Piecewise linear interpolation"""

    def __init__(self, tenors: List[float], yields: List[float]):
        if len(tenors) != len(yields):
            raise ValueError("Tenors and yields must have the same length.")

        self._tenors, self._yields = tenors, yields

    def value(self, dt: float) -> float:
        if dt <= self._tenors[0]:
            return self._yields[0]
        if dt >= self._tenors[-1]:
            return self._yields[-1]
        i = bisect_right(self._tenors, dt)
        t0, t1 = self._tenors[i - 1], self._tenors[i]
        y0, y1 = self._yields[i - 1], self._yields[i]
        return y0 + (dt - t0) / (t1 - t0) * (y1 - y0)

    @classmethod
    def make(cls, curve: "Curve", **_) -> "PiecewiseLinear":
        return cls(curve.tenors, curve.values)


@dataclass
class Curve:
    tenors: List[float]
    values: List[float]
    evaluation_date: Optional[date] = None
    model: Optional[CurveModel] = None

    def __post_init__(self):
        if len(self.tenors) != len(self.values):
            raise ValueError("tenors and values must have the same length.")

        paired = sorted(zip(self.tenors, self.values))
        self.tenors, self.values = [t for t, _ in paired], [v for _, v in paired]

        if self.model is None:
            self.model = PiecewiseLinear(self.tenors, self.values)

    def __iter__(self):
        yield from zip(self.tenors, self.values)

    def value(self, dt: float) -> float:
        return self.model.value(dt)

    def calibrate(self, model: type[CurveModel], **kwargs):
        self.model = model.make(self, **kwargs)

    def average(
        self,
        min_tenor: Optional[float] = None,
        max_tenor: Optional[float] = None,
        clamp_negative: bool = False,
    ) -> float:
        filtered = [
            v
            for t, v in zip(self.tenors, self.values)
            if (min_tenor is None or t >= min_tenor)
            and (max_tenor is None or t <= max_tenor)
        ]
        if not filtered:
            raise ValueError(f"no tenors found in range [{min_tenor}, {max_tenor}].")
        if clamp_negative:
            filtered = [max(v, 0.0) for v in filtered]
        return sum(filtered) / len(filtered)


@dataclass(frozen=True)
class NelsonSiegel(CurveModel):
    """Nelson-Siegel model for the yield curve term structure (Nelson & Siegel, 1987)"""

    b0: float
    b1: float
    b2: float
    tau: float
    rmse: Optional[float] = None

    def __str__(self):
        lines = [
            "Nelson-Siegel Model:",
            f"  Level (b0):     {self.b0:.6f}",
            f"  Slope (b1):     {self.b1:.6f}",
            f"  Curvature (b2): {self.b2:.6f}",
            f"  Tau:            {self.tau:.6f}",
            f"  Cost (RMSE):    {self.rmse:.6f}",
        ]
        return "\n".join(lines)

    @classmethod
    def make(cls, curve: Curve, **kwargs) -> "NelsonSiegel":
        return cls._fit(curve, **kwargs)

    def value(self, dt: float) -> float:
        return self._scalar(dt, self.b0, self.b1, self.b2, self.tau)

    @staticmethod
    def _scalar(dt: float, b0: float, b1: float, b2: float, tau: float) -> float:
        if dt < 1e-6:
            return b0
        x = dt / tau
        factor = (1 - exp(-x)) / x
        return b0 + b1 * factor + b2 * (factor - exp(-x))

    @staticmethod
    def _cost(tenors: np.ndarray, empirical: np.ndarray, params: np.ndarray) -> float:
        b0, b1, b2, tau = params
        x = tenors / tau
        x_safe = np.where(x < 1e-6, 1.0, x)
        factor = (1 - np.exp(-x_safe)) / x_safe
        predicted = np.where(
            x < 1e-6, b0, b0 + b1 * factor + b2 * (factor - np.exp(-x_safe))
        )
        return float(np.sqrt(np.mean((empirical - predicted) ** 2)))

    @staticmethod
    def _fit(
        curve: Curve,
        initial_guess: Optional[List[float]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> "NelsonSiegel":
        if bounds is None:
            bounds = [(-0.05, 0.25), (-0.25, 0.25), (-0.25, 0.25), (0.05, 30.0)]

        tenors = np.array(curve.tenors)
        values = np.array(curve.values)

        starting_points = [
            initial_guess if initial_guess is not None else [0.03, -0.02, 0.02, 2.0],
            [0.04, 0.02, -0.01, 1.5],
            [0.03, 0.00, 0.03, 3.0],
            [0.06, -0.03, 0.01, 5.0],
        ]

        best_result = None
        for x0 in starting_points:
            res = minimize(
                lambda x: NelsonSiegel._cost(tenors, values, x),
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": 10_000},
            )
            if best_result is None or res.fun < best_result.fun:
                best_result = res

        if best_result is None or not np.isfinite(best_result.fun):
            raise RuntimeError(
                "Nelson-Siegel optimization failed: no feasible solution found."
            )

        b0, b1, b2, tau = best_result.x
        return NelsonSiegel(
            b0=b0,
            b1=b1,
            b2=b2,
            tau=tau,
            rmse=NelsonSiegel._cost(tenors, values, best_result.x),
        )
