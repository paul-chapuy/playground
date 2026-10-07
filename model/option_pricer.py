from abc import ABC, abstractmethod
from math import erf, exp, log, pi, sqrt

import numpy as np
from instruments.option import ExerciseStyle, OptionType
from numba import njit
from scipy.optimize import brentq


class OptionPricer(ABC):
    """Abstract base class for vanilla option pricers (European or American)."""

    def __init__(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
    ):
        if S <= 0:
            raise ValueError(f"Spot must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike must be positive, got {K}")
        if T < 0:
            raise ValueError(f"Time to expiry must be non-negative, got {T}")
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = 0.0
        self.q = 0.0

    @staticmethod
    def make(
        exercise_style: ExerciseStyle,
        S: float,
        K: float,
        r: float,
        T: float,
    ) -> "OptionPricer":
        match exercise_style:
            case ExerciseStyle.European:
                return BlackScholesMerton(S, K, r, T)
            case ExerciseStyle.American:
                return CoxRossRubinstein(S, K, r, T)
            case _:
                raise ValueError(f"Unsupported exercise style: {exercise_style}")

    def set_volatility(self, sigma: float) -> None:
        if sigma < 0:
            raise ValueError(f"Volatility must be non-negative, got {sigma}")
        self.sigma = sigma

    def set_dividend_yield(self, q: float) -> None:
        self.q = q

    @abstractmethod
    def price(
        self,
        option_type: OptionType,
        S: float = None,
        r: float = None,
        T: float = None,
        sigma: float = None,
        q: float = None,
    ) -> float: ...

    def delta(self, option_type: OptionType, h: float = 1e-4) -> float:
        up = self.price(option_type, S=self.S + h)
        down = self.price(option_type, S=self.S - h)
        return (up - down) / (2 * h)

    def gamma(self, option_type: OptionType, h: float = 1e-4) -> float:
        up = self.price(option_type, S=self.S + h)
        mid = self.price(option_type)
        down = self.price(option_type, S=self.S - h)
        return (up - 2 * mid + down) / (h**2)

    def vega(self, option_type: OptionType, h: float = 1e-4) -> float:
        up = self.price(option_type, sigma=self.sigma + h)
        down = self.price(option_type, sigma=self.sigma - h)
        return (up - down) / (2 * h)

    def theta(self, option_type: OptionType, h: float = 1 / 365) -> float:
        return self.price(option_type, T=max(self.T - h, 0.0)) - self.price(option_type)

    def rho(self, option_type: OptionType, h: float = 1e-4) -> float:
        up = self.price(option_type, r=self.r + h)
        down = self.price(option_type, r=self.r - h)
        return (up - down) / (2 * h)

    def implied_volatility(
        self,
        option_type: OptionType,
        market_price: float,
        q: float,
        initial_guess: float = None,
    ) -> float:
        """Solve for implied volatility via Brent's method. Does not mutate object state."""

        def f(sigma):
            return self.price(option_type, sigma=sigma, q=q) - market_price

        a, b = 1e-4, 5.0
        if initial_guess is not None:
            a = max(1e-4, 0.5 * initial_guess)
            b = min(5.0, 1.5 * initial_guess)

        fa, fb = f(a), f(b)
        if fa * fb > 0:
            raise ValueError(
                f"IV bracket [{a:.4f}, {b:.4f}] does not straddle zero "
                f"(f(a)={fa:.6f}, f(b)={fb:.6f}). "
                "Market price may be outside model range or initial_guess bracket too narrow."
            )

        return brentq(f, a, b, maxiter=100, xtol=1e-6, rtol=1e-6)


class BlackScholesMerton(OptionPricer):
    """Closed-form pricing model for European options (Black-Scholes-Merton, 1973)."""

    def __init__(self, S: float, K: float, r: float, T: float):
        super().__init__(S, K, r, T)

    def price(
        self,
        option_type: OptionType,
        S: float = None,
        r: float = None,
        T: float = None,
        sigma: float = None,
        q: float = None,
    ) -> float:
        _S = S if S is not None else self.S
        _r = r if r is not None else self.r
        _T = T if T is not None else self.T
        _sigma = sigma if sigma is not None else self.sigma
        _q = q if q is not None else self.q

        if _T <= 0:
            return (
                max(_S - self.K, 0.0)
                if option_type == OptionType.Call
                else max(self.K - _S, 0.0)
            )
        if _sigma <= 0:
            intrinsic = (
                max(_S - self.K, 0.0)
                if option_type == OptionType.Call
                else max(self.K - _S, 0.0)
            )
            return intrinsic * exp(-_r * _T)

        return BlackScholesMerton._price(
            _S,
            self.K,
            _r,
            _q,
            _sigma,
            _T,
            is_call=(option_type == OptionType.Call),
        )

    @staticmethod
    def _d1_d2(
        S: float, K: float, r: float, q: float, sigma: float, T: float
    ) -> tuple[float, float]:
        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        return d1, d1 - sigma * sqrt(T)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return exp(-0.5 * x * x) / sqrt(2.0 * pi)

    @staticmethod
    @njit(fastmath=False)
    def _price(
        S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool
    ) -> float:
        def norm_cdf(x: float) -> float:
            return 0.5 * (1.0 + erf(x / sqrt(2.0)))

        F = S * exp((r - q) * T)
        DF = exp(-r * T)
        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        call_price = DF * (F * norm_cdf(d1) - K * norm_cdf(d2))
        if is_call:
            return call_price

        return call_price - DF * (F - K)

    def delta(self, option_type: OptionType, h: float = 1e-4) -> float:
        if self.T <= 0 or self.sigma <= 0:
            return super().delta(option_type, h)
        d1, _ = BlackScholesMerton._d1_d2(
            self.S, self.K, self.r, self.q, self.sigma, self.T
        )
        eq_T = exp(-self.q * self.T)
        nd1 = BlackScholesMerton._norm_cdf(d1)
        return eq_T * nd1 if option_type == OptionType.Call else eq_T * (nd1 - 1.0)

    def gamma(self, option_type: OptionType, h: float = 1e-4) -> float:
        if self.T <= 0 or self.sigma <= 0:
            return super().gamma(option_type, h)
        d1, _ = BlackScholesMerton._d1_d2(
            self.S, self.K, self.r, self.q, self.sigma, self.T
        )
        return (
            exp(-self.q * self.T)
            * BlackScholesMerton._norm_pdf(d1)
            / (self.S * self.sigma * sqrt(self.T))
        )

    def vega(self, option_type: OptionType, h: float = 1e-4) -> float:
        if self.T <= 0 or self.sigma <= 0:
            return super().vega(option_type, h)
        d1, _ = BlackScholesMerton._d1_d2(
            self.S, self.K, self.r, self.q, self.sigma, self.T
        )
        return (
            self.S
            * exp(-self.q * self.T)
            * BlackScholesMerton._norm_pdf(d1)
            * sqrt(self.T)
        )

    def theta(self, option_type: OptionType, h: float = 1 / 365) -> float:
        """Analytic per-calendar-day theta (dollar P&L per day)."""
        if self.T <= 0 or self.sigma <= 0:
            return super().theta(option_type, h)
        S, K, r, q, sigma, T = self.S, self.K, self.r, self.q, self.sigma, self.T
        d1, d2 = BlackScholesMerton._d1_d2(S, K, r, q, sigma, T)
        nd1 = BlackScholesMerton._norm_cdf(d1)
        nd2 = BlackScholesMerton._norm_cdf(d2)
        npd1 = BlackScholesMerton._norm_pdf(d1)
        eq_T, er_T = exp(-q * T), exp(-r * T)
        decay = -S * eq_T * npd1 * sigma / (2.0 * sqrt(T))
        if option_type == OptionType.Call:
            theta_annual = decay - r * K * er_T * nd2 + q * S * eq_T * nd1
        else:
            theta_annual = (
                decay + r * K * er_T * (1.0 - nd2) - q * S * eq_T * (1.0 - nd1)
            )
        return theta_annual / 365.0


class CoxRossRubinstein(OptionPricer):
    """Binomial tree model for American options (Cox, Ross & Rubinstein, 1979)."""

    def __init__(self, S: float, K: float, r: float, T: float, n_steps: int = 200):
        super().__init__(S, K, r, T)
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        self.n_steps = n_steps

    def price(
        self,
        option_type: OptionType,
        S: float = None,
        r: float = None,
        T: float = None,
        sigma: float = None,
        q: float = None,
    ) -> float:

        _S = S if S is not None else self.S
        _r = r if r is not None else self.r
        _T = T if T is not None else self.T
        _sigma = sigma if sigma is not None else self.sigma
        _q = q if q is not None else self.q

        if _T <= 0:
            return (
                max(_S - self.K, 0.0)
                if option_type == OptionType.Call
                else max(self.K - _S, 0.0)
            )
        if _sigma <= 0:
            intrinsic = (
                max(_S - self.K, 0.0)
                if option_type == OptionType.Call
                else max(self.K - _S, 0.0)
            )
            return intrinsic * exp(-_r * _T)

        return CoxRossRubinstein._price(
            _S,
            self.K,
            _r,
            _q,
            _sigma,
            _T,
            is_call=(option_type == OptionType.Call),
            N=self.n_steps,
        )

    @staticmethod
    @njit(fastmath=False)
    def _price(
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        is_call: bool,
        N: int,
    ) -> float:
        dt = T / N
        u = exp(sigma * sqrt(dt))
        d = 1.0 / u
        p = (exp((r - q) * dt) - d) / (u - d)
        df = exp(-r * dt)

        ST = np.empty(N + 1)
        for j in range(N + 1):
            ST[j] = S * u ** (N - j) * d**j

        payoff = np.empty(N + 1)
        for j in range(N + 1):
            payoff[j] = max(ST[j] - K, 0.0) if is_call else max(K - ST[j], 0.0)

        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                ST[j] = ST[j] / u
                cont_val = df * (p * payoff[j] + (1.0 - p) * payoff[j + 1])
                exercise_val = max(ST[j] - K, 0.0) if is_call else max(K - ST[j], 0.0)
                payoff[j] = max(cont_val, exercise_val)

        return payoff[0]
