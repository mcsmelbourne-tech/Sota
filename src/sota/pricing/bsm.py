"""Black–Scholes–Merton pricing, greeks, implied volatility, and a CRR binomial for American marks.

Conventions: `T` in years, rates continuous, `vega` per 1.00 of vol, `theta` per calendar day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


@dataclass(frozen=True, slots=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float


def _intrinsic(S: float, K: float, is_call: bool) -> float:
    return max(S - K, 0.0) if is_call else max(K - S, 0.0)


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> Greeks:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        px = _intrinsic(S, K, is_call)
        itm = (S > K) if is_call else (S < K)
        delta = (1.0 if is_call else -1.0) if itm else 0.0
        return Greeks(px, delta, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf1 = _pdf(d1)
    sqrt_t = math.sqrt(T)

    if is_call:
        px = S * disc_q * _cdf(d1) - K * disc_r * _cdf(d2)
        delta = disc_q * _cdf(d1)
        theta_annual = (
            -S * disc_q * pdf1 * sigma / (2.0 * sqrt_t)
            - r * K * disc_r * _cdf(d2)
            + q * S * disc_q * _cdf(d1)
        )
    else:
        px = K * disc_r * _cdf(-d2) - S * disc_q * _cdf(-d1)
        delta = -disc_q * _cdf(-d1)
        theta_annual = (
            -S * disc_q * pdf1 * sigma / (2.0 * sqrt_t)
            + r * K * disc_r * _cdf(-d2)
            - q * S * disc_q * _cdf(-d1)
        )

    gamma = disc_q * pdf1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf1 * sqrt_t
    return Greeks(px, delta, gamma, vega, theta_annual / 365.0)


def price(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    return greeks(S, K, T, r, q, sigma, is_call).price


def implied_vol(
    target: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float | None:
    """Solve BSM for sigma. Returns None when the price violates no-arbitrage bounds or has no time value."""
    if T <= 0.0 or S <= 0.0 or K <= 0.0 or target <= 0.0:
        return None
    fwd_intrinsic = (
        max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        if is_call
        else max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    )
    if target <= fwd_intrinsic + 1e-10:
        return None
    if price(S, K, T, r, q, lo, is_call) > target or price(S, K, T, r, q, hi, is_call) < target:
        return None

    # Brenner–Subrahmanyam seed, then safeguarded Newton.
    sigma = min(max(math.sqrt(2.0 * math.pi / T) * target / S, lo), hi)
    for _ in range(max_iter):
        g = greeks(S, K, T, r, q, sigma, is_call)
        diff = g.price - target
        if abs(diff) < tol:
            return sigma
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        nxt = sigma - diff / g.vega if g.vega > 1e-12 else 0.5 * (lo + hi)
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        sigma = nxt
    return sigma


def crr_american(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool, steps: int = 128
) -> float:
    """Cox–Ross–Rubinstein binomial price with early exercise."""
    if T <= 0.0 or sigma <= 0.0:
        return _intrinsic(S, K, is_call)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    values = [_intrinsic(S * (u ** (steps - i)) * (d**i), K, is_call) for i in range(steps + 1)]
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            cont = disc * (p * values[i] + (1.0 - p) * values[i + 1])
            spot = S * (u ** (step - i)) * (d**i)
            values[i] = max(cont, _intrinsic(spot, K, is_call))
    return values[0]
