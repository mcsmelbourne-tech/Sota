"""Per-expiry implied-volatility smile smoothing for noisy end-of-day quotes.

Post-close NBBO snapshots have wide, inconsistent spreads on single names. Marking and pricing legs
off a weighted quadratic smile in log-moneyness (tighter quotes weigh more, outliers dropped) gives
stable, close-to-arbitrage-free marks; the quoted half-spread is still charged as the trading cost.
"""

from __future__ import annotations

import math

import numpy as np

from sota.models import Chain, OptionContract, Quote
from sota.pricing.bsm import greeks as bsm_greeks

MIN_IV, MAX_IV = 0.03, 5.0


def _fit_smile(x: np.ndarray, iv: np.ndarray, w: np.ndarray, clip: float) -> np.ndarray | None:
    """Weighted quadratic fit with one robust re-fit excluding gross outliers."""
    coefs = np.polyfit(x, iv, 2, w=w)
    resid = iv - np.polyval(coefs, x)
    scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) or float(np.std(resid))
    if scale > 0:
        keep = np.abs(resid) <= clip * scale
        if keep.sum() >= 4 and keep.sum() < len(x):
            coefs = np.polyfit(x[keep], iv[keep], 2, w=w[keep])
    return coefs


def smooth_chain(
    chain: Chain,
    *,
    min_quotes: int = 6,
    max_rel_half_spread: float = 0.5,
    min_abs_delta: float = 0.05,
    clip: float = 3.0,
) -> int:
    """Replace per-contract IV/greeks with smile-fitted values and set `Quote.fair`.

    Expiries with too few usable quotes keep their raw IV and get `fair = mid`.
    Returns the number of expiries that were smoothed.
    """
    chain.ensure_greeks()
    by_expiry: dict = {}
    for c, q in chain.quotes.items():
        by_expiry.setdefault(c.expiry, []).append((c, q))

    smoothed = 0
    for expiry, items in by_expiry.items():
        T = (expiry - chain.asof).days / 365.0
        fwd = chain.spot * math.exp((chain.rate - chain.div_yield) * T)
        xs, ivs, ws = [], [], []
        for c, q in items:
            if (
                q.iv is None
                or q.delta is None
                or not q.is_valid
                or q.bid <= 0.0
                or abs(q.delta) < min_abs_delta
                or q.half_spread > max_rel_half_spread * q.mid
            ):
                continue
            xs.append(math.log(c.strike / fwd))
            ivs.append(q.iv)
            ws.append(1.0 / (q.half_spread / q.mid + 0.02))
        if T <= 0.0 or len(xs) < min_quotes:
            for _, q in items:
                q.fair = q.mid if q.is_valid else None
            continue

        coefs = _fit_smile(np.asarray(xs), np.asarray(ivs), np.asarray(ws), clip)
        lo_x, hi_x = min(xs), max(xs)
        for c, q in items:
            x = min(max(math.log(c.strike / fwd), lo_x), hi_x)  # flat extrapolation beyond quotes
            iv = float(min(max(np.polyval(coefs, x), MIN_IV), MAX_IV))
            g = bsm_greeks(
                chain.spot, c.strike, T, chain.rate, chain.div_yield, iv, c.right.is_call
            )
            q.iv, q.delta, q.gamma, q.vega, q.theta = iv, g.delta, g.gamma, g.vega, g.theta
            q.fair = g.price
        smoothed += 1
    return smoothed


def leg_is_liquid(contract: OptionContract, q: Quote | None, max_rel_half_spread: float) -> bool:
    return (
        q is not None
        and q.is_valid
        and q.bid > 0.0
        and q.half_spread <= max_rel_half_spread * max(q.mark, 1e-9)
    )
