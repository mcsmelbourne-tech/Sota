"""Hedge resolver (paper App. E.3, Table 10): delta-hedge volatility families inside a no-trade band."""

from __future__ import annotations

import math


def hedge_trade(
    option_dollar_delta: float, hedge_shares: int, spot: float, nav: float, band_frac: float
) -> int:
    """Signed shares to trade so net dollar delta returns to the band edge; 0 when inside the band."""
    if spot <= 0.0:
        return 0
    net = option_dollar_delta + hedge_shares * spot
    band = band_frac * nav
    if abs(net) <= band:
        return 0
    target = math.copysign(band, net)
    return int(round((target - net) / spot))


def whalley_wilmott_half_width(
    spot: float, gamma: float, cost_rate: float, risk_aversion: float, rate: float, tte: float
) -> float:
    """Asymptotic no-transaction band half-width in delta units (Whalley & Wilmott, 1997)."""
    if risk_aversion <= 0.0:
        return 0.0
    return (1.5 * cost_rate * spot * gamma * gamma * math.exp(-rate * tte) / risk_aversion) ** (
        1.0 / 3.0
    )
