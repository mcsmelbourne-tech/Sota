"""Portfolio-level risk snapshot used by the size resolver's caps (paper Table 10)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sota.models import MULTIPLIER, Chain, Package, Position


def scenario_risk_per_share(
    pkg_delta: float,
    pkg_gamma: float,
    pkg_vega: float,
    pkg_theta: float,
    spot: float,
    atm_iv: float,
    kappa: float,
) -> float:
    """Paper Eq. 21: 1-sigma adverse spot move, relative IV shock `kappa`, and one day of carry."""
    sigma_hat = atm_iv / math.sqrt(252.0)
    move = spot * sigma_hat
    risk = (
        abs(pkg_delta) * move
        - 0.5 * pkg_gamma * move * move
        + abs(pkg_vega) * atm_iv * kappa
        - pkg_theta
    )
    return max(risk, 0.0)


def package_scenario_risk(pkg: Package, spot: float, atm_iv: float, kappa: float) -> float:
    return scenario_risk_per_share(pkg.delta, pkg.gamma, pkg.vega, pkg.theta, spot, atm_iv, kappa)


def current_greeks(pkg: Package, chain: Chain) -> tuple[float, float, float, float] | None:
    """(delta, gamma, vega, theta) per share at today's quotes; None if any leg lacks greeks."""
    d = g = v = t = 0.0
    for leg in pkg.legs:
        q = chain.quote(leg.contract)
        if q is None or not q.has_greeks:
            return None
        d += leg.ratio * q.delta  # type: ignore[operator]
        g += leg.ratio * q.gamma  # type: ignore[operator]
        v += leg.ratio * q.vega  # type: ignore[operator]
        t += leg.ratio * q.theta  # type: ignore[operator]
    return d, g, v, t


@dataclass(frozen=True, slots=True)
class PortfolioRisk:
    nav: float
    open_positions: int = 0
    positions_by_underlying: dict[str, int] = field(default_factory=dict)
    scenario_risk_total: float = 0.0  # dollars
    scenario_risk_by_underlying: dict[str, float] = field(default_factory=dict)
    dollar_delta: float = 0.0  # net option + hedge dollar delta


def build_portfolio_risk(
    nav: float,
    positions: list[Position],
    chains: dict[str, Chain],
    atm_iv: dict[str, float],
    hedge_shares: dict[str, int],
    kappa: float,
) -> PortfolioRisk:
    count_u: dict[str, int] = {}
    risk_u: dict[str, float] = {}
    dollar_delta = 0.0
    for pos in positions:
        if not pos.is_open:
            continue
        u = pos.package.underlying
        count_u[u] = count_u.get(u, 0) + 1
        chain = chains.get(u)
        if chain is None:
            continue
        greeks = current_greeks(pos.package, chain)
        if greeks is None:
            greeks = (pos.package.delta, pos.package.gamma, pos.package.vega, pos.package.theta)
        d, g, v, t = greeks
        iv = atm_iv.get(u) or 0.3
        r = scenario_risk_per_share(d, g, v, t, chain.spot, iv, kappa) * MULTIPLIER * pos.units
        risk_u[u] = risk_u.get(u, 0.0) + r
        dollar_delta += d * chain.spot * MULTIPLIER * pos.units
    for u, shares in hedge_shares.items():
        chain = chains.get(u)
        if chain is not None:
            dollar_delta += shares * chain.spot
    return PortfolioRisk(
        nav=nav,
        open_positions=sum(count_u.values()),
        positions_by_underlying=count_u,
        scenario_risk_total=sum(risk_u.values()),
        scenario_risk_by_underlying=risk_u,
        dollar_delta=dollar_delta,
    )
