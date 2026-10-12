"""Portfolio-level risk snapshot used by the size resolver's caps (paper Table 10)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from sota.models import MULTIPLIER, Chain, Package, Position
from sota.pricing.bsm import price as bsm_price


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


def reprice_risk_per_share(
    pkg: Package,
    leg_ivs: Sequence[float],
    spot: float,
    asof: date,
    atm_iv: float,
    kappa: float,
    sigma_mult: float,
    rate: float = 0.0,
) -> float:
    """Worst one-day loss from full BSM repricing: spot +/- sigma_mult daily sigmas x IV +/- kappa.

    Unlike the second-order expansion, this captures the convexity loss of short-gamma structures
    (condors, butterflies) beyond the gamma/theta break-even, which otherwise prices at ~zero risk.
    """
    sigma_hat = atm_iv / math.sqrt(252.0)

    def value(s: float, dt_days: float, iv_mult: float) -> float:
        total = 0.0
        for leg, iv in zip(pkg.legs, leg_ivs, strict=True):
            c = leg.contract
            T = max((c.expiry - asof).days - dt_days, 0.0) / 365.0
            total += leg.ratio * bsm_price(s, c.strike, T, rate, 0.0, iv * iv_mult, c.right.is_call)
        return total

    base = value(spot, 0.0, 1.0)
    worst = 0.0
    for ds in (-sigma_mult * sigma_hat, 0.0, sigma_mult * sigma_hat):
        for dv in (1.0 - kappa, 1.0, 1.0 + kappa):
            worst = min(worst, value(spot * math.exp(ds), 1.0, dv) - base)
    return -worst


def package_scenario_risk(
    pkg: Package,
    spot: float,
    atm_iv: float,
    kappa: float,
    *,
    asof: date | None = None,
    sigma_mult: float = 1.0,
    leg_ivs: Sequence[float] | None = None,
) -> float:
    """Per-share scenario risk: full repricing when `asof` and leg IVs are known, else Eq. 21."""
    ivs = leg_ivs if leg_ivs is not None else pkg.leg_ivs
    if asof is not None and len(ivs) == len(pkg.legs) and all(iv > 0 for iv in ivs):
        return reprice_risk_per_share(pkg, ivs, spot, asof, atm_iv, kappa, sigma_mult)
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


def current_leg_ivs(pkg: Package, chain: Chain) -> list[float]:
    """Per-leg IV from today's chain, falling back to the IV recorded at resolution."""
    out: list[float] = []
    for i, leg in enumerate(pkg.legs):
        q = chain.quote(leg.contract)
        iv = q.iv if q is not None and q.iv else (pkg.leg_ivs[i] if i < len(pkg.leg_ivs) else 0.0)
        out.append(iv)
    return out


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
    sigma_mult: float = 1.0,
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
        d = greeks[0]
        iv = atm_iv.get(u) or 0.3
        r_share = package_scenario_risk(
            pos.package,
            chain.spot,
            iv,
            kappa,
            asof=chain.asof,
            sigma_mult=sigma_mult,
            leg_ivs=current_leg_ivs(pos.package, chain),
        )
        risk_u[u] = risk_u.get(u, 0.0) + r_share * MULTIPLIER * pos.units
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
