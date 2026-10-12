from datetime import date

import pytest

from sota.models import Leg, OptionContract, Right, Side, Tenor
from sota.resolvers.contract import (
    max_loss_per_share,
    resolve_expiry,
    resolve_package,
    resolve_strike,
)
from sota.resolvers.hedge import hedge_trade, whalley_wilmott_half_width
from sota.resolvers.size import resolve_size
from sota.risk import PortfolioRisk, build_portfolio_risk
from sota.strategies.families import FAMILIES


def test_resolve_expiry_prefers_bucket_nearest_anchor(chain):
    for tenor in Tenor:
        exp = resolve_expiry(chain, tenor)
        assert exp is not None
        lo, hi = tenor.bounds
        assert lo <= (exp - chain.asof).days <= hi


def test_resolve_strike_nearest_delta(chain):
    expiry = resolve_expiry(chain, Tenor.T8_30)
    c = resolve_strike(chain, expiry, Right.PUT, 0.25)
    assert c is not None
    q = chain.quote(c)
    assert abs(abs(q.delta) - 0.25) < 0.05


@pytest.mark.parametrize("code", list(FAMILIES))
@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_resolve_package_all_families(chain, code, side):
    pkg = resolve_package(chain, code, side, Tenor.T31_90)
    assert pkg is not None, code
    assert pkg.max_loss is not None  # every family is defined-risk
    assert pkg.hedged == FAMILIES[code].hedged
    assert len(pkg.leg_deltas) == len(pkg.legs)
    if code in ("cv", "ib", "ic"):
        assert pkg.is_credit
    if code in ("ol", "dv", "ls", "lg"):
        assert not pkg.is_credit


def test_directional_delta_signs(chain):
    assert resolve_package(chain, "ol", Side.LONG, Tenor.T8_30).delta > 0
    assert resolve_package(chain, "ol", Side.SHORT, Tenor.T8_30).delta < 0
    assert resolve_package(chain, "cv", Side.LONG, Tenor.T8_30).delta > 0  # bull put spread
    straddle = resolve_package(chain, "ls", Side.LONG, Tenor.T8_30)
    assert abs(straddle.delta) < 0.2 and straddle.gamma > 0 and straddle.vega > 0


def test_max_loss():
    u, e = "X", date(2030, 1, 1)
    long_call = [Leg(OptionContract(u, e, 100.0, Right.CALL), 1)]
    assert max_loss_per_share(long_call, 5.0) == 5.0
    naked_short_call = [Leg(OptionContract(u, e, 100.0, Right.CALL), -1)]
    assert max_loss_per_share(naked_short_call, -5.0) is None
    bull_put = [
        Leg(OptionContract(u, e, 95.0, Right.PUT), 1),
        Leg(OptionContract(u, e, 100.0, Right.PUT), -1),
    ]
    assert max_loss_per_share(bull_put, -2.0) == pytest.approx(3.0)


def test_size_resolver_targets_risk_and_caps(chain, limits):
    pkg = resolve_package(chain, "ic", Side.LONG, Tenor.T8_30)
    nav = 1_000_000.0
    res = resolve_size(pkg, chain.spot, 0.25, limits, PortfolioRisk(nav=nav))
    assert res.accepted
    assert res.units * res.risk_per_unit <= limits.target_risk_frac * nav
    assert (
        res.units + 1
    ) * res.risk_per_unit > limits.target_risk_frac * nav or res.binding != "target risk"

    full = PortfolioRisk(nav=nav, open_positions=limits.max_positions)
    assert not resolve_size(pkg, chain.spot, 0.25, limits, full).accepted
    per_u = PortfolioRisk(
        nav=nav, positions_by_underlying={pkg.underlying: limits.max_positions_per_underlying}
    )
    assert not resolve_size(pkg, chain.spot, 0.25, limits, per_u).accepted
    risk_full = PortfolioRisk(nav=nav, scenario_risk_total=limits.risk_cap_total * nav)
    assert not resolve_size(pkg, chain.spot, 0.25, limits, risk_full).accepted


def test_build_portfolio_risk_counts(chain, limits):
    from sota.models import Position

    pkg = resolve_package(chain, "ls", Side.LONG, Tenor.T8_30)
    pos = Position("p1", pkg, 3, chain.asof, pkg.net_premium, 1.0, pkg.expiry)
    risk = build_portfolio_risk(1e6, [pos], {"AAA": chain}, {"AAA": 0.25}, {}, limits.iv_shock)
    assert risk.open_positions == 1 and risk.positions_by_underlying == {"AAA": 1}
    assert risk.scenario_risk_total > 0


def test_repricing_risk_exceeds_taylor_for_short_gamma(chain, limits):
    """Full repricing must see the convexity loss a 1-sigma Taylor expansion nets out."""
    from sota.risk import package_scenario_risk, scenario_risk_per_share

    ic = resolve_package(chain, "ic", Side.LONG, Tenor.T8_30)
    taylor = scenario_risk_per_share(ic.delta, ic.gamma, ic.vega, ic.theta, chain.spot, 0.25, 0.10)
    repriced = package_scenario_risk(ic, chain.spot, 0.25, 0.10, asof=chain.asof, sigma_mult=2.0)
    assert repriced > taylor > 0
    assert repriced <= ic.max_loss, "a one-day scenario cannot lose more than the max loss"

    ls = resolve_package(chain, "ls", Side.LONG, Tenor.T8_30)
    long_gamma = package_scenario_risk(ls, chain.spot, 0.25, 0.10, asof=chain.asof, sigma_mult=2.0)
    assert 0 < long_gamma <= abs(ls.net_premium), "a long straddle can only lose its premium"

    # Without a date (or without leg IVs) the Taylor estimate is used.
    assert package_scenario_risk(ic, chain.spot, 0.25, 0.10) == pytest.approx(taylor)


def test_size_resolver_floors_risk_at_the_spread(chain, limits):
    from sota.risk import package_scenario_risk

    pkg = resolve_package(chain, "ic", Side.LONG, Tenor.T8_30)
    res = resolve_size(pkg, chain.spot, 0.25, limits, PortfolioRisk(nav=1e6), chain.asof)
    scenario = package_scenario_risk(
        pkg, chain.spot, 0.25, 0.10, asof=chain.asof, sigma_mult=limits.risk_sigma
    )
    assert res.risk_per_unit == pytest.approx(max(scenario, pkg.half_spread) * 100)
    assert res.accepted and res.units * res.risk_per_unit <= limits.target_risk_frac * 1e6


def test_hedge_band():
    assert hedge_trade(4_000.0, 0, 100.0, 1_000_000.0, 0.005) == 0  # inside the $5,000 band
    trade = hedge_trade(12_000.0, 0, 100.0, 1_000_000.0, 0.005)
    assert trade == -70  # back to the band edge, not to zero
    assert hedge_trade(-12_000.0, 0, 100.0, 1_000_000.0, 0.005) == 70
    assert whalley_wilmott_half_width(100.0, 0.02, 0.001, 1.0, 0.0, 0.1) > 0
