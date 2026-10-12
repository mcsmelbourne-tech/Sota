"""Contract resolver (paper App. E.1): tenor bucket -> expiry, |delta| coordinates -> strikes."""

from __future__ import annotations

from datetime import date

from sota.models import Chain, Leg, OptionContract, Package, Right, Side, Tenor
from sota.pricing.surface import leg_is_liquid
from sota.strategies.families import FAMILIES, normalize_coords


def resolve_expiry(chain: Chain, tenor: Tenor, min_dte: int = 2) -> date | None:
    """Closest listed expiry to the bucket anchor, preferring expiries inside the bucket."""
    lo, hi = tenor.bounds
    candidates = [e for e in chain.expiries() if (e - chain.asof).days >= min_dte]
    if not candidates:
        return None
    in_bucket = [e for e in candidates if lo <= (e - chain.asof).days <= hi]
    pool = in_bucket or candidates
    return min(pool, key=lambda e: abs((e - chain.asof).days - tenor.anchor))


def resolve_strike(
    chain: Chain,
    expiry: date,
    right: Right,
    target_delta: float,
    max_rel_half_spread: float = 0.5,
) -> OptionContract | None:
    """Contract at `expiry`/`right` whose |delta| is nearest the target, among liquid quotes."""
    best: OptionContract | None = None
    best_err = float("inf")
    for c in chain.contracts(expiry, right):
        q = chain.quote(c)
        if q is None or q.delta is None or not leg_is_liquid(c, q, max_rel_half_spread):
            continue
        err = abs(abs(q.delta) - target_delta)
        if err < best_err:
            best, best_err = c, err
    return best


def max_loss_per_share(legs: list[Leg], net_premium: float) -> float | None:
    """Worst expiry P&L per share, or None when the structure has unbounded loss."""
    if sum(leg.ratio for leg in legs if leg.contract.right.is_call) < 0:
        return None
    points = [0.0] + sorted({leg.contract.strike for leg in legs})
    worst = min(
        sum(leg.ratio * leg.contract.intrinsic(s) for leg in legs) - net_premium for s in points
    )
    return max(-worst, 0.0)


def resolve_package(
    chain: Chain,
    family: str,
    side: Side,
    tenor: Tenor,
    coords: dict[str, float] | None = None,
    *,
    min_dte: int = 2,
    max_rel_half_spread: float = 0.5,
) -> Package | None:
    """Map a strategy decision (u, f, omega, theta, kappa) to exact contracts with per-share pricing."""
    spec = FAMILIES.get(family)
    if spec is None:
        return None
    coords = normalize_coords(spec, coords)
    expiry = resolve_expiry(chain, tenor, min_dte)
    if expiry is None:
        return None
    chain.ensure_greeks()

    leg_specs = spec.legs(side, coords)
    merged: dict[OptionContract, int] = {}
    for ls in leg_specs:
        c = resolve_strike(chain, expiry, ls.right, ls.delta, max_rel_half_spread)
        if c is None:
            return None
        merged[c] = merged.get(c, 0) + ls.ratio
    merged = {c: r for c, r in merged.items() if r != 0}
    if len(merged) != len({(ls.right, ls.delta) for ls in leg_specs}):
        return None  # strikes collapsed onto the same contract: degenerate structure

    legs = [
        Leg(c, r)
        for c, r in sorted(merged.items(), key=lambda kv: (kv[0].strike, kv[0].right.value))
    ]
    net_premium = delta = gamma = vega = theta = half_spread = 0.0
    leg_deltas: list[float] = []
    leg_ivs: list[float] = []
    for leg in legs:
        q = chain.quote(leg.contract)
        assert q is not None and q.has_greeks
        net_premium += leg.ratio * q.mark
        delta += leg.ratio * q.delta  # type: ignore[operator]
        gamma += leg.ratio * q.gamma  # type: ignore[operator]
        vega += leg.ratio * q.vega  # type: ignore[operator]
        theta += leg.ratio * q.theta  # type: ignore[operator]
        half_spread += abs(leg.ratio) * q.half_spread
        leg_deltas.append(abs(q.delta))  # type: ignore[arg-type]
        leg_ivs.append(q.iv)  # type: ignore[arg-type]

    return Package(
        family=family,
        underlying=chain.underlying,
        side=side,
        tenor=tenor,
        expiry=expiry,
        legs=legs,
        coords=coords,
        hedged=spec.hedged,
        net_premium=net_premium,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        max_loss=max_loss_per_share(legs, net_premium),
        half_spread=half_spread,
        leg_deltas=leg_deltas,
        leg_ivs=leg_ivs,
    )
