"""Shared decision pipeline: chains -> features -> proposals -> priced candidates (used by backtest, paper, live)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from sota.config import RiskLimits
from sota.data.base import ChainSource
from sota.features import RV_WINDOW, MarketFeatures, compute_features
from sota.models import Chain, Position
from sota.pricing.surface import smooth_chain
from sota.resolvers.contract import resolve_package
from sota.resolvers.size import resolve_size
from sota.risk import build_portfolio_risk
from sota.selectors.base import PricedCandidate, Proposal

DEFAULT_ATM_IV = 0.30


def prepare_chain(chain: Chain, limits: RiskLimits | None = None) -> Chain:
    """Solve greeks and, if enabled, replace noisy per-contract marks with smile-fitted values."""
    chain.ensure_greeks()
    if limits is not None and limits.smooth_surface and not chain.smoothed:
        smooth_chain(chain)
        chain.smoothed = True
    return chain


def load_chains(
    source: ChainSource, universe: Sequence[str], asof: date, limits: RiskLimits | None = None
) -> dict[str, Chain]:
    chains: dict[str, Chain] = {}
    for u in universe:
        chain = source.get_chain(u, asof)
        if chain is not None and chain.quotes:
            chains[u] = prepare_chain(chain, limits)
    return chains


def compute_state(
    source: ChainSource,
    chains: dict[str, Chain],
    asof: date,
    prev_chains: dict[str, Chain] | None = None,
    prev_iv: dict[str, float] | None = None,
) -> dict[str, MarketFeatures]:
    features: dict[str, MarketFeatures] = {}
    for u, chain in chains.items():
        closes = list(source.get_closes(u, asof, RV_WINDOW + 1))
        if not closes or abs(closes[-1] - chain.spot) > 1e-9:
            closes.append(chain.spot)
        features[u] = compute_features(
            chain,
            closes,
            prev_chain=(prev_chains or {}).get(u),
            prev_iv=(prev_iv or {}).get(u),
        )
    return features


def atm_iv_map(features: dict[str, MarketFeatures]) -> dict[str, float]:
    return {u: f.iv for u, f in features.items() if not math.isnan(f.iv) and f.iv > 0}


def price_candidates(
    proposals: Sequence[Proposal],
    chains: dict[str, Chain],
    atm_iv: dict[str, float],
    nav: float,
    positions: list[Position],
    hedge_shares: dict[str, int],
    limits: RiskLimits,
) -> list[PricedCandidate]:
    """Resolve each open-proposal into contracts and a size; drop anything the resolvers reject."""
    risk = build_portfolio_risk(
        nav, positions, chains, atm_iv, hedge_shares, limits.iv_shock, limits.risk_sigma
    )
    out: list[PricedCandidate] = []
    per_u: dict[str, int] = {}
    for p in proposals:
        if p.action != "open":
            continue
        if limits.families is not None and p.family not in limits.families:
            continue
        chain = chains.get(p.underlying)
        if chain is None or per_u.get(p.underlying, 0) >= limits.candidates_per_underlying:
            continue
        pkg = resolve_package(
            chain,
            p.family,
            p.side,
            p.tenor,
            p.coords,
            min_dte=limits.force_close_dte + 1,
            max_rel_half_spread=limits.max_rel_half_spread,
        )
        if pkg is None:
            continue
        size = resolve_size(
            pkg, chain.spot, atm_iv.get(p.underlying, DEFAULT_ATM_IV), limits, risk, chain.asof
        )
        if not size.accepted:
            continue
        per_u[p.underlying] = per_u.get(p.underlying, 0) + 1
        out.append(PricedCandidate(p, pkg, size.units, size.risk_per_unit, size.capital_per_unit))
    return out
