"""Hindsight oracle (paper Table 2): picks the best-performing candidate using realized future outcomes.

Not tradable; it bounds the profitability available inside the structured strategy space and
sanity-checks the environment (it should be strongly positive on real data).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from sota.data.base import ChainSource
from sota.features import MarketFeatures
from sota.models import Chain, Side, Tenor
from sota.resolvers.contract import resolve_package
from sota.risk import package_scenario_risk
from sota.selectors.base import BaseSelector, PortfolioView, Proposal
from sota.strategies.families import FAMILIES


class HindsightOracle(BaseSelector):
    name = "oracle"

    def __init__(
        self,
        source: ChainSource,
        tenors: tuple[Tenor, ...] = (Tenor.T8_30, Tenor.T31_90),
        per_underlying: int = 1,
        max_lookback_days: int = 5,
    ) -> None:
        self.source = source
        self.tenors = tenors
        self.per_underlying = per_underlying
        self.max_lookback_days = max_lookback_days

    def _future_chain(self, underlying: str, target: date) -> Chain | None:
        for back in range(self.max_lookback_days + 1):
            chain = self.source.get_chain(underlying, target - timedelta(days=back))
            if chain is not None and chain.quotes:
                chain.ensure_greeks()
                return chain
        return None

    def propose(
        self, features: dict[str, MarketFeatures], portfolio: PortfolioView
    ) -> list[Proposal]:
        proposals: list[Proposal] = []
        for u, f in features.items():
            chain = self.source.get_chain(u, portfolio.asof)
            if chain is None:
                continue
            chain.ensure_greeks()
            atm_iv = f.iv if not math.isnan(f.iv) and f.iv > 0 else 0.3
            scored: list[tuple[float, Proposal]] = []
            for tenor in self.tenors:
                future = self._future_chain(u, portfolio.asof + timedelta(days=tenor.anchor))
                if future is None:
                    continue
                for code, spec in FAMILIES.items():
                    sides = (
                        (Side.LONG, Side.SHORT)
                        if spec.category == "directional" or code == "bf"
                        else (Side.LONG,)
                    )
                    for side in sides:
                        pkg = resolve_package(chain, code, side, tenor)
                        if pkg is None:
                            continue
                        fwd = pkg.value(future)
                        if fwd is None:
                            fwd = (
                                pkg.intrinsic_value(future.spot)
                                if future.asof >= pkg.expiry
                                else None
                            )
                        if fwd is None:
                            continue
                        risk = package_scenario_risk(pkg, chain.spot, atm_iv, 0.10)
                        if risk <= 0:
                            continue
                        score = (fwd - pkg.net_premium) / risk
                        if score > 0:
                            scored.append(
                                (
                                    score,
                                    Proposal(
                                        u,
                                        code,
                                        side,
                                        tenor,
                                        dict(pkg.coords),
                                        f"oracle score {score:.2f}",
                                    ),
                                )
                            )
            scored.sort(key=lambda t: t[0], reverse=True)
            proposals.extend(p for _, p in scored[: self.per_underlying])
        return proposals
