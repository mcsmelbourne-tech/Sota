"""Point-in-time trading environment (paper §3.1, App. D–E): marking, exits, execution, hedging, reward."""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from sota.config import Costs, RiskLimits
from sota.data.base import ChainSource
from sota.data.splits import apply_splits, splits_between
from sota.features import MarketFeatures
from sota.models import MULTIPLIER, Chain, Package, Position
from sota.pipeline import DEFAULT_ATM_IV, atm_iv_map, compute_state, load_chains, price_candidates
from sota.resolvers.hedge import hedge_trade
from sota.resolvers.size import resolve_size
from sota.risk import build_portfolio_risk, current_greeks
from sota.selectors.base import PortfolioView, PricedCandidate, Selector, SelectorUnavailable

log = logging.getLogger(__name__)

SHARE_HALF_SPREAD = 1e-4  # 1 bp (paper Table 10)


@dataclass(slots=True)
class Trade:
    asof: date
    position_id: str
    action: str
    underlying: str
    family: str
    units: int
    price_per_share: float
    fees: float
    pnl: float | None = None
    note: str = ""


@dataclass(slots=True)
class StepResult:
    asof: date
    nav: float
    reward: float
    opened: list[Position]
    closed: list[Position]
    features: dict[str, MarketFeatures]


@dataclass
class BacktestResult:
    nav: list[tuple[date, float]]
    trades: list[Trade]
    positions: list[Position]
    rewards: list[float] = field(default_factory=list)
    aborted: str | None = None  # set when the selector became unavailable before the last date

    @property
    def nav_values(self) -> list[float]:
        return [v for _, v in self.nav]

    def trade_pnls(self) -> list[float]:
        return [p.realized_pnl for p in self.positions if p.realized_pnl is not None]


class TradingEnv:
    def __init__(
        self,
        source: ChainSource,
        universe: Sequence[str],
        limits: RiskLimits,
        costs: Costs,
        initial_nav: float = 1_000_000.0,
    ) -> None:
        self.source = source
        self.universe = list(universe)
        self.limits = limits
        self.costs = costs
        self.initial_nav = initial_nav
        self.cash = initial_nav
        self.positions: list[Position] = []
        self.hedge_shares: dict[str, int] = {}
        self.nav_history: list[tuple[date, float]] = []
        self.rewards: list[float] = []
        self.trades: list[Trade] = []
        self.ruined = False
        self._last_mark: dict[str, float] = {}
        self._last_spot: dict[str, float] = {}
        self._prev_chain: dict[str, Chain] = {}
        self._prev_iv: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    def apply_splits(self, chains: dict[str, Chain], asof: date) -> None:
        for pos in self.open_positions:
            u = pos.package.underlying
            chain = chains.get(u)
            strikes = {c.strike for c in chain.quotes} if chain is not None else None
            for ex, ratio in apply_splits(pos, asof, strikes):
                self._last_mark[pos.id] = self._last_mark.get(pos.id, pos.entry_premium) / ratio
                self._log(
                    Trade(
                        asof,
                        pos.id,
                        "split",
                        u,
                        pos.package.family,
                        pos.units,
                        pos.entry_premium,
                        0.0,
                        note=f"{ratio:g}:1 on {ex}",
                    )
                )
        prev = self.nav_history[-1][0] if self.nav_history else None
        if prev is None:
            return
        for u in list(self.hedge_shares):
            for _, ratio in splits_between(u, prev, asof):
                self.hedge_shares[u] = int(round(self.hedge_shares[u] * ratio))

    def _option_fee(self, pkg: Package, units: int) -> float:
        return self.costs.option_fee * pkg.contracts_per_unit * units

    def _current_half_spread(self, pkg: Package, chain: Chain | None) -> float:
        if chain is None:
            return pkg.half_spread
        total = 0.0
        for leg in pkg.legs:
            q = chain.quote(leg.contract)
            if q is None or not q.is_valid:
                return pkg.half_spread
            total += abs(leg.ratio) * q.half_spread
        return total

    def mark_nav(self, chains: dict[str, Chain]) -> float:
        total = self.cash
        for pos in self.open_positions:
            chain = chains.get(pos.package.underlying)
            v = pos.package.value(chain) if chain is not None else None
            if v is None:
                v = self._last_mark.get(pos.id, pos.entry_premium)
            total += pos.market_value(v)
        for u, shares in self.hedge_shares.items():
            spot = chains[u].spot if u in chains else self._last_spot.get(u, 0.0)
            total += shares * spot
        return total

    def _log(self, trade: Trade) -> None:
        self.trades.append(trade)

    # ------------------------------------------------------------------ position lifecycle
    def open_position(self, pkg: Package, units: int, asof: date, rationale: str = "") -> Position:
        fill = pkg.net_premium + pkg.half_spread * self.costs.half_spread_mult
        fee = self._option_fee(pkg, units)
        self.cash -= fill * MULTIPLIER * units + fee
        pos = Position(
            id=uuid.uuid4().hex[:8],
            package=pkg,
            units=units,
            entry_date=asof,
            entry_premium=fill,
            stop_loss_per_share=self.limits.stop_loss_frac * abs(fill),
            force_close_date=pkg.expiry - timedelta(days=self.limits.force_close_dte),
            fees_paid=fee,
            tags={"rationale": rationale} if rationale else {},
        )
        self.positions.append(pos)
        self._last_mark[pos.id] = pkg.net_premium
        self._log(
            Trade(
                asof,
                pos.id,
                "open",
                pkg.underlying,
                pkg.family,
                units,
                fill,
                fee,
                note=pkg.describe(),
            )
        )
        return pos

    def close_position(self, pos: Position, chain: Chain | None, asof: date, reason: str) -> None:
        v = pos.package.value(chain) if chain is not None else None
        if v is None:
            v = self._last_mark.get(pos.id, pos.entry_premium)
        fill = v - self._current_half_spread(pos.package, chain) * self.costs.half_spread_mult
        fee = self._option_fee(pos.package, pos.units)
        self.cash += fill * MULTIPLIER * pos.units - fee
        pos.fees_paid += fee
        pos.exit_date, pos.exit_premium, pos.status = asof, fill, reason
        pos.realized_pnl = (fill - pos.entry_premium) * MULTIPLIER * pos.units - pos.fees_paid
        self._log(
            Trade(
                asof,
                pos.id,
                reason,
                pos.package.underlying,
                pos.package.family,
                pos.units,
                fill,
                fee,
                pos.realized_pnl,
            )
        )

    def settle_expired(self, pos: Position, spot: float, asof: date) -> None:
        pkg = pos.package
        v = pkg.intrinsic_value(spot)
        assigned = any(leg.ratio < 0 and leg.contract.intrinsic(spot) > 0 for leg in pkg.legs)
        fee = self.costs.assignment_fee if assigned else 0.0
        self.cash += v * MULTIPLIER * pos.units - fee
        pos.fees_paid += fee
        pos.exit_date, pos.exit_premium, pos.status = asof, v, "expired"
        pos.realized_pnl = (v - pos.entry_premium) * MULTIPLIER * pos.units - pos.fees_paid
        self._log(
            Trade(
                asof,
                pos.id,
                "expired",
                pkg.underlying,
                pkg.family,
                pos.units,
                v,
                fee,
                pos.realized_pnl,
            )
        )

    def _trade_shares(
        self, underlying: str, shares: int, spot: float, asof: date, note: str
    ) -> None:
        if shares == 0:
            return
        cost = abs(shares) * spot * (SHARE_HALF_SPREAD) + abs(shares) * self.costs.share_commission
        self.cash -= shares * spot + cost
        self.hedge_shares[underlying] = self.hedge_shares.get(underlying, 0) + shares
        self._log(Trade(asof, "-", "hedge", underlying, "-", shares, spot, cost, note=note))

    # ------------------------------------------------------------------ one decision date
    def manage_exits(self, chains: dict[str, Chain], asof: date) -> list[Position]:
        closed: list[Position] = []
        for pos in self.open_positions:
            u = pos.package.underlying
            chain = chains.get(u)
            if pos.package.expiry <= asof:
                spot = chain.spot if chain is not None else self._last_spot.get(u, 0.0)
                self.settle_expired(pos, spot, asof)
                closed.append(pos)
                continue
            if chain is None:
                continue
            v = pos.package.value(chain)
            if v is None:
                continue
            self._last_mark[pos.id] = v
            if asof >= pos.force_close_date:
                self.close_position(pos, chain, asof, "force_close")
                closed.append(pos)
            elif pos.pnl_per_share(v) <= -pos.stop_loss_per_share:
                self.close_position(pos, chain, asof, "stop")
                closed.append(pos)
        return closed

    def compute_state(self, chains: dict[str, Chain], asof: date) -> dict[str, MarketFeatures]:
        features = compute_state(self.source, chains, asof, self._prev_chain, self._prev_iv)
        for u, f in features.items():
            if not math.isnan(f.iv):
                self._prev_iv[u] = f.iv
        return features

    def execute(
        self,
        chosen: Sequence[PricedCandidate],
        chains: dict[str, Chain],
        atm_iv: dict[str, float],
        asof: date,
    ) -> list[Position]:
        opened: list[Position] = []
        for cand in chosen[: self.limits.max_order_lines]:
            chain = chains[cand.package.underlying]
            nav = self.mark_nav(chains)
            risk = build_portfolio_risk(
                nav,
                self.positions,
                chains,
                atm_iv,
                self.hedge_shares,
                self.limits.iv_shock,
                self.limits.risk_sigma,
            )
            size = resolve_size(
                cand.package,
                chain.spot,
                atm_iv.get(chain.underlying, DEFAULT_ATM_IV),
                self.limits,
                risk,
                asof,
            )
            if not size.accepted:
                continue
            units = min(size.units, cand.units)
            opened.append(self.open_position(cand.package, units, asof, cand.proposal.rationale))
        return opened

    def rebalance_hedges(self, chains: dict[str, Chain], asof: date, nav: float) -> None:
        if not self.limits.hedge:
            return
        for u, chain in chains.items():
            hedged = [
                p for p in self.open_positions if p.package.underlying == u and p.package.hedged
            ]
            current = self.hedge_shares.get(u, 0)
            if not hedged:
                if current:
                    self._trade_shares(u, -current, chain.spot, asof, "unwind")
                continue
            option_dd = 0.0
            for pos in hedged:
                g = current_greeks(pos.package, chain)
                delta = g[0] if g is not None else pos.package.delta
                option_dd += delta * chain.spot * MULTIPLIER * pos.units
            trade = hedge_trade(option_dd, current, chain.spot, nav, self.limits.hedge_band_frac)
            self._trade_shares(u, trade, chain.spot, asof, f"option $delta {option_dd:,.0f}")

    def accrue_borrow(self, chains: dict[str, Chain]) -> None:
        for u, shares in self.hedge_shares.items():
            if shares < 0 and u in chains:
                self.cash -= abs(shares) * chains[u].spot * self.costs.borrow_rate / 365.0

    def step(self, asof: date, selector: Selector | None) -> StepResult:
        chains = load_chains(self.source, self.universe, asof, self.limits)
        for u, chain in chains.items():
            self._last_spot[u] = chain.spot

        self.apply_splits(chains, asof)
        closed = self.manage_exits(chains, asof)
        features = self.compute_state(chains, asof)
        atm_iv = atm_iv_map(features)
        nav = self.mark_nav(chains)
        view = PortfolioView(asof, nav, self.cash, self.positions, dict(self._last_mark))

        opened: list[Position] = []
        if selector is not None and not self.ruined:
            proposals = selector.propose(features, view)
            for p in proposals:
                if p.action == "close" and p.position_id:
                    pos = next((x for x in self.open_positions if x.id == p.position_id), None)
                    if pos is not None:
                        self.close_position(
                            pos, chains.get(pos.package.underlying), asof, "selector_close"
                        )
                        closed.append(pos)
            candidates = price_candidates(
                proposals, chains, atm_iv, nav, self.positions, self.hedge_shares, self.limits
            )
            chosen = selector.decide(candidates, view, self.limits.max_order_lines)
            opened = self.execute(chosen, chains, atm_iv, asof)

        self.rebalance_hedges(chains, asof, nav)
        self.accrue_borrow(chains)

        nav_new = self.mark_nav(chains)
        prev_nav = self.nav_history[-1][1] if self.nav_history else self.initial_nav
        reward = math.log(nav_new / prev_nav) if nav_new > 0 and prev_nav > 0 else -math.inf
        self.nav_history.append((asof, nav_new))
        self.rewards.append(reward)
        self._prev_chain = chains

        if nav_new < self.limits.ruin_floor_frac * self.initial_nav and not self.ruined:
            self.ruined = True
            for pos in self.open_positions:
                self.close_position(pos, chains.get(pos.package.underlying), asof, "ruin")
                closed.append(pos)
        return StepResult(asof, nav_new, reward, opened, closed, features)

    def run(self, dates: Sequence[date], selector: Selector | None) -> BacktestResult:
        aborted: str | None = None
        for i, d in enumerate(dates, 1):
            try:
                step = self.step(d, selector)
            except SelectorUnavailable as exc:
                aborted = f"{d}: {exc}"
                break
            log.info(
                "%s day %d/%d nav=%.0f open=%d opened=%d closed=%d",
                d,
                i,
                len(dates),
                step.nav,
                len(self.open_positions),
                len(step.opened),
                len(step.closed),
            )
            if self.ruined:
                break
        return BacktestResult(
            list(self.nav_history),
            list(self.trades),
            list(self.positions),
            list(self.rewards),
            aborted,
        )
