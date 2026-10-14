"""Daily decision runner for paper (auto-execute) and live (advisory tickets) modes.

Both modes share the pipeline; they differ only in how a decision is acted on:
paper -> `Broker.submit_package`, live -> `ManualTicketBroker.issue_*` (never an order).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from sota.config import Settings
from sota.data.base import ChainSource
from sota.data.splits import apply_splits
from sota.execution.base import Broker, Ticket
from sota.execution.manual import ManualTicketBroker
from sota.features import MarketFeatures
from sota.journal import Journal
from sota.models import MULTIPLIER, Chain, Position
from sota.notify.base import Notifier
from sota.pipeline import atm_iv_map, compute_state, load_chains, price_candidates
from sota.resolvers.hedge import hedge_trade
from sota.risk import current_greeks
from sota.selectors.base import (
    PortfolioView,
    PricedCandidate,
    Proposal,
    Selector,
    SelectorUnavailable,
)

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    asof: date
    mode: str
    nav: float
    features: dict[str, MarketFeatures]
    proposals: list[Proposal] = field(default_factory=list)
    candidates: list[PricedCandidate] = field(default_factory=list)
    chosen: list[PricedCandidate] = field(default_factory=list)
    opened: list[Position] = field(default_factory=list)
    closed: list[Position] = field(default_factory=list)
    tickets: list[Ticket] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.mode} run {self.asof}: NAV ${self.nav:,.0f}"]
        lines.append(
            f"  features for {len(self.features)} underlyings, {len(self.proposals)} proposals, "
            f"{len(self.candidates)} priced, {len(self.chosen)} chosen"
        )
        for p in self.opened:
            lines.append(f"  opened {p.id}: {p.package.describe()} x{p.units}")
        for p in self.closed:
            lines.append(f"  closed {p.id} ({p.status}): pnl ${p.realized_pnl or 0:,.0f}")
        for t in self.tickets:
            lines.append(
                f"  ticket {t.id}: {t.intent} {t.package.underlying} {t.package.family} x{t.units}"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


class Runner:
    def __init__(
        self,
        settings: Settings,
        source: ChainSource,
        selector: Selector,
        broker: Broker | ManualTicketBroker,
        journal: Journal,
        notifier: Notifier,
    ) -> None:
        self.settings = settings
        self.mode = settings.mode.value
        self.limits = settings.limits
        self.costs = settings.effective_costs
        self.source = source
        self.selector = selector
        self.broker = broker
        self.journal = journal
        self.notifier = notifier

    @property
    def is_live(self) -> bool:
        return isinstance(self.broker, ManualTicketBroker)

    def _slip(self, half_spread: float) -> float:
        """Price concession the cost model assumes on every fill (backtest and paper agree)."""
        return self.costs.half_spread_mult * half_spread

    # ------------------------------------------------------------------ exits
    def _close_paper(
        self, pos: Position, value: float, reason: str, asof: date, chain: Chain | None = None
    ) -> None:
        half_spread = pos.package.half_spread
        if chain is not None:
            quotes = [chain.quote(leg.contract) for leg in pos.package.legs]
            if all(q is not None and q.is_valid for q in quotes):
                half_spread = sum(
                    abs(leg.ratio) * q.half_spread
                    for leg, q in zip(pos.package.legs, quotes, strict=True)
                )
        # A resting order at the mark does not fill in the last minutes; pay the modelled slippage.
        limit = value - self._slip(half_spread)
        order_id = self.broker.submit_package(pos.package, pos.units, limit, "close")
        fee = self.costs.option_fee * pos.package.contracts_per_unit * pos.units
        pos.fees_paid += fee
        pos.exit_date, pos.exit_premium, pos.status = asof, limit, reason
        pos.realized_pnl = (limit - pos.entry_premium) * MULTIPLIER * pos.units - pos.fees_paid
        pos.tags["close_order_id"] = order_id
        pos.tags["close_fill"] = "assumed"
        self.journal.save_position(pos, self.mode)

    def manage_exits(
        self, positions: list[Position], chains: dict[str, Chain], asof: date, report: RunReport
    ) -> None:
        for pos in positions:
            pkg = pos.package
            chain = chains.get(pkg.underlying)
            if pkg.expiry <= asof:
                spot = chain.spot if chain is not None else 0.0
                v = pkg.intrinsic_value(spot)
                pos.exit_date, pos.exit_premium, pos.status = asof, v, "expired"
                pos.realized_pnl = (v - pos.entry_premium) * MULTIPLIER * pos.units - pos.fees_paid
                self.journal.save_position(pos, self.mode)
                report.closed.append(pos)
                if self.is_live:
                    self.notifier.send(
                        f"SOTA: {pkg.underlying} {pkg.family} expired",
                        f"Position {pos.id} expired; verify settlement in your broker app.",
                    )
                continue
            if chain is None:
                continue
            v = pkg.value(chain)
            if v is None:
                continue
            reason = None
            if asof >= pos.force_close_date:
                reason = "force_close"
            elif pos.pnl_per_share(v) <= -pos.stop_loss_per_share:
                reason = "stop"
            if reason is None:
                continue
            if self.is_live:
                broker: ManualTicketBroker = self.broker  # type: ignore[assignment]
                if not broker.pending_close_exists(pos.id):
                    report.tickets.append(broker.issue_close(pos, v, reason))
            else:
                self._close_paper(pos, v, reason, asof, chain)
                report.closed.append(pos)

    # ------------------------------------------------------------------ entries
    def execute(self, chosen: list[PricedCandidate], asof: date, report: RunReport) -> None:
        for cand in chosen[: self.limits.max_order_lines]:
            pkg = cand.package
            if self.is_live:
                broker: ManualTicketBroker = self.broker  # type: ignore[assignment]
                report.tickets.append(broker.issue_open(cand, self.limits))
                continue
            limit = pkg.net_premium + self._slip(pkg.half_spread)
            order_id = self.broker.submit_package(pkg, cand.units, limit, "open")
            fee = self.costs.option_fee * pkg.contracts_per_unit * cand.units
            pos = Position(
                id=uuid.uuid4().hex[:8],
                package=pkg,
                units=cand.units,
                entry_date=asof,
                entry_premium=limit,
                stop_loss_per_share=self.limits.stop_loss_frac * abs(limit),
                force_close_date=pkg.expiry - timedelta(days=self.limits.force_close_dte),
                fees_paid=fee,
                tags={
                    "order_id": order_id,
                    "fill": "assumed",
                    "rationale": cand.proposal.rationale,
                },
            )
            self.journal.save_position(pos, self.mode)
            report.opened.append(pos)

    def rebalance_hedges(
        self, positions: list[Position], chains: dict[str, Chain], nav: float, report: RunReport
    ) -> None:
        hedges = self.journal.hedge_shares(self.mode)
        for u, chain in chains.items():
            hedged = [
                p for p in positions if p.is_open and p.package.underlying == u and p.package.hedged
            ]
            current = hedges.get(u, 0)
            if not hedged:
                trade = -current
            else:
                option_dd = 0.0
                for pos in hedged:
                    g = current_greeks(pos.package, chain)
                    delta = g[0] if g is not None else pos.package.delta
                    option_dd += delta * chain.spot * MULTIPLIER * pos.units
                trade = hedge_trade(
                    option_dd, current, chain.spot, nav, self.limits.hedge_band_frac
                )
            if trade == 0:
                continue
            self.broker.submit_shares(u, trade)
            self.journal.set_hedge_shares(self.mode, u, current + trade)
            report.notes.append(f"hedge {u}: {trade:+d} shares")

    # ------------------------------------------------------------------ main entry point
    def run_once(self, asof: date | None = None) -> RunReport:
        asof = asof or date.today()
        chains = load_chains(self.source, self.settings.universe, asof, self.limits)
        acct = self.broker.account()
        # A paper account can be sized as if it were the (smaller) real account.
        nav = min(acct.nav, self.settings.sizing_nav) if self.settings.sizing_nav else acct.nav
        features = compute_state(self.source, chains, asof)
        report = RunReport(asof, self.mode, nav, features)

        positions = self.journal.list_positions(self.mode, "open")
        for pos in positions:
            chain = chains.get(pos.package.underlying)
            strikes = {c.strike for c in chain.quotes} if chain is not None else None
            for ex, ratio in apply_splits(pos, asof, strikes):
                self.journal.save_position(pos, self.mode)
                report.notes.append(
                    f"{pos.package.underlying} {ratio:g}:1 split on {ex}: position {pos.id} "
                    f"re-expressed as {pos.units} units; verify your broker shows the same"
                )
        self.manage_exits(positions, chains, asof, report)
        positions = [p for p in positions if p.is_open]

        marks = {}
        for p in positions:
            chain = chains.get(p.package.underlying)
            v = p.package.value(chain) if chain is not None else None
            if v is not None:
                marks[p.id] = v
                # EOD mark kept with the position so the cockpit shows P&L without live chain calls
                p.tags["mark"], p.tags["mark_date"] = f"{v:.4f}", asof.isoformat()
                self.journal.save_position(p, self.mode)
        view = PortfolioView(asof, nav, acct.cash, positions, marks)
        atm_iv = atm_iv_map(features)

        try:
            report.proposals = self.selector.propose(features, view)
            report.candidates = price_candidates(
                report.proposals,
                chains,
                atm_iv,
                nav,
                positions,
                self.journal.hedge_shares(self.mode),
                self.limits,
            )
            report.chosen = self.selector.decide(
                report.candidates, view, self.limits.max_order_lines
            )
            self.execute(report.chosen, asof, report)
        except SelectorUnavailable as exc:
            # Exits were already managed above; only new entries are skipped today.
            report.notes.append(f"SELECTOR UNAVAILABLE, no new positions: {exc}")
            log.error("selector unavailable: %s", exc)

        if not self.is_live and self.limits.hedge:
            self.rebalance_hedges(positions + report.opened, chains, nav, report)

        self.journal.record_nav(asof, self.mode, acct.nav, acct.cash)
        self.journal.record_decision(
            asof,
            self.mode,
            self.selector.name,
            "run",
            {
                "nav": acct.nav,
                "cash": acct.cash,
                "features": {u: f.as_dict() for u, f in features.items()},
                "spots": {u: c.spot for u, c in chains.items()},
                "proposals": [asdict(p) for p in report.proposals],
                "candidates": len(report.candidates),
                "chosen": [c.package.describe() for c in report.chosen],
                "opened": [
                    {"id": p.id, "text": p.package.describe(), "units": p.units}
                    for p in report.opened
                ],
                "closed": [
                    {
                        "id": p.id,
                        "text": p.package.describe(),
                        "reason": p.status,
                        "pnl": p.realized_pnl,
                    }
                    for p in report.closed
                ],
                "tickets": [t.id for t in report.tickets],
                "notes": list(report.notes),
            },
        )
        self.notifier.send(f"SOTA {self.mode} run {asof}", report.summary())
        return report

    def reconcile(self) -> list[str]:
        """Paper mode: replace assumed fills with the broker's, drop unfilled opens, revive unfilled closes.

        Runs before each decision so the journal never diverges from the account: an entry whose
        DAY order expired never existed; an exit whose order did not fill is still a live position
        and goes back to the open book (the next cycle re-evaluates its stop / force-close).
        """
        notes: list[str] = []
        if self.is_live or not hasattr(self.broker, "order_status"):
            return notes
        done = {"filled"}
        gone = {"canceled", "expired", "rejected", "replaced"}
        for pos in self.journal.list_positions(self.mode, None):
            if pos.is_open and pos.tags.get("fill") in {"assumed", "assumed_mid"}:
                order_id = pos.tags.get("order_id")
                if not order_id:
                    continue
                status, price, qty = self.broker.order_status(order_id)  # type: ignore[attr-defined]
                if status in done or (qty and status in gone):
                    if price is not None:
                        pos.entry_premium = price if pos.package.net_premium >= 0 else -price
                        pos.stop_loss_per_share = self.limits.stop_loss_frac * abs(
                            pos.entry_premium
                        )
                    if qty and qty < pos.units:
                        pos.fees_paid *= qty / pos.units
                        pos.units = qty
                    pos.tags["fill"] = "confirmed"
                    notes.append(f"{pos.id} filled x{pos.units} at {pos.entry_premium:+.2f}")
                elif status in gone:
                    pos.status = "unfilled"
                    notes.append(f"{pos.id} {status} unfilled; removed")
                else:
                    continue
                self.journal.save_position(pos, self.mode)
            elif not pos.is_open and pos.tags.get("close_fill") == "assumed":
                order_id = pos.tags.get("close_order_id")
                if not order_id:
                    continue
                status, price, qty = self.broker.order_status(order_id)  # type: ignore[attr-defined]
                if status in done or (qty and status in gone):
                    if qty and qty < pos.units:
                        # partial close: the remainder is still held; keep it as its own open position
                        per_unit_fee = self.costs.option_fee * pos.package.contracts_per_unit
                        rest = Position(
                            id=uuid.uuid4().hex[:8],
                            package=pos.package,
                            units=pos.units - qty,
                            entry_date=pos.entry_date,
                            entry_premium=pos.entry_premium,
                            stop_loss_per_share=pos.stop_loss_per_share,
                            force_close_date=pos.force_close_date,
                            fees_paid=per_unit_fee * (pos.units - qty),
                            tags={**pos.tags, "fill": "confirmed", "split_from": pos.id},
                        )
                        for k in ("close_order_id", "close_fill"):
                            rest.tags.pop(k, None)
                        self.journal.save_position(rest, self.mode)
                        pos.units = qty
                        pos.fees_paid = 2 * per_unit_fee * qty  # entry + exit for the closed units
                        notes.append(f"{pos.id} partially closed x{qty}; {rest.id} holds the rest")
                    if price is not None:
                        pos.exit_premium = price if (pos.exit_premium or 0) >= 0 else -price
                    pos.realized_pnl = (
                        (pos.exit_premium or 0.0) - pos.entry_premium
                    ) * MULTIPLIER * pos.units - pos.fees_paid
                    pos.tags["close_fill"] = "confirmed"
                    notes.append(f"{pos.id} close filled at {pos.exit_premium:+.2f}")
                elif status in gone:
                    fee = self.costs.option_fee * pos.package.contracts_per_unit * pos.units
                    pos.fees_paid = max(pos.fees_paid - fee, 0.0)
                    pos.tags["close_attempts"] = str(int(pos.tags.get("close_attempts", "0")) + 1)
                    pos.tags["last_close_reason"] = pos.status
                    pos.status, pos.exit_date, pos.exit_premium, pos.realized_pnl = (
                        "open",
                        None,
                        None,
                        None,
                    )
                    for k in ("close_order_id", "close_fill"):
                        pos.tags.pop(k, None)
                    notes.append(f"{pos.id} close {status} unfilled; position reopened")
                else:
                    continue
                self.journal.save_position(pos, self.mode)
        return notes
