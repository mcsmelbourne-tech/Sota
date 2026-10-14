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

    # ------------------------------------------------------------------ exits
    def _close_paper(self, pos: Position, value: float, reason: str, asof: date) -> None:
        order_id = self.broker.submit_package(pos.package, pos.units, value, "close")
        fee = self.costs.option_fee * pos.package.contracts_per_unit * pos.units
        pos.fees_paid += fee
        pos.exit_date, pos.exit_premium, pos.status = asof, value, reason
        pos.realized_pnl = (value - pos.entry_premium) * MULTIPLIER * pos.units - pos.fees_paid
        pos.tags["close_order_id"] = order_id
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
                self._close_paper(pos, v, reason, asof)
                report.closed.append(pos)

    # ------------------------------------------------------------------ entries
    def execute(self, chosen: list[PricedCandidate], asof: date, report: RunReport) -> None:
        for cand in chosen[: self.limits.max_order_lines]:
            pkg = cand.package
            if self.is_live:
                broker: ManualTicketBroker = self.broker  # type: ignore[assignment]
                report.tickets.append(broker.issue_open(cand, self.limits))
                continue
            order_id = self.broker.submit_package(pkg, cand.units, pkg.net_premium, "open")
            fee = self.costs.option_fee * pkg.contracts_per_unit * cand.units
            pos = Position(
                id=uuid.uuid4().hex[:8],
                package=pkg,
                units=cand.units,
                entry_date=asof,
                entry_premium=pkg.net_premium,
                stop_loss_per_share=self.limits.stop_loss_frac * abs(pkg.net_premium),
                force_close_date=pkg.expiry - timedelta(days=self.limits.force_close_dte),
                fees_paid=fee,
                tags={
                    "order_id": order_id,
                    "fill": "assumed_mid",
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
                "features": {u: f.as_dict() for u, f in features.items()},
                "proposals": [asdict(p) for p in report.proposals],
                "chosen": [c.package.describe() for c in report.chosen],
            },
        )
        self.notifier.send(f"SOTA {self.mode} run {asof}", report.summary())
        return report

    def reconcile(self) -> list[str]:
        """Paper mode: replace assumed mid fills with actual fills; drop unfilled orders."""
        notes: list[str] = []
        if self.is_live or not hasattr(self.broker, "order_status"):
            return notes
        for pos in self.journal.list_positions(self.mode, "open"):
            if pos.tags.get("fill") != "assumed_mid" or "order_id" not in pos.tags:
                continue
            status, price = self.broker.order_status(pos.tags["order_id"])  # type: ignore[attr-defined]
            if status == "filled":
                if price is not None:
                    pos.entry_premium = price if pos.package.net_premium >= 0 else -abs(price)
                    pos.stop_loss_per_share = self.limits.stop_loss_frac * abs(pos.entry_premium)
                pos.tags["fill"] = "confirmed"
                notes.append(f"{pos.id} filled at {pos.entry_premium:.2f}")
            elif status in {"canceled", "expired", "rejected"}:
                pos.status = "unfilled"
                notes.append(f"{pos.id} {status}; removed")
            self.journal.save_position(pos, self.mode)
        return notes
