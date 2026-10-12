"""Live mode: converts decisions into order tickets for manual entry in an external app (e.g. Robinhood).

This adapter has no order-placement code path. `submit_*` exist only to satisfy the Broker
interface and always raise `LiveModeOrderBlocked`.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta

from sota.config import Costs, RiskLimits
from sota.execution.base import AccountInfo, Intent, LiveModeOrderBlocked, Ticket
from sota.journal import Journal
from sota.models import MULTIPLIER, Package, Position
from sota.notify.base import Notifier
from sota.selectors.base import PricedCandidate


def format_ticket(t: Ticket) -> str:
    pkg = t.package
    lines = [
        f"{t.intent.upper()} {pkg.underlying} {pkg.family} ({pkg.side.value}, {pkg.tenor.value}) x{t.units}"
    ]
    for leg in pkg.legs:
        ratio = leg.ratio if t.intent == "open" else -leg.ratio
        verb = "BUY " if ratio > 0 else "SELL"
        lines.append(f"  {verb} {abs(ratio)}x {leg.contract.describe()}")
    kind = "DEBIT" if t.limit_price > 0 else "CREDIT"
    lines.append(f"  limit: {kind} {abs(t.limit_price):.2f} per share (mid)")
    if t.intent == "open":
        max_loss = (
            "unbounded" if pkg.max_loss is None else f"${pkg.max_loss * MULTIPLIER * t.units:,.0f}"
        )
        lines.append(
            f"  max loss {max_loss} | scenario risk ${t.scenario_risk:,.0f} | capital ${t.capital:,.0f}"
        )
        lines.append(
            f"  stop: close if P&L <= -{t.stop_loss_per_share:.2f}/share | force close by {t.force_close_date}"
        )
    if t.rationale:
        lines.append(f"  why: {t.rationale}")
    lines.append(
        f"  confirm: sota live confirm {t.id} --price <fill> | skip: sota live skip {t.id}"
    )
    return "\n".join(lines)


class ManualTicketBroker:
    mode = "live"

    def __init__(self, journal: Journal, notifier: Notifier, nav: float, costs: Costs) -> None:
        self.journal = journal
        self.notifier = notifier
        self.costs = costs
        self._nav = nav

    # -- Broker interface -----------------------------------------------------
    def account(self) -> AccountInfo:
        return AccountInfo(self._nav, self._nav)

    def set_nav(self, nav: float) -> None:
        self._nav = nav

    def submit_package(self, pkg: Package, units: int, limit_price: float, intent: Intent) -> str:
        raise LiveModeOrderBlocked("live mode never places orders; a ticket was expected instead")

    def submit_shares(self, underlying: str, qty: int) -> str:
        raise LiveModeOrderBlocked("live mode never places orders; hedge manually if desired")

    # -- tickets --------------------------------------------------------------
    def issue_open(self, cand: PricedCandidate, limits: RiskLimits) -> Ticket:
        pkg = cand.package
        t = Ticket(
            id=uuid.uuid4().hex[:8],
            created=datetime.now(),
            status="pending",
            intent="open",
            package=pkg,
            units=cand.units,
            limit_price=pkg.net_premium,
            scenario_risk=cand.risk_per_unit * cand.units,
            capital=cand.capital_per_unit * cand.units,
            stop_loss_per_share=limits.stop_loss_frac * abs(pkg.net_premium),
            force_close_date=pkg.expiry - timedelta(days=limits.force_close_dte),
            rationale=cand.proposal.rationale,
        )
        self.journal.save_ticket(t)
        self.notifier.send(
            f"SOTA ticket {t.id}: open {pkg.underlying} {pkg.family}", format_ticket(t)
        )
        return t

    def issue_close(self, pos: Position, value_per_share: float, reason: str) -> Ticket:
        t = Ticket(
            id=uuid.uuid4().hex[:8],
            created=datetime.now(),
            status="pending",
            intent="close",
            package=pos.package,
            units=pos.units,
            limit_price=value_per_share,
            scenario_risk=0.0,
            capital=0.0,
            stop_loss_per_share=pos.stop_loss_per_share,
            force_close_date=pos.force_close_date,
            rationale=reason,
            position_id=pos.id,
        )
        self.journal.save_ticket(t)
        self.notifier.send(
            f"SOTA ticket {t.id}: CLOSE {pos.package.underlying} {pos.package.family} ({reason})",
            format_ticket(t),
        )
        return t

    def pending_close_exists(self, position_id: str) -> bool:
        return any(t.position_id == position_id for t in self.journal.list_tickets("pending"))

    def confirm(
        self, ticket_id: str, fill_price: float, units: int | None = None, asof: date | None = None
    ) -> Position:
        t = self.journal.get_ticket(ticket_id)
        if t is None:
            raise KeyError(f"unknown ticket {ticket_id}")
        if t.status != "pending":
            raise ValueError(f"ticket {ticket_id} is already {t.status}")
        asof = asof or date.today()
        units = units or t.units
        fee = self.costs.option_fee * t.package.contracts_per_unit * units

        if t.intent == "open":
            pos = Position(
                id=uuid.uuid4().hex[:8],
                package=t.package,
                units=units,
                entry_date=asof,
                entry_premium=fill_price,
                stop_loss_per_share=t.stop_loss_per_share,
                force_close_date=t.force_close_date,
                fees_paid=fee,
                tags={"ticket": t.id, "rationale": t.rationale},
            )
            t.position_id = pos.id
        else:
            assert t.position_id is not None
            pos = self.journal.get_position(t.position_id)
            if pos is None:
                raise KeyError(f"ticket {ticket_id} refers to unknown position {t.position_id}")
            pos.fees_paid += fee
            pos.exit_date, pos.exit_premium, pos.status = asof, fill_price, "closed"
            pos.realized_pnl = (
                fill_price - pos.entry_premium
            ) * MULTIPLIER * pos.units - pos.fees_paid

        t.status, t.fill_price, t.filled_units, t.filled_at = (
            "filled",
            fill_price,
            units,
            datetime.now(),
        )
        self.journal.save_position(pos, self.mode)
        self.journal.save_ticket(t)
        return pos

    def skip(self, ticket_id: str) -> Ticket:
        t = self.journal.get_ticket(ticket_id)
        if t is None:
            raise KeyError(f"unknown ticket {ticket_id}")
        t.status = "skipped"
        self.journal.save_ticket(t)
        return t
