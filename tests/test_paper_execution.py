"""Paper execution: orders priced at the modelled fill, and reconcile keeping the journal honest."""

from __future__ import annotations

from datetime import timedelta

import pytest

from sota.config import Mode
from sota.execution.base import AccountInfo
from sota.journal import Journal
from sota.models import MULTIPLIER, Side, Tenor
from sota.notify.base import LogNotifier
from sota.runner import Runner
from sota.selectors.base import BaseSelector, Proposal


class FakePaperBroker:
    """Records orders; the test scripts each order's eventual status via `outcomes`."""

    mode = "paper"

    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.outcomes: dict[str, tuple[str, float | None, int]] = {}

    def account(self) -> AccountInfo:
        return AccountInfo(50_000.0, 50_000.0)

    def submit_package(self, pkg, units, limit_price, intent) -> str:
        oid = f"o{len(self.orders) + 1}"
        self.orders.append(
            {"id": oid, "pkg": pkg, "units": units, "limit": limit_price, "intent": intent}
        )
        return oid

    def submit_shares(self, underlying, qty) -> str:
        return "s"

    def order_status(self, order_id):
        return self.outcomes.get(order_id, ("new", None, 0))


class OpenOnce(BaseSelector):
    name = "open_once"

    def propose(self, features, portfolio):
        if portfolio.positions:
            return []
        return [Proposal("AAA", "dv", Side.LONG, Tenor.T8_30, rationale="t")]


@pytest.fixture()
def paper(settings, source):
    from sota.config import Costs

    s = settings.model_copy(
        update={
            "mode": Mode.PAPER,
            "cost_preset": "robinhood",
            "costs": Costs(half_spread_mult=0.5),
        }
    )
    journal = Journal(":memory:")
    broker = FakePaperBroker()
    return Runner(s, source, OpenOnce(), broker, journal, LogNotifier()), broker, journal


def test_open_orders_pay_the_modelled_slippage(paper, source):
    runner, broker, journal = paper
    report = runner.run_once(source.start)
    assert len(report.opened) == 1 and len(broker.orders) == 1
    pos, order = report.opened[0], broker.orders[0]
    pkg = pos.package
    expected = pkg.net_premium + 0.5 * pkg.half_spread
    assert order["limit"] == pytest.approx(expected) and order["limit"] > pkg.net_premium
    assert pos.entry_premium == pytest.approx(expected), "journal books the same assumed fill"
    assert pos.stop_loss_per_share == pytest.approx(0.6 * abs(expected))
    assert pos.tags["fill"] == "assumed"


def test_reconcile_confirms_fill_and_drops_expired_open(paper, source):
    runner, broker, journal = paper
    pos = runner.run_once(source.start).opened[0]
    oid = pos.tags["order_id"]

    broker.outcomes[oid] = ("filled", abs(pos.entry_premium) + 0.03, pos.units)
    notes = runner.reconcile()
    got = journal.get_position(pos.id)
    assert got.tags["fill"] == "confirmed" and "filled" in notes[0]
    sign = 1 if pos.package.net_premium >= 0 else -1
    assert got.entry_premium == pytest.approx(sign * (abs(pos.entry_premium) + 0.03))
    assert got.stop_loss_per_share == pytest.approx(0.6 * abs(got.entry_premium))
    assert runner.reconcile() == [], "confirmed positions are not touched again"

    # an expired DAY order on a fresh journal: the entry never existed
    fresh = Journal(":memory:")
    r = Runner(runner.settings, source, OpenOnce(), broker, fresh, LogNotifier())
    p2 = r.run_once(source.start).opened[0]
    broker.outcomes[p2.tags["order_id"]] = ("expired", None, 0)
    notes = r.reconcile()
    assert fresh.get_position(p2.id).status == "unfilled" and "removed" in notes[0]
    assert fresh.list_positions("paper", "open") == []


def _order(broker: FakePaperBroker, order_id: str) -> dict:
    return next(o for o in broker.orders if o["id"] == order_id)


def test_reconcile_revives_an_unfilled_close(paper, source):
    runner, broker, journal = paper
    pos = runner.run_once(source.start).opened[0]
    broker.outcomes[pos.tags["order_id"]] = ("filled", abs(pos.entry_premium), pos.units)
    runner.reconcile()

    # force the exit on the next cycle
    pos = journal.get_position(pos.id)
    pos.force_close_date = source.start
    journal.save_position(pos, "paper")
    report = runner.run_once(source.start + timedelta(days=1))
    assert [p.id for p in report.closed] == [pos.id]
    closed = journal.get_position(pos.id)
    close_order = _order(broker, closed.tags["close_order_id"])
    assert close_order["intent"] == "close" and closed.status == "force_close"
    assert closed.tags["close_fill"] == "assumed"
    # the close is priced below the mark by the modelled slippage
    chain = source.get_chain("AAA", source.start + timedelta(days=1))
    value = closed.package.value(chain)
    assert close_order["limit"] < value and closed.exit_premium == pytest.approx(
        close_order["limit"]
    )

    # the DAY order expires unfilled -> the position is still held
    broker.outcomes[close_order["id"]] = ("expired", None, 0)
    notes = runner.reconcile()
    revived = journal.get_position(pos.id)
    assert revived.is_open and revived.exit_date is None and revived.realized_pnl is None
    assert "close_order_id" not in revived.tags and revived.tags["close_attempts"] == "1"
    assert revived.tags["last_close_reason"] == "force_close"
    assert any("reopened" in n for n in notes)
    assert revived.fees_paid == pytest.approx(pos.fees_paid), "exit fee was refunded"

    # next cycle tries again; this time it fills
    report = runner.run_once(source.start + timedelta(days=2))
    assert pos.id in [p.id for p in report.closed]
    retry = journal.get_position(pos.id)
    broker.outcomes[retry.tags["close_order_id"]] = ("filled", 1.23, pos.units)
    runner.reconcile()
    done = journal.get_position(pos.id)
    assert done.status == "force_close" and done.tags["close_fill"] == "confirmed"
    sign = 1 if (closed.exit_premium or 0) >= 0 else -1
    assert done.exit_premium == pytest.approx(sign * 1.23)
    assert done.realized_pnl == pytest.approx(
        (done.exit_premium - done.entry_premium) * MULTIPLIER * done.units - done.fees_paid
    )


def test_reconcile_splits_a_partial_close(paper, source):
    runner, broker, journal = paper
    pos = runner.run_once(source.start).opened[0]
    broker.outcomes[pos.tags["order_id"]] = ("filled", abs(pos.entry_premium), pos.units)
    runner.reconcile()
    pos = journal.get_position(pos.id)
    if pos.units < 2:
        pos.units = 2
    units = pos.units
    pos.force_close_date = source.start
    journal.save_position(pos, "paper")
    runner.run_once(source.start + timedelta(days=1))
    closing = journal.get_position(pos.id)
    broker.outcomes[closing.tags["close_order_id"]] = ("expired", 1.0, 1)  # 1 of N filled
    notes = runner.reconcile()
    closed = journal.get_position(pos.id)
    rest = [p for p in journal.list_positions("paper", "open") if p.tags.get("split_from")]
    assert closed.units == 1 and not closed.is_open
    assert len(rest) == 1 and rest[0].units == units - 1 and rest[0].tags["split_from"] == pos.id
    per_unit_fee = 0.04 * pos.package.contracts_per_unit
    assert closed.fees_paid == pytest.approx(2 * per_unit_fee)
    assert rest[0].fees_paid == pytest.approx(per_unit_fee * (units - 1))
    assert any("partially closed" in n for n in notes)
