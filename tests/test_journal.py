from datetime import date, datetime

from sota.execution.base import Ticket
from sota.journal import (
    Journal,
    package_from_dict,
    package_to_dict,
    position_from_dict,
    position_to_dict,
)
from sota.models import Position, Side, Tenor
from sota.resolvers.contract import resolve_package


def test_package_round_trip(chain):
    pkg = resolve_package(chain, "ib", Side.LONG, Tenor.T8_30)
    back = package_from_dict(package_to_dict(pkg))
    assert back.legs == pkg.legs and back.net_premium == pkg.net_premium
    assert back.side is Side.LONG and back.tenor is Tenor.T8_30 and back.hedged


def test_journal_positions_and_tickets(chain):
    journal = Journal(":memory:")
    pkg = resolve_package(chain, "dv", Side.SHORT, Tenor.T31_90)
    pos = Position(
        "p1",
        pkg,
        2,
        chain.asof,
        pkg.net_premium,
        0.6 * pkg.net_premium,
        pkg.expiry,
        tags={"x": "y"},
    )
    journal.save_position(pos, "live")
    assert [p.id for p in journal.list_positions("live", "open")] == ["p1"]
    assert journal.list_positions("paper", "open") == []
    loaded = journal.get_position("p1")
    assert position_to_dict(loaded) == position_to_dict(position_from_dict(position_to_dict(pos)))

    t = Ticket(
        "t1",
        datetime(2025, 1, 2, 9, 30),
        "pending",
        "open",
        pkg,
        2,
        pkg.net_premium,
        500.0,
        2000.0,
        1.0,
        date(2025, 2, 1),
    )
    journal.save_ticket(t)
    assert journal.get_ticket("t1").package.legs == pkg.legs
    assert [x.id for x in journal.list_tickets("pending")] == ["t1"]
    t.status = "skipped"
    journal.save_ticket(t)
    assert journal.list_tickets("pending") == []

    journal.record_nav(date(2025, 1, 2), "live", 50_000.0)
    journal.record_nav(date(2025, 1, 3), "live", 51_000.0)
    assert journal.nav_series("live")[-1] == (date(2025, 1, 3), 51_000.0)
    journal.set_hedge_shares("paper", "AAA", -40)
    assert journal.hedge_shares("paper") == {"AAA": -40}
