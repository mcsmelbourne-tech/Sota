"""Live mode must never place an order: it only issues tickets that the user executes elsewhere."""

from pathlib import Path

import pytest

from sota.config import Costs, Mode
from sota.execution.base import LiveModeOrderBlocked
from sota.execution.manual import ManualTicketBroker, format_ticket
from sota.factory import build_broker
from sota.journal import Journal
from sota.models import Side, Tenor
from sota.notify.base import LogNotifier
from sota.resolvers.contract import resolve_package
from sota.runner import Runner
from sota.selectors.base import BaseSelector, PricedCandidate, Proposal


class AlwaysOpen(BaseSelector):
    name = "always"

    def propose(self, features, portfolio):
        return [Proposal(u, "ic", Side.LONG, Tenor.T31_90, rationale="test") for u in features]


def test_manual_broker_blocks_orders(chain):
    broker = ManualTicketBroker(Journal(":memory:"), LogNotifier(), 100_000.0, Costs())
    pkg = resolve_package(chain, "ls", Side.LONG, Tenor.T8_30)
    with pytest.raises(LiveModeOrderBlocked):
        broker.submit_package(pkg, 1, pkg.net_premium, "open")
    with pytest.raises(LiveModeOrderBlocked):
        broker.submit_shares("AAA", 10)


def test_factory_live_returns_manual_broker(settings):
    live = settings.model_copy(update={"mode": Mode.LIVE})
    broker = build_broker(live, Journal(":memory:"), LogNotifier())
    assert isinstance(broker, ManualTicketBroker)


def test_manual_module_has_no_broker_sdk_import():
    src = Path(__file__).resolve().parents[1] / "src" / "sota" / "execution" / "manual.py"
    assert "alpaca" not in src.read_text().lower()


def test_paper_broker_guard_reads_enum_value():
    from alpaca.common.enums import BaseURL
    from pydantic import SecretStr

    from sota.config import AlpacaSettings
    from sota.execution.alpaca_paper import AlpacaPaperBroker, assert_paper_host

    assert_paper_host(BaseURL.TRADING_PAPER)  # enum, as alpaca-py stores it
    assert_paper_host("https://paper-api.alpaca.markets")  # plain string
    with pytest.raises(RuntimeError, match="refusing to run"):
        assert_paper_host(BaseURL.TRADING_LIVE)
    with pytest.raises(RuntimeError, match="refusing to run"):
        assert_paper_host("BaseURL.TRADING_PAPER")  # the enum's repr must not pass either

    broker = AlpacaPaperBroker(AlpacaSettings(api_key=SecretStr("k"), secret_key=SecretStr("s")))
    assert broker.client._base_url is BaseURL.TRADING_PAPER  # construction makes no network call


def test_live_runner_issues_tickets_and_confirms(settings, source):
    live = settings.model_copy(update={"mode": Mode.LIVE, "cost_preset": "robinhood"})
    journal = Journal(":memory:")
    notifier = LogNotifier()
    broker = ManualTicketBroker(journal, notifier, 100_000.0, live.effective_costs)
    runner = Runner(live, source, AlwaysOpen(), broker, journal, notifier)

    report = runner.run_once(
        source.start
    )  # would raise LiveModeOrderBlocked if any order were attempted
    assert report.tickets and all(
        t.intent == "open" and t.status == "pending" for t in report.tickets
    )
    assert report.opened == [] and journal.list_positions("live", "open") == []
    assert any("ticket" in title for title, _ in notifier.sent)
    text = format_ticket(report.tickets[0])
    assert "BUY" in text and "SELL" in text and "confirm: sota live confirm" in text

    t = report.tickets[0]
    pos = broker.confirm(t.id, t.limit_price, asof=source.start)
    assert pos.is_open and journal.get_ticket(t.id).status == "filled"
    assert journal.list_positions("live", "open")[0].id == pos.id
    assert pos.fees_paid == pytest.approx(0.04 * pos.package.contracts_per_unit * pos.units)

    close_ticket = broker.issue_close(pos, pos.entry_premium + 1.0, "stop")
    closed = broker.confirm(close_ticket.id, pos.entry_premium + 1.0, asof=source.start)
    assert closed.status == "closed" and closed.realized_pnl == pytest.approx(
        100 * pos.units - closed.fees_paid
    )
    assert journal.list_positions("live", "open") == []

    with pytest.raises(ValueError):
        broker.confirm(t.id, 1.0)


def test_live_runner_does_not_duplicate_close_tickets(settings, source):
    live = settings.model_copy(update={"mode": Mode.LIVE})
    journal = Journal(":memory:")
    broker = ManualTicketBroker(journal, LogNotifier(), 100_000.0, live.effective_costs)
    runner = Runner(live, source, AlwaysOpen(), broker, journal, LogNotifier())
    first = runner.run_once(source.start)
    t = first.tickets[0]
    pos = broker.confirm(t.id, t.limit_price, asof=source.start)
    pos.force_close_date = source.start  # force an exit on the next run
    journal.save_position(pos, "live")

    second = runner.run_once(source.start)
    closes = [x for x in second.tickets if x.intent == "close"]
    assert len(closes) == 1 and closes[0].position_id == pos.id
    third = runner.run_once(source.start)
    assert not [x for x in third.tickets if x.intent == "close"]


def test_priced_candidate_dataclass(chain):
    pkg = resolve_package(chain, "ol", Side.LONG, Tenor.T8_30)
    cand = PricedCandidate(Proposal("AAA", "ol", Side.LONG, Tenor.T8_30), pkg, 3, 100.0, 500.0)
    assert cand.units == 3 and cand.package is pkg
