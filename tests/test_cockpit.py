"""Pre-registered forward-test scorecard and the cockpit API that serves it."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from sota.scorecard import build_scorecard, max_drawdown, monthly_returns

D0 = date(2026, 10, 14)


def _series(returns: list[float], start: float = 50_000.0, step_days: int = 1):
    out, nav, d = [], start, D0
    for r in [0.0, *returns]:
        nav *= 1 + r
        out.append((d, nav))
        d += timedelta(days=step_days)
    return out


def test_max_drawdown_and_monthly_returns():
    assert max_drawdown([100, 120, 90, 130, 65]) == pytest.approx(0.5)
    series = [
        (date(2026, 10, 14), 100.0),
        (date(2026, 10, 30), 110.0),
        (date(2026, 11, 2), 99.0),
        (date(2026, 11, 30), 121.0),
        (date(2026, 12, 1), 121.0),
    ]
    assert monthly_returns(series) == [
        ("2026-10", pytest.approx(0.10)),
        ("2026-11", pytest.approx(0.10)),
        ("2026-12", pytest.approx(0.0)),
    ]


def test_scorecard_is_pending_then_provisional():
    empty = build_scorecard([], [])
    assert empty.start is None and "waiting" in empty.verdict
    card = build_scorecard(_series([0.01, -0.005]), [])
    assert card.provisional and card.days_elapsed == 2
    by = {c.key: c for c in card.criteria}
    assert by["return"].status == "pass" and by["drawdown"].status == "pass"
    assert by["trade_rate"].status == "pending" and by["worst_month"].status == "pending"
    assert "indicative" in card.verdict


def test_scorecard_six_month_verdict_stops_on_two_failures(chain):
    from sota.models import Position, Side, Tenor
    from sota.resolvers.contract import resolve_package

    # 200 days: a slow bleed to -12% with a 20% drawdown in the middle
    rets = [-0.0006] * 100 + [-0.02] * 10 + [0.005] * 20 + [-0.0006] * 70
    series = _series(rets)
    pkg = resolve_package(chain, "ol", Side.LONG, Tenor.T8_30)
    positions = [
        Position(f"p{i}", pkg, 1, D0 + timedelta(days=3 * i), 1.0, 0.6, pkg.expiry, status="stop")
        for i in range(20)
    ]
    card = build_scorecard(series, positions)
    by = {c.key: c for c in card.criteria}
    assert not card.provisional
    assert by["return"].status == "fail" and by["drawdown"].status == "fail"
    assert by["trade_rate"].status == "pass", f"20 trades in 200 days ≈ 36/yr: {by['trade_rate']}"
    assert by["worst_month"].status == "fail"
    assert card.failures >= 2 and card.verdict.startswith("STOP")


def test_cockpit_api_serves_summary_positions_and_page(settings, source, monkeypatch, tmp_path):
    """The web app reads only the journal; a paper run's records must round-trip into the API."""
    from sota.config import Mode
    from sota.execution.manual import ManualTicketBroker
    from sota.journal import Journal
    from sota.models import Side, Tenor
    from sota.notify.base import LogNotifier
    from sota.runner import Runner
    from sota.selectors.base import BaseSelector, Proposal
    from sota.web import app as web

    class Once(BaseSelector):
        name = "once"

        def propose(self, features, portfolio):
            if portfolio.positions:
                return []
            return [Proposal(u, "ol", Side.LONG, Tenor.T8_30, rationale="test") for u in features]

    live = settings.model_copy(update={"mode": Mode.LIVE})
    journal = Journal(live.journal_path)
    broker = ManualTicketBroker(journal, LogNotifier(), 50_000.0, live.effective_costs)
    runner = Runner(live, source, Once(), broker, journal, LogNotifier())
    first = runner.run_once(source.start)
    t = first.tickets[0]
    broker.confirm(t.id, t.limit_price, asof=source.start)
    runner.run_once(source.start + timedelta(days=1))  # records a mark on the open position
    journal.close()

    monkeypatch.setattr(web, "Settings", lambda: live)
    client = TestClient(web.app)

    s = client.get("/api/summary?mode=live").json()
    assert s["mode"] == "live" and s["open_count"] == 1 and len(s["nav_series"]) == 2
    assert s["scorecard"]["start"] == source.start.isoformat()
    assert {c["key"] for c in s["scorecard"]["criteria"]} == {
        "return",
        "drawdown",
        "trade_rate",
        "worst_month",
    }
    assert s["market"]["next_decision"].endswith("15:50:00-04:00") or s["market"][
        "next_decision"
    ].endswith("15:50:00-05:00")

    open_pos = client.get("/api/positions?mode=live&status=open").json()
    assert len(open_pos) == 1 and open_pos[0]["mark"] is not None and open_pos[0]["pnl"] is not None
    assert open_pos[0]["legs"] and open_pos[0]["label"].startswith("ol long")
    assert client.get("/api/positions?mode=live&status=closed").json() == []

    runs = client.get("/api/runs?mode=live").json()
    assert len(runs) == 2 and runs[0]["asof"] > runs[1]["asof"] and "features" in runs[0]
    assert client.get("/api/market?mode=live").json()["asof"] == runs[0]["asof"]

    page = client.get("/")
    assert page.status_code == 200 and "Equity curve" in page.text and "Forward test" in page.text


def test_market_status_clock():
    from datetime import datetime

    from sota.web.app import ET, market_status

    friday_noon = datetime(2026, 10, 16, 12, 0, tzinfo=ET)
    st = market_status(friday_noon)
    assert st["is_open"] and st["next_decision"].startswith("2026-10-16T15:50")
    friday_late = datetime(2026, 10, 16, 16, 30, tzinfo=ET)
    st = market_status(friday_late)
    assert not st["is_open"] and st["next_decision"].startswith("2026-10-19T15:50"), "skips weekend"


def test_telegram_html_and_chat_ids():
    import json

    import httpx

    from sota.notify.channels import TelegramNotifier, telegram_chat_ids

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(
                200,
                json={
                    "result": [
                        {"message": {"chat": {"id": 42, "first_name": "Wes", "last_name": "L"}}},
                        {"message": {"chat": {"id": 42, "first_name": "Wes"}}},
                        {"channel_post": {"chat": {"id": -100, "title": "alerts"}}},
                    ]
                },
            )
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert telegram_chat_ids("tok", client) == [("-100", "alerts"), ("42", "Wes L")]
    TelegramNotifier("tok", "42", client).send("SOTA paper run <1>", "opened a1_b & c")
    assert seen["body"]["parse_mode"] == "HTML"
    assert "&lt;1&gt;" in seen["body"]["text"] and "&amp;" in seen["body"]["text"]
    assert seen["body"]["text"].startswith("<b>") and "<pre>" in seen["body"]["text"]


def test_telegram_failures_never_leak_the_token(caplog):
    import logging

    import httpx

    from sota.notify.channels import TelegramError, TelegramNotifier

    token = "8977:SECRET-TOKEN"

    def forbidden(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"ok": False, "description": "Forbidden: bot was blocked by the user"}
        )

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    for transport in (forbidden, broken):
        notifier = TelegramNotifier(
            token, "42", httpx.Client(transport=httpx.MockTransport(transport))
        )
        with pytest.raises(TelegramError) as info:
            notifier.send("t", "b")
        assert "SECRET" not in str(info.value) and "api.telegram.org" not in str(info.value)

    # the httpx request log line, which embeds the URL, is suppressed for Telegram only
    with caplog.at_level(logging.INFO, logger="httpx"):
        logging.getLogger("httpx").info(
            'HTTP Request: POST https://api.telegram.org/bot%s/sendMessage "HTTP/1.1 200 OK"', token
        )
        logging.getLogger("httpx").info("HTTP Request: POST https://api.x.ai/v1/chat/completions")
    assert "SECRET" not in caplog.text and "api.x.ai" in caplog.text
