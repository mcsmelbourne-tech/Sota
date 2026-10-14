"""Cockpit web app: JSON API over the journal plus a single-page UI (static/index.html).

Everything shown comes from the journal, so the page never calls a broker or data vendor; marks are
the EOD values the last run recorded with each position.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sota.config import Settings
from sota.execution.manual import ManualTicketBroker, format_ticket
from sota.journal import Journal, position_to_dict, ticket_to_dict
from sota.models import MULTIPLIER, Position
from sota.notify.base import LogNotifier
from sota.scorecard import build_scorecard, max_drawdown

app = FastAPI(title="SOTA cockpit")
STATIC = Path(__file__).parent / "static"
ET = ZoneInfo("America/New_York")


def _journal() -> Journal:
    return Journal(Settings().journal_path)


def _mode(mode: str | None) -> str:
    return mode or Settings().mode.value


class Confirm(BaseModel):
    price: float
    units: int | None = None


# ----------------------------------------------------------------------------- read models
def position_view(p: Position) -> dict[str, Any]:
    """Journal position plus the derived numbers the table shows (mark, P&L, DTE, stop level)."""
    d = position_to_dict(p)
    pkg = p.package
    mark = float(p.tags["mark"]) if "mark" in p.tags else None
    value = p.exit_premium if p.exit_premium is not None else mark
    pnl = p.realized_pnl
    if pnl is None and value is not None:
        pnl = (value - p.entry_premium) * MULTIPLIER * p.units - p.fees_paid
    pnl_pct = (
        (value - p.entry_premium) / abs(p.entry_premium)
        if value is not None and p.entry_premium
        else None
    )
    d.update(
        label=f"{pkg.family} {pkg.side.value} {pkg.tenor.value}",
        legs=[
            f"{leg.ratio:+d} {leg.contract.expiry:%b %d} {leg.contract.strike:g}{leg.contract.right.value}"
            for leg in pkg.legs
        ],
        is_credit=pkg.is_credit,
        mark=mark,
        mark_date=p.tags.get("mark_date"),
        value=value,
        pnl=pnl,
        pnl_pct=pnl_pct,
        stop_level=p.entry_premium - p.stop_loss_per_share,
        dte=(pkg.expiry - date.today()).days,
        held_days=((p.exit_date or date.today()) - p.entry_date).days,
        max_loss_total=None if pkg.max_loss is None else pkg.max_loss * MULTIPLIER * p.units,
        rationale=p.tags.get("rationale", ""),
    )
    return d


def market_status(now: datetime | None = None) -> dict[str, Any]:
    """Regular-session clock in ET; holidays are not known here (the runner checks the broker)."""
    now = now or datetime.now(ET)
    minutes = now.hour * 60 + now.minute
    weekday = now.weekday() < 5
    decision = now.replace(hour=15, minute=50, second=0, microsecond=0)
    if not weekday or now >= decision:
        decision += timedelta(days=1)
        while decision.weekday() >= 5:
            decision += timedelta(days=1)
    return {
        "et_time": now.strftime("%H:%M"),
        "is_open": weekday and 9 * 60 + 30 <= minutes < 16 * 60,
        "next_decision": decision.isoformat(),
        "seconds_to_decision": int((decision - now).total_seconds()),
    }


# ----------------------------------------------------------------------------- api
@app.get("/api/summary")
def summary(mode: str | None = None) -> dict[str, Any]:
    s = Settings()
    m = _mode(mode)
    j = _journal()
    series = j.nav_series(m)
    all_positions = j.list_positions(m, None)
    open_pos = [p for p in all_positions if p.is_open]
    closed = [p for p in all_positions if not p.is_open and p.status != "unfilled"]
    values = [v for _, v in series]
    nav = values[-1] if values else None
    prev = values[-2] if len(values) > 1 else (values[0] if values else None)
    start_nav = values[0] if values else s.initial_nav
    open_views = [position_view(p) for p in open_pos]
    wins = [p for p in closed if (p.realized_pnl or 0) > 0]
    last = j.list_decisions(m, limit=1)
    return {
        "mode": m,
        "selector": s.selector,
        "universe": s.universe,
        "families": s.limits.families,
        "cost_preset": s.cost_preset,
        "market": market_status(),
        "last_run": last[0]["asof"] if last else None,
        "nav": nav,
        "start_nav": start_nav,
        "day_change": (nav - prev) if nav is not None and prev is not None else None,
        "total_return": (nav / start_nav - 1.0) if nav and start_nav else None,
        "max_drawdown": max_drawdown(values) if values else None,
        "unrealized": sum(v["pnl"] for v in open_views if v["pnl"] is not None),
        "realized": sum(p.realized_pnl or 0.0 for p in closed),
        "fees": sum(p.fees_paid for p in all_positions),
        "open_count": len(open_pos),
        "open_max_loss": sum(v["max_loss_total"] or 0.0 for v in open_views),
        "closed_count": len(closed),
        "win_rate": len(wins) / len(closed) if closed else None,
        "scorecard": build_scorecard(series, all_positions).as_dict(),
        "nav_series": [{"date": d.isoformat(), "nav": v} for d, v in series],
    }


@app.get("/api/positions")
def positions(mode: str | None = None, status: str = "open") -> list[dict]:
    rows = _journal().list_positions(_mode(mode), None)
    if status == "open":
        rows = [p for p in rows if p.is_open]
    elif status == "closed":
        rows = [p for p in rows if not p.is_open and p.status != "unfilled"]
        rows.sort(key=lambda p: p.exit_date or p.entry_date, reverse=True)
    return [position_view(p) for p in rows]


@app.get("/api/runs")
def runs(mode: str | None = None, limit: int = 30) -> list[dict]:
    return _journal().list_decisions(_mode(mode), limit=limit)


@app.get("/api/market")
def market(mode: str | None = None) -> dict[str, Any]:
    """The 10 features and spots recorded by the most recent run."""
    rows = _journal().list_decisions(_mode(mode), limit=1)
    if not rows:
        return {"asof": None, "features": {}, "spots": {}}
    return {
        "asof": rows[0]["asof"],
        "features": rows[0].get("features", {}),
        "spots": rows[0].get("spots", {}),
    }


@app.get("/api/nav")
def nav(mode: str | None = None) -> list[dict]:
    return [{"date": d.isoformat(), "nav": v} for d, v in _journal().nav_series(_mode(mode))]


@app.get("/api/tickets")
def tickets(status: str | None = "pending") -> list[dict]:
    return [
        {**ticket_to_dict(t), "text": format_ticket(t)} for t in _journal().list_tickets(status)
    ]


@app.post("/api/tickets/{ticket_id}/confirm")
def confirm(ticket_id: str, body: Confirm) -> dict:
    s = Settings()
    broker = ManualTicketBroker(_journal(), LogNotifier(), s.initial_nav, s.effective_costs)
    try:
        pos = broker.confirm(ticket_id, body.price, body.units, date.today())
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return position_to_dict(pos)


@app.post("/api/tickets/{ticket_id}/skip")
def skip(ticket_id: str) -> dict:
    s = Settings()
    broker = ManualTicketBroker(_journal(), LogNotifier(), s.initial_nav, s.effective_costs)
    try:
        return ticket_to_dict(broker.skip(ticket_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
