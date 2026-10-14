"""SQLite journal of tickets, positions, NAV and decisions. Everything the tool proposes or does is recorded."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sota.execution.base import Ticket
from sota.models import Leg, OptionContract, Package, Position, Right, Side, Tenor

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY, created TEXT NOT NULL, status TEXT NOT NULL, intent TEXT NOT NULL,
    underlying TEXT, family TEXT, side TEXT, tenor TEXT, units INTEGER, limit_price REAL,
    scenario_risk REAL, capital REAL, stop_loss REAL, force_close TEXT, rationale TEXT,
    position_id TEXT, fill_price REAL, filled_units INTEGER, filled_at TEXT, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY, mode TEXT NOT NULL, status TEXT NOT NULL, underlying TEXT, family TEXT,
    opened TEXT, units INTEGER, entry_premium REAL, exit_date TEXT, exit_premium REAL,
    realized_pnl REAL, fees_paid REAL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nav (
    asof TEXT NOT NULL, mode TEXT NOT NULL, nav REAL NOT NULL, cash REAL, PRIMARY KEY (asof, mode)
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, asof TEXT NOT NULL, mode TEXT NOT NULL,
    selector TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hedges (
    underlying TEXT NOT NULL, mode TEXT NOT NULL, shares INTEGER NOT NULL, PRIMARY KEY (underlying, mode)
);
"""


# ----------------------------------------------------------------------------- serialization
def package_to_dict(pkg: Package) -> dict[str, Any]:
    return {
        "family": pkg.family,
        "underlying": pkg.underlying,
        "side": pkg.side.value,
        "tenor": pkg.tenor.value,
        "expiry": pkg.expiry.isoformat(),
        "legs": [
            {
                "expiry": leg.contract.expiry.isoformat(),
                "strike": leg.contract.strike,
                "right": leg.contract.right.value,
                "ratio": leg.ratio,
            }
            for leg in pkg.legs
        ],
        "coords": pkg.coords,
        "hedged": pkg.hedged,
        "net_premium": pkg.net_premium,
        "delta": pkg.delta,
        "gamma": pkg.gamma,
        "vega": pkg.vega,
        "theta": pkg.theta,
        "max_loss": pkg.max_loss,
        "half_spread": pkg.half_spread,
        "leg_deltas": pkg.leg_deltas,
        "leg_ivs": pkg.leg_ivs,
    }


def package_from_dict(d: dict[str, Any]) -> Package:
    return Package(
        family=d["family"],
        underlying=d["underlying"],
        side=Side(d["side"]),
        tenor=Tenor(d["tenor"]),
        expiry=date.fromisoformat(d["expiry"]),
        legs=[
            Leg(
                OptionContract(
                    d["underlying"],
                    date.fromisoformat(leg["expiry"]),
                    float(leg["strike"]),
                    Right(leg["right"]),
                ),
                int(leg["ratio"]),
            )
            for leg in d["legs"]
        ],
        coords=dict(d.get("coords", {})),
        hedged=bool(d["hedged"]),
        net_premium=float(d["net_premium"]),
        delta=float(d["delta"]),
        gamma=float(d["gamma"]),
        vega=float(d["vega"]),
        theta=float(d["theta"]),
        max_loss=None if d.get("max_loss") is None else float(d["max_loss"]),
        half_spread=float(d.get("half_spread", 0.0)),
        leg_deltas=list(d.get("leg_deltas", [])),
        leg_ivs=list(d.get("leg_ivs", [])),
    )


def position_to_dict(pos: Position) -> dict[str, Any]:
    return {
        "id": pos.id,
        "package": package_to_dict(pos.package),
        "units": pos.units,
        "entry_date": pos.entry_date.isoformat(),
        "entry_premium": pos.entry_premium,
        "stop_loss_per_share": pos.stop_loss_per_share,
        "force_close_date": pos.force_close_date.isoformat(),
        "status": pos.status,
        "fees_paid": pos.fees_paid,
        "exit_date": pos.exit_date.isoformat() if pos.exit_date else None,
        "exit_premium": pos.exit_premium,
        "realized_pnl": pos.realized_pnl,
        "tags": pos.tags,
    }


def position_from_dict(d: dict[str, Any]) -> Position:
    return Position(
        id=d["id"],
        package=package_from_dict(d["package"]),
        units=int(d["units"]),
        entry_date=date.fromisoformat(d["entry_date"]),
        entry_premium=float(d["entry_premium"]),
        stop_loss_per_share=float(d["stop_loss_per_share"]),
        force_close_date=date.fromisoformat(d["force_close_date"]),
        status=d.get("status", "open"),
        fees_paid=float(d.get("fees_paid", 0.0)),
        exit_date=date.fromisoformat(d["exit_date"]) if d.get("exit_date") else None,
        exit_premium=d.get("exit_premium"),
        realized_pnl=d.get("realized_pnl"),
        tags=dict(d.get("tags", {})),
    )


def ticket_to_dict(t: Ticket) -> dict[str, Any]:
    return {
        "id": t.id,
        "created": t.created.isoformat(),
        "status": t.status,
        "intent": t.intent,
        "package": package_to_dict(t.package),
        "units": t.units,
        "limit_price": t.limit_price,
        "scenario_risk": t.scenario_risk,
        "capital": t.capital,
        "stop_loss_per_share": t.stop_loss_per_share,
        "force_close_date": t.force_close_date.isoformat(),
        "rationale": t.rationale,
        "position_id": t.position_id,
        "fill_price": t.fill_price,
        "filled_units": t.filled_units,
        "filled_at": t.filled_at.isoformat() if t.filled_at else None,
        "tags": t.tags,
    }


def ticket_from_dict(d: dict[str, Any]) -> Ticket:
    return Ticket(
        id=d["id"],
        created=datetime.fromisoformat(d["created"]),
        status=d["status"],
        intent=d["intent"],
        package=package_from_dict(d["package"]),
        units=int(d["units"]),
        limit_price=float(d["limit_price"]),
        scenario_risk=float(d["scenario_risk"]),
        capital=float(d["capital"]),
        stop_loss_per_share=float(d["stop_loss_per_share"]),
        force_close_date=date.fromisoformat(d["force_close_date"]),
        rationale=d.get("rationale", ""),
        position_id=d.get("position_id"),
        fill_price=d.get("fill_price"),
        filled_units=d.get("filled_units"),
        filled_at=datetime.fromisoformat(d["filled_at"]) if d.get("filled_at") else None,
        tags=dict(d.get("tags", {})),
    )


# ----------------------------------------------------------------------------- journal
class Journal:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # tickets -----------------------------------------------------------------
    def save_ticket(self, t: Ticket) -> None:
        pkg = t.package
        with self._tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO tickets VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    t.id,
                    t.created.isoformat(),
                    t.status,
                    t.intent,
                    pkg.underlying,
                    pkg.family,
                    pkg.side.value,
                    pkg.tenor.value,
                    t.units,
                    t.limit_price,
                    t.scenario_risk,
                    t.capital,
                    t.stop_loss_per_share,
                    t.force_close_date.isoformat(),
                    t.rationale,
                    t.position_id,
                    t.fill_price,
                    t.filled_units,
                    t.filled_at.isoformat() if t.filled_at else None,
                    json.dumps(ticket_to_dict(t)),
                ),
            )

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        row = self._conn.execute(
            "SELECT payload FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        return ticket_from_dict(json.loads(row["payload"])) if row else None

    def list_tickets(self, status: str | None = None, limit: int = 200) -> list[Ticket]:
        sql = "SELECT payload FROM tickets"
        args: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            args = (status,)
        sql += " ORDER BY created DESC LIMIT ?"
        rows = self._conn.execute(sql, (*args, limit)).fetchall()
        return [ticket_from_dict(json.loads(r["payload"])) for r in rows]

    # positions ---------------------------------------------------------------
    def save_position(self, pos: Position, mode: str) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    pos.id,
                    mode,
                    pos.status,
                    pos.package.underlying,
                    pos.package.family,
                    pos.entry_date.isoformat(),
                    pos.units,
                    pos.entry_premium,
                    pos.exit_date.isoformat() if pos.exit_date else None,
                    pos.exit_premium,
                    pos.realized_pnl,
                    pos.fees_paid,
                    json.dumps(position_to_dict(pos)),
                ),
            )

    def get_position(self, position_id: str) -> Position | None:
        row = self._conn.execute(
            "SELECT payload FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
        return position_from_dict(json.loads(row["payload"])) if row else None

    def list_positions(self, mode: str, status: str | None = "open") -> list[Position]:
        sql = "SELECT payload FROM positions WHERE mode = ?"
        args: list[Any] = [mode]
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY opened"
        rows = self._conn.execute(sql, args).fetchall()
        return [position_from_dict(json.loads(r["payload"])) for r in rows]

    # hedges ------------------------------------------------------------------
    def hedge_shares(self, mode: str) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT underlying, shares FROM hedges WHERE mode = ?", (mode,)
        ).fetchall()
        return {r["underlying"]: int(r["shares"]) for r in rows}

    def set_hedge_shares(self, mode: str, underlying: str, shares: int) -> None:
        with self._tx() as c:
            c.execute("INSERT OR REPLACE INTO hedges VALUES (?,?,?)", (underlying, mode, shares))

    # nav / decisions ---------------------------------------------------------
    def record_nav(self, asof: date, mode: str, nav: float, cash: float | None = None) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO nav VALUES (?,?,?,?)", (asof.isoformat(), mode, nav, cash)
            )

    def nav_series(self, mode: str) -> list[tuple[date, float]]:
        rows = self._conn.execute(
            "SELECT asof, nav FROM nav WHERE mode = ? ORDER BY asof", (mode,)
        ).fetchall()
        return [(date.fromisoformat(r["asof"]), float(r["nav"])) for r in rows]

    def record_decision(
        self, asof: date, mode: str, selector: str, kind: str, payload: Any
    ) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO decisions (asof, mode, selector, kind, payload) VALUES (?,?,?,?,?)",
                (asof.isoformat(), mode, selector, kind, json.dumps(payload, default=str)),
            )

    def list_decisions(self, mode: str, kind: str = "run", limit: int = 60) -> list[dict[str, Any]]:
        """Most recent first; each row is {asof, selector, **payload}."""
        rows = self._conn.execute(
            "SELECT asof, selector, payload FROM decisions WHERE mode = ? AND kind = ? "
            "ORDER BY asof DESC, id DESC LIMIT ?",
            (mode, kind, limit),
        ).fetchall()
        return [
            {"asof": r["asof"], "selector": r["selector"], **json.loads(r["payload"])} for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
