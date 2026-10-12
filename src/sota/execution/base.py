"""Execution interfaces. Only the paper broker can place orders; live mode is advisory by construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Protocol

from sota.models import Package

Intent = Literal["open", "close"]


class LiveModeOrderBlocked(RuntimeError):
    """Raised by any attempt to place an order while running in live (advisory) mode."""


@dataclass(frozen=True, slots=True)
class AccountInfo:
    nav: float
    cash: float


@dataclass(slots=True)
class Ticket:
    """An order ticket for manual entry in an external trading app."""

    id: str
    created: datetime
    status: Literal["pending", "filled", "skipped"]
    intent: Intent
    package: Package
    units: int
    limit_price: float  # per share; > 0 debit, < 0 credit
    scenario_risk: float  # dollars for the whole ticket
    capital: float  # dollars for the whole ticket
    stop_loss_per_share: float
    force_close_date: date
    rationale: str = ""
    position_id: str | None = None  # the position this ticket opens or closes
    fill_price: float | None = None
    filled_units: int | None = None
    filled_at: datetime | None = None
    tags: dict[str, str] = field(default_factory=dict)


class Broker(Protocol):
    mode: str

    def account(self) -> AccountInfo: ...

    def submit_package(self, pkg: Package, units: int, limit_price: float, intent: Intent) -> str:
        """Place a multi-leg limit order; returns the broker order id."""
        ...

    def submit_shares(self, underlying: str, qty: int) -> str:
        """Market order for `qty` shares (signed); returns the broker order id."""
        ...
