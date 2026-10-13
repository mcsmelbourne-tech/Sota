"""Selector interface: strategy-level proposals, then a final decision over priced candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol

from sota.features import MarketFeatures
from sota.models import Package, Position, Side, Tenor

Action = Literal["open", "close"]


class SelectorUnavailable(RuntimeError):
    """The selector cannot make decisions at all (quota exhausted, bad credentials, unknown model).

    Unlike a transient failure, retrying tomorrow with the same configuration will not help, so
    callers must surface it instead of silently trading as if the selector had proposed nothing.
    """


@dataclass(slots=True)
class Proposal:
    """A strategy decision (u, f, omega, theta, kappa) before contract resolution."""

    underlying: str
    family: str
    side: Side
    tenor: Tenor
    coords: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    action: Action = "open"
    position_id: str | None = None  # for action == "close"

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.underlying, self.family, self.side.value, self.tenor.value)


@dataclass(slots=True)
class PricedCandidate:
    """A proposal after the resolvers produced exact contracts, pricing and a size."""

    proposal: Proposal
    package: Package
    units: int
    risk_per_unit: float
    capital_per_unit: float


@dataclass(slots=True)
class PortfolioView:
    asof: date
    nav: float
    cash: float
    positions: list[Position]
    marks: dict[str, float] = field(default_factory=dict)  # position id -> value per share

    def pnl_pct(self, pos: Position) -> float | None:
        mark = self.marks.get(pos.id)
        if mark is None or not pos.entry_premium:
            return None
        return pos.pnl_per_share(mark) / abs(pos.entry_premium)

    def open_positions(self, underlying: str | None = None) -> list[Position]:
        return [
            p
            for p in self.positions
            if p.is_open and (underlying is None or p.package.underlying == underlying)
        ]


class Selector(Protocol):
    name: str

    def propose(
        self, features: dict[str, MarketFeatures], portfolio: PortfolioView
    ) -> list[Proposal]: ...

    def decide(
        self, candidates: list[PricedCandidate], portfolio: PortfolioView, max_lines: int
    ) -> list[PricedCandidate]: ...


class BaseSelector:
    """Default two-stage behaviour: keep candidates in proposal order, up to `max_lines`."""

    name = "base"

    def propose(
        self, features: dict[str, MarketFeatures], portfolio: PortfolioView
    ) -> list[Proposal]:
        return []

    def decide(
        self, candidates: list[PricedCandidate], portfolio: PortfolioView, max_lines: int
    ) -> list[PricedCandidate]:
        return candidates[:max_lines]
