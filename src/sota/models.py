"""Core domain types: contracts, quotes, chains, packages, positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from sota.pricing.bsm import greeks as bsm_greeks
from sota.pricing.bsm import implied_vol

MULTIPLIER = 100  # shares per option contract


class Right(StrEnum):
    CALL = "C"
    PUT = "P"

    @property
    def is_call(self) -> bool:
        return self is Right.CALL


class Side(StrEnum):
    """Orientation: bullish/bearish for directional families, long/short structure otherwise."""

    LONG = "long"
    SHORT = "short"


class Tenor(StrEnum):
    """Tenor buckets in calendar days to expiry (paper Table 10) with resolver anchors (App. E.1)."""

    T0_7 = "0-7"
    T8_30 = "8-30"
    T31_90 = "31-90"
    T91_180 = "91-180"

    @property
    def anchor(self) -> int:
        return {Tenor.T0_7: 5, Tenor.T8_30: 14, Tenor.T31_90: 45, Tenor.T91_180: 120}[self]

    @property
    def bounds(self) -> tuple[int, int]:
        lo, hi = self.value.split("-")
        return int(lo), int(hi)

    @classmethod
    def for_dte(cls, dte: int) -> Tenor:
        for t in cls:
            lo, hi = t.bounds
            if lo <= dte <= hi:
                return t
        return cls.T91_180 if dte > 180 else cls.T0_7


@dataclass(frozen=True, slots=True)
class OptionContract:
    underlying: str
    expiry: date
    strike: float
    right: Right

    @property
    def occ_symbol(self) -> str:
        return (
            f"{self.underlying}{self.expiry:%y%m%d}{self.right.value}"
            f"{int(round(self.strike * 1000)):08d}"
        )

    @classmethod
    def from_occ(cls, symbol: str) -> OptionContract:
        root, tail = symbol[:-15], symbol[-15:]
        expiry = date(2000 + int(tail[0:2]), int(tail[2:4]), int(tail[4:6]))
        return cls(root, expiry, int(tail[7:]) / 1000.0, Right(tail[6]))

    def dte(self, asof: date) -> int:
        return (self.expiry - asof).days

    def intrinsic(self, spot: float) -> float:
        return max(spot - self.strike, 0.0) if self.right.is_call else max(self.strike - spot, 0.0)

    def describe(self) -> str:
        return f"{self.underlying} {self.expiry:%Y-%m-%d} {self.strike:g}{self.right.value}"


@dataclass(slots=True)
class Quote:
    bid: float
    ask: float
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None  # per calendar day, per share
    open_interest: int | None = None
    volume: int | None = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def half_spread(self) -> float:
        return 0.5 * (self.ask - self.bid)

    @property
    def is_valid(self) -> bool:
        return self.ask > 0.0 and self.ask >= self.bid >= 0.0

    @property
    def has_greeks(self) -> bool:
        return None not in (self.iv, self.delta, self.gamma, self.vega, self.theta)


@dataclass
class Chain:
    """Point-in-time option chain for one underlying."""

    underlying: str
    asof: date
    spot: float
    quotes: dict[OptionContract, Quote]
    rate: float = 0.0
    div_yield: float = 0.0
    _index: dict[tuple[date, Right], list[OptionContract]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def expiries(self) -> list[date]:
        return sorted({c.expiry for c in self.quotes})

    def contracts(
        self, expiry: date | None = None, right: Right | None = None
    ) -> list[OptionContract]:
        if expiry is not None and right is not None:
            if self._index is None:
                idx: dict[tuple[date, Right], list[OptionContract]] = {}
                for c in sorted(self.quotes, key=lambda c: (c.expiry, c.strike, c.right.value)):
                    idx.setdefault((c.expiry, c.right), []).append(c)
                self._index = idx
            return self._index.get((expiry, right), [])
        return sorted(
            (
                c
                for c in self.quotes
                if (expiry is None or c.expiry == expiry) and (right is None or c.right == right)
            ),
            key=lambda c: (c.expiry, c.strike, c.right.value),
        )

    def quote(self, contract: OptionContract) -> Quote | None:
        return self.quotes.get(contract)

    def total_open_interest(self) -> int | None:
        ois = [q.open_interest for q in self.quotes.values() if q.open_interest is not None]
        return sum(ois) if ois else None

    def ensure_greeks(self) -> None:
        """Fill missing IV/greeks from the mid price with BSM; quotes that cannot be solved keep None."""
        for c, q in self.quotes.items():
            if q.has_greeks or not q.is_valid:
                continue
            T = c.dte(self.asof) / 365.0
            if q.iv is None:
                q.iv = implied_vol(
                    q.mid, self.spot, c.strike, T, self.rate, self.div_yield, c.right.is_call
                )
            if q.iv is None:
                continue
            g = bsm_greeks(self.spot, c.strike, T, self.rate, self.div_yield, q.iv, c.right.is_call)
            q.delta = g.delta if q.delta is None else q.delta
            q.gamma = g.gamma if q.gamma is None else q.gamma
            q.vega = g.vega if q.vega is None else q.vega
            q.theta = g.theta if q.theta is None else q.theta


@dataclass(frozen=True, slots=True)
class Leg:
    contract: OptionContract
    ratio: int  # signed: +1 long, -1 short, -2 short body of a butterfly

    @property
    def is_long(self) -> bool:
        return self.ratio > 0


@dataclass
class Package:
    """A resolved, priced strategy: exact contracts plus per-share aggregates."""

    family: str
    underlying: str
    side: Side
    tenor: Tenor
    expiry: date
    legs: list[Leg]
    coords: dict[str, float]
    hedged: bool  # volatility families are delta-hedged
    net_premium: float  # per share; > 0 debit paid, < 0 credit received
    delta: float
    gamma: float
    vega: float
    theta: float  # per calendar day
    max_loss: float | None  # per share; None when unbounded
    half_spread: float  # per share, summed over legs
    leg_deltas: list[float] = field(default_factory=list)  # realized |delta| per leg at resolution

    @property
    def contracts_per_unit(self) -> int:
        return sum(abs(leg.ratio) for leg in self.legs)

    @property
    def is_credit(self) -> bool:
        return self.net_premium < 0.0

    def value(self, chain: Chain) -> float | None:
        """Mark-to-mid liquidation value per share (negative means it costs to close)."""
        total = 0.0
        for leg in self.legs:
            q = chain.quote(leg.contract)
            if q is None or not q.is_valid:
                return None
            total += leg.ratio * q.mid
        return total

    def intrinsic_value(self, spot: float) -> float:
        return sum(leg.ratio * leg.contract.intrinsic(spot) for leg in self.legs)

    def payoff_at_expiry(self, spot: float) -> float:
        return self.intrinsic_value(spot) - self.net_premium

    def describe(self) -> str:
        legs = ", ".join(f"{leg.ratio:+d} {leg.contract.describe()}" for leg in self.legs)
        kind = "credit" if self.is_credit else "debit"
        return f"[{self.family}/{self.side.value}/{self.tenor.value}] {legs} | {kind} {abs(self.net_premium):.2f}"


@dataclass
class Position:
    id: str
    package: Package
    units: int
    entry_date: date
    entry_premium: float  # per share, same sign convention as Package.net_premium
    stop_loss_per_share: float  # close when pnl_per_share <= -stop_loss_per_share
    force_close_date: date
    status: str = "open"
    fees_paid: float = 0.0
    exit_date: date | None = None
    exit_premium: float | None = None
    realized_pnl: float | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def pnl_per_share(self, value_per_share: float) -> float:
        return value_per_share - self.entry_premium

    def market_value(self, value_per_share: float) -> float:
        return value_per_share * MULTIPLIER * self.units
