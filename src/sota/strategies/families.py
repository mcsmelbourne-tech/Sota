"""The nine option-strategy families (paper Table 4) with default coordinates and admitted ranges (Table 11).

Coordinates are |delta| targets on a 0.05 grid; the contract resolver turns them into strikes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sota.models import Right, Side

MIN_DELTA = 0.05
MAX_DELTA = 0.95
DELTA_STEP = 0.05

Category = Literal["directional", "volatility"]


@dataclass(frozen=True, slots=True)
class CoordSpec:
    name: str
    default: float
    lo: float
    hi: float


@dataclass(frozen=True, slots=True)
class LegSpec:
    right: Right
    delta: float  # target |delta|
    ratio: int  # signed contract ratio


@dataclass(frozen=True)
class FamilySpec:
    code: str
    name: str
    category: Category
    summary: str
    coords: tuple[CoordSpec, ...]
    builder: Callable[[Side, dict[str, float]], list[LegSpec]]

    @property
    def hedged(self) -> bool:
        return self.category == "volatility"

    def legs(self, side: Side, coords: dict[str, float]) -> list[LegSpec]:
        return [
            LegSpec(leg.right, min(max(leg.delta, MIN_DELTA), MAX_DELTA), leg.ratio)
            for leg in self.builder(side, coords)
        ]


def _outright(side: Side, c: dict[str, float]) -> list[LegSpec]:
    right = Right.CALL if side is Side.LONG else Right.PUT
    return [LegSpec(right, c["delta"], +1)]


def _debit_vertical(side: Side, c: dict[str, float]) -> list[LegSpec]:
    right = Right.CALL if side is Side.LONG else Right.PUT
    return [LegSpec(right, c["delta"], +1), LegSpec(right, c["delta"] - c["width"], -1)]


def _credit_vertical(side: Side, c: dict[str, float]) -> list[LegSpec]:
    # bullish = bull put spread; bearish = bear call spread
    right = Right.PUT if side is Side.LONG else Right.CALL
    return [LegSpec(right, c["delta"], -1), LegSpec(right, c["delta"] - c["width"], +1)]


def _risk_reversal(side: Side, c: dict[str, float]) -> list[LegSpec]:
    long_r, short_r = (Right.CALL, Right.PUT) if side is Side.LONG else (Right.PUT, Right.CALL)
    return [
        LegSpec(long_r, c["delta"], +1),
        LegSpec(short_r, c["delta"], -1),
        LegSpec(short_r, c["delta"] - c["wing"], +1),
    ]


def _straddle(side: Side, c: dict[str, float]) -> list[LegSpec]:
    return [LegSpec(Right.CALL, c["call_delta"], +1), LegSpec(Right.PUT, c["put_delta"], +1)]


def _iron(side: Side, c: dict[str, float], wing_key: str) -> list[LegSpec]:
    d, w = c["delta"], c[wing_key]
    return [
        LegSpec(Right.CALL, d, -1),
        LegSpec(Right.PUT, d, -1),
        LegSpec(Right.CALL, d - w, +1),
        LegSpec(Right.PUT, d - w, +1),
    ]


def _butterfly(side: Side, c: dict[str, float]) -> list[LegSpec]:
    right = Right.CALL if side is Side.LONG else Right.PUT
    ctr = c["center"]
    return [
        LegSpec(right, ctr + c["width_itm"], +1),
        LegSpec(right, ctr, -2),
        LegSpec(right, ctr - c["width_otm"], +1),
    ]


FAMILIES: dict[str, FamilySpec] = {
    f.code: f
    for f in (
        FamilySpec(
            "ol",
            "Outright",
            "directional",
            "Long call (bullish) or long put (bearish): leveraged directional exposure, loss capped at premium.",
            (CoordSpec("delta", 0.55, 0.20, 0.80),),
            _outright,
        ),
        FamilySpec(
            "dv",
            "Debit vertical",
            "directional",
            "Buy a nearer-the-money option, sell a further one: defined-risk directional with lower cost.",
            (CoordSpec("delta", 0.55, 0.20, 0.80), CoordSpec("width", 0.10, 0.05, 0.30)),
            _debit_vertical,
        ),
        FamilySpec(
            "cv",
            "Credit vertical",
            "directional",
            "Sell an OTM option, buy a further OTM one: collect premium on a directional view, defined risk.",
            (CoordSpec("delta", 0.35, 0.10, 0.55), CoordSpec("width", 0.10, 0.05, 0.30)),
            _credit_vertical,
        ),
        FamilySpec(
            "dg",
            "Defined-risk reversal",
            "directional",
            "Long OTM option one side, short OTM option the other side, plus a protective wing: skew-financed direction.",
            (CoordSpec("delta", 0.35, 0.15, 0.60), CoordSpec("wing", 0.20, 0.05, 0.35)),
            _risk_reversal,
        ),
        FamilySpec(
            "ls",
            "Long straddle",
            "volatility",
            "Long ATM call and put: profits from large moves or rising IV; pays time decay.",
            (CoordSpec("call_delta", 0.50, 0.10, 0.55), CoordSpec("put_delta", 0.50, 0.10, 0.55)),
            _straddle,
        ),
        FamilySpec(
            "lg",
            "Long strangle",
            "volatility",
            "Long OTM call and put: cheaper long volatility with wider breakevens.",
            (CoordSpec("call_delta", 0.25, 0.10, 0.55), CoordSpec("put_delta", 0.25, 0.10, 0.55)),
            _straddle,
        ),
        FamilySpec(
            "ib",
            "Iron butterfly",
            "volatility",
            "Short ATM straddle with protective wings: short volatility, maximal theta, defined risk.",
            (CoordSpec("delta", 0.50, 0.10, 0.55), CoordSpec("wing", 0.25, 0.05, 0.40)),
            lambda s, c: _iron(s, c, "wing"),
        ),
        FamilySpec(
            "ic",
            "Iron condor",
            "volatility",
            "Short OTM strangle with protective wings: short volatility over a range, defined risk.",
            (CoordSpec("delta", 0.30, 0.10, 0.55), CoordSpec("width", 0.10, 0.05, 0.30)),
            lambda s, c: _iron(s, c, "width"),
        ),
        FamilySpec(
            "bf",
            "Butterfly",
            "volatility",
            "Long wings, short 2x body (calls if bullish, puts if bearish): bets the underlying pins near the body.",
            (
                CoordSpec("center", 0.45, 0.25, 0.70),
                CoordSpec("width_itm", 0.10, 0.05, 0.30),
                CoordSpec("width_otm", 0.10, 0.05, 0.30),
            ),
            _butterfly,
        ),
    )
}

FAMILY_CODES = tuple(FAMILIES)


def snap(value: float, step: float = DELTA_STEP) -> float:
    return round(round(value / step) * step, 6)


def normalize_coords(
    spec: FamilySpec, coords: dict[str, float] | None, step: float = DELTA_STEP
) -> dict[str, float]:
    """Fill defaults, clip to the admitted range, and snap to the delta grid (paper App. G.5)."""
    coords = coords or {}
    out: dict[str, float] = {}
    for cs in spec.coords:
        v = coords.get(cs.name, cs.default)
        v = min(max(float(v), cs.lo), cs.hi)
        out[cs.name] = min(max(snap(v, step), cs.lo), cs.hi)
    return out


def describe_families() -> str:
    lines = []
    for f in FAMILIES.values():
        coords = ", ".join(f"{c.name}={c.default:g} [{c.lo:g}-{c.hi:g}]" for c in f.coords)
        lines.append(f"- {f.code} ({f.name}, {f.category}): {f.summary} Coords: {coords}")
    return "\n".join(lines)
