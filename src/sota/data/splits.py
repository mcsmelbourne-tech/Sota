"""Stock splits: point-in-time closes and positions carried across an ex-date.

Theta serves raw (unadjusted) prices. On a whole-number split the OCC adjusts standard contracts by
dividing the strike and multiplying the contract count by the ratio, so a held position keeps its
value but changes identity; closes before the ex-date must be divided to keep returns continuous.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import date

from sota.models import Leg, Package, Position

# (ex-date, ratio): ratio new shares per old share.
SPLITS: dict[str, list[tuple[date, float]]] = {
    "NVDA": [(date(2021, 7, 20), 4.0), (date(2024, 6, 10), 10.0)],
    "TSLA": [(date(2020, 8, 31), 5.0), (date(2022, 8, 25), 3.0)],
    "AAPL": [(date(2020, 8, 31), 4.0)],
    "AMZN": [(date(2022, 6, 6), 20.0)],
    "GOOGL": [(date(2022, 7, 18), 20.0)],
}


def splits_between(underlying: str, after: date, through: date) -> list[tuple[date, float]]:
    """Splits with ex-date in (after, through], oldest first."""
    return [(ex, r) for ex, r in SPLITS.get(underlying, []) if after < ex <= through]


def adjust_closes(
    underlying: str, series: Iterable[tuple[date, float]], asof: date
) -> list[tuple[date, float]]:
    """Closes as seen on `asof`: splits already effective are folded into earlier prices, later ones not."""
    effective = [(ex, r) for ex, r in SPLITS.get(underlying, []) if ex <= asof]
    if not effective:
        return list(series)
    out: list[tuple[date, float]] = []
    for d, close in series:
        factor = 1.0
        for ex, r in effective:
            if d < ex:
                factor *= r
        out.append((d, close / factor))
    return out


def _snap(strike: float, strikes: Iterable[float] | None) -> float:
    """OCC rounds adjusted strikes to the cent; prefer the listed strike when one is within that."""
    strike = round(strike, 2)
    if strikes is None:
        return strike
    nearest = min(strikes, key=lambda k: abs(k - strike), default=strike)
    return nearest if abs(nearest - strike) <= 0.011 else strike


def split_package(pkg: Package, ratio: float, strikes: Iterable[float] | None = None) -> Package:
    listed = list(strikes) if strikes is not None else None
    legs = [
        Leg(replace(leg.contract, strike=_snap(leg.contract.strike / ratio, listed)), leg.ratio)
        for leg in pkg.legs
    ]
    return replace(
        pkg,
        legs=legs,
        net_premium=pkg.net_premium / ratio,
        half_spread=pkg.half_spread / ratio,
        max_loss=None if pkg.max_loss is None else pkg.max_loss / ratio,
        gamma=pkg.gamma * ratio,  # per-share delta is scale-free; gamma, vega and theta are not
        vega=pkg.vega / ratio,
        theta=pkg.theta / ratio,
    )


def apply_splits(
    pos: Position, asof: date, strikes: Iterable[float] | None = None
) -> list[tuple[date, float]]:
    """Re-express an open position in post-split terms for every ex-date since entry not yet applied.

    Idempotent: applied ex-dates are recorded in `pos.tags["splits"]`. Returns the splits applied.
    """
    done = set(pos.tags.get("splits", "").split())
    applied: list[tuple[date, float]] = []
    listed = list(strikes) if strikes is not None else None
    for ex, ratio in splits_between(pos.package.underlying, pos.entry_date, asof):
        if ex.isoformat() in done:
            continue
        pos.package = split_package(pos.package, ratio, listed)
        pos.units = int(round(pos.units * ratio))
        pos.entry_premium /= ratio
        pos.stop_loss_per_share /= ratio
        done.add(ex.isoformat())
        applied.append((ex, ratio))
    if applied:
        pos.tags["splits"] = " ".join(sorted(done))
    return applied
