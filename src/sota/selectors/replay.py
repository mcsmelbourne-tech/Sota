"""Re-open the trades a previous backtest recorded, so exit rules and family vetoes can be tested
without re-running the selector that chose them.

Entries come from `<name>_proposals.json` written by `sota backtest --out`. Each is re-resolved from
its coordinates on the recorded entry date, so contracts match the original run while sizing, exits
and the family allow-list follow the current settings. The counterfactual is approximate: the
original selector would have seen a different portfolio and might have chosen differently.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from sota.features import MarketFeatures
from sota.models import Position, Side, Tenor
from sota.selectors.base import BaseSelector, PortfolioView, Proposal


def proposals_from_positions(positions: list[Position]) -> list[dict]:
    return [
        {
            "entry_date": p.entry_date.isoformat(),
            "underlying": p.package.underlying,
            "family": p.package.family,
            "side": p.package.side.value,
            "tenor": p.package.tenor.value,
            "coords": p.package.coords,
            "units": p.units,
            "rationale": p.tags.get("rationale", ""),
        }
        for p in positions
    ]


class ReplaySelector(BaseSelector):
    name = "replay"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.by_date: dict[date, list[Proposal]] = defaultdict(list)
        for row in json.loads(self.path.read_text()):
            self.by_date[date.fromisoformat(row["entry_date"])].append(
                Proposal(
                    row["underlying"],
                    row["family"],
                    Side(row["side"]),
                    Tenor(row["tenor"]),
                    dict(row["coords"]),
                    row.get("rationale", ""),
                )
            )

    def propose(
        self, features: dict[str, MarketFeatures], portfolio: PortfolioView
    ) -> list[Proposal]:
        return [p for p in self.by_date.get(portfolio.asof, []) if p.underlying in features]
