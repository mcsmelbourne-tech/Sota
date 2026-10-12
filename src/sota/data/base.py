"""Data-source interface shared by backtest and live modes."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from sota.models import Chain


class ChainSource(Protocol):
    def get_chain(self, underlying: str, asof: date) -> Chain | None:
        """Point-in-time chain (quotes as of the close on `asof`), or None if unavailable."""
        ...

    def get_closes(self, underlying: str, asof: date, n: int) -> list[float]:
        """`n` daily closes ending at `asof` (inclusive), oldest first; fewer if history is short."""
        ...
