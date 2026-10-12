"""Local Parquet cache of point-in-time chains and daily closes (data/<provider>/...)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from sota.models import Chain, OptionContract, Quote, Right

CHAIN_SCHEMA = {
    "expiry": pl.Utf8,
    "strike": pl.Float64,
    "right": pl.Utf8,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "iv": pl.Float64,
    "open_interest": pl.Int64,
    "volume": pl.Int64,
    "spot": pl.Float64,
}


class ChainStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _chain_path(self, underlying: str, asof: date) -> Path:
        return self.root / "chains" / underlying / f"{asof:%Y%m%d}.parquet"

    def _closes_path(self, underlying: str) -> Path:
        return self.root / "closes" / f"{underlying}.parquet"

    # chains -------------------------------------------------------------------
    def has_chain(self, underlying: str, asof: date) -> bool:
        return self._chain_path(underlying, asof).exists()

    def save_chain(self, chain: Chain) -> None:
        rows = [
            {
                "expiry": c.expiry.isoformat(),
                "strike": c.strike,
                "right": c.right.value,
                "bid": q.bid,
                "ask": q.ask,
                "iv": q.iv,
                "open_interest": q.open_interest,
                "volume": q.volume,
                "spot": chain.spot,
            }
            for c, q in chain.quotes.items()
        ]
        path = self._chain_path(chain.underlying, chain.asof)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows, schema=CHAIN_SCHEMA).write_parquet(path)

    def load_chain(self, underlying: str, asof: date, rate: float = 0.0) -> Chain | None:
        path = self._chain_path(underlying, asof)
        if not path.exists():
            return None
        df = pl.read_parquet(path)
        if df.is_empty():
            return None
        quotes: dict[OptionContract, Quote] = {}
        for r in df.iter_rows(named=True):
            contract = OptionContract(
                underlying, date.fromisoformat(r["expiry"]), float(r["strike"]), Right(r["right"])
            )
            quotes[contract] = Quote(
                bid=float(r["bid"]),
                ask=float(r["ask"]),
                iv=r["iv"],
                open_interest=r["open_interest"],
                volume=r["volume"],
            )
        return Chain(underlying, asof, float(df["spot"][0]), quotes, rate=rate)

    # closes -------------------------------------------------------------------
    def save_closes(self, underlying: str, series: list[tuple[date, float]]) -> None:
        path = self._closes_path(underlying)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = pl.DataFrame(
            {"date": [d.isoformat() for d, _ in series], "close": [c for _, c in series]},
            schema={"date": pl.Utf8, "close": pl.Float64},
        )
        if path.exists():
            new = pl.concat([pl.read_parquet(path), new]).unique(subset=["date"], keep="last")
        new.sort("date").write_parquet(path)

    def load_closes(self, underlying: str) -> list[tuple[date, float]]:
        path = self._closes_path(underlying)
        if not path.exists():
            return []
        df = pl.read_parquet(path).sort("date")
        return [
            (date.fromisoformat(d), float(c)) for d, c in zip(df["date"], df["close"], strict=True)
        ]
