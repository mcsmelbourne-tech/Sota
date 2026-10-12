"""Alpaca market data (read-only): live option-chain snapshots and daily closes.

Used by paper mode and by live/advisory mode. This module never imports the trading client.
Open interest is not part of the snapshot, so the `doi` feature is NaN on this source.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from sota.config import AlpacaSettings
from sota.models import Chain, OptionContract, Quote


def _creds(settings: AlpacaSettings) -> tuple[str, str]:
    if settings.api_key is None or settings.secret_key is None:
        raise RuntimeError(
            "Alpaca credentials missing: set SOTA_ALPACA__API_KEY / SOTA_ALPACA__SECRET_KEY"
        )
    return settings.api_key.get_secret_value(), settings.secret_key.get_secret_value()


class AlpacaChainSource:
    def __init__(self, settings: AlpacaSettings, max_dte: int = 200, rate: float = 0.0) -> None:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient

        key, secret = _creds(settings)
        self.options = OptionHistoricalDataClient(key, secret)
        self.stocks = StockHistoricalDataClient(key, secret)
        self.feed = settings.feed
        self.max_dte = max_dte
        self.rate = rate

    def spot(self, underlying: str) -> float:
        from alpaca.data.requests import StockLatestTradeRequest

        trades = self.stocks.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=underlying)
        )
        return float(trades[underlying].price)

    def get_chain(self, underlying: str, asof: date | None = None) -> Chain | None:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionChainRequest

        asof = asof or date.today()
        req = OptionChainRequest(
            underlying_symbol=underlying,
            feed=OptionsFeed(self.feed),
            expiration_date_gte=asof,
            expiration_date_lte=asof + timedelta(days=self.max_dte),
        )
        snapshots = self.options.get_option_chain(req)
        quotes: dict[OptionContract, Quote] = {}
        for symbol, snap in snapshots.items():
            q = getattr(snap, "latest_quote", None)
            if q is None:
                continue
            try:
                contract = OptionContract.from_occ(symbol)
            except (ValueError, IndexError):
                continue
            iv = getattr(snap, "implied_volatility", None)
            quotes[contract] = Quote(
                bid=float(q.bid_price or 0.0),
                ask=float(q.ask_price or 0.0),
                iv=float(iv) if iv else None,
            )
        if not quotes:
            return None
        return Chain(underlying, asof, self.spot(underlying), quotes, rate=self.rate)

    def get_closes(self, underlying: str, asof: date, n: int) -> list[float]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        start = asof - timedelta(days=int(n * 1.6) + 10)
        bars = self.stocks.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=underlying,
                timeframe=TimeFrame.Day,
                start=datetime.combine(start, time.min),
                end=datetime.combine(asof, time.max),
            )
        )
        series = bars.data.get(underlying, [])
        return [float(b.close) for b in series][-n:]
