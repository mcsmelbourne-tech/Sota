"""ThetaDataSource parsing and caching, exercised through a fake transport (no network)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sota.config import ThetaSettings
from sota.data.theta import REST_PATHS, LibraryTransport, RestTransport, ThetaDataSource
from sota.models import OptionContract, Right

ASOF = date(2024, 11, 4)
EXP = date(2024, 11, 15)


def _rows(style: str) -> dict[str, list[dict[str, Any]]]:
    """`library` rows carry date/datetime objects and None; `rest` rows carry strings and zeros."""
    lib = style == "library"
    exp: Any = EXP if lib else EXP.isoformat()
    ts: Any = datetime(2024, 11, 4, 17, 15) if lib else "2024-11-04T17:15:00.000"
    missing: Any = None if lib else 0.0
    contracts = [
        (220.0, "call", 5.1, 5.3, 0.28),
        (220.0, "put", 3.0, 3.2, 0.31),
        (230.0, "call", 1.0, 1.1, None),
    ]
    eod = [
        {
            "symbol": "AAPL",
            "expiration": exp,
            "strike": k,
            "right": r,
            "created": ts,
            "bid": b,
            "ask": a,
            "volume": 10,
        }
        for k, r, b, a, _ in contracts
    ]
    # an expired contract that must be filtered out
    eod.append(
        {
            "symbol": "AAPL",
            "expiration": EXP - timedelta(days=30),
            "strike": 200.0,
            "right": "call",
            "bid": 1,
            "ask": 2,
        }
    )
    greeks = [
        {
            "symbol": "AAPL",
            "expiration": exp,
            "strike": k,
            "right": r,
            "implied_vol": iv if iv is not None else missing,
            "underlying_price": 222.5,
        }
        for k, r, _, _, iv in contracts
    ]
    oi = [
        {
            "symbol": "AAPL",
            "expiration": exp,
            "strike": 220.0,
            "right": "call",
            "open_interest": 1500,
        }
    ]
    closes = [
        {
            "created": datetime(2024, 11, d, 17, 15) if lib else f"2024-11-{d:02d}T17:15:00.000",
            "close": 220.0 + d,
        }
        for d in (1, 4)
    ]
    return {
        "option_history_eod": eod,
        "option_history_greeks_eod": greeks,
        "option_history_open_interest": oi,
        "stock_history_eod": closes,
    }


class FakeTransport:
    def __init__(self, style: str) -> None:
        self.rows = _rows(style)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def fetch(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((endpoint, params))
        return list(self.rows[endpoint])


@pytest.mark.parametrize("style", ["library", "rest"])
def test_fetch_chain_parses_both_row_styles(tmp_path, style):
    transport = FakeTransport(style)
    src = ThetaDataSource(ThetaSettings(), tmp_path, transport=transport)
    chain = src.get_chain("AAPL", ASOF)
    assert chain is not None and chain.spot == 222.5
    assert set(chain.quotes) == {
        OptionContract("AAPL", EXP, 220.0, Right.CALL),
        OptionContract("AAPL", EXP, 220.0, Right.PUT),
        OptionContract("AAPL", EXP, 230.0, Right.CALL),
    }
    c220 = chain.quote(OptionContract("AAPL", EXP, 220.0, Right.CALL))
    assert c220.bid == 5.1 and c220.ask == 5.3 and c220.iv == 0.28 and c220.open_interest == 1500
    c230 = chain.quote(OptionContract("AAPL", EXP, 230.0, Right.CALL))
    assert c230.iv is None and c230.open_interest is None  # missing IV/OI stay None, never 0 or NaN
    assert all(p["expiration"] == "*" and p["start_date"] == ASOF for _, p in transport.calls)


def test_chain_is_cached_to_parquet(tmp_path):
    transport = FakeTransport("library")
    src = ThetaDataSource(ThetaSettings(), tmp_path, transport=transport)
    first = src.get_chain("AAPL", ASOF)
    n_calls = len(transport.calls)
    again = ThetaDataSource(ThetaSettings(), tmp_path, transport=transport).get_chain("AAPL", ASOF)
    assert len(transport.calls) == n_calls, "second source must be served from the Parquet cache"
    assert again is not None and again.spot == first.spot and set(again.quotes) == set(first.quotes)


class TierGatedTransport(FakeTransport):
    """Free/Value tiers: greeks (and on Free, open interest) raise instead of returning rows."""

    def __init__(self, style: str, gated: tuple[str, ...]) -> None:
        super().__init__(style)
        self.gated = gated

    def fetch(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((endpoint, params))
        if endpoint in self.gated:
            raise PermissionError(f"{endpoint}: subscription tier does not include this endpoint")
        return list(self.rows[endpoint])


def test_lower_tiers_degrade_gracefully(tmp_path):
    transport = TierGatedTransport(
        "library", ("option_history_greeks_eod", "option_history_open_interest")
    )
    src = ThetaDataSource(ThetaSettings(), tmp_path, transport=transport)
    chain = src.get_chain("AAPL", ASOF)
    assert chain is not None
    assert chain.spot == 224.0, "spot falls back to the stock EOD close when greeks are unavailable"
    assert all(q.iv is None and q.open_interest is None for q in chain.quotes.values())
    chain.ensure_greeks()
    assert all(q.iv is not None for q in chain.quotes.values()), "IV is solved locally from mid"
    src.get_chain("AAPL", ASOF + timedelta(days=1))
    gated_calls = [e for e, _ in transport.calls if e in transport.gated]
    assert len(gated_calls) == 2, "a gated endpoint is tried once per session, not once per day"
    assert all(p.get("max_dte") == 200 for e, p in transport.calls if e.startswith("option_"))


def test_closes_and_trading_days(tmp_path):
    src = ThetaDataSource(ThetaSettings(), tmp_path, transport=FakeTransport("rest"))
    assert src.get_closes("AAPL", ASOF, 5) == [221.0, 224.0]
    assert src.trading_days("AAPL", date(2024, 11, 1), ASOF) == [date(2024, 11, 1), ASOF]


def test_fetch_closes_chunks_long_ranges(tmp_path):
    """stock_history_eod rejects spans over 365 days; long ranges must be split."""
    transport = FakeTransport("library")
    src = ThetaDataSource(ThetaSettings(), tmp_path, transport=transport)
    src.fetch_closes("AAPL", date(2024, 5, 1), date(2025, 10, 10))
    calls = [p for e, p in transport.calls if e == "stock_history_eod"]
    assert len(calls) == 2
    assert all((p["end_date"] - p["start_date"]).days < 365 for p in calls)
    assert calls[0]["start_date"] == date(2024, 5, 1) and calls[-1]["end_date"] == date(
        2025, 10, 10
    )
    assert calls[1]["start_date"] == calls[0]["end_date"] + timedelta(
        days=1
    )  # contiguous, no overlap


def test_cache_only_source_needs_no_credentials(tmp_path):
    src = ThetaDataSource(ThetaSettings(transport="library"), tmp_path)
    assert src._transport is None  # created lazily on first network call


def test_rest_transport_formats_dates_and_tolerates_errors():
    import httpx

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(472, text="no data")  # Theta's no-data style status

    client = httpx.Client(transport=httpx.MockTransport(handler))
    t = RestTransport("http://127.0.0.1:25503/v3", 5.0, client=client)
    assert (
        t.fetch("option_history_eod", symbol="AAPL", expiration="*", start_date=ASOF, end_date=ASOF)
        == []
    )
    assert REST_PATHS["option_history_eod"] in seen["url"]
    assert "start_date=20241104" in seen["url"] and "format=json" in seen["url"]


def test_library_transport_is_importable():
    assert (
        LibraryTransport.__init__ is not None
    )  # construction authenticates, so only check the import path


class _GrpcError(Exception):
    def __init__(self, name: str, text: str = "slow down") -> None:
        super().__init__(text)
        self._name, self._text = name, text

    def code(self):
        return type("Code", (), {"name": self._name})()

    def details(self):
        return self._text


class _FlakyClient:
    def __init__(self, failures: list[Exception]) -> None:
        self.failures, self.calls = failures, 0

    def option_history_eod(self, **params):
        import polars as pl

        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return pl.DataFrame({"symbol": ["AAPL"]})


def test_library_transport_retries_transient_grpc_errors():
    slept: list[float] = []
    client = _FlakyClient([_GrpcError("RESOURCE_EXHAUSTED"), _GrpcError("UNAVAILABLE")])
    t = LibraryTransport(None, client=client, backoff_s=1.0, sleep=slept.append)
    assert t.fetch("option_history_eod", symbol="AAPL") == [{"symbol": "AAPL"}]
    assert client.calls == 3 and slept == [1.0, 2.0]


def test_library_transport_does_not_retry_permission_errors():
    client = _FlakyClient([_GrpcError("PERMISSION_DENIED", "needs a value subscription")])
    t = LibraryTransport(None, client=client, sleep=lambda _: None)
    with pytest.raises(_GrpcError):
        t.fetch("option_history_eod", symbol="AAPL")
    assert client.calls == 1


def test_library_transport_gives_up_after_max_attempts():
    client = _FlakyClient([_GrpcError("UNAVAILABLE")] * 3)
    t = LibraryTransport(None, client=client, max_attempts=3, sleep=lambda _: None)
    with pytest.raises(_GrpcError):
        t.fetch("option_history_eod", symbol="AAPL")
    assert client.calls == 3


def test_throttle_spaces_requests():
    from sota.data.theta import _Throttle

    now = [100.0]
    slept: list[float] = []
    th = _Throttle(per_minute=20, clock=lambda: now[0], sleep=slept.append)
    th.wait()  # first call never waits
    th.wait()  # same instant -> must wait the full 3s interval
    now[0] += 10.0
    th.wait()  # well past the interval -> no wait
    assert slept == [pytest.approx(3.0)]
    assert _Throttle(per_minute=0, sleep=slept.append).interval == 0.0


def test_source_throttles_every_request(tmp_path):
    transport = FakeTransport("library")
    src = ThetaDataSource(ThetaSettings(requests_per_minute=20), tmp_path, transport=transport)
    slept: list[float] = []
    src._throttle._sleep = slept.append
    src.get_chain("AAPL", ASOF)
    assert len(slept) == len(transport.calls) - 1  # every request after the first is spaced
