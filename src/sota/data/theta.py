"""Theta Data point-in-time chains and daily closes, cached to Parquet.

Two interchangeable transports (``SOTA_THETA__TRANSPORT``):
  library  the ``thetadata`` package: direct gRPC with an API key, no Theta Terminal (default)
  rest     the Theta Terminal v3 local REST server (http://127.0.0.1:25503/v3)
Both serve the same endpoints with the same parameters and columns:
  option_history_eod            bid/ask/volume per contract (expiration="*" => whole chain, one day)
  option_history_greeks_eod     implied_vol + underlying_price (Standard tier; optional)
  option_history_open_interest
  stock_history_eod             daily closes
Greeks are recomputed locally from IV with BSM so conventions match the rest of the tool.
The transport is created on first use, so cache-only runs never need credentials.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx

from sota.config import ThetaSettings
from sota.data.store import ChainStore
from sota.models import Chain, OptionContract, Quote, Right

log = logging.getLogger(__name__)

_RIGHT = {"call": Right.CALL, "put": Right.PUT, "c": Right.CALL, "p": Right.PUT}

REST_PATHS = {
    "option_history_eod": "/option/history/eod",
    "option_history_greeks_eod": "/option/history/greeks/eod",
    "option_history_open_interest": "/option/history/open_interest",
    "stock_history_eod": "/stock/history/eod",
}


def _row_date(row: dict[str, Any]) -> date | None:
    for key in ("date", "created", "timestamp", "last_trade"):
        v = row.get(key)
        if v:
            try:
                return date.fromisoformat(str(v)[:10])
            except ValueError:
                continue
    return None


def _num(v: Any) -> float | None:
    """Finite float, or None for missing/NaN/non-numeric (Polars nulls arrive as None, REST as 0)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _grpc_code(exc: BaseException) -> str | None:
    code = getattr(exc, "code", None)
    status = code() if callable(code) else None
    return getattr(status, "name", None)


def error_detail(exc: BaseException) -> str:
    """One-line error text; gRPC errors carry the useful message in .details()."""
    details = getattr(exc, "details", None)
    text = details() if callable(details) else None
    return str(text or str(exc).strip().splitlines()[0])[:200]


class _Throttle:
    """Spaces requests at least 60/per_minute seconds apart; per_minute <= 0 disables."""

    def __init__(
        self,
        per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self.interval <= 0:
            return
        if self._last is not None:
            delay = self._last + self.interval - self._clock()
            if delay > 0:
                self._sleep(delay)
        self._last = self._clock()


class Transport(Protocol):
    def fetch(self, endpoint: str, **params: Any) -> list[dict[str, Any]]: ...


class RestTransport:
    """Theta Terminal v3 REST. Dates go out as YYYYMMDD; non-200 (incl. Theta's no-data codes) -> []."""

    def __init__(self, base_url: str, timeout_s: float, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout_s)

    def fetch(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        query = {k: (f"{v:%Y%m%d}" if isinstance(v, date) else v) for k, v in params.items()}
        query["format"] = "json"
        r = self.client.get(f"{self.base_url}{REST_PATHS[endpoint]}", params=query)
        if r.status_code != 200:
            log.debug("theta %s -> %s %s", endpoint, r.status_code, r.text[:200])
            return []
        data = r.json()
        if isinstance(data, dict):
            data = data.get("response") or data.get("data") or []
        return [row for row in data if isinstance(row, dict)]


class LibraryTransport:
    """`thetadata` package. Auth: explicit key, else THETADATA_API_KEY / .env / creds.txt discovery."""

    TRANSIENT = frozenset({"RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"})

    def __init__(
        self,
        api_key: str | None,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        max_attempts: int = 5,
        backoff_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_attempts, self.backoff_s, self._sleep = max_attempts, backoff_s, sleep
        self._factory = client_factory or (lambda: self._build_client(api_key))
        self.client = client if client is not None else self._factory()

    @staticmethod
    def _build_client(api_key: str | None) -> Any:
        from thetadata import ThetaClient

        # The library logs the full account record (ids, billing fields) at INFO on login.
        logging.getLogger("thetadata").setLevel(logging.WARNING)
        kwargs: dict[str, Any] = {"dataframe_type": "polars"}
        if api_key:
            kwargs["api_key"] = api_key
        return ThetaClient(**kwargs)

    def fetch(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        from thetadata.errors import NoDataFoundError

        delay = self.backoff_s
        reauthenticated = False
        for attempt in range(1, self.max_attempts + 1):
            try:
                return getattr(self.client, endpoint)(**params).to_dicts()
            except NoDataFoundError:
                return []
            except Exception as exc:
                code = _grpc_code(exc)
                # Theta allows one session per account; another client (or a restart) invalidates ours.
                if code == "UNAUTHENTICATED" and not reauthenticated:
                    log.warning("theta session invalid (%s); re-authenticating", error_detail(exc))
                    self.client = self._factory()
                    reauthenticated = True
                    continue
                if code not in self.TRANSIENT or attempt == self.max_attempts:
                    raise
                log.warning(
                    "theta %s: %s; retry %d/%d in %.0fs",
                    endpoint,
                    error_detail(exc),
                    attempt,
                    self.max_attempts - 1,
                    delay,
                )
                self._sleep(delay)
                delay *= 2
        return []


class ThetaDataSource:
    def __init__(
        self,
        settings: ThetaSettings,
        data_dir: Path,
        rate: float = 0.0,
        max_dte: int = 200,
        transport: Transport | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self._owns_transport = transport is None
        self.store = ChainStore(Path(data_dir) / "theta")
        self.rate = rate
        self.max_dte = max_dte
        self._closes_cache: dict[str, list[tuple[date, float]]] = {}
        self._unavailable: set[str] = set()
        self._throttle = _Throttle(settings.requests_per_minute)
        self._chains: OrderedDict[tuple[str, date], Chain] = OrderedDict()
        self._no_data: set[tuple[str, date]] = set()

    # ------------------------------------------------------------------ transport
    @property
    def transport(self) -> Transport:
        if self._transport is None:
            s = self.settings
            if s.transport == "library":
                self._transport = LibraryTransport(
                    s.api_key.get_secret_value() if s.api_key else None
                )
            else:
                self._transport = RestTransport(s.base_url, s.timeout_s)
        return self._transport

    def reset_transport(self) -> None:
        """Drop the live connection so the next request re-authenticates (after an outage)."""
        if self._owns_transport:
            self._transport = None

    def _rows(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        self._throttle.wait()
        return self.transport.fetch(endpoint, **params)

    def _optional_rows(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        """Tier-gated endpoints (greeks: Standard+, OI: Value+) are best-effort: warn once, then skip."""
        if endpoint in self._unavailable:
            return []
        try:
            return self._rows(endpoint, **params)
        except Exception as exc:  # noqa: BLE001 - permission/transport errors must not block the chain
            log.warning(
                "theta %s unavailable (%s); continuing without it", endpoint, error_detail(exc)
            )
            self._unavailable.add(endpoint)
            return []

    # ------------------------------------------------------------------ chains
    def fetch_chain(self, underlying: str, asof: date) -> Chain | None:
        day = {
            "symbol": underlying,
            "expiration": "*",
            "start_date": asof,
            "end_date": asof,
            "max_dte": self.max_dte,
        }
        eod = self._rows("option_history_eod", **day)
        if not eod:
            return None
        greeks = self._optional_rows("option_history_greeks_eod", **day)
        oi = self._optional_rows("option_history_open_interest", **day)

        def key(row: dict[str, Any]) -> tuple[str, float, str] | None:
            right = _RIGHT.get(str(row.get("right", "")).lower())
            if right is None or not row.get("expiration"):
                return None
            return str(row["expiration"])[:10], float(row["strike"]), right.value

        iv_by: dict[tuple[str, float, str], float] = {}
        spot: float | None = None
        for row in greeks:
            k = key(row)
            iv = _num(row.get("implied_vol"))
            if k and iv:
                iv_by[k] = iv
            if spot is None:
                spot = _num(row.get("underlying_price")) or None
        oi_by: dict[tuple[str, float, str], int] = {}
        for row in oi:
            k = key(row)
            v = _num(row.get("open_interest"))
            if k and v is not None:
                oi_by[k] = int(v)

        quotes: dict[OptionContract, Quote] = {}
        for row in eod:
            k = key(row)
            if k is None:
                continue
            expiry = date.fromisoformat(k[0])
            if not (0 < (expiry - asof).days <= self.max_dte):
                continue
            contract = OptionContract(underlying, expiry, k[1], Right(k[2]))
            quotes[contract] = Quote(
                bid=_num(row.get("bid")) or 0.0,
                ask=_num(row.get("ask")) or 0.0,
                iv=iv_by.get(k),
                open_interest=oi_by.get(k),
                volume=int(_num(row.get("volume")) or 0),
            )
        if not quotes:
            return None
        if spot is None:
            closes = self.get_closes(underlying, asof, 1)
            if not closes:
                return None
            spot = closes[-1]
        return Chain(underlying, asof, spot, quotes, rate=self.rate)

    def get_chain(self, underlying: str, asof: date) -> Chain | None:
        key = (underlying, asof)
        if key in self._chains:
            self._chains.move_to_end(key)
            return self._chains[key]
        if key in self._no_data or self._is_trading_day(underlying, asof) is False:
            return None

        chain = self.store.load_chain(underlying, asof, rate=self.rate)
        fetched = chain is None
        if chain is None:
            chain = self.fetch_chain(underlying, asof)
        if chain is None:
            self._no_data.add(key)
            return None

        # Solving IV is the slow step; persist it so later loads only recompute greeks.
        before = sum(1 for q in chain.quotes.values() if q.iv is not None)
        chain.ensure_greeks()
        after = sum(1 for q in chain.quotes.values() if q.iv is not None)
        if fetched or after > before:
            self.store.save_chain(chain)

        self._chains[key] = chain
        if len(self._chains) > self.CHAIN_LRU_SIZE:
            self._chains.popitem(last=False)
        return chain

    CHAIN_LRU_SIZE = 32  # one day's working set: today's chains plus the oracle's two look-aheads

    def _is_trading_day(self, underlying: str, asof: date) -> bool | None:
        """True/False when the cached closes calendar covers `asof`; None when unknown."""
        closes = self._closes(underlying)
        if not closes or not (closes[0][0] <= asof <= closes[-1][0]):
            return None
        return any(d == asof for d, _ in closes)

    # ------------------------------------------------------------------ closes
    MAX_EOD_SPAN_DAYS = 365  # server limit per stock_history_eod request

    def fetch_closes(self, underlying: str, start: date, end: date) -> list[tuple[date, float]]:
        out: list[tuple[date, float]] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=self.MAX_EOD_SPAN_DAYS - 1), end)
            rows = self._rows(
                "stock_history_eod", symbol=underlying, start_date=chunk_start, end_date=chunk_end
            )
            for row in rows:
                d = _row_date(row)
                close = _num(row.get("close"))
                if d is not None and close:
                    out.append((d, close))
            chunk_start = chunk_end + timedelta(days=1)
        return sorted(out)

    def _closes(self, underlying: str) -> list[tuple[date, float]]:
        if underlying not in self._closes_cache:
            self._closes_cache[underlying] = self.store.load_closes(underlying)
        return self._closes_cache[underlying]

    def ensure_closes(self, underlying: str, start: date, end: date) -> None:
        have = {d for d, _ in self._closes(underlying)}
        if have and min(have) <= start and max(have) >= end:
            return
        series = self.fetch_closes(underlying, start - timedelta(days=5), end)
        if series:
            self.store.save_closes(underlying, series)
            self._closes_cache[underlying] = self.store.load_closes(underlying)

    def get_closes(self, underlying: str, asof: date, n: int) -> list[float]:
        self.ensure_closes(underlying, asof - timedelta(days=int(n * 1.6) + 10), asof)
        return [c for d, c in self._closes(underlying) if d <= asof][-n:]

    def trading_days(self, underlying: str, start: date, end: date) -> list[date]:
        self.ensure_closes(underlying, start, end)
        return [d for d, _ in self._closes(underlying) if start <= d <= end]
