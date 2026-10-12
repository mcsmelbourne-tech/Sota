"""Synthetic market: GBM underlyings with a parametric IV surface and BSM quotes. For tests and demos."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from sota.models import Chain, OptionContract, Quote, Right
from sota.pricing.bsm import price as bsm_price


def weekdays(start: date, count: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


class SyntheticChainSource:
    def __init__(
        self,
        underlyings: list[str],
        start: date,
        days: int,
        seed: int = 0,
        spot0: float = 100.0,
        base_vol: float = 0.25,
        drift: float = 0.0,
        vol_of_vol: float = 0.15,
        skew: float = -0.20,
        curvature: float = 0.15,
        term_slope: float = 0.15,
        spread_frac: float = 0.02,
        max_dte: int = 200,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.dates = weekdays(start, days + 30)  # extra history for realized vol
        self.start = self.dates[30]
        self.max_dte = max_dte
        self.skew, self.curvature, self.term_slope, self.spread_frac = skew, curvature, term_slope, spread_frac
        self.closes: dict[str, dict[date, float]] = {}
        self.iv_level: dict[str, dict[date, float]] = {}
        for i, u in enumerate(underlyings):
            s = spot0 * (1 + 0.5 * i)
            iv = base_vol
            cl: dict[date, float] = {}
            lv: dict[date, float] = {}
            for d in self.dates:
                iv = max(0.08, iv * math.exp(vol_of_vol / math.sqrt(252) * self.rng.standard_normal() - 0.5 * (iv - base_vol)))
                s *= math.exp((drift - 0.5 * iv * iv) / 252 + iv / math.sqrt(252) * self.rng.standard_normal())
                cl[d] = s
                lv[d] = iv
            self.closes[u] = cl
            self.iv_level[u] = lv

    def trading_days(self, underlying: str, start: date, end: date) -> list[date]:
        return [d for d in self.dates if start <= d <= end]

    def _expiries(self, asof: date) -> list[date]:
        days_to_friday = (4 - asof.weekday()) % 7 or 7
        first = asof + timedelta(days=days_to_friday)
        out = [first + timedelta(weeks=k) for k in range(8)]
        out += [first + timedelta(weeks=k) for k in range(12, 30, 4)]
        return [e for e in out if (e - asof).days <= self.max_dte]

    def iv(self, u: str, asof: date, spot: float, strike: float, tte: float) -> float:
        base = self.iv_level[u][asof] * (1 + self.term_slope * (tte - 30 / 365))
        m = math.log(strike / spot) / (base * math.sqrt(max(tte, 1 / 365)))
        m = max(-1.5, min(1.5, m))  # bounded smile keeps delta monotonic in strike
        return max(0.05, base * (1 + self.skew * m + self.curvature * m * m))

    def get_chain(self, underlying: str, asof: date) -> Chain | None:
        closes = self.closes.get(underlying)
        if closes is None or asof not in closes:
            return None
        spot = closes[asof]
        quotes: dict[OptionContract, Quote] = {}
        for expiry in self._expiries(asof):
            tte = (expiry - asof).days / 365
            for k in range(-25, 26):
                strike = round(spot * (1 + 0.01 * k) * 2) / 2
                iv = self.iv(underlying, asof, spot, strike, tte)
                for right in (Right.CALL, Right.PUT):
                    mid = bsm_price(spot, strike, tte, 0.0, 0.0, iv, right.is_call)
                    hs = max(0.01, self.spread_frac * mid)
                    quotes[OptionContract(underlying, expiry, strike, right)] = Quote(
                        bid=max(mid - hs, 0.0), ask=mid + hs, iv=iv,
                        open_interest=int(1000 * math.exp(-abs(k) / 8)), volume=int(100 * math.exp(-abs(k) / 6)),
                    )
        return Chain(underlying, asof, spot, quotes)

    def get_closes(self, underlying: str, asof: date, n: int) -> list[float]:
        closes = self.closes.get(underlying, {})
        return [c for d, c in sorted(closes.items()) if d <= asof][-n:]
