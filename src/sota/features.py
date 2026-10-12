"""Market-state features summarizing the option-implied return distribution (paper App. C.1)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from sota.models import Chain, Right

RV_WINDOW = 21
NAN = float("nan")


@dataclass(frozen=True, slots=True)
class MarketFeatures:
    underlying: str
    asof: date
    spot: float
    ret: float  # session log return
    rv: float  # trailing 21d realized vol, annualized
    iv: float  # 30d ATM call implied vol
    dv: float  # change in iv (open->close when open IV is known, else day over day)
    wedge: float  # iv - rv
    ts: float  # term structure: 90d ATM - 30d ATM
    sk: float  # skew: 30d 25-delta put - 25-delta call
    bf: float  # curvature: 30d 25-delta butterfly
    fi: float  # delta-weighted flow imbalance (NaN without trade data)
    doi: float  # relative change in total open interest (NaN without prior chain)

    @property
    def iv_rv_ratio(self) -> float:
        return self.iv / self.rv if self.rv > 0 and not math.isnan(self.iv) else NAN

    def as_dict(self) -> dict[str, float]:
        return {
            "ret": self.ret,
            "rv": self.rv,
            "iv": self.iv,
            "dv": self.dv,
            "wedge": self.wedge,
            "ts": self.ts,
            "sk": self.sk,
            "bf": self.bf,
            "fi": self.fi,
            "doi": self.doi,
        }


class IVSurface:
    """Interpolates implied vol at (|delta|, right, calendar days) from a chain.

    Linear in |delta| within an expiry; linear in total variance across expiries; flat beyond the ends.
    """

    def __init__(self, chain: Chain) -> None:
        by_exp: dict[date, dict[Right, list[tuple[float, float]]]] = {}
        for c, q in chain.quotes.items():
            if q.iv is None or q.delta is None or not q.is_valid or c.dte(chain.asof) <= 0:
                continue
            by_exp.setdefault(c.expiry, {Right.CALL: [], Right.PUT: []})[c.right].append(
                (abs(q.delta), q.iv)
            )
        self._slices: list[tuple[int, dict[Right, tuple[np.ndarray, np.ndarray]]]] = []
        for exp in sorted(by_exp):
            sl: dict[Right, tuple[np.ndarray, np.ndarray]] = {}
            for right, pts in by_exp[exp].items():
                if len(pts) < 2:
                    continue
                pts.sort()
                deltas, idx = np.unique([p[0] for p in pts], return_index=True)
                ivs = np.asarray([p[1] for p in pts])[idx]
                if len(deltas) >= 2:
                    sl[right] = (deltas, ivs)
            if sl:
                self._slices.append(((exp - chain.asof).days, sl))

    @staticmethod
    def _at_delta(sl: tuple[np.ndarray, np.ndarray], delta: float) -> float:
        deltas, ivs = sl
        return float(np.interp(delta, deltas, ivs))

    def iv(self, delta: float, right: Right, tenor_days: int) -> float | None:
        slices = [(d, sl[right]) for d, sl in self._slices if right in sl]
        if not slices:
            return None
        lower = [s for s in slices if s[0] <= tenor_days]
        upper = [s for s in slices if s[0] >= tenor_days]
        if lower and upper:
            d0, s0 = lower[-1]
            d1, s1 = upper[0]
            iv0 = self._at_delta(s0, delta)
            if d0 == d1:
                return iv0
            iv1 = self._at_delta(s1, delta)
            w = (tenor_days - d0) / (d1 - d0)
            var = iv0 * iv0 * d0 + w * (iv1 * iv1 * d1 - iv0 * iv0 * d0)
            return math.sqrt(max(var, 0.0) / tenor_days)
        if lower:
            return self._at_delta(lower[-1][1], delta)
        return self._at_delta(upper[0][1], delta)


def realized_vol(closes: Sequence[float], window: int = RV_WINDOW) -> float:
    """Zero-mean realized volatility over the trailing `window` daily log returns, annualized."""
    arr = np.asarray(closes, dtype=float)
    n = min(window, len(arr) - 1)
    if n < 2:
        return NAN
    r = np.diff(np.log(arr[-(n + 1) :]))
    return float(math.sqrt(252.0 / n * float(np.sum(r * r))))


def flow_imbalance(trades: Iterable[tuple[float, int, float]]) -> float:
    """Delta-weighted signed demand / total delta transacted, from (qty, sign, delta) trades."""
    num = den = 0.0
    for qty, sign, delta in trades:
        w = qty * abs(delta)
        num += sign * w
        den += w
    return num / den if den > 0 else NAN


def _sub(a: float | None, b: float | None) -> float:
    return NAN if a is None or b is None else a - b


def compute_features(
    chain: Chain,
    closes: Sequence[float],
    *,
    prev_chain: Chain | None = None,
    prev_iv: float | None = None,
    iv_open: float | None = None,
    trades: Iterable[tuple[float, int, float]] | None = None,
) -> MarketFeatures:
    """Build the 10-feature state. `closes` must end with the close on `chain.asof`."""
    chain.ensure_greeks()
    surf = IVSurface(chain)
    iv30 = surf.iv(0.50, Right.CALL, 30)
    iv90 = surf.iv(0.50, Right.CALL, 90)
    p25 = surf.iv(0.25, Right.PUT, 30)
    c25 = surf.iv(0.25, Right.CALL, 30)

    ret = math.log(closes[-1] / closes[-2]) if len(closes) >= 2 and closes[-2] > 0 else NAN
    rv = realized_vol(closes)
    iv = iv30 if iv30 is not None else NAN

    if iv_open is not None:
        dv = iv - iv_open
    elif prev_iv is not None:
        dv = iv - prev_iv
    else:
        dv = NAN

    bf = NAN if p25 is None or c25 is None or iv30 is None else 0.5 * (p25 + c25) - iv30

    doi = NAN
    if prev_chain is not None:
        oi_now, oi_prev = chain.total_open_interest(), prev_chain.total_open_interest()
        if oi_now is not None and oi_prev:
            doi = (oi_now - oi_prev) / oi_prev

    return MarketFeatures(
        underlying=chain.underlying,
        asof=chain.asof,
        spot=chain.spot,
        ret=ret,
        rv=rv,
        iv=iv,
        dv=dv,
        wedge=iv - rv,
        ts=_sub(iv90, iv30),
        sk=_sub(p25, c25),
        bf=bf,
        fi=flow_imbalance(trades) if trades is not None else NAN,
        doi=doi,
    )
