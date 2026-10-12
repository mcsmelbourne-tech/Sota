"""Portfolio performance metrics (paper App. E.4)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

NAN = float("nan")


@dataclass(frozen=True, slots=True)
class Metrics:
    total_return: float
    sharpe: float
    calmar: float
    sortino: float
    ann_vol: float
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float
    n_days: int
    n_trades: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def as_row(self) -> dict[str, str]:
        pct = lambda x: "nan" if math.isnan(x) else f"{100 * x:.2f}%"  # noqa: E731
        num = lambda x: "nan" if math.isnan(x) else f"{x:.2f}"  # noqa: E731
        return {
            "TR": pct(self.total_return),
            "ASR": num(self.sharpe),
            "ACR": num(self.calmar),
            "ASoR": num(self.sortino),
            "AVOL": pct(self.ann_vol),
            "MDD": pct(self.max_drawdown),
            "WR": pct(self.win_rate),
            "PLR": num(self.profit_loss_ratio),
            "days": str(self.n_days),
            "trades": str(self.n_trades),
        }


def max_drawdown(nav: Sequence[float]) -> float:
    arr = np.asarray(nav, dtype=float)
    if arr.size < 2:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    return float(np.max((peaks - arr) / peaks))


def compute_metrics(
    nav: Sequence[float], trade_pnls: Sequence[float] = (), rf: float = 0.0, periods: int = 252
) -> Metrics:
    arr = np.asarray(nav, dtype=float)
    n_days = int(arr.size)
    if n_days < 2 or arr[0] <= 0:
        return Metrics(NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, n_days, len(trade_pnls))

    r = arr[1:] / arr[:-1] - 1.0
    total_return = float(arr[-1] / arr[0] - 1.0)
    ann_ret = periods * float(np.mean(r))
    ann_vol = math.sqrt(periods) * float(np.std(r, ddof=1)) if r.size > 1 else NAN
    mdd = max_drawdown(arr)
    downside = math.sqrt(periods) * math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))

    excess = ann_ret - rf
    sharpe = excess / ann_vol if ann_vol and ann_vol > 0 else NAN
    calmar = excess / mdd if mdd > 0 else NAN
    sortino = excess / downside if downside > 0 else NAN

    pnls = np.asarray(trade_pnls, dtype=float)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    win_rate = float(wins.size / pnls.size) if pnls.size else NAN
    plr = float(np.mean(wins) / abs(np.mean(losses))) if wins.size and losses.size else NAN

    return Metrics(
        total_return, sharpe, calmar, sortino, ann_vol, mdd, win_rate, plr, n_days, int(pnls.size)
    )
