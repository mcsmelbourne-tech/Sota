"""Pre-registered pass/fail criteria for the forward (paper) test.

Written down on 2026-10-14, before the first paper run, so the verdict cannot be fitted to the
outcome. The thresholds come from the no-condor replays of the three historical years (worst year
-9.8% / MDD 12.4%, 23-41 trades per year) and from the account size: a 15% drawdown on $50k is the
most the user said he would sit through.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from sota.models import Position

Status = Literal["pass", "warn", "fail", "pending"]

CHECKPOINT_6M_DAYS = 182
CHECKPOINT_12M_DAYS = 365
MAX_DRAWDOWN = 0.15
TRADES_PER_YEAR = (20, 50)
TRADES_WARN_BAND = 5  # within this many trades/yr of the band is a warning, not a failure
WORST_MONTH = -0.08
MIN_DAYS_FOR_TRADE_RATE = 60
TBILL_PROXY = 0.04  # 12-month hurdle: the strategy must beat cash

RULES = (
    "C1  Total return after costs > 0 at 6 months; > ~4% (cash) at 12 months.",
    f"C2  Maximum drawdown < {MAX_DRAWDOWN:.0%} at every point.",
    f"C3  {TRADES_PER_YEAR[0]}-{TRADES_PER_YEAR[1]} trades per year (annualized after "
    f"{MIN_DAYS_FOR_TRADE_RATE} days) - fewer means the live chains starve the resolver, "
    "more means behaviour changed.",
    f"C4  No calendar month worse than {WORST_MONTH:.0%}.",
    "Verdict: two or more failures at the 6-month checkpoint end the experiment; all four must "
    "pass at 12 months before any real-money (advisory) use, and then only at a fraction of size.",
)


@dataclass
class Criterion:
    key: str
    label: str
    status: Status
    value: str
    target: str
    detail: str = ""


@dataclass
class Scorecard:
    start: date | None
    asof: date | None
    start_nav: float | None
    nav: float | None
    days_elapsed: int
    checkpoint_6m: date | None
    checkpoint_12m: date | None
    provisional: bool  # before the 6-month checkpoint statuses are indicative only
    criteria: list[Criterion] = field(default_factory=list)
    failures: int = 0
    verdict: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("start", "asof", "checkpoint_6m", "checkpoint_12m"):
            d[k] = d[k].isoformat() if d[k] else None
        d["rules"] = list(RULES)
        return d


def max_drawdown(values: list[float]) -> float:
    peak, worst = float("-inf"), 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, 1.0 - v / peak)
    return worst


def monthly_returns(series: list[tuple[date, float]]) -> list[tuple[str, float]]:
    """Calendar-month returns from an EOD NAV series; the first month starts at its first point."""
    out: list[tuple[str, float]] = []
    if not series:
        return out
    prev_close = series[0][1]
    month = series[0][0].strftime("%Y-%m")
    last = prev_close
    for d, v in series:
        m = d.strftime("%Y-%m")
        if m != month:
            out.append((month, last / prev_close - 1.0 if prev_close else 0.0))
            prev_close, month = last, m
        last = v
    out.append((month, last / prev_close - 1.0 if prev_close else 0.0))
    return out


def build_scorecard(
    series: list[tuple[date, float]], positions: list[Position], asof: date | None = None
) -> Scorecard:
    if not series:
        return Scorecard(
            None, None, None, None, 0, None, None, True, [], 0, "waiting for first run"
        )
    start, start_nav = series[0]
    asof = asof or series[-1][0]
    nav = series[-1][1]
    days = max((asof - start).days, 0)
    cp6, cp12 = (
        start + timedelta(days=CHECKPOINT_6M_DAYS),
        start + timedelta(days=CHECKPOINT_12M_DAYS),
    )
    provisional = days < CHECKPOINT_6M_DAYS
    crit: list[Criterion] = []

    tr = nav / start_nav - 1.0
    hurdle = 0.0 if days < CHECKPOINT_12M_DAYS else TBILL_PROXY
    crit.append(
        Criterion(
            "return",
            "Total return after costs",
            "pass" if tr > hurdle else "fail",
            f"{tr:+.1%}",
            f"> {hurdle:.0%}",
            f"${nav - start_nav:+,.0f} on ${start_nav:,.0f}",
        )
    )

    mdd = max_drawdown([v for _, v in series])
    crit.append(
        Criterion(
            "drawdown",
            "Maximum drawdown",
            "fail" if mdd >= MAX_DRAWDOWN else "warn" if mdd >= MAX_DRAWDOWN * 2 / 3 else "pass",
            f"{mdd:.1%}",
            f"< {MAX_DRAWDOWN:.0%}",
            "peak to trough on daily closes",
        )
    )

    traded = [p for p in positions if p.status != "unfilled" and p.entry_date >= start]
    if days < MIN_DAYS_FOR_TRADE_RATE:
        rate_status: Status = "pending"
        rate_detail = f"{len(traded)} so far; annualized after {MIN_DAYS_FOR_TRADE_RATE} days"
        rate_value = f"{len(traded)} trades"
    else:
        rate = len(traded) * 365.0 / days
        lo, hi = TRADES_PER_YEAR
        if lo <= rate <= hi:
            rate_status = "pass"
        elif lo - TRADES_WARN_BAND <= rate <= hi + TRADES_WARN_BAND:
            rate_status = "warn"
        else:
            rate_status = "fail"
        rate_detail = f"{len(traded)} trades in {days} days"
        rate_value = f"{rate:.0f} / yr"
    crit.append(
        Criterion(
            "trade_rate",
            "Trade rate",
            rate_status,
            rate_value,
            f"{TRADES_PER_YEAR[0]}-{TRADES_PER_YEAR[1]} / yr",
            rate_detail,
        )
    )

    months = monthly_returns(series)
    complete = months[:-1] if len(months) > 1 else []
    if not complete:
        month_status: Status = "pending"
        month_value = f"{months[-1][1]:+.1%} MTD" if months else "-"
        month_detail = "needs one complete calendar month"
    else:
        worst_m, worst_r = min(complete, key=lambda x: x[1])
        month_status = "fail" if worst_r <= WORST_MONTH else "pass"
        month_value = f"{worst_r:+.1%}"
        month_detail = f"worst complete month {worst_m}; MTD {months[-1][1]:+.1%}"
    crit.append(
        Criterion(
            "worst_month",
            "Worst calendar month",
            month_status,
            month_value,
            f"> {WORST_MONTH:.0%}",
            month_detail,
        )
    )

    failures = sum(1 for c in crit if c.status == "fail")
    if provisional:
        verdict = f"day {days} of {CHECKPOINT_6M_DAYS}: indicative only until {cp6}"
    elif days < CHECKPOINT_12M_DAYS:
        verdict = (
            "STOP: two or more criteria failed at 6 months"
            if failures >= 2
            else f"continue to the 12-month checkpoint on {cp12}"
        )
    else:
        verdict = (
            "PASS: eligible for small-size advisory use"
            if failures == 0 and all(c.status != "pending" for c in crit)
            else "FAIL: do not trade real money on this configuration"
        )
    return Scorecard(
        start, asof, start_nav, nav, days, cp6, cp12, provisional, crit, failures, verdict
    )
