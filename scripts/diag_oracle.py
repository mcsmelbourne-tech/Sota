"""Diagnose the oracle run: where do the huge daily NAV moves come from?"""

import logging
from collections import Counter, defaultdict
from datetime import date

from sota.backtest.env import TradingEnv
from sota.backtest.metrics import compute_metrics
from sota.config import Settings
from sota.data.theta import ThetaDataSource
from sota.models import MULTIPLIER
from sota.selectors.oracle import HindsightOracle

logging.disable(logging.WARNING)
s = Settings()
src = ThetaDataSource(s.theta, s.data_dir)
dates = src.trading_days("SPY", date(2024, 9, 3), date(2025, 8, 29))
env = TradingEnv(src, s.universe, s.limits, s.effective_costs, s.initial_nav)
res = env.run(dates, HindsightOracle(src))

nav = res.nav_values
nav_by_date = dict(res.nav)
rets = [(dates[i], nav[i] / nav[i - 1] - 1) for i in range(1, len(nav))]
m = compute_metrics(nav, res.trade_pnls())
print(f"NAV end {nav[-1]:,.0f}; peak {max(nav):,.0f}")
print("metrics:", {k: v for k, v in m.as_row().items()})
print("largest daily moves:")
for d, r in sorted(rets, key=lambda t: -abs(t[1]))[:8]:
    print(f"  {d} {r:+.1%}")

print("\nattribution of the three worst days:")
hedge_trades = [t for t in res.trades if t.action == "hedge"]
for d, r in sorted(rets, key=lambda t: t[1])[:3]:
    p = dates[dates.index(d) - 1]
    rows = []
    for pos in res.positions:
        if pos.entry_date > p or (pos.exit_date is not None and pos.exit_date < d):
            continue
        cp, cd = src.get_chain(pos.package.underlying, p), src.get_chain(pos.package.underlying, d)
        vp, vd = pos.package.value(cp), pos.package.value(cd)
        if vp is None or vd is None:
            continue
        rows.append(
            (
                (vd - vp) * MULTIPLIER * pos.units,
                f"{pos.package.family} {pos.package.underlying} x{pos.units} "
                f"({vp:+.2f}->{vd:+.2f}, spot {cp.spot:.0f}->{cd.spot:.0f})",
            )
        )
    shares = defaultdict(int)
    for t in hedge_trades:
        if t.asof <= p:
            shares[t.underlying] += t.units
    for u, sh in shares.items():
        if sh:
            cp, cd = src.get_chain(u, p), src.get_chain(u, d)
            rows.append((sh * (cd.spot - cp.spot), f"hedge {u} {sh:+d} sh"))
    rows.sort()
    print(f"  {d} ({r:+.1%}, NAV {nav_by_date[p]:,.0f}): explained {sum(x for x, _ in rows):+,.0f}")
    for x, label in rows[:4] + rows[-2:]:
        print(f"    {x:+12,.0f}  {label}")

print("\nposition sizing at entry (max loss as % of NAV at entry):")
sizes = []
for p in res.positions:
    n = nav_by_date.get(p.entry_date, s.initial_nav)
    ml = (p.package.max_loss or 0) * MULTIPLIER * p.units / n
    prem = abs(p.entry_premium) * MULTIPLIER * p.units / n
    sizes.append((ml, prem, p))
sizes.sort(key=lambda t: -t[0])
print(
    f"  median max-loss {sorted(t[0] for t in sizes)[len(sizes) // 2]:.1%}, "
    f">5% NAV: {sum(1 for t in sizes if t[0] > 0.05)}/{len(sizes)}"
)
for ml, prem, p in sizes[:6]:
    print(
        f"  {p.entry_date} {p.package.family}/{p.package.side.value} {p.package.underlying} x{p.units} "
        f"maxloss {ml:.1%} premium {prem:.1%} -> pnl {p.realized_pnl or 0:+,.0f} ({p.status})"
    )

print("\nrealized pnl by family / exit reason:")
by_fam: dict[str, list[float]] = defaultdict(list)
for p in res.positions:
    if p.realized_pnl is not None:
        by_fam[p.package.family].append(p.realized_pnl)
for fam, pnls in sorted(by_fam.items(), key=lambda kv: -sum(kv[1])):
    print(
        f"  {fam}: n={len(pnls)} total {sum(pnls):+,.0f} worst {min(pnls):+,.0f} best {max(pnls):+,.0f}"
    )
print("  exits:", dict(Counter(p.status for p in res.positions)))

print("\nhedging:")
hedges = [t for t in res.trades if t.action == "hedge"]
print(
    f"  {len(hedges)} hedge trades; largest |shares x price| = "
    f"{max((abs(t.units) * t.price_per_share for t in hedges), default=0):,.0f}"
)
print(f"  final hedge shares: { {u: v for u, v in env.hedge_shares.items() if v} }")

print("\nquote quality of entered legs (at entry):")
wide = zero_bid = total = 0
for p in res.positions:
    chain = src.get_chain(p.package.underlying, p.entry_date)
    for leg in p.package.legs:
        q = chain.quote(leg.contract) if chain else None
        if q is None:
            continue
        total += 1
        zero_bid += q.bid <= 0
        wide += q.half_spread > 0.25 * max(q.mid, 1e-9)
print(f"  legs {total}: zero bid {zero_bid}, half-spread > 25% of mid {wide}")

print("\nworst and best butterfly: leg quotes at entry and exit")
bfs = [p for p in res.positions if p.package.family == "bf" and p.realized_pnl is not None]
for p in (min(bfs, key=lambda p: p.realized_pnl), max(bfs, key=lambda p: p.realized_pnl)):
    pkg = p.package
    print(
        f"  {pkg.describe()} x{p.units}: entry {p.entry_date} @ {p.entry_premium:+.2f}, "
        f"exit {p.exit_date} @ {p.exit_premium:+.2f} ({p.status}) pnl {p.realized_pnl:+,.0f}"
    )
    for when in (p.entry_date, p.exit_date):
        chain = src.get_chain(pkg.underlying, when)
        print(f"    {when} spot {chain.spot:.2f}")
        for leg in pkg.legs:
            q = chain.quote(leg.contract)
            print(
                f"      {leg.ratio:+d} {leg.contract.strike:g}{leg.contract.right.value}: "
                f"bid {q.bid:.2f} ask {q.ask:.2f} (mid {q.mid:.2f}, intrinsic {leg.contract.intrinsic(chain.spot):.2f})"
            )
