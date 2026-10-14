"""Split handling: point-in-time closes and positions carried across an ex-date."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from sota.config import Costs, RiskLimits, ThetaSettings
from sota.data.splits import SPLITS, adjust_closes, apply_splits, split_package
from sota.data.theta import ThetaDataSource
from sota.models import Side, Tenor
from sota.resolvers.contract import resolve_package

REPO_CACHE = Path(__file__).resolve().parents[1] / "data"
NVDA_SPLIT = date(2024, 6, 10)


def test_adjust_closes_is_point_in_time(monkeypatch):
    monkeypatch.setitem(SPLITS, "XYZ", [(date(2024, 6, 10), 10.0)])
    series = [(date(2024, 6, 6), 1200.0), (date(2024, 6, 7), 1210.0), (date(2024, 6, 10), 122.0)]
    # the day before the split, history is still in old terms
    assert adjust_closes("XYZ", series[:2], date(2024, 6, 7)) == series[:2]
    # from the ex-date on, earlier closes are divided so returns stay continuous
    assert adjust_closes("XYZ", series, date(2024, 6, 10)) == [
        (date(2024, 6, 6), 120.0),
        (date(2024, 6, 7), 121.0),
        (date(2024, 6, 10), 122.0),
    ]
    assert adjust_closes("SPY", series, date(2024, 6, 10)) == series


def test_split_package_and_position_keep_value(chain, monkeypatch):
    monkeypatch.setitem(SPLITS, "AAA", [(chain.asof + timedelta(days=3), 4.0)])
    pkg = resolve_package(chain, "ic", Side.SHORT, Tenor.T8_30)
    assert pkg is not None
    adjusted = split_package(pkg, 4.0, strikes=[leg.contract.strike / 4.0 for leg in pkg.legs])
    assert [leg.contract.strike for leg in adjusted.legs] == [
        pytest.approx(leg.contract.strike / 4.0) for leg in pkg.legs
    ]
    assert adjusted.net_premium == pytest.approx(pkg.net_premium / 4.0)
    assert adjusted.max_loss == pytest.approx(pkg.max_loss / 4.0)
    assert adjusted.delta == pkg.delta and adjusted.gamma == pytest.approx(pkg.gamma * 4.0)

    from sota.models import Position

    pos = Position(
        "p1", pkg, 3, chain.asof, pkg.net_premium, 0.6 * abs(pkg.net_premium), pkg.expiry
    )
    before = pos.entry_premium * pos.units
    assert apply_splits(pos, chain.asof + timedelta(days=2)) == [], "not effective yet"
    applied = apply_splits(pos, chain.asof + timedelta(days=5))
    assert applied == [(chain.asof + timedelta(days=3), 4.0)]
    assert pos.units == 12 and pos.entry_premium * pos.units == pytest.approx(before)
    assert pos.stop_loss_per_share == pytest.approx(0.6 * abs(pkg.net_premium) / 4.0)
    assert apply_splits(pos, chain.asof + timedelta(days=9)) == [], "idempotent"
    assert pos.tags["splits"] == (chain.asof + timedelta(days=3)).isoformat()


def test_snap_prefers_listed_strike():
    from sota.data.splits import _snap

    assert _snap(700 / 3, [233.33, 235.0]) == 233.33  # OCC cent rounding, as listed
    assert _snap(700 / 3, [240.0]) == 233.33  # nothing close: keep the cent-rounded value
    assert _snap(120.0, None) == 120.0


@pytest.mark.skipif(
    not (REPO_CACHE / "theta" / "chains" / "NVDA" / "20240610.parquet").exists(),
    reason="needs the real NVDA cache around the 2024-06-10 split",
)
def test_nvda_split_in_real_cache():
    """Raw Theta closes drop 10x on the ex-date; the point-in-time series must not."""
    src = ThetaDataSource(ThetaSettings(), REPO_CACHE)
    before = src.get_closes("NVDA", NVDA_SPLIT - timedelta(days=1), 3)
    after = src.get_closes("NVDA", NVDA_SPLIT, 3)
    assert before[-1] > 1000 and after[-1] < 200, "raw data is unadjusted"
    assert all(abs(a / b - 1) < 0.1 for a, b in zip(after[:-1], after[1:], strict=False)), (
        f"adjusted history is continuous: {after}"
    )
    assert after[-2] == pytest.approx(before[-1] / 10.0)

    from sota.backtest.env import TradingEnv

    env = TradingEnv(src, ["NVDA"], RiskLimits(hedge=False), Costs(), 50_000)
    d0 = NVDA_SPLIT - timedelta(days=3)  # Friday 2024-06-07
    chain0 = src.get_chain("NVDA", d0)
    pkg = resolve_package(chain0, "ic", Side.SHORT, Tenor.T31_90)
    assert pkg is not None
    pos = env.open_position(pkg, 1, d0)
    env.nav_history.append((d0, env.mark_nav({"NVDA": chain0})))
    nav0 = env.nav_history[-1][1]
    step = env.step(NVDA_SPLIT, None)
    assert pos.units == 10 and all(leg.contract.strike < 300 for leg in pos.package.legs)
    chain1 = src.get_chain("NVDA", NVDA_SPLIT)
    assert pos.package.value(chain1) is not None, "re-expressed legs are found in the new chain"
    assert abs(step.nav / nav0 - 1) < 0.05, f"no fake P&L across the split: {nav0} -> {step.nav}"
    assert pos.is_open or pos.status != "expired"
