from datetime import timedelta

from sota.backtest.env import TradingEnv
from sota.backtest.metrics import compute_metrics
from sota.config import Costs, RiskLimits
from sota.models import Side, Tenor
from sota.selectors.base import BaseSelector, Proposal
from sota.selectors.oracle import HindsightOracle
from sota.selectors.threshold import ThresholdSelector


class OneOfEach(BaseSelector):
    """Opens one straddle and one iron condor per underlying on the first day only."""

    name = "one_of_each"

    def propose(self, features, portfolio):
        if portfolio.positions:
            return []
        out = []
        for u in features:
            out.append(Proposal(u, "ls", Side.LONG, Tenor.T8_30))
            out.append(Proposal(u, "ic", Side.LONG, Tenor.T31_90))
        return out


def _dates(source, n):
    return source.trading_days("AAA", source.start, source.start + timedelta(days=400))[:n]


def test_env_no_selector_keeps_nav(source):
    env = TradingEnv(source, ["AAA", "BBB"], RiskLimits(), Costs(), 1_000_000)
    result = env.run(_dates(source, 5), None)
    assert all(abs(v - 1_000_000) < 1e-6 for v in result.nav_values)


def test_env_lifecycle_and_hedging(source):
    limits = RiskLimits()
    env = TradingEnv(source, ["AAA", "BBB"], limits, Costs(), 1_000_000)
    dates = _dates(source, 60)
    result = env.run(dates, OneOfEach())
    assert len(result.positions) == 4
    assert all(not p.is_open for p in result.positions), (
        "all positions should have exited within 60 days"
    )
    assert {p.status for p in result.positions} <= {"stop", "force_close", "expired"}
    assert any(t.action == "hedge" for t in result.trades), "volatility families are delta hedged"
    assert all(p.fees_paid > 0 for p in result.positions)
    for p in result.positions:
        assert p.units * p.package.max_loss * 100 <= limits.nav_frac_cap * 1_000_000 * (1 + 1e-9)
    assert env.hedge_shares.get("AAA", 0) == 0 and env.hedge_shares.get("BBB", 0) == 0
    m = compute_metrics(result.nav_values, result.trade_pnls())
    assert m.n_trades == 4 and m.n_days == 60


def test_threshold_selector_runs(source):
    env = TradingEnv(source, ["AAA", "BBB"], RiskLimits(), Costs(), 1_000_000)
    result = env.run(_dates(source, 40), ThresholdSelector(participation_gate=False))
    assert len(result.nav_values) == 40
    assert all(v > 0 for v in result.nav_values)


def test_replay_reproduces_recorded_run_and_honours_new_rules(source, tmp_path):
    import json

    from sota.selectors.replay import ReplaySelector, proposals_from_positions

    dates = _dates(source, 60)
    original = TradingEnv(source, ["AAA", "BBB"], RiskLimits(), Costs(), 1_000_000).run(
        dates, OneOfEach()
    )
    path = tmp_path / "proposals.json"
    path.write_text(json.dumps(proposals_from_positions(original.positions)))

    replay = TradingEnv(source, ["AAA", "BBB"], RiskLimits(), Costs(), 1_000_000).run(
        dates, ReplaySelector(path)
    )
    assert replay.nav_values == original.nav_values
    assert [(p.entry_date, p.package.describe()) for p in replay.positions] == [
        (p.entry_date, p.package.describe()) for p in original.positions
    ]

    no_condors = TradingEnv(
        source, ["AAA", "BBB"], RiskLimits(families=["ls"]), Costs(), 1_000_000
    ).run(dates, ReplaySelector(path))
    assert {p.package.family for p in no_condors.positions} == {"ls"}


def test_hedging_can_be_disabled(source):
    env = TradingEnv(source, ["AAA", "BBB"], RiskLimits(hedge=False), Costs(), 1_000_000)
    result = env.run(_dates(source, 30), OneOfEach())
    assert len(result.positions) == 4
    assert not any(t.action == "hedge" for t in result.trades)
    assert not env.hedge_shares


def test_small_account_trades_whole_contracts(source):
    limits = RiskLimits(hedge=False, target_risk_frac=0.01)
    env = TradingEnv(source, ["AAA", "BBB"], limits, Costs(), 25_000)
    result = env.run(_dates(source, 30), OneOfEach())
    assert result.positions, "a $25k account should still fit some 1-contract structures"
    for p in result.positions:
        assert p.units >= 1
        assert p.units * p.package.max_loss * 100 <= limits.nav_frac_cap * 25_000 * (1 + 1e-9)


def test_family_allow_list(source):
    limits = RiskLimits(families=["ic"])
    env = TradingEnv(source, ["AAA", "BBB"], limits, Costs(), 1_000_000)
    result = env.run(_dates(source, 10), OneOfEach())
    assert result.positions and {p.package.family for p in result.positions} == {"ic"}


def test_oracle_is_profitable_on_synthetic(source):
    env = TradingEnv(source, ["AAA", "BBB"], RiskLimits(), Costs(), 1_000_000)
    result = env.run(_dates(source, 50), HindsightOracle(source))
    m = compute_metrics(result.nav_values, result.trade_pnls())
    assert m.n_trades > 0
    assert m.total_return > 0, "hindsight oracle should make money when it can see the future"
