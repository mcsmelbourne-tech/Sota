import math

import pytest

from sota.backtest.metrics import compute_metrics, max_drawdown


def test_max_drawdown():
    assert max_drawdown([100, 110, 99, 120]) == pytest.approx(0.1)
    assert max_drawdown([100, 101, 102]) == 0.0


def test_compute_metrics_basic():
    m = compute_metrics([100, 110, 99], trade_pnls=[10, -5, 20, -10])
    assert m.total_return == pytest.approx(-0.01)
    assert m.max_drawdown == pytest.approx(0.1)
    assert m.win_rate == 0.5
    assert m.profit_loss_ratio == pytest.approx(15 / 7.5)
    assert m.n_days == 3 and m.n_trades == 4
    row = m.as_row()
    assert row["TR"] == "-1.00%" and row["WR"] == "50.00%"


def test_compute_metrics_degenerate():
    m = compute_metrics([100.0])
    assert math.isnan(m.total_return)
    flat = compute_metrics([100, 100, 100])
    assert flat.total_return == 0.0 and math.isnan(flat.sharpe)
