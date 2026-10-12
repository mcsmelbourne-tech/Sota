import math

import pytest

from sota.pricing.bsm import crr_american, greeks, implied_vol, price

S, K, T, R, Q, SIG = 100.0, 105.0, 0.5, 0.03, 0.01, 0.3


def test_put_call_parity():
    c = price(S, K, T, R, Q, SIG, True)
    p = price(S, K, T, R, Q, SIG, False)
    assert c - p == pytest.approx(S * math.exp(-Q * T) - K * math.exp(-R * T), abs=1e-9)


def test_greek_bounds_and_signs():
    c = greeks(S, K, T, R, Q, SIG, True)
    p = greeks(S, K, T, R, Q, SIG, False)
    assert 0 < c.delta < 1 and -1 < p.delta < 0
    assert c.gamma == pytest.approx(p.gamma) and c.gamma > 0
    assert c.vega == pytest.approx(p.vega) and c.vega > 0
    assert c.theta < 0 and p.theta < 0  # per calendar day, long options decay


def test_implied_vol_round_trip():
    for sigma in (0.08, 0.25, 0.9):
        for is_call in (True, False):
            px = price(S, K, T, R, Q, sigma, is_call)
            assert implied_vol(px, S, K, T, R, Q, is_call) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage_and_no_time_value():
    assert implied_vol(0.0, S, K, T, R, Q, True) is None
    deep_itm_intrinsic = max(S * math.exp(-Q * T) - 50 * math.exp(-R * T), 0.0)
    assert implied_vol(deep_itm_intrinsic, S, 50.0, T, R, Q, True) is None


def test_expired_option_is_intrinsic():
    g = greeks(110.0, 100.0, 0.0, R, Q, SIG, True)
    assert g.price == 10.0 and g.delta == 1.0 and g.gamma == 0.0


def test_crr_american_bounds():
    euro_put = price(S, K, T, R, 0.0, SIG, False)
    amer_put = crr_american(S, K, T, R, 0.0, SIG, False, steps=200)
    assert amer_put >= euro_put - 1e-6
    euro_call = price(S, K, T, R, 0.0, SIG, True)
    amer_call = crr_american(S, K, T, R, 0.0, SIG, True, steps=200)
    assert amer_call == pytest.approx(euro_call, rel=5e-3)  # no dividends: no early exercise
