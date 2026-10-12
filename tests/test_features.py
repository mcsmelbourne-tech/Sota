import math

import pytest

from sota.features import IVSurface, compute_features, flow_imbalance, realized_vol
from sota.models import Right


def test_iv_surface_reproduces_synthetic_shape(chain, source):
    surf = IVSurface(chain)
    atm30 = surf.iv(0.50, Right.CALL, 30)
    atm90 = surf.iv(0.50, Right.CALL, 90)
    p25 = surf.iv(0.25, Right.PUT, 30)
    c25 = surf.iv(0.25, Right.CALL, 30)
    assert atm30 is not None and atm90 is not None and p25 is not None and c25 is not None
    base = source.iv_level["AAA"][chain.asof]
    assert atm30 == pytest.approx(base, rel=0.15)
    assert atm90 > atm30  # positive term slope in the synthetic surface
    assert p25 > c25  # negative skew coefficient: puts richer than calls
    assert 0.5 * (p25 + c25) > atm30  # positive curvature


def test_compute_features_fields(chain, source):
    closes = source.get_closes("AAA", chain.asof, 22)
    f = compute_features(chain, closes)
    assert f.underlying == "AAA" and f.spot == chain.spot
    assert not math.isnan(f.ret) and not math.isnan(f.rv) and f.rv > 0
    assert f.wedge == pytest.approx(f.iv - f.rv)
    assert f.ts > 0 and f.sk > 0 and f.bf > 0
    assert math.isnan(f.dv) and math.isnan(f.fi) and math.isnan(f.doi)


def test_compute_features_with_prev_chain(chain, source):
    prev = source.get_chain("AAA", source.dates[source.dates.index(chain.asof) - 1])
    closes = source.get_closes("AAA", chain.asof, 22)
    f = compute_features(chain, closes, prev_chain=prev, prev_iv=0.20)
    assert not math.isnan(f.doi)
    assert f.dv == pytest.approx(f.iv - 0.20)


def test_realized_vol():
    assert realized_vol([100, 100, 100, 100]) == 0.0
    assert math.isnan(realized_vol([100.0]))
    closes = [
        100 * math.exp(0.01 * i) for i in range(30)
    ]  # constant log return -> rv = 0.01*sqrt(252)
    assert realized_vol(closes) == pytest.approx(0.01 * math.sqrt(252))


def test_flow_imbalance():
    assert flow_imbalance([(10, 1, 0.5), (10, -1, 0.5)]) == 0.0
    assert flow_imbalance([(10, 1, 0.5)]) == 1.0
    assert math.isnan(flow_imbalance([]))
