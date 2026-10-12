import math

import numpy as np
import pytest

from sota.models import OptionContract, Right, Side, Tenor
from sota.pricing.surface import leg_is_liquid, smooth_chain
from sota.resolvers.contract import resolve_package, resolve_strike


def _perturb(chain, seed=0, noise=0.15, wide_every=7):
    """Scatter mids and widen some spreads to mimic noisy post-close NBBO."""
    rng = np.random.default_rng(seed)
    for i, (_c, q) in enumerate(
        sorted(chain.quotes.items(), key=lambda kv: (kv[0].expiry, kv[0].strike))
    ):
        mid = q.mid * float(np.exp(noise * rng.standard_normal()))
        hs = q.half_spread * (6.0 if i % wide_every == 0 else 1.0)
        q.bid, q.ask = max(mid - hs, 0.0), mid + hs
        q.iv = q.delta = q.gamma = q.vega = q.theta = None


def test_smoothing_recovers_the_true_surface(source, asof):
    clean = source.get_chain("AAA", asof)
    clean.ensure_greeks()
    noisy = source.get_chain("AAA", asof)
    _perturb(noisy)
    n = smooth_chain(noisy)
    assert n >= 10, "every liquid expiry gets a fitted smile"

    err_raw, err_fit, count = 0.0, 0.0, 0
    for c, q in clean.quotes.items():
        nq = noisy.quote(c)
        if q.iv is None or nq.fair is None or not (0.15 <= abs(q.delta) <= 0.85):
            continue
        err_raw += abs(nq.mid - q.mid)
        err_fit += abs(nq.fair - q.mid)
        count += 1
    assert count > 500
    assert err_fit < 0.4 * err_raw, (
        "fitted fair values are much closer to the truth than noisy mids"
    )


def test_mark_stays_inside_the_quoted_spread(chain):
    smooth_chain(chain)
    for q in chain.quotes.values():
        if q.fair is not None and q.is_valid:
            assert q.bid - 1e-12 <= q.mark <= q.ask + 1e-12


def test_illiquid_legs_are_excluded(chain):
    smooth_chain(chain)
    expiry = chain.expiries()[3]
    target = resolve_strike(chain, expiry, Right.PUT, 0.25)
    assert target is not None
    q = chain.quote(target)
    q.bid, q.ask = 0.0, 3 * q.ask  # make the natural pick untradable
    assert not leg_is_liquid(target, q, 0.25)
    again = resolve_strike(chain, expiry, Right.PUT, 0.25, max_rel_half_spread=0.25)
    assert again is not None and again != target


def test_package_priced_at_mark_and_arbitrage_rejected(chain, limits):
    from sota.resolvers.size import resolve_size
    from sota.risk import PortfolioRisk

    smooth_chain(chain)
    pkg = resolve_package(chain, "bf", Side.LONG, Tenor.T31_90)
    assert pkg is not None and pkg.max_loss is not None and pkg.max_loss > 0
    assert len(pkg.leg_ivs) == len(pkg.legs) and all(iv > 0 for iv in pkg.leg_ivs)
    # a credit butterfly (negative max loss) is a pricing artefact, never a trade
    pkg.max_loss = 0.0
    res = resolve_size(pkg, chain.spot, 0.25, limits, PortfolioRisk(nav=1e6), chain.asof)
    assert not res.accepted and "arbitrage" in res.binding


def test_smoothing_leaves_thin_expiries_at_mid(chain):
    thin = chain.expiries()[0]
    keep = {
        c: q for c, q in chain.quotes.items() if c.expiry != thin or abs(c.strike - chain.spot) < 2
    }
    chain.quotes = keep
    chain._index = None
    smooth_chain(chain, min_quotes=6)
    for c, q in chain.quotes.items():
        if c.expiry == thin:
            assert q.fair == pytest.approx(q.mid) and math.isfinite(q.mark)


def test_occ_round_trip_unaffected():
    c = OptionContract("SPY", __import__("datetime").date(2025, 5, 16), 500.0, Right.PUT)
    assert OptionContract.from_occ(c.occ_symbol) == c
