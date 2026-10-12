import pytest

from sota.models import Right, Side
from sota.strategies.families import FAMILIES, describe_families, normalize_coords, snap

EXPECTED_LEGS = {"ol": 1, "dv": 2, "cv": 2, "dg": 3, "ls": 2, "lg": 2, "ib": 4, "ic": 4, "bf": 3}


@pytest.mark.parametrize("code", list(FAMILIES))
def test_leg_counts_and_defaults(code):
    spec = FAMILIES[code]
    legs = spec.legs(Side.LONG, normalize_coords(spec, None))
    assert len(legs) == EXPECTED_LEGS[code]
    assert all(0.05 <= leg.delta <= 0.95 for leg in legs)


def test_structures():
    coords = normalize_coords(FAMILIES["ic"], None)
    ic = FAMILIES["ic"].legs(Side.LONG, coords)
    assert sum(leg.ratio for leg in ic) == 0 and sum(1 for leg in ic if leg.ratio < 0) == 2
    bf = FAMILIES["bf"].legs(Side.LONG, normalize_coords(FAMILIES["bf"], None))
    assert sorted(leg.ratio for leg in bf) == [-2, 1, 1] and all(
        leg.right is Right.CALL for leg in bf
    )
    bf_put = FAMILIES["bf"].legs(Side.SHORT, normalize_coords(FAMILIES["bf"], None))
    assert all(leg.right is Right.PUT for leg in bf_put)
    cv_bull = FAMILIES["cv"].legs(Side.LONG, normalize_coords(FAMILIES["cv"], None))
    assert all(leg.right is Right.PUT for leg in cv_bull) and cv_bull[0].ratio == -1
    dg = FAMILIES["dg"].legs(Side.LONG, normalize_coords(FAMILIES["dg"], None))
    assert [(leg.right, leg.ratio) for leg in dg] == [
        (Right.CALL, 1),
        (Right.PUT, -1),
        (Right.PUT, 1),
    ]


def test_normalize_coords_clips_and_snaps():
    spec = FAMILIES["ic"]
    out = normalize_coords(spec, {"delta": 0.93, "width": 0.07})
    assert out["delta"] == 0.55  # clipped to admitted range
    assert out["width"] == 0.05  # snapped to the 0.05 grid
    assert snap(0.337) == 0.35


def test_hedged_categories():
    assert {c for c, f in FAMILIES.items() if f.hedged} == {"ls", "lg", "ib", "ic", "bf"}
    text = describe_families()
    assert all(f"- {code} (" in text for code in FAMILIES)
