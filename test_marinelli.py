import pytest
import math

from marinelli import PitFlow, PitFlowCommonUnits

testpit = PitFlow(
    drawdown_stab=6,
    trans_h=20 / (24 * 60 * 60),
    radius_eff=math.sqrt(40 * 100 / math.pi),
    recharge=761 / (1000 * 365.25 * 24 * 60 * 60) * 0.1,
)


def test_get_depression_radius():
    assert math.isclose(testpit.radius_infl, 1088.212649, rel_tol=1e-6)


def test_get_depression_at_r_100():
    assert math.isclose(testpit.get_depression_at_r(100), 1.951781, rel_tol=1e-6)


def test_get_depression_at_r_neg20():
    assert testpit.get_depression_at_r(-20) == 6


def test_get_depression_at_r_1100():
    assert testpit.get_depression_at_r(1100) == 0


def test_get_significant_radius():
    assert math.isclose(testpit.get_significant_radius(), 244.000377, rel_tol=1e-6)


def test_get_depression_radius_commonunits():
    testpit = PitFlowCommonUnits(
        drawdown_stab=6,
        trans_h_md=20,
        area=40 * 100,
        precip=761,
    )
    assert math.isclose(testpit.radius_infl, 1088.212649)
