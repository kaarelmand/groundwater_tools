import pytest
import numpy as np
import math

from marinelli import PitFlow, PitFlowCommonUnits, get_nice_intervals

testpit = PitFlow(
    drawdown_stab=6,
    cond_h=20 / (24 * 60 * 60),
    radius_eff=np.sqrt(40 * 100 / np.pi),
    recharge=761 / (1000 * 365.25 * 24 * 60 * 60) * 0.1,
)

testpit_anisotropic = PitFlow(
    drawdown_stab=6,
    cond_h=20 / (24 * 60 * 60),
    radius_eff=np.sqrt(40 * 100 / np.pi),
    recharge=761 / (1000 * 365.25 * 24 * 60 * 60) * 0.1,
    anisotropy=0.1,
)

testpit_commonunits = PitFlowCommonUnits(
    drawdown_stab=6,
    cond_h_md=20,
    area=40 * 100,
    recharge_mm_yr=761 * 0.1,
)


def test_radius_infl():
    assert math.isclose(testpit.radius_infl, 1088.212649, rel_tol=1e-6)


def test_get_drawdown_at_r_100():
    assert math.isclose(testpit.get_drawdown_at_r(100), 1.951781, rel_tol=1e-6)


def test_get_drawdown_at_r_neg20():
    assert testpit.get_drawdown_at_r(-20) == 6


def test_get_drawdown_at_r_1100():
    assert testpit.get_drawdown_at_r(1100) == 0


def test_get_r_at_drawdown():
    assert math.isclose(testpit.radius_at_1m, 244.000377, rel_tol=1e-6)


def test_inflow_zone1():
    assert math.isclose(testpit.inflow_zone1, 0.008962, rel_tol=1e-4)


def test_inflow_zone2():
    assert math.isclose(testpit_anisotropic.inflow_zone2, 0.062688, rel_tol=1e-5)


def test_radius_infl_commonunits():
    assert math.isclose(testpit_commonunits.radius_infl, 1088.212649)


def test_get_nice_intervals():
    assert (
        get_nice_intervals(174.10374226797754) == np.array([50.0, 100.0, 150.0, 200.0])
    ).all()
