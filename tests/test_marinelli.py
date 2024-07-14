import pytest  # noqa: F401
import numpy as np
import math

from src.groundwater_tools.marinelli import (
    PitFlow,
    PitFlowCommonUnits,
    get_nice_intervals,
)


@pytest.fixture
def testpit():
    return PitFlow(
        drawdown_stab=6,
        area=40 * 100,
        recharge=761 / (1000 * 365.25 * 24 * 60 * 60) * 0.1,
        precipitation=761 / (1000 * 365.25 * 24 * 60 * 60),
        cond_h=20 / (24 * 60 * 60),
    )


@pytest.fixture
def testpit_anisotropic():
    return PitFlow(
        drawdown_stab=6,
        area=40 * 100,
        recharge=761 / (1000 * 365.25 * 24 * 60 * 60) * 0.1,
        precipitation=761 / (1000 * 365.25 * 24 * 60 * 60),
        cond_h=20 / (24 * 60 * 60),
        anisotropy=0.1,
    )


@pytest.fixture
def testpit_commonunits():
    return PitFlowCommonUnits(
        drawdown_stab=6,
        cond_h_md=20,
        area=40 * 100,
        recharge_mm_yr=761 * 0.1,
        precipitation_mm_yr=761,
    )


def test_radius_infl(testpit):
    assert math.isclose(testpit.radius_infl, 1088.212649, rel_tol=1e-6)


def test_get_drawdown_at_r_100(testpit):
    assert math.isclose(testpit.get_drawdown_at_r(100), 1.951781, rel_tol=1e-6)


def test_get_drawdown_at_r_neg20(testpit):
    assert testpit.get_drawdown_at_r(-20) == 6


def test_get_drawdown_at_r_1100(testpit):
    assert testpit.get_drawdown_at_r(1100) == 0


def test_get_r_at_drawdown(testpit):
    assert math.isclose(testpit.radius_at_1m, 244.000377, rel_tol=1e-6)


def test_inflow_zone1(testpit):
    assert math.isclose(testpit.inflow_zone1, 0.008962, rel_tol=1e-4)


def test_inflow_zone2(testpit_anisotropic):
    assert math.isclose(testpit_anisotropic.inflow_zone2, 0.062688, rel_tol=1e-5)


def test_radius_infl_commonunits(testpit_commonunits):
    assert math.isclose(testpit_commonunits.radius_infl, 1088.212649)


def test_get_nice_intervals():
    assert (
        get_nice_intervals(174.10374226797754) == np.array([50.0, 100.0, 150.0, 200.0])
    ).all()
