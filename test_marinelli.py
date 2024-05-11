import pytest
import math

from marinelli import PitFlow, PitFlowCommonUnits


def test_get_depression_radius():
    testpit = PitFlow(
        drawdown_stab=6,
        trans=20 / (24 * 60 * 60),
        radius_eff=math.sqrt(40 * 100 / math.pi),
        recharge=761 / (1000 * 365.25 * 24 * 60 * 60) * 0.1,
    )
    assert math.isclose(testpit.radius_infl, 1088.212649)


def test_get_depression_radius_commonunits():
    testpit = PitFlowCommonUnits(
        drawdown_stab=6,
        trans_md=20,
        area=40 * 100,
        precip=761,
    )
    assert math.isclose(testpit.radius_infl, 1088.212649)
