import math
from scipy import optimize


# TODO: integration test
class PitFlow:
    """All internal values are in meters and seconds."""

    def __init__(self, drawdown_stab, trans, radius_eff, recharge, drawdown_edge=0):
        """h_edge=0 is conservative, hence default."""
        self.drawdown_stab = drawdown_stab
        self.trans = trans
        self.radius_eff = radius_eff
        self.recharge = recharge
        self.drawdown_edge = drawdown_edge
        self.radius_infl = self.get_depression_radius()
        self.radius_infl_from_edge = self.radius_infl - self.radius_eff

    def get_marinelli_niccoli_h_0(self, radius_infl):
        """Marinelli and Niccoli 2000 formula describing horizontal groundwater flow
        into pit. radius_infl will be correct if return value is 0."""
        radius_infl = radius_infl.item()
        radius_term = (
            radius_infl**2 * math.log(radius_infl / self.radius_eff)
            - (radius_infl**2 - self.radius_eff**2) / 2
        )
        right_term = math.sqrt(
            self.drawdown_edge**2 + (self.recharge / self.trans) * radius_term
        )
        return self.drawdown_stab - right_term

    # TODO: test
    def get_depression_radius(self, radius_start=10000):
        """Find optimum depression radius through the Marinelli and Niccoli 2000
        formula."""
        return optimize.fsolve(func=self.get_marinelli_niccoli_h_0, x0=radius_start)[0]

    # TODO: test
    def get_depression_at_r(self, radius_from_wall):
        """Return depression at length `radius_from_wall` according to Marinelli &
        Niccoli (2000)."""
        if radius_from_wall < 0:
            return -self.drawdown_stab
        elif radius_from_wall > self.radius_infl - self.radius_eff:
            return 0
        else:
            radius = radius_from_wall + self.radius_eff
            radius_term = (
                self.radius_infl**2 * math.log(radius / self.radius_eff)
                - (radius**2 - self.radius_eff**2) / 2
            )
            sqrt_term = math.sqrt(
                self.drawdown_edge**2 + (self.recharge / self.trans) * radius_term
            )
            return sqrt_term - self.drawdown_stab


# TODO: integration test
class PitFlowCommonUnits(PitFlow):
    """Same as PitFlow but expects input in more convenient forms:
    `trans` (horizontal transmissivity) in m/d
    `area` (total pit area) in m^2
    `precipitation` in mm/yr
    `infil_coef=0.1` is a typical value for most of Estonia.
    """

    def __init__(
        self, drawdown_stab, trans_md, area, precip, drawdown_edge=0, infil_coef=0.1
    ):
        self.trans_md = trans_md
        self.area = area
        self.precip = precip
        self.infil_coef = infil_coef
        super().__init__(
            drawdown_stab=drawdown_stab,
            trans=self.transmissivity_to_m_sec(),
            radius_eff=self.get_effective_radius(),
            recharge=self.get_recharge(),
            drawdown_edge=drawdown_edge,
        )

    def transmissivity_to_m_sec(self):
        """Unit conversion of transmissivity from m/d to m/s."""
        return self.trans_md / (24 * 60 * 60)

    def get_effective_radius(self):
        """Return ideal circularized radius from true quarry area."""
        return math.sqrt(self.area / math.pi)

    def get_recharge(self):
        """Return recharge in m/sec, input needs to be in mm/yr. infil_coef=0.1 is typical
        for Estonia."""
        precip_m_sec = self.precip / (1000 * 365.25 * 24 * 60 * 60)
        return precip_m_sec * self.infil_coef


pit2 = PitFlowCommonUnits(6, 20, 40 * 100, 761)
print(pit2.radius_infl)
