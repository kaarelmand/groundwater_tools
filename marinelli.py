import math
from scipy import optimize


class PitFlow:
    """All internal values are in meters and seconds."""

    def __init__(self, table_stab, trans, radius_eff, recharge, table_edge=0):
        """h_edge=0 is conservative, hence default."""
        self.table_stab = table_stab
        self.trans = trans
        self.radius_eff = radius_eff
        self.recharge = recharge
        self.table_edge = table_edge
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
            self.table_edge**2 + (self.recharge / self.trans) * radius_term
        )
        return self.table_stab - right_term

    def get_depression_radius(self, radius_start=10000):
        """Find optimum depression radius through the Marinelli and Niccoli 2000
        formula."""
        return optimize.fsolve(func=self.get_marinelli_niccoli_h_0, x0=radius_start)[0]


class PitFlowCommonUnits(PitFlow):
    """Same as PitFlow but expects input in more convenient forms:
    `trans` (horizontal transmissivity) in m/d
    `area` (total pit area) in m^2
    `precipitation` in mm/yr
    `infil_coef=0.1` is a typical value for most of Estonia.
    """

    def __init__(self, table_stab, trans, area, precip, table_edge=0, infil_coef=0.1):
        self.trans = trans
        self.area = area
        self.precip = precip
        self.infil_coef = infil_coef
        super().__init__(
            table_stab=table_stab,
            trans=self.transmissivity_to_m_sec(),
            radius_eff=self.get_effective_radius(),
            recharge=self.get_recharge(),
            table_edge=table_edge,
        )

    def transmissivity_to_m_sec(self):
        """Unit conversion of transmissivity from m/d to m/s."""
        return self.trans / (24 * 60 * 60)

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
