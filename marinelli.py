import math
import numpy as np
from scipy import optimize


class PitFlow:
    """All parameters are excpected to be in meters and seconds.
    anisotropy defaults to 1 (i.e. sandstone), limestones can be 0.1. Only used for
        calculating bottom flow, in any case.
    h_edge=0 is conservative, hence default.
    """

    def __init__(
        self,
        drawdown_stab,
        trans_h,
        radius_eff,
        recharge,
        anisotropy=1,
        drawdown_edge=0,
    ):
        self.drawdown_stab = drawdown_stab
        self.trans_h = trans_h
        self.radius_eff = radius_eff
        self.recharge = recharge
        self.drawdown_edge = drawdown_edge
        self.anisotropy = anisotropy
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
            self.drawdown_edge**2 + (self.recharge / self.trans_h) * radius_term
        )
        return self.drawdown_stab - right_term

    def get_depression_radius(self, radius_start=10000):
        """Find optimum depression radius through the Marinelli and Niccoli 2000
        formula."""
        return optimize.fsolve(func=self.get_marinelli_niccoli_h_0, x0=radius_start)[0]

    def get_depression_at_r(self, radius_from_wall):
        """Return depression at length `radius_from_wall` according to Marinelli &
        Niccoli (2000)."""
        if radius_from_wall < 0:
            return self.drawdown_stab
        elif radius_from_wall > self.radius_infl_from_edge:
            return 0
        else:
            radius = radius_from_wall + self.radius_eff
            radius_term = (
                self.radius_infl**2 * math.log(radius / self.radius_eff)
                - (radius**2 - self.radius_eff**2) / 2
            )
            sqrt_term = math.sqrt(
                self.drawdown_edge**2 + (self.recharge / self.trans_h) * radius_term
            )
            return self.drawdown_stab - sqrt_term

    def balance_depression_threshold(self, radius_from_wall, threshold=1):
        """Helper function to find radius where depression is at threshold."""
        return self.get_depression_at_r(radius_from_wall[0]) - threshold

    def get_significant_radius(self, significant_threshold=1, x0=10):
        """Return radius at which depression equals given threshold (default = 1 m).
        Unfortunately, has to be iterative as analytical solution is tough."""
        return optimize.fsolve(
            func=self.balance_depression_threshold,
            x0=x0,
            args=(significant_threshold),
        )[0]

    def draw_drawdown_curve(self, ax, line_buffer=(0.15, 1.5), lims=None, **kwargs):
        """Draw the groundwater drawdown curve on an existing matplotlib axes.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes on which to draw the curve.
            line_buffer (tuple, optional): Tuple stating how far to extend the curve
                back from the pit wall and forward past the radius of influence, as
                relative to the radius of influence. Defaults to (0.15, 1.5).
            lims (tuple or None, optional): If present, a tuple that overrides
                `line_buffer` with absolute values. Defaults to None.

        Returns:
            list of matplotlib.lines.Line2D: a list of lines on the specified matplotlib
                axes.
        """
        if not lims:
            lims = (
                -(self.radius_infl_from_edge * line_buffer[0]),
                self.radius_infl_from_edge * line_buffer[1],
            )
        radiuses = np.linspace(*lims, 1000)
        drawdowns = [self.get_depression_at_r(r) for r in radiuses]
        return ax.plot(radiuses, drawdowns, **kwargs)

    # TODO: test
    def get_zone1_inflow(self):
        """Get horizontal inflow from zone 1, i.e. pit walls.

        Returns:
            float: Inflow in m3/sec.
        """
        return self.recharge * math.pi * (self.radius_infl**2 - self.radius_eff**2)

    # TODO: test
    def get_zone2_inflow(self, pit_lake_depth=0):
        """Get vertical inflow from zone 2, i.e. pit bottom.

        Args:
            pit_lake_depth (float, optional): Depth of the water column overlying the
                pit bottom. Defaults to 0.

        Returns:
            float: Inflow in m3/sec.
        """
        anisotropy_term = math.sqrt(self.trans_h / (self.trans_h * self.anisotropy))
        return (
            4
            * self.radius_eff
            * (self.trans_h / anisotropy_term)
            * (self.drawdown_stab - pit_lake_depth)
        )


class PitFlowCommonUnits(PitFlow):
    """Same as PitFlow but expects input in more convenient forms:
    `trans_h_md` (horizontal transmissivity) in m/d
    `area` (total pit area) in m^2
    `precipitation` in mm/yr
    `infil_coef=0.1` is a typical value for most of Estonia.
    """

    def __init__(
        self,
        drawdown_stab,
        trans_h_md,
        anisotropy,
        area,
        precip,
        drawdown_edge=0,
        infil_coef=0.1,
    ):
        self.trans_h_md = trans_h_md
        self.area = area
        self.precip = precip
        self.infil_coef = infil_coef
        super().__init__(
            drawdown_stab=drawdown_stab,
            trans_h=self.transmissivity_to_m_sec(),
            anisotropy=anisotropy,
            radius_eff=self.get_effective_radius(),
            recharge=self.get_recharge(),
            drawdown_edge=drawdown_edge,
        )

    def transmissivity_to_m_sec(self):
        """Unit conversion of transmissivity from m/d to m/s."""
        return self.trans_h_md / (24 * 60 * 60)

    def get_effective_radius(self):
        """Return ideal circularized radius from true quarry area."""
        return math.sqrt(self.area / math.pi)

    def get_recharge(self):
        """Return recharge in m/sec, input needs to be in mm/yr. infil_coef=0.1 is
        typical for Estonia."""
        precip_m_sec = self.precip / (1000 * 365.25 * 24 * 60 * 60)
        return precip_m_sec * self.infil_coef
