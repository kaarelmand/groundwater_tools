from collections.abc import MutableMapping
from functools import cached_property
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

MINUTE_TO_SEC = 60
HOUR_TO_SEC = 60 * MINUTE_TO_SEC
DAY_TO_SEC = 24 * HOUR_TO_SEC
YEAR_TO_SEC = 365.25 * DAY_TO_SEC
QUARTER_TO_SEC = YEAR_TO_SEC / 4
M_TO_MM = 1000
M3_TO_L = 1000


class PitFlow:
    """All parameters are expected to be in meters and seconds.
    anisotropy defaults to 1 (i.e. sandstone), limestones can be 0.1. Only used for
        calculating bottom flow, in any case.
    h_edge=0 is conservative, hence default.
    """

    _inputs_info = {
        "drawdown_stab": ("Drawdown in pit center", "m"),
        "drawdown_edge": ("Height of inflow at pit edge", "m"),
        "depth_pitlake": ("Water depth in the pit lake", "m"),
        "area": ("Area of pit", "m^2"),
        "recharge": ("Recharge", "m/sec"),
        "precipitation": ("Precipitation", "m/sec"),
        "period_snow_accumulation": ("Duration of snow accumulation in Winter", "sec"),
        "period_melting": ("Duration of snow melt in Spring", "sec"),
        "cond_h": ("Hydraulic conductivity (horizontal)", "m/sec"),
        "anisotropy": ("Anisotropy of hydraulic conductivity", ""),
    }
    _outputs_info = {
        "radius_eff": ("Effective radius of pit", "m"),
        "radius_infl": ("Radius of influence from pit centre", "m"),
        "radius_infl_from_edge": ("Radius of influence from pit edge", "m"),
        "radius_at_1m": ("Radius at drawdown of 1 m, from pit edge", "m"),
        "inflow_zone1": ("Zone 1 inflow", "m^3/sec"),
        "inflow_zone2": ("Zone 2 inflow", "m^3/sec"),
        "inflow_zones_both": ("Zone 1 and 2 inflow", "m^3/sec"),
        "inflow_precipitation": ("Inflow from precipitation in pit", "m^3/sec"),
        "inflow_precipitation_zone1": (
            "Inflow from zone 1 and precipitation in pit",
            "m^3/sec",
        ),
        "inflow_meltwater": ("Inflow from meltwater", "m^3/sec"),
        "inflow_meltwater_zone1": ("Inflow from zone 1 and meltwater", "m^3/sec"),
    }

    def __init__(
        self,
        drawdown_stab,
        area,
        recharge,
        precipitation,
        cond_h,
        anisotropy=1,
        drawdown_edge=0,
        depth_pitlake=0,
        period_snow_accumulation=90 * DAY_TO_SEC,
        period_melting=14 * DAY_TO_SEC,
    ):
        self.drawdown_stab = drawdown_stab
        self.drawdown_edge = drawdown_edge
        self.depth_pitlake = depth_pitlake
        self.area = area
        self.recharge = recharge
        self.precipitation = precipitation
        self.period_snow_accumulation = period_snow_accumulation
        self.period_melting = period_melting
        self.cond_h = cond_h
        self.anisotropy = anisotropy

    def _get_marinelli_niccoli_h_0(self, radius_infl):
        """Marinelli and Niccoli 2000 formula describing horizontal groundwater flow
        into pit. radius_infl will be correct if return value is 0."""
        radius_infl = radius_infl.item()
        radius_term = (
            radius_infl**2 * np.log(radius_infl / self.radius_eff)
            - (radius_infl**2 - self.radius_eff**2) / 2
        )
        right_term = np.sqrt(
            self.drawdown_edge**2 + (self.recharge / self.cond_h) * radius_term
        )
        return self.drawdown_stab - right_term

    # TODO: test
    @cached_property
    def radius_eff(self):
        """Find effective (circular) radius using area (m)."""
        return np.sqrt(self.area / np.pi)

    @cached_property
    def radius_infl(self, radius_start=10000):
        """Find optimum drawdown radius (m) through the Marinelli and Niccoli 2000
        formula."""
        return optimize.fsolve(func=self._get_marinelli_niccoli_h_0, x0=radius_start)[0]

    @cached_property
    def radius_infl_from_edge(self):
        """Find optimum drawdown radius (m) from the edge of the pit, not the center."""
        return self.radius_infl - self.radius_eff

    def get_drawdown_at_r(self, radius_from_wall):
        """Return drawdown at length `radius_from_wall` according to Marinelli &
        Niccoli (2000)."""
        if radius_from_wall < 0:
            return self.drawdown_stab
        elif radius_from_wall > self.radius_infl_from_edge:
            return 0
        else:
            radius = radius_from_wall + self.radius_eff
            radius_term = (
                self.radius_infl**2 * np.log(radius / self.radius_eff)
                - (radius**2 - self.radius_eff**2) / 2
            )
            sqrt_term = np.sqrt(
                self.drawdown_edge**2 + (self.recharge / self.cond_h) * radius_term
            )
            return self.drawdown_stab - sqrt_term

    def _balance_drawdown_threshold(self, radius_from_wall, threshold=1):
        """Helper function to find radius where drawdown is at threshold."""
        return self.get_drawdown_at_r(radius_from_wall[0]) - threshold

    def get_r_at_drawdown(self, drawdown, x0=10):
        """Return radius at which drawdown equals given threshold.
        Unfortunately, has to be iterative as analytical solution is tough."""
        return optimize.fsolve(
            func=self._balance_drawdown_threshold,
            x0=x0,
            args=(drawdown),
        )[0]

    @cached_property
    def radius_at_1m(self):
        """Radius from wall at which estimated water table drawdown is 1 m."""
        return self.get_r_at_drawdown(1)

    @cached_property
    def inflow_zone1(self):
        """Horizontal inflow into pit from zone 1, i.e. pit walls (m^3/s)."""
        return self.recharge * np.pi * (self.radius_infl**2 - self.radius_eff**2)

    @cached_property
    def inflow_zone2(self):
        """Vertical inflow into pit from zone 2, i.e. pit bottom (m^3/sec)."""
        anisotropy_term = np.sqrt(self.cond_h / (self.cond_h * self.anisotropy))
        return (
            4
            * self.radius_eff
            * (self.cond_h / anisotropy_term)
            * (self.drawdown_stab - self.depth_pitlake)
        )

    # TODO: test
    @cached_property
    def inflow_zones_both(self):
        """Combined inflow into pit from zones 1 and 2, i.e. pit walls and bottom
        (m^3/sec)."""
        return self.inflow_zone1 + self.inflow_zone2

    # TODO: test
    @cached_property
    def inflow_precipitation(self):
        """Inflow from precipitation into the pit (m^3/sec)."""
        return self.precipitation * self.area

    # TODO: test
    @cached_property
    def inflow_precipitation_zone1(self):
        """Inflow from precipitation into the pit and zone 1, i.e. pit walls
        (m^3/sec)."""
        return self.inflow_precipitation + self.inflow_zone1

    # TODO: test
    @cached_property
    def inflow_meltwater(self):
        """Inflow from meltwater in spring (m^3/sec)."""
        return (
            self.precipitation
            * self.period_snow_accumulation
            / self.period_melting
            * self.area
        )

    # TODO: test
    @cached_property
    def inflow_meltwater_zone1(self):
        """Inflow from meltwater in spring and zone 1, i.e. pit walls (m^3/sec)."""
        return self.inflow_meltwater + self.inflow_zone1

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
        radii = np.linspace(*lims, 1000)
        drawdowns = [self.get_drawdown_at_r(r) for r in radii]
        return ax.plot(radii, drawdowns, **kwargs)

    # TODO: test conversions
    def report(self, drawdown_points=None, volume="m^3", rate="sec", sort="False"):
        """pandas.DataFrame report of model inputs and commonly-needed model
        outputs."""
        report_table = (
            pd.DataFrame(self._inputs_info | self._outputs_info)
            .T.reset_index()[[0, "index", 1]]
            .rename(columns={0: "Parameter", 1: "Unit", "index": "Value"})
        )
        report_table["Value"] = report_table["Value"].apply(lambda x: getattr(self, x))

        if drawdown_points is None:
            drawdown_points = get_nice_intervals(self.radius_infl_from_edge)
        drawdown_rows = pd.DataFrame(
            (
                [
                    f"Drawdown at {point} m from pit wall",
                    self.get_drawdown_at_r(point),
                    "m",
                ]
                for point in drawdown_points
            ),
            columns=["Parameter", "Value", "Unit"],
        )
        report_table = pd.concat([report_table, drawdown_rows]).reset_index(drop=True)

        if volume == "m^3":
            pass
        elif volume == "l":
            report_table = unit_convert(report_table, "m^3", "l", M3_TO_L)
        else:
            raise (
                ValueError(
                    f"'{volume}' not recognized as value for `volume`."
                    "Needs to be one of 'm^3' or 'l'."
                )
            )

        if rate == "sec":
            pass
        elif rate == "hr":
            report_table = unit_convert(report_table, "sec", "hr", HOUR_TO_SEC)
        elif rate == "day":
            report_table = unit_convert(report_table, "sec", "day", DAY_TO_SEC)
        elif rate == "quarter":
            report_table = unit_convert(report_table, "sec", "quarter", QUARTER_TO_SEC)
        elif rate == "yr":
            report_table = unit_convert(report_table, "sec", "yr", YEAR_TO_SEC)
        else:
            raise (
                ValueError(
                    f"'{rate}' not recognized as value for `time`."
                    "Needs to be one of 'sec', 'hr', 'day', 'quarter', or 'yr'."
                )
            )

        return report_table

    def __repr__(self):
        fmt_output = f"{self.__class__.__name__}("
        for param in self._inputs_info.keys():
            fmt_output += f"\n{param}={getattr(self, param)},"
        return fmt_output + "\n)"

    def __str__(self):
        fmt_output = f"{self.__class__.__name__}:"
        for param, (name, unit) in self._inputs_info.items():
            fmt_output += f"\n{name}: {getattr(self, param)} {unit}"
        return fmt_output

    def _repr_html_(self):
        fmt_output = (
            '<table border="1">'
            f'<tr><th colspan="3">{self.__class__.__name__}</th></tr>'
        )
        for param, (name, unit) in self._inputs_info.items():
            fmt_output += (
                f"\n<tr><td>{name}</td>"
                f"<td>{getattr(self, param):.6f}</td>"
                f"<td>{unit}</td></tr>"
            )
        return fmt_output + "</table>"


class PitFlowCommonUnits(PitFlow):
    """Same as PitFlow but expects input in more convenient forms:
    `recharge_mm_yr` in mm/yr
    `precipitation_mm_yr` in mm/yr
    `period_snow_accumulation_d` in days
    `period_melting_d` in days
    `cond_h_md` (horizontal hydraulic conductivity) in m/day
    """

    _inputs_info = dict(PitFlow._inputs_info)
    for param in [
        "recharge",
        "precipitation",
        "period_snow_accumulation",
        "period_melting",
        "cond_h",
    ]:
        _ = _inputs_info.pop(param)
    _inputs_info.update(
        {
            "recharge_mm_yr": ("Recharge", "mm/yr"),
            "precipitation_mm_yr": ("Precipitation", "mm/yr"),
            "period_snow_accumulation_d": (
                "Duration of snow accumulation in Winter",
                "day",
            ),
            "period_melting_d": ("Duration of snow melt in Spring", "day"),
            "cond_h_md": ("Hydraulic conductivity (horizontal)", "m/day"),
        }
    )

    def __init__(
        self,
        drawdown_stab,
        area,
        recharge_mm_yr,
        precipitation_mm_yr,
        cond_h_md,
        anisotropy=1,
        drawdown_edge=0,
        depth_pitlake=0,
        period_snow_accumulation_d=90,
        period_melting_d=14,
    ):
        self.recharge_mm_yr = recharge_mm_yr
        self.precipitation_mm_yr = precipitation_mm_yr
        self.period_snow_accumulation_d = period_snow_accumulation_d
        self.period_melting_d = period_melting_d
        self.cond_h_md = cond_h_md
        super().__init__(
            drawdown_stab=drawdown_stab,
            drawdown_edge=drawdown_edge,
            depth_pitlake=depth_pitlake,
            area=area,
            recharge=recharge_mm_yr / YEAR_TO_SEC / M_TO_MM,
            precipitation=precipitation_mm_yr / YEAR_TO_SEC / M_TO_MM,
            period_snow_accumulation=self.period_snow_accumulation_d * DAY_TO_SEC,
            period_melting=self.period_melting_d * DAY_TO_SEC,
            cond_h=cond_h_md / DAY_TO_SEC,
            anisotropy=anisotropy,
        )

    def __repr__(self):
        fmt_output = f"{self.__class__.__name__}("
        for param in self._inputs_info.keys():
            if param not in [
                "recharge",
                "precipitation",
                "period_snow_accumulation",
                "period_melting",
                "cond_h",
            ]:
                fmt_output += f"\n{param}={getattr(self, param)},"
        return fmt_output + "\n)"


class PitFlowCollection(MutableMapping):
    """Methods to show and collect different drawdown scenarios.
    Pass any of the parameters needed for PitFlow as either a
    list of parameters or a single numerical parameter. A PitFlow instance is added per
    each item in the product of all lists. Returns a dict-like PitFlowCollection
    object, where names are attributes that change."""

    def __init__(self, flowclass=PitFlow, **kwargs):
        multiples = {}
        singles = {}
        for name, val in kwargs.items():
            if type(val) is list:
                multiples[name] = val
            else:
                singles[name] = val
        for arg_lst in product(*multiples.values()):
            instance_params = {
                param: val for param, val in zip(multiples.keys(), arg_lst)
            }
            name_lst = [(f"{param} = {val}") for param, val in instance_params.items()]
            flow_name = ", ".join(name_lst)
            self.__dict__[flow_name] = flowclass(**instance_params, **singles)

    def __setitem__(self, key, value):
        if type(value) != PitFlow:
            raise TypeError(
                f"Attempted to set a `{type(value)}` as value."
                "Values must be of type `PitFlow`."
            )
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def draw_drawdown_curves(self, ax, line_buffer=(0.15, 1.5), lims=None, **kwargs):
        if not lims:
            max_infl = max(
                flow.radius_infl_from_edge for flow in self.__dict__.values()
            )
            lims = (-(max_infl * line_buffer[0]), max_infl * line_buffer[1])
        return [
            flow.draw_drawdown_curve(ax, lims=lims, label=name, **kwargs)
            for name, flow in self.__dict__.items()
        ]

    def draw_drawdown_figure(
        self,
        ylabel="Water table drawdown, m",
        xlabel="Distance from pit edge, m",
        axkwargs=dict(),
        linekwargs=dict(),
    ):
        fig, ax = plt.subplots(**axkwargs)
        self.draw_drawdown_curves(ax, **linekwargs)
        ax.invert_yaxis()
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax

    def report(self, drawdown_points=None, **kwargs):
        if drawdown_points is None:
            max_infl = max(
                flow.radius_infl_from_edge for flow in self.__dict__.values()
            )
            drawdown_points = get_nice_intervals(max_infl)

        reports = []
        for i, flow in enumerate(self.__dict__.values()):
            single_report = flow.report(drawdown_points=drawdown_points, **kwargs).T
            single_report.columns = single_report.loc["Parameter"]
            single_report.columns.name = ""
            single_report.drop("Parameter", inplace=True)
            if i == 0:
                single_report = single_report.reindex(["Unit", "Value"])
            else:
                single_report.drop("Unit", inplace=True)
            reports.append(single_report)
        fullreport = pd.concat(reports).reset_index(drop=True)
        return fullreport.rename(index={0: "Units"})


class PitFlowCommonUnitsCollection(PitFlowCollection):
    def __init__(self, **kwargs):
        super().__init__(self, flowclass=PitFlowCommonUnits, **kwargs)


def get_nice_intervals(end, num_bounds=(4, 8)):
    """Return rounded intervals from 0 to `end`."""
    mag = np.floor(np.log10(end))
    digit1 = end // 10**mag % 10
    digit2 = end // 10 ** (mag - 1) % 10

    if digit2 >= 5:
        max = (digit1 + 1) * 10**mag
    elif digit2 < 5:
        max = (digit1 + 0.5) * 10**mag

    num = int(max / (0.5 * 10**mag))
    if num > num_bounds[1]:
        if digit2 < 5:
            max += 0.5 * 10**mag
        num = int(np.ceil(num / 2))
    elif num < num_bounds[0]:
        num *= 2
    return np.linspace(0, max, num + 1)[1:]


# TODO: test
def unit_convert(df, unit_from, unit_to, coef, unit_col="Unit", value_col="Value"):
    """Helper function to convert the PitFlow.report() table units.

    Args:
        df (pd.DataFrame): DataFrame to convert.
        unit_from (str): unit as exists in the DataFrame.
        unit_to (str): unit to be converted to.
        coef (float, int): conversion factor.
        unit_col (str, optional): Column name that stores units. Defaults to "Unit".
        value_col (str, optional): Column name that stores values. Defaults to "Value".

    Returns:
        pd.DataFrame: DataFrame with converted values.
    """
    convert_mask = df[unit_col].str.contains(unit_from, regex=False)
    df.loc[convert_mask, value_col] = df.loc[convert_mask, value_col] * coef
    df.loc[convert_mask, unit_col] = df.loc[convert_mask, unit_col].str.replace(
        unit_from, unit_to
    )
    return df
