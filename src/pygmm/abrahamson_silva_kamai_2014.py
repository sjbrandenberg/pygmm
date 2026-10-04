"""Abrahamson, Silva, and Kamai (2014, :cite:`abrahamson14`) model."""

import numpy as np

from . import model
from .types import ArrayLike

__author__ = "Albert Kottke"


class AbrahamsonSilvaKamai2014(model.GroundMotionModel):
    """Abrahamson, Silva, and Kamai (2014, :cite:`abrahamson14`) model.

    This model was developed for active tectonic regions as part of the
    NGA-West2 effort.

    The model is vectorized. Each scenario value (``mag``, ``dist_rup``,
    ``dist_jb``, ``dist_x``, ``dist_y0``, ``dist_crjb``, ``v_s30``,
    ``depth_1_0``, ``depth_tor``, ``dip``, ``width``, ``mechanism``,
    ``region``, ``vs_source``, ``is_aftershock``, and ``on_hanging_wall``) can
    be a scalar or an array, and the arrays are broadcast against each other.
    Values that are *None* are estimated as for a scalar scenario (``width``
    from ``mag`` and ``dip``, and ``depth_tor`` from ``mag``). For a scalar
    scenario, the response and standard deviation have one value per period,
    as in other models. For arrays of N scenarios, they have shape
    (N, periods), so, for example, ``pga`` has shape (N,) and ``spec_accels``
    has shape (N, 22).

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario
    ims : str or sequence of str, optional
        intensity measures to compute: "pga", "pgv", spectral periods such as
        "psa_1p000" (1.0 s), "psa_ngawest2_21" (the 21 NGA-West2 comparison
        periods), and/or "psa_all" (all 22 periods). Computing only
        the needed intensity measures is much faster for large vectorized
        scenarios. If *None* (default), all intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> import pygmm
    >>> s = pygmm.Scenario(
    ...     mag=np.array([6.0, 7.0]), dist_rup=np.array([10.0, 50.0]),
    ...     dist_jb=np.array([8.0, 48.0]), dip=90.0, v_s30=760.0,
    ...     mechanism=np.array(["SS", "RS"]))
    >>> pygmm.AbrahamsonSilvaKamai2014(s, ims=["pga"]).pga.shape
    (2,)
    >>> m = pygmm.AbrahamsonSilvaKamai2014(s, ims=["pga", "psa_ngawest2_21"])
    >>> m.spec_accels.shape
    (2, 21)

    """

    NAME = "Abrahamson, Silva, & Kamai (2014)"
    ABBREV = "ASK14"

    # Reference velocity (m/sec)
    V_REF = 1180.0

    # Load the coefficients for the model
    COEFF = model.load_data_file("abrahamson_silva_kamai_2014.csv", 2)

    PERIODS = COEFF["period"]

    INDICES_PSA = np.arange(22)
    INDEX_PGA = -2
    INDEX_PGV = -1

    PARAMS = [
        model.NumericParameter("dist_rup", True, None, 300),
        model.NumericParameter("dist_jb", True),
        model.NumericParameter("mag", True, 3, 8.5),
        model.NumericParameter("v_s30", True, 180, 1000),
        model.NumericParameter("depth_1_0", False),
        model.NumericParameter("depth_tor", False),
        model.NumericParameter("dip", True),
        model.NumericParameter("dist_crjb", False, default=15),
        model.NumericParameter("dist_x", False),
        model.NumericParameter("dist_y0", False),
        model.NumericParameter("width", False),
        model.CategoricalParameter("mechanism", True, ["SS", "NS", "RS"]),
        model.CategoricalParameter(
            "region",
            False,
            ["global", "california", "china", "italy", "japan", "taiwan"],
            "global",
        ),
        model.CategoricalParameter(
            "vs_source", False, ["measured", "inferred"], "measured"
        ),
        model.CategoricalParameter("is_aftershock", False, [True, False], False),
        model.CategoricalParameter("on_hanging_wall", False, [True, False], False),
    ]

    def _check_inputs(self) -> None:
        """Check the inputs."""
        super()._check_inputs()
        s = self._scenario
        if s["width"] is None:
            s["width"] = self.calc_width(s.mag, s.dip)

        if s["depth_tor"] is None:
            s["depth_tor"] = self.calc_depth_tor(s.mag)

    def __init__(self, scenario: model.Scenario, ims=None):
        """Initialize the model.

        Args:
            scenario (:class:`pygmm.model.Scenario`): earthquake scenario.
            ims (str or sequence of str, optional): intensity measures to
                compute. If *None* (default), all intensity measures are
                computed.
        """
        super().__init__(scenario, ims)
        c = self._coeff_rows(self.COEFF)
        # The source and path terms do not depend on the site condition, so
        # they are shared by the reference and site responses
        ln_resp_source = self._calc_ln_resp_source(c)
        # Compute the response at the reference velocity
        resp_ref = np.exp(self._calc_ln_resp(c, ln_resp_source, self.V_REF, np.nan))

        self._ln_resp = self._calc_ln_resp(
            c, ln_resp_source, self._scenario.v_s30, resp_ref
        )
        self._ln_std, self._tau, self._phi = self._calc_ln_std(c, resp_ref)
        # vs_source only affects the standard deviation, so the response and
        # standard deviation are broadcast to the same shape
        shape = np.broadcast_shapes(self._ln_resp.shape, self._ln_std.shape)
        if self._ln_resp.shape != shape:
            self._ln_resp = np.broadcast_to(self._ln_resp, shape).copy()

    def _calc_ln_resp_source(self, c) -> np.ndarray:
        """Calculate the terms of the response that do not depend on the site.

        These are the magnitude scaling, hanging-wall, depth to top of rupture,
        style of faulting, and aftershock terms.

        Parameters
        ----------
        c : :class:`numpy.recarray`
            coefficients for the periods to compute

        Returns
        -------
        ln_resp_source: :class:`np.ndarray`
            sum of the terms of the natural log of the response.

        """
        s = self._scenario
        # Scenario values get a trailing axis to broadcast against the
        # coefficients, which have one value per period
        mag = model.as_column(s.mag)
        mechanism = model.as_column(s.mechanism)

        # Magnitude scaling
        f1 = self._calc_f1(c)

        # Hanging-wall term, which is only applied for sites on the hanging wall
        on_hanging_wall = model.as_column(model.equals(s.on_hanging_wall, True))
        if np.any(on_hanging_wall):
            f4 = np.where(on_hanging_wall, self._calc_f4(c), 0.0)
        else:
            f4 = 0.0

        # Depth to top of rupture term
        f6 = c.a15 * np.clip(model.as_column(s.depth_tor) / 20, 0, 1)
        # Style of faulting. Strike-slip earthquakes have no style of faulting
        # term.
        taper = np.clip(mag - 4, 0, 1)
        f7 = np.where(model.equals(mechanism, "RS"), c.a11 * taper, 0.0)
        f8 = np.where(model.equals(mechanism, "NS"), c.a12 * taper, 0.0)

        # Aftershock term
        is_aftershock = model.as_column(model.equals(s.is_aftershock, True))
        f11 = np.where(
            is_aftershock,
            c.a14 * np.clip(1 - (model.as_column(s.dist_crjb) - 5) / 10, 0, 1),
            0.0,
        )

        return f1 + f4 + f6 + f7 + f8 + f11

    def _calc_ln_resp(
        self, c, ln_resp_source: ArrayLike, v_s30: ArrayLike, resp_ref: ArrayLike
    ) -> np.ndarray:
        """Calculate the natural logarithm of the response.

        Parameters
        ----------
        c : :class:`numpy.recarray`
            coefficients for the periods to compute
        ln_resp_source : array_like
            terms of the natural log of the response that do not depend on the
            site, see :meth:`_calc_ln_resp_source`.
        v_s30 : float or array_like
            site condition. Set `v_s30` to the reference
            velocity (e.g., 1180 m/s) for the reference response.
        resp_ref :  array_like, optional
            response at the reference condition. Required if `v_s30` is not
            equal to reference velocity.

        Returns
        -------
        ln_resp: :class:`np.ndarray`
            natural log of the response.

        """
        s = self._scenario
        v_s30 = model.as_column(v_s30)
        dist_rup = model.as_column(s.dist_rup)
        region = model.as_column(s.region)

        site_term = self._calc_site_term(
            c,
            resp_ref,
            v_s30,
            None if s.depth_1_0 is None else model.as_column(s.depth_1_0),
            s.region if np.ndim(s.region) == 0 else region,
        )

        # Regional terms, which are only computed for the regions in the
        # scenarios. Other regions have no regional term.
        conds = []
        choices = []
        is_taiwan = model.equals(region, "taiwan")
        if np.any(is_taiwan):
            vs_ratio = np.minimum(v_s30, self._calc_v_1(c)) / c.v_lin
            conds.append(is_taiwan)
            choices.append(c.a31 * np.log(vs_ratio) + c.a25 * dist_rup)
        is_china = model.equals(region, "china")
        if np.any(is_china):
            conds.append(is_china)
            choices.append(c.a28 * dist_rup)
        is_japan = model.equals(region, "japan")
        if np.any(is_japan):
            f13 = self._interp_coeffs(
                [150, 250, 350, 450, 600, 850, 1150],
                [c.a36, c.a37, c.a38, c.a39, c.a40, c.a41, c.a42],
                v_s30,
            )
            conds.append(is_japan)
            choices.append(f13 + c.a29 * dist_rup)
        freg = np.select(conds, choices, default=0.0) if conds else 0.0

        return ln_resp_source + freg + site_term

    @classmethod
    def calc_site_term(
        cls,
        resp_ref: ArrayLike,
        v_s30: float,
        depth_1_0: float,
        region: str = "california",
    ):
        """Calculate the site term, which includes site and basin effects.

        Parameters
        ----------
        resp_ref :  array_like, optional
            response at the reference condition
        v_s30 : float or array_like
            site condition. Set `v_s30` to the reference
            velocity (e.g., 1180 m/s) for the reference response. An array of
            values with a trailing axis (shape (N, 1)) gives results with shape
            (N, periods).
        depth_1_0 : float or array_like
            depth to the 1.0 km∕s shear-wave velocity horizon beneath the site,
            :math:`Z_{1.0}` in (km). If *None*, then no basin term is applied.
        region : str or array_like, optional
            region of basin model. Valid options: 'california', 'japan'. If
            *None*, then 'california' is used as the default value.

        Returns
        -------
        site_term: :class:`np.ndarray`
            site term that is applied to the natural log response.
        """
        return cls._calc_site_term(cls.COEFF, resp_ref, v_s30, depth_1_0, region)

    @classmethod
    def _calc_site_term(cls, c, resp_ref, v_s30, depth_1_0, region) -> np.ndarray:
        """Calculate the site term for the periods in the coefficients `c`."""
        vs_ratio = np.minimum(v_s30, cls._calc_v_1(c)) / c.v_lin
        # Linear site model, or nonlinear model for velocities less than v_lin
        f5 = np.where(
            vs_ratio < 1,
            c.a10 * np.log(vs_ratio)
            - c.b * np.log(resp_ref + c.c)
            + c.b * np.log(resp_ref + c.c * vs_ratio**c.n),
            (c.a10 + c.b * c.n) * np.log(vs_ratio),
        )

        # Basin term
        if depth_1_0 is None or np.all(v_s30 == cls.V_REF):
            # No basin response
            f10 = 0.0
        else:
            # Ratio between site depth_1_0 and model center
            ln_depth_ratio = np.log(
                (depth_1_0 + 0.01) / (cls.calc_depth_1_0(v_s30, region) + 0.01)
            )
            slope = cls._interp_coeffs(
                [150, 250, 400, 700], [c.a43, c.a44, c.a45, c.a46], v_s30
            )
            # No basin response at the reference velocity
            f10 = np.where(v_s30 == cls.V_REF, 0.0, slope * ln_depth_ratio)

        return f5 + f10

    @staticmethod
    def _calc_v_1(c) -> np.ndarray:
        """Calculate the period dependent limiting velocity :math:`V_1` (m/s)."""
        return np.exp(-0.35 * np.log(np.clip(c.period, 0.5, 3) / 0.5) + np.log(1500))

    @staticmethod
    def _interp_coeffs(xp: ArrayLike, fps: list, x: ArrayLike) -> np.ndarray:
        """Linearly interpolate coefficients with respect to a scenario value.

        This gives the same values as :func:`scipy.interpolate.interp1d` with the
        end values used outside of `xp`, but broadcasts the scenario values
        against the coefficients, which have one value per period.

        Parameters
        ----------
        xp : array_like
            increasing values at which the coefficients are defined
        fps : list of array_like
            coefficients at each value of `xp`, each with one value per period
        x : float or array_like
            scenario values, with a trailing axis for arrays (see
            :func:`pygmm.model.as_column`)

        Returns
        -------
        values : :class:`np.ndarray`
            interpolated coefficients
        """
        xp = np.asarray(xp)
        x = np.asarray(x)
        # Index of the upper end of the interval, as in interp1d
        hi = np.clip(np.searchsorted(xp, x), 1, len(xp) - 1)
        lo = hi - 1
        fp_lo = np.choose(lo, fps)
        x_lo = xp[lo]
        slope = (np.choose(hi, fps) - fp_lo) / (xp[hi] - x_lo)
        values = slope * (x - x_lo) + fp_lo
        return np.where(x < xp[0], fps[0], np.where(x > xp[-1], fps[-1], values))

    def _calc_ln_std(
        self, c, psa_ref: ArrayLike
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """Calculate the logarithmic standard deviation.

        Parameters
        ----------
        c : :class:`numpy.recarray`
            coefficients for the periods to compute
        psa_ref : array_like
           spectral accelerations at the reference condition

        Returns
        -------
        ln_std :  :class:`np.ndarray`
            logarithmic standard deviation.
        """
        s = self._scenario
        mag = model.as_column(s.mag)
        v_s30 = model.as_column(s.v_s30)

        # Japan has a distance dependent within-event standard deviation, and
        # other regions have a magnitude dependent one that depends on the
        # source of v_s30
        transition = np.clip((mag - 4) / 2, 0, 1)
        phi_al = np.select(
            [
                model.equals(model.as_column(s.region), "japan"),
                model.equals(model.as_column(s.vs_source), "measured"),
            ],
            [
                c.s5
                + (c.s6 - c.s5)
                * np.clip((model.as_column(s.dist_rup) - 30) / 50, 0, 1),
                c.s1m + (c.s2m - c.s1m) * transition,
            ],
            # Inferred (estimated) v_s30
            default=c.s1e + (c.s2e - c.s1e) * transition,
        )

        tau_al = c.s3 + (c.s4 - c.s3) * np.clip((mag - 5) / 2, 0, 1)
        tau_b = tau_al

        # Remove period independent site amplification uncertainty of 0.4
        phi_amp = 0.4
        phi_b = np.sqrt(np.maximum(phi_al**2 - phi_amp**2, 0))

        # The partial derivative of the amplification with respect to
        # the reference intensity, which is zero for the linear site model
        deriv = (-c.b * psa_ref) / (psa_ref + c.c) + (c.b * psa_ref) / (
            psa_ref + c.c * (v_s30 / c.v_lin) ** c.n
        )
        deriv = np.where(v_s30 >= c.v_lin, 0.0, deriv)
        tau = tau_b * (1 + deriv)
        phi = np.sqrt(phi_b**2 * (1 + deriv) ** 2 + phi_amp**2)

        ln_std = np.sqrt(phi**2 + tau**2)
        return ln_std, tau, phi

    @staticmethod
    def calc_width(mag: ArrayLike, dip: ArrayLike) -> ArrayLike:
        r"""Compute the fault width based on equation in NGW2 spreadsheet.

        This equation is not provided in the paper.

        Parameters
        ----------
        mag : float or array_like
            moment magnitude of the event (:math:`M_w`)
        dip : float or array_like
            Fault dip angle (:math:`\phi`, deg)

        Returns
        -------
        width : float or :class:`np.ndarray`
            estimated fault width (:math:`W`, km)
        """
        return np.minimum(18 / np.sin(np.radians(dip)), 10 ** (-1.75 + 0.45 * mag))

    @staticmethod
    def calc_depth_tor(mag: ArrayLike) -> ArrayLike:
        """Calculate the depth to top of rupture (km).

        Parameters
        ----------
        mag : float or array_like
            moment magnitude of the event (:math:`M_w`)

        Returns
        -------
        depth_tor : float or :class:`np.ndarray`
            estimated depth to top of rupture (km)
        """
        return np.interp(mag, [5.0, 7.2], [7.8, 0])

    @staticmethod
    def calc_depth_1_0(v_s30: ArrayLike, region: ArrayLike = "california") -> ArrayLike:
        """Estimate the depth to 1 km/sec horizon (:math:`Z_{1.0}`) based on
        :math:`V_{s30}` and region.

        This is based on equations 18 and 19 in the :cite:`abrahamson14`
        and differs from the equations in the :cite:`chiou14`.

        Parameters
        ----------
        v_s30 : float or array_like
            time-averaged shear-wave velocity over the top 30 m
            of the site (:math:`V_{s30}`, m/s).
        region : str or array_like, optional
            region of basin model. Valid options: 'california', 'japan'. If
            *None*, then 'california' is used as the default value. An array
            of regions is broadcast against `v_s30`.

        Returns
        -------
        depth_1_0 : float or :class:`np.ndarray`
            depth to a shear-wave velocity of 1,000 m/sec
            (:math:`Z_{1.0}`, km).

        """

        def calc(power, v_ref, slope):
            return (
                np.exp(
                    slope
                    * np.log(
                        (v_s30**power + v_ref**power) / (1360.0**power + v_ref**power)
                    )
                )
                / 1000
            )

        # Japan (equation 19) and global (equation 18) relations
        japan = (2, 412, -5.23 / 2)
        world = (4, 610, -7.67 / 4)
        if np.ndim(region) == 0:
            return calc(*japan) if region in ["japan"] else calc(*world)
        is_japan = model.equals(region, "japan")
        return np.where(is_japan, calc(*japan), calc(*world))

    def _calc_f1(self, c) -> np.ndarray:
        """Calculate the magnitude scaling parameter f1."""
        s = self._scenario
        mag = model.as_column(s.mag)
        dist_rup = model.as_column(s.dist_rup)

        # Magnitude dependent taper
        dist = np.sqrt(dist_rup**2 + (c.c4 - (c.c4 - 1) * np.clip(5 - mag, 0, 1)) ** 2)

        # Magnitude scaling for magnitudes less than or equal to m2, and
        # greater than m2
        small = mag <= c.m2
        f1 = c.a1 + np.where(
            small,
            c.a4 * (c.m2 - c.m1)
            + c.a8 * (8.5 - c.m2) ** 2
            + c.a6 * (mag - c.m2)
            + c.a7 * (mag - c.m2) ** 2
            + (c.a2 + c.a3 * (c.m2 - c.m1)) * np.log(dist)
            + c.a17 * dist_rup,
            c.a8 * (8.5 - mag) ** 2
            + (c.a2 + c.a3 * (mag - c.m1)) * np.log(dist)
            + c.a17 * dist_rup,
        )

        # Linear magnitude scaling for magnitudes greater than m2, with a
        # change of slope at m1
        f1 = f1 + np.where(
            small, 0.0, np.where(mag <= c.m1, c.a4 * (mag - c.m1), c.a5 * (mag - c.m1))
        )

        return f1

    def _calc_f4(self, c) -> np.ndarray:
        """Calculate the hanging-wall parameter f4."""
        s = self._scenario
        if s.dist_x is None:
            raise ValueError(
                "dist_x is required for sites on the hanging wall (on_hanging_wall)"
            )
        mag = model.as_column(s.mag)
        dip = model.as_column(s.dip)
        dist_x = model.as_column(s.dist_x)

        t1 = np.minimum(90 - dip, 60) / 45
        # Constant from page 1041
        a2hw = 0.2
        t2 = np.select(
            [mag <= 5.5, mag < 6.5],
            [0.0, 1 + a2hw * (mag - 6.5) - (1 - a2hw) * (mag - 6.5) ** 2],
            default=1 + a2hw * (mag - 6.5),
        )

        # Constants defined on page 1040
        r1 = model.as_column(s.width) * np.cos(np.radians(dip))
        r2 = 3 * r1
        h1 = 0.25
        h2 = 1.5
        h3 = -0.75
        t3 = np.select(
            [dist_x < r1, dist_x < r2],
            [
                h1 + h2 * (dist_x / r1) + h3 * (dist_x / r1) ** 2,
                1 - ((dist_x - r1) / (r2 - r1)),
            ],
            default=0.0,
        )

        t4 = np.clip(1 - model.as_column(s.depth_tor) ** 2 / 100, 0, 1)

        if s.dist_y0 is None:
            t5 = np.clip(1 - model.as_column(s.dist_jb) / 30, 0, 1)
        else:
            dist_y1 = dist_x * np.tan(np.radians(20))
            t5 = np.clip(1 - (model.as_column(s.dist_y0) - dist_y1) / 5, 0, 1)

        f4 = c.a13 * t1 * t2 * t3 * t4 * t5

        return f4
