"""Boore, Stewart, Seyhan, and Atkinson (2014) ground motion model."""

import logging
from typing import Optional

import numpy as np

from . import model
from .chiou_youngs_2014 import ChiouYoungs2014 as CY14
from .types import ArrayLike

__author__ = "Albert Kottke"


class BooreStewartSeyhanAtkinson2014(model.GroundMotionModel):
    """Boore, Stewart, Seyhan, and Atkinson (2014, :cite:`boore14`) model.

    This model was developed for active tectonic regions as part of the
    NGA-West2 effort.

    The BSSA14 model defines the following distance attenuation models:

        +--------------+-------------------------------+
        | Name         | Description                   |
        +--------------+-------------------------------+
        | global       | Global; California and Taiwan |
        +--------------+-------------------------------+
        | china_turkey | China and Turkey              |
        +--------------+-------------------------------+
        | italy_japan  | Italy and Japan               |
        +--------------+-------------------------------+

    and the following basin region models:

        +--------+---------------------+
        | Name   | Description         |
        +========+=====================+
        | global | Global / California |
        +--------+---------------------+
        | japan  | Japan               |
        +--------+---------------------+

    These are simplified into one regional parameter with the following
    possibilities:

        +-------------+--------------+------------+
        | Region      | Attenuation  | Basin      |
        +=============+==============+============+
        | global      | global       | global     |
        +-------------+--------------+------------+
        | california  | global       | global     |
        +-------------+--------------+------------+
        | china       | china_turkey | global     |
        +-------------+--------------+------------+
        | italy       | italy_japan  | global     |
        +-------------+--------------+------------+
        | japan       | italy_japan  | japan      |
        +-------------+--------------+------------+
        | new zealand | italy_japan  | global     |
        +-------------+--------------+------------+
        | taiwan      | global       | global     |
        +-------------+--------------+------------+
        | turkey      | china_turkey | global     |
        +-------------+--------------+------------+

    The model is vectorized. Each scenario value (``mag``, ``dist_jb``,
    ``v_s30``, ``depth_1_0``, ``mechanism``, and ``region``) can be a scalar or
    an array, and the arrays are broadcast against each other. For a scalar
    scenario, the response and standard deviation have one value per period,
    as in other models. For arrays of N scenarios, they have shape
    (N, periods), so, for example, ``pga`` has shape (N,).

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario
    ims : str or sequence of str, optional
        intensity measures to compute: "pga", "pgv", spectral periods such as
        "psa_1p000" (1.0 s), "psa_ngawest2_21" (the 21 NGA-West2 comparison
        periods), and/or "psa_all" (all 105 periods). Computing only
        the needed intensity measures is much faster for large vectorized
        scenarios. If *None* (default), all intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> import pygmm
    >>> s = pygmm.Scenario(
    ...     mag=np.array([6.0, 7.0]), dist_jb=np.array([10.0, 50.0]),
    ...     v_s30=300.0, mechanism=np.array(["SS", "RS"]))
    >>> pygmm.BooreStewartSeyhanAtkinson2014(s, ims=["pga"]).pga.shape
    (2,)
    >>> m = pygmm.BooreStewartSeyhanAtkinson2014(s, ims=["pga", "psa_ngawest2_21"])
    >>> m.spec_accels.shape
    (2, 21)

    """

    NAME = "Boore, Stewart, Seyhan, and Atkinson (2014)"
    ABBREV = "BSSA14"

    # Reference shear-wave velocity in m/sec
    V_REF = 760.0

    # Load the coefficients for the model
    COEFF = model.load_data_file("boore_stewart_seyhan_atkinson-2014.csv", 2)
    PERIODS = COEFF["period"]

    INDEX_PGV = 0
    INDEX_PGA = 1
    INDICES_PSA = np.arange(2, 107)

    LIMITS = dict(
        mag=(3.0, 8.5),
        dist_jb=(0.0, 300.0),
        v_s30=(150.0, 1500.0),
    )

    PARAMS = [
        model.NumericParameter("mag", True, 3, 8.5),
        model.NumericParameter("depth_1_0", False),
        model.NumericParameter("dist_jb", True, None, 300.0),
        model.NumericParameter("v_s30", True, 150.0, 1500.0),
        model.CategoricalParameter("mechanism", False, ["U", "SS", "NS", "RS"], "U"),
        model.CategoricalParameter(
            "region",
            False,
            [
                "global",
                "california",
                "china",
                "italy",
                "japan",
                "new_zealand",
                "taiwan",
                "turkey",
            ],
            "global",
        ),
    ]

    def __init__(self, scenario: model.Scenario, ims=None):
        """Initialize the model.

        Args:
            scenario (:class:`pygmm.model.Scenario`): earthquake scenario.
            ims (str or sequence of str, optional): intensity measures to
                compute. If *None* (default), all intensity measures are
                computed.
        """
        super().__init__(scenario, ims)
        # The nonlinear site term depends on PGA at the reference condition,
        # which only needs the PGA coefficients
        c_ref = self.COEFF[[self.INDEX_PGA]]
        pga_ref = np.exp(self._calc_ln_resp(np.nan, c_ref)[..., 0])
        self._ln_resp = self._calc_ln_resp(pga_ref)
        self._ln_std, self._tau, self._phi = self._calc_ln_std()

    def _check_inputs(self) -> None:
        """Check the inputs."""
        super()._check_inputs()
        s = self._scenario
        if np.ndim(s.mechanism) > 0 or np.ndim(s.mag) > 0:
            self._check_mag_by_mechanism()
            return
        # Mechanism specific limits
        if s.mechanism == "SS":
            _min, _max = 3.0, 8.5
            if not (_min <= s.mag <= _max):
                logging.warning(
                    "Magnitude (%g) exceeds recommended bounds (%g to %g)"
                    " for a strike-slip earthquake!",
                    s.mag,
                    _min,
                    _max,
                )
        elif s.mechanism == "NS":
            _min, _max = 3.0, 7.0
            if not (_min <= s.mag <= _max):
                logging.warning(
                    "Magnitude (%g) exceeds recommended bounds (%g to %g)"
                    " for a normal-slip earthquake!",
                    s.mag,
                    _min,
                    _max,
                )

    def _check_mag_by_mechanism(self) -> None:
        """Check mechanism specific magnitude limits for vectorized scenarios."""
        s = self._scenario
        mag, mechanism = np.broadcast_arrays(np.asarray(s.mag), np.asarray(s.mechanism))
        for option, name, _min, _max in [
            ("SS", "strike-slip", 3.0, 8.5),
            ("NS", "normal-slip", 3.0, 7.0),
        ]:
            is_option = model.equals(mechanism, option)
            outside = is_option & ~((_min <= mag) & (mag <= _max))
            if np.any(outside):
                logging.warning(
                    "Magnitude (%d of %d %s earthquakes, %g to %g) exceeds"
                    " recommended bounds (%g to %g)!",
                    np.count_nonzero(outside),
                    np.count_nonzero(is_option),
                    name,
                    np.min(mag[outside]),
                    np.max(mag[outside]),
                    _min,
                    _max,
                )

    def _calc_ln_resp(self, pga_ref: ArrayLike, c=None) -> np.ndarray:
        """Calculate the natural logarithm of the response.

        Parameters
        ----------
        pga_ref : float or array_like
            peak ground acceleration (g) at the reference
            condition. If :class:`np.nan`, then no site term is applied.
        c : :class:`numpy.recarray`, optional
            coefficients for the periods to compute. If *None*, the
            coefficients for the requested intensity measures are used.

        Returns
        -------
        ln_resp : class:`np.array`:
            natural log of the response

        """
        s = self._scenario
        if c is None:
            c = self._coeff_rows(self.COEFF)
        # Scenario values get a trailing axis to broadcast against the
        # coefficients, which have one value per period
        mag = model.as_column(s.mag)
        mechanism = model.as_column(s.mechanism)
        region = model.as_column(s.region)

        # Compute the event term
        ########################
        event = np.select(
            [
                model.equals(mechanism, "SS"),
                model.equals(mechanism, "NS"),
                model.equals(mechanism, "RS"),
            ],
            [c.e_1, c.e_2, c.e_3],
            # Unspecified
            default=c.e_0,
        )

        mask = mag <= c.M_h
        event = event + np.where(
            mask,
            c.e_4 * (mag - c.M_h) + c.e_5 * (mag - c.M_h) ** 2,
            c.e_6 * (mag - c.M_h),
        )

        # Compute the distance terms
        ############################
        dc_3 = np.select(
            [
                model.equals(region, "china") | model.equals(region, "turkey"),
                model.equals(region, "italy") | model.equals(region, "japan"),
            ],
            [c.dc_3ct, c.dc_3ij],
            # 'global', 'california', 'new_zealand', 'taiwan'
            default=c.dc_3global,
        )

        dist = np.sqrt(model.as_column(s.dist_jb) ** 2 + c.h**2)
        path = (c.c_1 + c.c_2 * (mag - c.M_ref)) * np.log(dist / c.R_ref) + (
            c.c_3 + dc_3
        ) * (dist - c.R_ref)

        if np.all(np.isnan(pga_ref)):
            # Reference condition. No site effect
            site = 0
        else:
            # Compute the site term. The basin model uses the Japan relation for
            # Z1.0 for the Japan region, and the global relation otherwise.
            site = self._calc_site_term(
                c,
                model.as_column(pga_ref),
                model.as_column(s.v_s30),
                None if s.depth_1_0 is None else model.as_column(s.depth_1_0),
                s.region if np.ndim(s.region) == 0 else region,
            )

        ln_resp = event + path + site
        return ln_resp

    @classmethod
    def calc_site_term(
        cls,
        pga_ref: float,
        v_s30: float,
        depth_1_0: Optional[float],
        region: str = "california",
    ) -> ArrayLike:
        """Calculate the site term, which includes site and basin effects.

        Parameters
        ----------
        pga_ref : float
            peak ground acceleration (g) at the reference
            condition. If :class:`np.nan`, then no site term is applied.
        v_s30 : float
            site condition. Set `v_s30` to the reference
            velocity (e.g., 1180 m/s) for the reference response.
        depth_1_0 : float
            depth to the 1.0 km∕s shear-wave velocity horizon beneath the site,
            :math:`Z_{1.0}` in (km).
        region : str, optional
            region of basin model. Valid options: 'california', 'japan'. If
            *None*, then 'california' is used as the default value.

        Returns
        -------
        site_term: :class:`np.ndarray`
            site term that is applied to the natural log response.
        """
        return cls._calc_site_term(cls.COEFF, pga_ref, v_s30, depth_1_0, region)

    @staticmethod
    def _calc_site_term(c, pga_ref, v_s30, depth_1_0, region) -> ArrayLike:
        """Calculate the site term for the periods in the coefficients `c`."""
        f_lin = c.c * np.log(np.minimum(v_s30, c.V_c) / c.V_ref)

        # Add the nonlinearity to the site term
        f_2 = c.f_4 * (
            np.exp(c.f_5 * (np.minimum(v_s30, 760) - 360.0))
            - np.exp(c.f_5 * (760.0 - 360.0))
        )
        f_nl = c.f_1 + f_2 * np.log((pga_ref + c.f_3) / c.f_3)

        # Compute the average from the Chiou and Youngs (2014)
        # model convert from m to km.
        ln_mz1 = np.log(CY14.calc_depth_1_0(v_s30, region))

        if depth_1_0 is not None:
            delta_depth_1_0 = depth_1_0 - np.exp(ln_mz1)
        else:
            delta_depth_1_0 = 0.0

        # Add the basin effect to the site term for periods of 0.65 s and longer
        F_dz1 = np.where(
            c.period >= 0.65, np.minimum(c.f_6 * delta_depth_1_0, c.f_7), 0.0
        )

        site = f_lin + f_nl + F_dz1

        return site

    def _calc_ln_std(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Calculate the logarithmic standard deviation.

        Returns
        -------
        ln_std : class:`np.array`:
            natural log standard deviation

        """
        c = self._coeff_rows(self.COEFF)
        s = self._scenario
        mag = model.as_column(s.mag)

        # Uncertainty model
        tau = c.tau_1 + (c.tau_2 - c.tau_1) * (np.clip(mag, 4.5, 5.5) - 4.5)
        phi = c.phi_1 + (c.phi_2 - c.phi_1) * (np.clip(mag, 4.5, 5.5) - 4.5)

        # Modify phi for Vs30
        phi = phi - c.dphi_V * np.clip(
            np.log(c.V_2 / model.as_column(s.v_s30)) / np.log(c.V_2 / c.V_1), 0, 1
        )

        # Modify phi for R
        phi = phi + c.dphi_R * np.clip(
            # Maximum added for zero distance caes
            np.log(np.maximum(model.as_column(s.dist_jb), 0.1) / c.R_1)
            / np.log(c.R_2 / c.R_1),
            0,
            1,
        )

        ln_std = np.sqrt(phi**2 + tau**2)
        return ln_std, tau, phi
