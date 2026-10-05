"""NGA-Subduction USGS (2018) ground motion model."""

import numpy as np

from . import model


class NgaSubductionUsgs2018(model.GroundMotionModel):
    """NGA-Subduction USGS (2018) ground motion model.

    Preliminary PEER NGA-Subduction ground motion model for subduction
    interface and intraslab events, as implemented by the U.S. Geological
    Survey for the 2018 update of the National Seismic Hazard Model. The model
    is an update of the BC Hydro model (Abrahamson et al., 2016) that was
    fit to the NGA-Subduction dataset and calibrated for use in Cascadia only.

    Reference: Abrahamson, N., Kuehn, N., Gulerce, Z., Gregor, N., Bozorgnia,
    Y., Parker, G., Stewart, J., Chiou, B., Idriss, I. M., Campbell, K., and
    Youngs, R. (2018). Update of the BC Hydro subduction ground-motion model
    using the NGA-Subduction dataset. PEER Report No. 2018/02, Pacific
    Earthquake Engineering Research Center, Berkeley, CA.

    The implementation follows ``NgaSubductionUsgs_2018`` in the USGS
    nshmp-haz code (``Gmm.NGA_SUB_USGS_INTERFACE``, ``NGA_SUB_USGS_SLAB``,
    and the ``*_NO_EPI`` variants), including its coefficients and the fixes
    to typos in the report noted there (a '+' in the nonlinear site term,
    :math:`a_3 = 0.1`, and slab epistemic adjustments of 0.3 at 7.5 and 10 s).
    The exception is :math:`a_{14}` for PGA, which is -0.233 in nshmp-haz. Here
    it is -0.223, the 0.01 s value, as in OpenQuake, because all of the other
    PGA coefficients equal the 0.01 s coefficients. This only affects
    intraslab events (PGA about 4 to 5% larger). As in nshmp-haz:

    * The Cascadia adjustment (``adj_int`` or ``adj_slab``) is added to the
      mean. The reference PGA (:math:`V_{s30}` = 1000 m/sec) used by the
      nonlinear site term does not include the adjustment.
    * Forearc and backarc sites are not distinguished.
    * ``depth_hyp`` is only used for intraslab events and is limited to
      100 km.
    * The standard deviation is :math:`\\sqrt{\\phi^2 + \\tau^2}` with
      :math:`\\phi = 0.62`.
    * With ``epistemic=True`` (default), the epistemic uncertainty of the
      Cascadia adjustment is represented by three branches with means
      :math:`\\mu + \\Delta_{lo}`, :math:`\\mu`, and :math:`\\mu +
      \\Delta_{hi}` and weights of 0.2, 0.6, and 0.2. :math:`\\Delta_{lo}` and
      :math:`\\Delta_{hi}` are -0.3 and 0.3 for interface events, and
      period-dependent values (``adj_slab_epi_lo`` and ``adj_slab_epi_hi``)
      for intraslab events. As in nshmp-haz, the branches are collapsed to a
      single median, :math:`\\ln \\sum_i w_i \\exp(\\mu_i)`, and the weighted
      sum of the standard deviations, which is the same for all branches.
      With ``epistemic=False``, the central branch (:math:`\\mu`) is used.

    The model is vectorized. Each scenario value (``mag``, ``dist_rup``,
    ``depth_hyp``, ``v_s30``, and ``event_type``) can be a scalar or an
    array, and the arrays are broadcast against each other. For a scalar
    scenario, the response and standard deviation have one value per period,
    as in other models. For arrays of N scenarios, they have shape
    (N, periods), so, for example, ``pga`` has shape (N,). ``depth_hyp`` is
    only needed for intraslab events, so it can be *None* if none of the
    events are intraslab events. Results for events that are neither
    interface nor intraslab events are *NaN*.

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario
    epistemic : bool, optional
        if *True* (default), the epistemic branches of the Cascadia adjustment
        are included (``Gmm.NGA_SUB_USGS_INTERFACE`` and
        ``Gmm.NGA_SUB_USGS_SLAB`` in nshmp-haz). If *False*, only the central
        branch is used (the ``*_NO_EPI`` variants).
    ims : str or sequence of str, optional
        intensity measures to compute: "pga", spectral periods such as
        "psa_1p000" (1.0 s), "psa_ngawest2_21" (the 21 NGA-West2 comparison
        periods), and/or "psa_all" (all 24 periods). Computing only the
        needed intensity measures is much faster for large vectorized
        scenarios. If *None* (default), all intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_subduction_usgs_2018 import NgaSubductionUsgs2018
    >>> s = Scenario(
    ...     mag=np.array([7.0, 9.0]), dist_rup=np.array([80.0, 100.0]),
    ...     depth_hyp=60.0, v_s30=400.0,
    ...     event_type=np.array(["intraslab", "interface"]))
    >>> m = NgaSubductionUsgs2018(s)
    >>> m.pga.shape, m.spec_accels.shape
    ((2,), (2, 24))
    >>> NgaSubductionUsgs2018(s, epistemic=False, ims=["pga"]).pga.shape
    (2,)
    >>> m = NgaSubductionUsgs2018(s, ims=["pga", "psa_0p200", "psa_1p000"])
    >>> m.psa_ims
    ['psa_0p200', 'psa_1p000']
    >>> m.spec_accels.shape
    (2, 2)

    """

    NAME = "NGA-Subduction USGS (2018)"
    ABBREV = "NGASUB18"

    # Load the coefficients for the model
    COEFF = model.load_data_file("nga_subduction_usgs_2018.csv", 1)
    PERIODS = COEFF["period"]

    INDEX_PGA = 0
    INDICES_PSA = np.arange(1, 25)

    # Reference shear-wave velocity (m/sec)
    V_REF = 1000.0

    # Constants
    N = 1.18
    C = 1.88
    C_4 = 10.0
    A_3 = 0.1
    A_5 = 0.0
    A_9 = 0.4
    A_10 = 1.73
    C_1_SLAB = 7.2
    PHI = 0.62
    # Interface epistemic adjustments
    ADJ_INT_EPI_LO = -0.3
    ADJ_INT_EPI_HI = 0.3
    # Weights of the low, central, and high epistemic branches
    EPI_WEIGHTS = (0.2, 0.6, 0.2)

    # Recommended limits. The hypocentral depth limits are for intraslab events
    # (nshmp-haz Earthquakes.SLAB_DEPTH_RANGE).
    LIMITS = dict(
        mag=(5.0, 9.5),
        dist_rup=(0.0, 1000.0),
        v_s30=(150.0, 1000.0),
        depth_hyp=(20.0, 700.0),
    )

    PARAMS = [
        model.NumericParameter("mag", True, 5.0, 9.5),
        model.NumericParameter("dist_rup", True, 0.0, 1000.0),
        # Only used for intraslab events, so values for interface events are not
        # checked against the limits
        model.NumericParameter("depth_hyp", False),
        model.NumericParameter("v_s30", True, 150.0, 1000.0),
        model.CategoricalParameter("event_type", True, ["interface", "intraslab"]),
    ]

    def __init__(self, scenario: model.Scenario, epistemic: bool = True, ims=None):
        """Initialize the model.

        Args:
            scenario (:class:`pygmm.model.Scenario`): earthquake scenario.
            epistemic (bool, optional): if *True* (default), include the
                epistemic branches of the Cascadia adjustment.
            ims (str or sequence of str, optional): intensity measures to
                compute. If *None* (default), all intensity measures are
                computed.
        """
        super().__init__(scenario, ims)
        self._epistemic = bool(epistemic)
        self._ln_resp = self._calc_ln_resp()
        self._ln_std = self._calc_ln_std()

    @property
    def epistemic(self) -> bool:
        """If the epistemic branches of the Cascadia adjustment are included."""
        return self._epistemic

    def _calc_ln_resp(self) -> np.ndarray:
        """Calculate the natural logarithm of the response.

        Returns
        -------
        ln_resp : class:`np.array`:
            natural log of the response

        """
        s = self._scenario
        c = self._coeff_rows(self.COEFF)
        # Scenario values get a trailing axis to broadcast against the
        # coefficients, which have one value per period
        event_type = model.as_column(s.event_type)
        is_slab = model.equals(event_type, "intraslab")
        is_interface = model.equals(event_type, "interface")

        if s.depth_hyp is None:
            if np.any(is_slab):
                raise ValueError("depth_hyp is required for intraslab events")
            # Not used for interface events
            depth_hyp = np.nan
        else:
            depth_hyp = model.as_column(s.depth_hyp)

        mag = model.as_column(s.mag)
        dist_rup = model.as_column(s.dist_rup)
        v_s30 = model.as_column(s.v_s30)

        # Reference PGA (V_s30 = 1000 m/sec), which only needs the PGA
        # coefficients. It is only used by the nonlinear site term (V_s30 < V_lin).
        c_pga = self.COEFF[[self.INDEX_PGA]]
        ln_pga_ref = self._calc_ln_mean(
            c_pga, is_slab, 0.0, mag, dist_rup, depth_hyp, self.V_REF
        )
        pga_ref = np.where(v_s30 < c.vlin, np.exp(ln_pga_ref), 0.0)

        ln_resp = self._calc_ln_mean(
            c, is_slab, pga_ref, mag, dist_rup, depth_hyp, v_s30
        )
        # Cascadia adjustment
        ln_resp = ln_resp + np.where(is_slab, c.adj_slab, c.adj_int)

        if self._epistemic:
            # Collapse the branches: ln(sum_i w_i exp(mu + delta_i))
            #   = mu + ln(sum_i w_i exp(delta_i))
            w_lo, w_mid, w_hi = self.EPI_WEIGHTS
            epi_lo = np.where(is_slab, c.adj_slab_epi_lo, self.ADJ_INT_EPI_LO)
            epi_hi = np.where(is_slab, c.adj_slab_epi_hi, self.ADJ_INT_EPI_HI)
            ln_resp = ln_resp + np.log(
                w_lo * np.exp(epi_lo) + w_mid + w_hi * np.exp(epi_hi)
            )

        # The model is not defined for other event types
        return np.where(is_slab | is_interface, ln_resp, np.nan)

    def _calc_ln_mean(
        self, c, is_slab, pga_ref, mag, dist_rup, depth_hyp, v_s30
    ) -> np.ndarray:
        """Calculate the mean without the Cascadia adjustment.

        Parameters
        ----------
        c : :class:`numpy.recarray`
            coefficients for the periods to compute
        is_slab : array_like
            if the event is an intraslab event
        pga_ref : float or array_like
            peak ground acceleration (g) at the reference condition
            (V_s30 = 1000 m/sec). Only used if V_s30 < V_lin.
        mag : array_like
            moment magnitude
        dist_rup : array_like
            closest distance to the rupture (km)
        depth_hyp : array_like
            hypocentral depth (km), only used for intraslab events
        v_s30 : float or array_like
            time-averaged shear-wave velocity over the top 30 m (m/sec)

        Returns
        -------
        ln_mean : class:`np.array`:
            natural log of the response

        """
        # Magnitude scaling
        c_1 = np.where(is_slab, self.C_1_SLAB, c.c1int)
        f_mag = np.where(mag <= c_1, c.a4, self.A_5) * (mag - c_1) + c.a13 * (
            10 - mag
        ) * (10 - mag)

        # Depth scaling, only for intraslab events
        f_depth = np.where(is_slab, c.a11 * (np.minimum(depth_hyp, 100.0) - 60.0), 0.0)

        # Site response
        vs_ratio = np.minimum(v_s30, self.V_REF) / c.vlin
        ln_vs = np.log(vs_ratio)
        f_site = np.where(
            v_s30 < c.vlin,
            c.a12 * ln_vs
            - c.b * np.log(pga_ref + self.C)
            + c.b * np.log(pga_ref + self.C * vs_ratio**self.N),
            (c.a12 + c.b * self.N) * ln_vs,
        )

        delta_c_1 = np.where(is_slab, c.a4 * (self.C_1_SLAB - c.c1int), 0.0)
        f_path = (
            c.a2 + np.where(is_slab, c.a14, 0.0) + self.A_3 * (mag - 7.8)
        ) * np.log(dist_rup + self.C_4 * np.exp((mag - 6.0) * self.A_9))

        return (
            c.a1
            + delta_c_1
            + f_path
            + c.a6 * dist_rup
            + np.where(is_slab, self.A_10, 0.0)
            + f_mag
            + f_depth
            + f_site
        )

    def _calc_ln_std(self) -> np.ndarray:
        """Calculate the logarithmic standard deviation.

        The epistemic branches all have the same standard deviation, so it does
        not depend on ``epistemic``.

        Returns
        -------
        ln_std : class:`np.array`:
            natural log standard deviation

        """
        c = self._coeff_rows(self.COEFF)
        ln_std = np.hypot(self.PHI, c.tau)
        if np.ndim(self._ln_resp) > 1:
            # The standard deviation does not depend on the scenario, so it is
            # repeated for each scenario
            ln_std = np.broadcast_to(ln_std, np.shape(self._ln_resp))
        return ln_std
