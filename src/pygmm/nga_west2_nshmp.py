r"""USGS nshmp-lib versions of the NGA-West2 ground motion models.

The NGA-West2 models for active crustal regions as implemented in the USGS
nshmp-lib library (commit 44728a7d), which is used for the current U.S.
Geological Survey national seismic hazard models (NSHMs):

- :class:`AbrahamsonSilvaKamai2014Nshmp` (``AbrahamsonEtAl_2014``, ``ASK_14*``)
- :class:`BooreStewartSeyhanAtkinson2014Nshmp` (``BooreEtAl_2014``,
  ``BSSA_14*``)
- :class:`CampbellBozorgnia2014Nshmp` (``CampbellBozorgnia_2014``, ``CB_14*``)
- :class:`ChiouYoungs2014Nshmp` (``ChiouYoungs_2014``, ``CY_14*``)
- :class:`Idriss2014Nshmp` (``Idriss_2014``, ``IDRISS_14*``)
- :class:`NgaWest2NshmpTree`, the weighted logic trees of these models used by
  the 2023 conterminous U.S. (CONUS) NSHM for active crust
  (``TOTAL_TREE_CONUS_ACTIVE_CRUST_2023*``).

These are separate from the published implementations (e.g.,
:class:`~pygmm.abrahamson_silva_kamai_2014.AbrahamsonSilvaKamai2014`), which
are unchanged, because nshmp-lib differs from them in details such as the
treatment of missing basin depths, the hanging-wall conditions, and the
regional terms (only the global or California models are implemented). Each
class has the model options that select the nshmp-lib ``Gmm`` variants, and
``GMM_IDS`` maps the nshmp-lib ``Gmm`` ids to the options:

- ``epistemic`` (default *True*): the USGS additional epistemic uncertainty of
  the NGA-West2 models (``NgaUncertainty.NGA_WEST2``) is represented by three
  branches with means :math:`\mu - \epsilon`, :math:`\mu`, and :math:`\mu +
  \epsilon`, where :math:`\epsilon` depends on the magnitude and the
  Joyner-Boore distance (see :func:`epistemic_epsilon`), and weights of
  0.185, 0.63, and 0.185. As in nshmp-lib (``GroundMotions.combine``), the
  branches are collapsed to a single median, :math:`\ln \sum_i w_i
  \exp(\mu_i)`, and the standard deviation, which is the same for all
  branches. The ``_BASE``, ``_VS30_MEASURED``, and ``_PRVI`` variants have
  ``epistemic=False``.
- ``basin`` (``_BASIN``): the USGS deep basin model, which keeps the basin
  (sediment depth) term of the model only for spectral periods longer than
  0.5 s and scales it from 0 at :math:`Z_{1.0}` of 0.3 km to the full term at
  0.5 km (ASK14, BSSA14, and CY14), or, for CB14, from the :math:`V_{S30}`
  based (reference depth) term at :math:`Z_{2.5}` of 1 km to the full term at
  3 km. The scaled term is multiplied by 0.585 at 0.75 s. This is the variant
  used by the current CONUS NSHM (nshm-conus 6.2.0), with weights of 0.25 for
  ASK14, BSSA14, CB14, and CY14.
- ``cybershake`` (``_CYBERSHAKE``, requires ``basin=True``): the 2023 NSHM
  CyberShake based basin terms for spectral periods longer than 1.9 s
  (alternative basin coefficients and a constant of 0.1 added to the basin
  term), which are used in the Los Angeles region.
- ``prvi`` (``_PRVI``): the 2025 Puerto Rico and U.S. Virgin Islands
  adjustments of the constant and linear site coefficients.
- ``vs30_measured`` (``_VS30_MEASURED``, ASK14 and CY14): the aleatory
  variability for a measured :math:`V_{S30}`. nshmp-lib otherwise uses the
  values for an inferred :math:`V_{S30}`.

As in nshmp-lib:

- The style of faulting is given by ``mechanism`` ("SS", "NS", "RS", or "U"
  for unspecified). nshmp-lib converts the rake to the style of faulting
  with :func:`mechanism_from_rake` (reverse for rakes from 45 to 135 degrees,
  normal for -135 to -45 degrees, and strike-slip otherwise). "U"
  (unspecified, a rake of NaN) uses the unspecified coefficient of BSSA14, and
  no style of faulting term (as for strike-slip) in the other models.
- A ``depth_1_0`` (:math:`Z_{1.0}`) or ``depth_2_5`` (:math:`Z_{2.5}`) of
  *None* or NaN gives no basin term in ASK14, BSSA14, and CY14 (the same as
  the model's :math:`V_{S30}` based reference depth) and the :math:`V_{S30}`
  based reference depth term in CB14.
- The scenario values are not clipped to the model limits (values outside of
  the limits only give a warning).

The models are vectorized. Each scenario value can be a scalar or an array,
and the arrays are broadcast against each other. For a scalar scenario, the
response and standard deviation have one value per period. For arrays of N
scenarios, they have shape (N, periods), so, for example, ``pga`` has shape
(N,). All models provide PGA, PGV (except Idriss (2014)), and the 21
NGA-West2 comparison periods.

The scenario values (with the nshmp-lib names) are:

======================  ============  ==========================================
pygmm                   nshmp-lib     used by
======================  ============  ==========================================
``mag``                 ``Mw``        all
``dist_rup``            ``rRup``      ASK14, CB14, CY14, Idriss (2014)
``dist_jb``             ``rJB``       ASK14, BSSA14, CB14, CY14, and the
                                      epistemic uncertainty (all models)
``dist_x``              ``rX``        ASK14, CB14, CY14
``dip``                 ``dip``       ASK14, CB14, CY14 (deg)
``width``               ``width``     ASK14, CB14 (down-dip width, km)
``depth_tor``           ``zTor``      ASK14, CB14, CY14 (km)
``depth_hyp``           ``zHyp``      CB14 (km)
``mechanism``           ``rake``      all
``v_s30``               ``vs30``      all (m/sec)
``depth_1_0``           ``z1p0``      ASK14, BSSA14, CY14 (km)
``depth_2_5``           ``z2p5``      CB14 (km)
======================  ============  ==========================================

If ``width`` is *None*, it is computed from the depth to the bottom of the
rupture, ``depth_bor`` (:math:`Z_{bor}`), as :math:`(Z_{bor} - Z_{tor}) /
\sin \delta`. If ``depth_hyp`` is *None*, it is computed as in nshmp-lib for
fault sources (``Faults.hypocentralDepth``): :math:`Z_{tor} + W \sin(\delta) /
2`, the middle of the rupture (:math:`(Z_{tor} + Z_{bor}) / 2`). So for fault
and gridded sources described by ``m``, ``rjb``, ``rrup``, ``rx``, ``dip``,
``ztor``, ``zbor``, ``mechanism``, ``vs30``, ``z1p0``, and ``z2p5``, use
``mag``, ``dist_jb``, ``dist_rup``, ``dist_x``, ``dip``, ``depth_tor``,
``depth_bor``, ``mechanism``, ``v_s30``, ``depth_1_0``, and ``depth_2_5``. The
``ry0`` and ``rx1`` distances are not used by nshmp-lib.

References
----------
USGS nshmp-lib, ``gov.usgs.earthquake.nshmp.gmm.AbrahamsonEtAl_2014``,
``BooreEtAl_2014``, ``CampbellBozorgnia_2014``, ``ChiouYoungs_2014``,
``Idriss_2014``, ``GmmUtils``, ``GroundMotions``, and ``GmmTotalTree``, commit
44728a7d (https://code.usgs.gov/ghsc/nshmp/nshmp-lib).
"""

import numpy as np

from . import model
from .types import ArrayLike

#: Weights of the epistemic branches (mu - epsilon, mu, mu + epsilon)
EPI_WTS = (0.185, 0.63, 0.185)

#: NGA-West2 additional epistemic uncertainty (nshmp-lib
#: ``NgaUncertainty.NGA_WEST2``), in natural log units. Rows are for Joyner-Boore
#: distances less than 10 km, from 10 to 30 km, and 30 km and greater, and the
#: columns are for magnitudes less than 6, from 6 to 7, and 7 and greater.
EPI_NGA_WEST2 = np.array(
    [
        [0.37, 0.25, 0.40],
        [0.22, 0.23, 0.36],
        [0.22, 0.23, 0.33],
    ]
)

# USGS deep basin model (GmmUtils)
BASIN_Z1P0_UPPER = 0.3
BASIN_Z1P0_LOWER = 0.5
BASIN_Z2P5_UPPER = 1.0
BASIN_Z2P5_LOWER = 3.0
# Constant added to the CyberShake basin terms
CY_CSIM = 0.1

MECHANISMS = ["U", "SS", "NS", "RS"]


def epistemic_epsilon(mag: ArrayLike, dist_jb: ArrayLike) -> np.ndarray:
    r"""USGS NGA-West2 additional epistemic uncertainty.

    The :math:`\epsilon` of the epistemic branches with means :math:`\mu -
    \epsilon`, :math:`\mu`, and :math:`\mu + \epsilon` (nshmp-lib
    ``NgaUncertainty.NGA_WEST2``; see :data:`EPI_NGA_WEST2`).

    Parameters
    ----------
    mag : float or array_like
        moment magnitude
    dist_jb : float or array_like
        Joyner-Boore distance (km)

    Returns
    -------
    epsilon : :class:`np.ndarray`
        epistemic uncertainty in natural log units

    Examples
    --------
    >>> from pygmm.nga_west2_nshmp import epistemic_epsilon
    >>> epistemic_epsilon([5.5, 6.5, 7.5], [5.0, 20.0, 50.0]).tolist()
    [0.37, 0.23, 0.33]
    """
    mag = np.asarray(mag, dtype=float)
    dist_jb = np.asarray(dist_jb, dtype=float)
    mi = np.where(mag < 6, 0, np.where(mag < 7, 1, 2))
    ri = np.where(dist_jb < 10, 0, np.where(dist_jb < 30, 1, 2))
    return EPI_NGA_WEST2[ri, mi]


def epistemic_ln_factor(mag: ArrayLike, dist_jb: ArrayLike) -> np.ndarray:
    r"""Change of the natural log of the median from the epistemic branches.

    :math:`\ln(0.185 e^{-\epsilon} + 0.63 + 0.185 e^{\epsilon})`, which is
    added to the natural log of the median of the central branch to give the
    collapsed median, :math:`\ln \sum_i w_i \exp(\mu_i)`.

    Parameters
    ----------
    mag : float or array_like
        moment magnitude
    dist_jb : float or array_like
        Joyner-Boore distance (km)

    Returns
    -------
    ln_factor : :class:`np.ndarray`
        change of the natural log of the median
    """
    eps = epistemic_epsilon(mag, dist_jb)
    w_lo, w_mid, w_hi = EPI_WTS
    return np.log(w_lo * np.exp(-eps) + w_mid + w_hi * np.exp(eps))


def mechanism_from_rake(rake: ArrayLike) -> np.ndarray:
    """Style of faulting from the rake, as in nshmp-lib.

    nshmp-lib (``GmmUtils.rakeToFaultStyle_NSHMP``) divides the styles on 45
    degree diagonals: reverse ("RS") for rakes from 45 to 135 degrees, normal
    ("NS") for rakes from -135 to -45 degrees, strike-slip ("SS") otherwise,
    and unspecified ("U") for a rake of NaN.

    Parameters
    ----------
    rake : float or array_like
        rake angle (deg)

    Returns
    -------
    mechanism : :class:`np.ndarray`
        mechanism ("SS", "NS", "RS", or "U")

    Examples
    --------
    >>> from pygmm.nga_west2_nshmp import mechanism_from_rake
    >>> mechanism_from_rake([0.0, 90.0, -90.0, 180.0, float("nan")]).tolist()
    ['SS', 'RS', 'NS', 'SS', 'U']
    """
    rake = np.asarray(rake, dtype=float)
    return np.select(
        [np.isnan(rake), (rake >= 45) & (rake <= 135), (rake >= -135) & (rake <= -45)],
        ["U", "RS", "NS"],
        "SS",
    )


def basin_scale(
    periods: ArrayLike, depth: ArrayLike, upper: float, lower: float
) -> np.ndarray:
    """USGS deep basin scaling (``GmmUtils.deltaZ1scale`` and ``deltaZ25scale``).

    The scale is 0 for periods of 0.5 s and shorter (and PGA and PGV), depths
    of `upper` or less, or NaN depths, increases linearly to 1 at `lower`, and
    is multiplied by 0.585 at 0.75 s.

    Parameters
    ----------
    periods : array_like
        spectral periods (s), with 0 for PGA and -1 for PGV
    depth : float or array_like
        basin depth (km)
    upper : float
        depth with a scale of 0 (km)
    lower : float
        depth with a scale of 1 (km)

    Returns
    -------
    scale : :class:`np.ndarray`
        basin scale
    """
    periods = np.asarray(periods)
    depth = np.asarray(depth, dtype=float)
    scale = (np.clip(depth, upper, lower) - upper) / (lower - upper)
    scale = np.where(periods == 0.75, scale * 0.585, scale)
    # NaN depths compare False
    return np.where((periods > 0.5) & (depth > upper), scale, 0.0)


def _interp_extrap(xs, ys, x) -> np.ndarray:
    """Linear interpolation that extrapolates the end segments.

    nshmp-lib ``Interpolator.findY``. `ys` is a list of per-period values and
    `x` has a trailing axis.
    """
    xs = np.asarray(xs, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(xs)
    idx = np.searchsorted(xs, x, side="left")
    exact = xs[np.clip(idx, 0, n - 1)] == x
    i = np.clip(np.where(exact, idx, idx - 1), 0, n - 2)
    y1 = np.choose(i, ys)
    y2 = np.choose(i + 1, ys)
    x1 = xs[i]
    x2 = xs[i + 1]
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def _load_coeffs(name: str, name_prvi: str = None):
    """Load the coefficients, and the coefficients with the PRVI values."""
    coeff = model.load_data_file(name, 1)
    if name_prvi is None:
        return coeff, None
    prvi = model.load_data_file(name_prvi, 1)
    assert np.array_equal(coeff["period"], prvi["period"])
    coeff_prvi = coeff.copy()
    for key in prvi.dtype.names:
        coeff_prvi[key] = prvi[key]
    return coeff, coeff_prvi


class _NgaWest2NshmpBase(model.GroundMotionModel):
    """Shared code of the nshmp-lib NGA-West2 models."""

    #: nshmp-lib coefficients, with periods 0 for PGA and -1 for PGV
    COEFF = None
    #: Coefficients with the PRVI values
    COEFF_PRVI = None
    PERIODS = np.array([])

    INDEX_PGV = 0
    INDEX_PGA = 1
    INDICES_PSA = np.arange(2, 23)

    #: Model options of the nshmp-lib ``Gmm`` ids
    GMM_IDS = {}

    #: Model options supported by the model
    OPTIONS = ("epistemic",)

    def __init__(self, scenario: model.Scenario, ims=None, **options):
        """Initialize the model."""
        for key in options:
            if key not in self.OPTIONS:
                raise TypeError(f"{self.NAME} does not have the option {key!r}")
        self._epistemic = bool(options.get("epistemic", True))
        self._basin = bool(options.get("basin", False))
        self._cybershake = bool(options.get("cybershake", False))
        self._prvi = bool(options.get("prvi", False))
        self._vs30_measured = bool(options.get("vs30_measured", False))
        if self._cybershake and not self._basin:
            raise ValueError(
                "cybershake=True requires basin=True (nshmp-lib *_14_CYBERSHAKE)"
            )
        super().__init__(scenario, ims)
        self._rows = self._coeff_rows(np.arange(len(self.PERIODS)))
        coeff = self.COEFF_PRVI if self._prvi else self.COEFF
        self._c = coeff[self._rows]
        self._c_pga = coeff[[self.INDEX_PGA]] if self.INDEX_PGA is not None else None
        self._periods = self.PERIODS[self._rows]
        ln_resp, ln_std = self._calc()
        if self._epistemic:
            ln_resp = ln_resp + epistemic_ln_factor(
                model.as_column(self._value("mag")),
                model.as_column(self._value("dist_jb")),
            )
        # Some terms do not depend on all of the scenario values
        shape = np.broadcast_shapes(np.shape(ln_resp), np.shape(ln_std))
        self._ln_resp = np.array(np.broadcast_to(ln_resp, shape), dtype=float)
        self._ln_std = np.array(np.broadcast_to(ln_std, shape), dtype=float)

    @property
    def epistemic(self) -> bool:
        """If the USGS epistemic branches are included."""
        return self._epistemic

    @property
    def basin(self) -> bool:
        """If the USGS deep basin model is used."""
        return self._basin

    @property
    def cybershake(self) -> bool:
        """If the CyberShake basin terms are used."""
        return self._cybershake

    @property
    def prvi(self) -> bool:
        """If the PRVI coefficients are used."""
        return self._prvi

    @property
    def vs30_measured(self) -> bool:
        """If the aleatory variability for a measured V_s30 is used."""
        return self._vs30_measured

    def _value(self, name: str) -> np.ndarray:
        """Scenario value as a float array, which is required."""
        value = self._scenario[name]
        if value is None:
            raise ValueError(f"{name} is required by {self.NAME}")
        return np.asarray(value, dtype=float)

    def _column(self, name: str, default=None) -> np.ndarray:
        """Scenario value with a trailing axis for the periods.

        Optional values that are *None* use `default` (e.g., NaN).
        """
        value = self._scenario[name]
        if value is None and default is not None:
            return np.asarray(default, dtype=float)
        return model.as_column(self._value(name))

    def _width(self) -> np.ndarray:
        """Rupture width, or the width from the depth to the bottom of rupture."""
        if self._scenario["width"] is not None:
            return self._column("width")
        if self._scenario["depth_bor"] is None:
            raise ValueError(f"width or depth_bor is required by {self.NAME}")
        return (self._column("depth_bor") - self._column("depth_tor")) / np.sin(
            np.radians(self._column("dip"))
        )

    def _mechanism(self) -> dict:
        """Masks of the styles of faulting."""
        mechanism = model.as_column(self._scenario["mechanism"])
        return {m: model.equals(mechanism, m) for m in MECHANISMS}

    def _cybershake_periods(self) -> np.ndarray:
        """Periods with the CyberShake terms (GmmUtils.cybershakeImt)."""
        return self._cybershake & (self._periods > 1.9)

    def _calc(self) -> tuple:
        raise NotImplementedError


class AbrahamsonSilvaKamai2014Nshmp(_NgaWest2NshmpBase):
    r"""Abrahamson, Silva, and Kamai (2014) model as implemented in nshmp-lib.

    The ``AbrahamsonEtAl_2014`` class of USGS nshmp-lib (commit 44728a7d),
    with the coefficients of ``ASK14.csv`` and ``ASK14_PRVI.csv`` (see
    :mod:`pygmm.nga_west2_nshmp` for the options, the scenario values, and the
    treatment of missing values). Compared with the published model
    (:class:`~pygmm.abrahamson_silva_kamai_2014.AbrahamsonSilvaKamai2014`),
    nshmp-lib:

    - has no regional or aftershock terms (the global model, with the
      California :math:`Z_{1.0}` reference depth),
    - applies the hanging-wall term for sites with :math:`R_x \ge 0`,
      :math:`R_{JB} <` 30 km, :math:`M >` 5.5, and :math:`Z_{tor} \le` 10 km
      (there is no ``on_hanging_wall`` value), with the :math:`R_{JB}` taper
      (equation 15b) and a dip taper of 1.33333333 for dips of 30 degrees or
      less,
    - has no basin (soil depth) term if ``depth_1_0`` is *None* or NaN, and
      interpolates the basin slope in :math:`V_{S30}` between 150, 250, 400,
      700, and 1000 m/sec (with linear extrapolation for :math:`V_{S30} <`
      150 m/sec),
    - uses the aleatory variability for an inferred :math:`V_{S30}` unless
      ``vs30_measured=True``.

    The nshmp-lib ``Gmm`` ids are (see :attr:`GMM_IDS`):

    ========================  =========================================
    nshmp-lib ``Gmm``         Options
    ========================  =========================================
    ``ASK_14``                (defaults)
    ``ASK_14_BASE``           ``epistemic=False``
    ``ASK_14_BASIN``          ``basin=True``
    ``ASK_14_CYBERSHAKE``     ``basin=True, cybershake=True``
    ``ASK_14_VS30_MEASURED``  ``epistemic=False, vs30_measured=True``
    ``ASK_14_PRVI``           ``epistemic=False, prvi=True``
    ========================  =========================================

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario with ``mag``, ``dist_rup``, ``dist_jb``,
        ``dist_x``, ``dip``, ``width`` (or ``depth_bor``), ``depth_tor``,
        ``mechanism``, ``v_s30``, and optionally ``depth_1_0``
    epistemic : bool, optional
        include the USGS epistemic branches (default *True*)
    basin : bool, optional
        use the USGS deep basin model (default *False*)
    cybershake : bool, optional
        use the CyberShake basin terms (default *False*, requires
        ``basin=True``)
    vs30_measured : bool, optional
        use the aleatory variability for a measured :math:`V_{S30}` (default
        *False*)
    prvi : bool, optional
        use the PRVI coefficients (default *False*)
    ims : str or sequence of str, optional
        intensity measures to compute: "pga", "pgv", spectral periods such as
        "psa_1p000" (1.0 s), "psa_ngawest2_21" (the 21 NGA-West2 comparison
        periods, which are all of the periods), and/or "psa_all". If *None*
        (default), all intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_west2_nshmp import AbrahamsonSilvaKamai2014Nshmp
    >>> s = Scenario(
    ...     mag=np.array([6.0, 7.0]), dist_rup=np.array([10.0, 50.0]),
    ...     dist_jb=np.array([8.0, 48.0]), dist_x=np.array([8.0, -48.0]),
    ...     dip=60.0, depth_tor=2.0, depth_bor=15.0, mechanism="RS",
    ...     v_s30=400.0, depth_1_0=np.array([np.nan, 0.6]))
    >>> m = AbrahamsonSilvaKamai2014Nshmp(s, basin=True)
    >>> m.pga.shape, m.spec_accels.shape
    ((2,), (2, 21))
    >>> m = AbrahamsonSilvaKamai2014Nshmp(
    ...     s, **AbrahamsonSilvaKamai2014Nshmp.GMM_IDS["ASK_14_BASE"], ims=["pga"])
    >>> m.pga.shape
    (2,)
    """

    NAME = "Abrahamson, Silva, & Kamai (2014) (USGS nshmp-lib)"
    ABBREV = "ASK14_NSHMP"

    COEFF, COEFF_PRVI = _load_coeffs(
        "abrahamson_silva_kamai_2014-nshmp.csv",
        "abrahamson_silva_kamai_2014-nshmp-prvi.csv",
    )
    PERIODS = COEFF["period"]

    GMM_IDS = {
        "ASK_14": dict(),
        "ASK_14_BASE": dict(epistemic=False),
        "ASK_14_BASIN": dict(basin=True),
        "ASK_14_CYBERSHAKE": dict(basin=True, cybershake=True),
        "ASK_14_VS30_MEASURED": dict(epistemic=False, vs30_measured=True),
        "ASK_14_PRVI": dict(epistemic=False, prvi=True),
    }
    OPTIONS = ("epistemic", "basin", "cybershake", "vs30_measured", "prvi")

    VS_RK = 1180.0
    PHI_AMP_SQ = 0.16
    A3 = 0.275
    A4 = -0.1
    A5 = -0.41
    M2 = 5.0
    N = 1.5
    C4 = 4.5
    A = 610.0**4
    B = 1360.0**4 + A
    A2_HW = 0.2
    H1 = 0.25
    H2 = 1.5
    H3 = -0.75
    VS_BINS = (150.0, 250.0, 400.0, 700.0, 1000.0)

    PARAMS = [
        model.NumericParameter("mag", True, 3.0, 8.5),
        model.NumericParameter("dist_rup", True, 0.0, 300.0),
        model.NumericParameter("dist_jb", True, 0.0, 300.0),
        model.NumericParameter("dist_x", True),
        model.NumericParameter("dip", True, 0.0, 90.0),
        model.NumericParameter("width", False),
        model.NumericParameter("depth_bor", False),
        model.NumericParameter("depth_tor", True),
        model.CategoricalParameter("mechanism", True, MECHANISMS),
        model.NumericParameter("v_s30", True, 180.0, 1000.0),
        model.NumericParameter("depth_1_0", False, 0.0, 3.0),
    ]

    def __init__(
        self,
        scenario: model.Scenario,
        epistemic: bool = True,
        basin: bool = False,
        cybershake: bool = False,
        vs30_measured: bool = False,
        prvi: bool = False,
        ims=None,
    ):
        """Initialize the model."""
        super().__init__(
            scenario,
            ims,
            epistemic=epistemic,
            basin=basin,
            cybershake=cybershake,
            vs30_measured=vs30_measured,
            prvi=prvi,
        )

    def _calc(self) -> tuple:
        c = self._c
        periods = self._periods
        mag = self._column("mag")
        dist_rup = self._column("dist_rup")
        dist_jb = self._column("dist_jb")
        dist_x = self._column("dist_x")
        dip = self._column("dip")
        depth_tor = self._column("depth_tor")
        v_s30 = self._column("v_s30")
        depth_1_0 = self._column("depth_1_0", np.nan)
        mech = self._mechanism()

        # Magnitude dependent taper -- equation 4
        c4mag = np.where(
            mag > 5,
            self.C4,
            np.where(mag > 4, self.C4 - (self.C4 - 1.0) * (5.0 - mag), 1.0),
        )
        # Equation 3
        dist = np.sqrt(dist_rup * dist_rup + c4mag * c4mag)
        ln_dist = np.log(dist)
        # Equation 2
        max_mag_sq = (8.5 - mag) * (8.5 - mag)
        mag_m1 = mag - c.M1
        m2_m1 = self.M2 - c.M1
        f1 = (c.a1 + c.a17 * dist_rup) + np.where(
            mag > c.M1,
            self.A5 * mag_m1 + c.a8 * max_mag_sq + (c.a2 + self.A3 * mag_m1) * ln_dist,
            np.where(
                mag >= self.M2,
                self.A4 * mag_m1
                + c.a8 * max_mag_sq
                + (c.a2 + self.A3 * mag_m1) * ln_dist,
                self.A4 * m2_m1
                + c.a8 * ((8.5 - self.M2) * (8.5 - self.M2))
                + c.a6 * (mag - self.M2)
                + (c.a2 + self.A3 * m2_m1) * ln_dist,
            ),
        )

        # Hanging-wall model
        on_hw = (dist_jb < 30) & (dist_x >= 0.0) & (mag > 5.5) & (depth_tor <= 10.0)
        if np.any(on_hw):
            width = self._width()
            # Dip taper -- equation 11
            t1 = np.where(dip > 30.0, (90.0 - dip) / 45, 1.33333333)
            # Magnitude taper -- equation 12
            d_m = mag - 6.5
            t2 = np.where(
                mag >= 6.5,
                1 + self.A2_HW * d_m,
                1 + self.A2_HW * d_m - (1 - self.A2_HW) * d_m * d_m,
            )
            # R_x taper -- equation 13
            r1 = width * np.cos(np.radians(dip))
            r2 = 3 * r1
            with np.errstate(divide="ignore", invalid="ignore"):
                rx_r1 = dist_x / r1
                t3 = np.where(
                    dist_x <= r1,
                    self.H1 + self.H2 * rx_r1 + self.H3 * rx_r1 * rx_r1,
                    np.where(dist_x <= r2, 1 - (dist_x - r1) / (r2 - r1), 0.0),
                )
            # Z_tor taper -- equation 14
            t4 = 1 - (depth_tor * depth_tor) / 100.0
            # R_JB taper -- equation 15b
            t5 = np.where(dist_jb == 0.0, 1.0, 1 - dist_jb / 30.0)
            f4 = np.where(on_hw, c.a13 * t1 * t2 * t3 * t4 * t5, 0.0)
        else:
            f4 = 0.0

        # Depth to top of rupture -- equation 16
        f6 = np.where(depth_tor < 20.0, c.a15 * (depth_tor / 20.0), c.a15)

        # Style of faulting (a11 = 0 for reverse) -- equations 5 and 6
        f78 = np.where(
            mech["NS"],
            np.where(mag > 5.0, c.a12, np.where(mag >= 4.0, c.a12 * (mag - 4), 0.0)),
            0.0,
        )

        # Soil depth model -- equations 17 and 18
        cy = self._cybershake_periods()
        v_s30_pow4 = v_s30 * v_s30 * v_s30 * v_s30
        z1_ref = np.exp(-7.67 / 4.0 * np.log((v_s30_pow4 + self.A) / self.B)) / 1000.0
        slope = _interp_extrap(
            self.VS_BINS,
            [
                c.a43,
                np.where(cy, c.a44cy, c.a44),
                np.where(cy, c.a45cy, c.a45),
                c.a46,
                c.a46,
            ],
            v_s30,
        )
        f10 = slope * np.log((depth_1_0 + 0.01) / (z1_ref + 0.01))
        f10 = np.where(cy, f10 + CY_CSIM, f10)
        f10 = np.where(np.isnan(depth_1_0), 0.0, f10)
        if self._basin:
            f10 = f10 * basin_scale(
                periods, depth_1_0, BASIN_Z1P0_UPPER, BASIN_Z1P0_LOWER
            )

        # Site response model -- equations 7 to 9
        v_1 = np.where(
            periods >= 3.0,
            800.0,
            np.where(
                periods > 0.5,
                np.exp(-0.35 * np.log(np.maximum(periods, 0.5) / 0.5) + np.log(1500.0)),
                1500.0,
            ),
        )
        vs30_s = np.where(v_s30 < v_1, v_s30, v_1)
        nonlinear = v_s30 < c.Vlin
        vs30_s_rk = np.where(self.VS_RK < v_1, self.VS_RK, v_1)
        f5_rk = (c.a10 + c.b * self.N) * np.log(vs30_s_rk / c.Vlin)
        sa_rock = np.where(nonlinear, np.exp(f1 + f78 + f5_rk + f4 + f6), 0.0)
        vs_ratio = vs30_s / c.Vlin
        f5 = np.where(
            nonlinear,
            c.a10 * np.log(vs_ratio)
            - c.b * np.log(sa_rock + c.c)
            + c.b * np.log(sa_rock + c.c * vs_ratio**self.N),
            (c.a10 + c.b * self.N) * np.log(vs_ratio),
        )

        # Equation 1 (no aftershock term)
        ln_resp = f1 + f78 + f5 + f4 + f6 + f10

        # Aleatory variability -- equations 24 to 30
        s1, s2 = (c.s1m, c.s2m) if self._vs30_measured else (c.s1e, c.s2e)
        phi_a = np.where(
            mag < 4.0, s1, np.where(mag > 6.0, s2, s1 + ((s2 - s1) / 2) * (mag - 4.0))
        )
        tau_b = np.where(
            mag < 5.0,
            c.s3,
            np.where(mag > 7.0, c.s4, c.s3 + ((c.s4 - c.s3) / 2) * (mag - 5.0)),
        )
        phi_b_sq = phi_a * phi_a - self.PHI_AMP_SQ
        d_amp = np.where(
            v_s30 >= c.Vlin,
            0.0,
            (-c.b * sa_rock) / (sa_rock + c.c)
            + (c.b * sa_rock) / (sa_rock + c.c * (v_s30 / c.Vlin) ** self.N),
        )
        d_amp_p1 = d_amp + 1.0
        phi_sq = phi_b_sq * d_amp_p1 * d_amp_p1 + self.PHI_AMP_SQ
        tau = tau_b * d_amp_p1
        ln_std = np.sqrt(phi_sq + tau * tau)
        return ln_resp, ln_std


class BooreStewartSeyhanAtkinson2014Nshmp(_NgaWest2NshmpBase):
    r"""Boore, Stewart, Seyhan, and Atkinson (2014) model as in nshmp-lib.

    The ``BooreEtAl_2014`` class of USGS nshmp-lib (commit 44728a7d), with
    the coefficients of ``BSSA14.csv`` and ``BSSA14_PRVI.csv`` (see
    :mod:`pygmm.nga_west2_nshmp` for the options, the scenario values, and the
    treatment of missing values). Compared with the published model
    (:class:`~pygmm.boore_stewart_seyhan_atkinson_2014.BooreStewartSeyhanAtkinson2014`),
    nshmp-lib uses the global (California and Taiwan) anelastic attenuation,
    and has no basin term if ``depth_1_0`` is *None* or NaN (the basin term
    is only used for spectral periods of 0.65 s and longer, not PGV).

    The nshmp-lib ``Gmm`` ids are (see :attr:`GMM_IDS`):

    ========================  =========================================
    nshmp-lib ``Gmm``         Options
    ========================  =========================================
    ``BSSA_14``               (defaults)
    ``BSSA_14_BASE``          ``epistemic=False``
    ``BSSA_14_BASIN``         ``basin=True``
    ``BSSA_14_CYBERSHAKE``    ``basin=True, cybershake=True``
    ``BSSA_14_PRVI``          ``epistemic=False, prvi=True``
    ========================  =========================================

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario with ``mag``, ``dist_jb``, ``mechanism``,
        ``v_s30``, and optionally ``depth_1_0``
    epistemic : bool, optional
        include the USGS epistemic branches (default *True*)
    basin : bool, optional
        use the USGS deep basin model (default *False*)
    cybershake : bool, optional
        use the CyberShake basin terms (default *False*, requires
        ``basin=True``)
    prvi : bool, optional
        use the PRVI coefficients (default *False*)
    ims : str or sequence of str, optional
        intensity measures to compute (see
        :class:`AbrahamsonSilvaKamai2014Nshmp`). If *None* (default), all
        intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_west2_nshmp import BooreStewartSeyhanAtkinson2014Nshmp
    >>> s = Scenario(
    ...     mag=np.array([6.0, 7.0]), dist_jb=np.array([8.0, 48.0]),
    ...     mechanism=np.array(["SS", "U"]), v_s30=400.0)
    >>> BooreStewartSeyhanAtkinson2014Nshmp(s, basin=True).spec_accels.shape
    (2, 21)
    """

    NAME = "Boore, Stewart, Seyhan, & Atkinson (2014) (USGS nshmp-lib)"
    ABBREV = "BSSA14_NSHMP"

    COEFF, COEFF_PRVI = _load_coeffs(
        "boore_stewart_seyhan_atkinson_2014-nshmp.csv",
        "boore_stewart_seyhan_atkinson_2014-nshmp-prvi.csv",
    )
    PERIODS = COEFF["period"]

    GMM_IDS = {
        "BSSA_14": dict(),
        "BSSA_14_BASE": dict(epistemic=False),
        "BSSA_14_BASIN": dict(basin=True),
        "BSSA_14_CYBERSHAKE": dict(basin=True, cybershake=True),
        "BSSA_14_PRVI": dict(epistemic=False, prvi=True),
    }
    OPTIONS = ("epistemic", "basin", "cybershake", "prvi")

    A = 570.94**4
    B = 1360.0**4 + A
    M_REF = 4.5
    R_REF = 1.0
    DC3_CA_TW = 0.0
    V_REF = 760.0
    F1 = 0.0
    F3 = 0.1
    V1 = 225.0
    V2 = 300.0

    PARAMS = [
        model.NumericParameter("mag", True, 3.0, 8.5),
        model.NumericParameter("dist_jb", True, 0.0, 400.0),
        model.CategoricalParameter("mechanism", True, MECHANISMS),
        model.NumericParameter("v_s30", True, 150.0, 1500.0),
        model.NumericParameter("depth_1_0", False, 0.0, 3.0),
    ]

    def __init__(
        self,
        scenario: model.Scenario,
        epistemic: bool = True,
        basin: bool = False,
        cybershake: bool = False,
        prvi: bool = False,
        ims=None,
    ):
        """Initialize the model."""
        super().__init__(
            scenario,
            ims,
            epistemic=epistemic,
            basin=basin,
            cybershake=cybershake,
            prvi=prvi,
        )

    def _source_path(self, c, mag, dist_jb, mech) -> np.ndarray:
        """Source and path terms -- equations 2 to 4."""
        fe = np.select(
            [mech["SS"], mech["RS"], mech["NS"]], [c.e1, c.e3, c.e2], default=c.e0
        )
        mag_mh = mag - c.Mh
        fe = fe + np.where(
            mag <= c.Mh, c.e4 * mag_mh + c.e5 * mag_mh * mag_mh, c.e6 * mag_mh
        )
        dist = np.sqrt(dist_jb * dist_jb + c.h * c.h)
        fp = (c.c1 + c.c2 * (mag - self.M_REF)) * np.log(dist / self.R_REF) + (
            c.c3 + self.DC3_CA_TW
        ) * (dist - self.R_REF)
        return fe, fp

    def _calc(self) -> tuple:
        c = self._c
        periods = self._periods
        mag = self._column("mag")
        dist_jb = self._column("dist_jb")
        v_s30 = self._column("v_s30")
        depth_1_0 = self._column("depth_1_0", np.nan)
        mech = self._mechanism()

        # Median PGA at the reference rock (V_s30 = 760 m/sec)
        fe_pga, fp_pga = self._source_path(self._c_pga, mag, dist_jb, mech)
        pga_rock = np.exp(fe_pga + fp_pga)

        fe, fp = self._source_path(c, mag, dist_jb, mech)

        # Linear site term -- equation 6
        vs_lin = np.where(v_s30 <= c.Vc, v_s30, c.Vc)
        ln_f_lin = c.c * np.log(vs_lin / self.V_REF)
        # Nonlinear site term -- equations 7 and 8
        f2 = c.f4 * (
            np.exp(c.f5 * (np.minimum(v_s30, 760.0) - 360.0))
            - np.exp(c.f5 * (760.0 - 360.0))
        )
        ln_f_nl = self.F1 + f2 * np.log((pga_rock + self.F3) / self.F3)

        # Basin depth term -- equations 9 to 11
        v_s30_pow4 = v_s30 * v_s30 * v_s30 * v_s30
        z1_ref = np.exp(-7.15 / 4.0 * np.log((v_s30_pow4 + self.A) / self.B)) / 1000.0
        d_z1 = np.where(np.isnan(depth_1_0), 0.0, depth_1_0 - z1_ref)
        cy = self._cybershake_periods()
        with np.errstate(divide="ignore", invalid="ignore"):
            f_dz1 = np.where(
                cy,
                np.where(d_z1 <= c.dz1cy, c.f6cy * d_z1, c.f7cy) + CY_CSIM,
                np.where(d_z1 <= c.f7 / c.f6, c.f6 * d_z1, c.f7),
            )
        f_dz1 = np.where(periods >= 0.65, f_dz1, 0.0)
        if self._basin:
            f_dz1 = f_dz1 * basin_scale(
                periods, depth_1_0, BASIN_Z1P0_UPPER, BASIN_Z1P0_LOWER
            )

        # Equations 1 and 5
        f_s = ln_f_lin + ln_f_nl + f_dz1
        ln_resp = fe + fp + f_s

        # Aleatory variability -- equations 13 to 17
        tau = np.where(
            mag >= 5.5,
            c.tau2,
            np.where(mag <= 4.5, c.tau1, c.tau1 + (c.tau2 - c.tau1) * (mag - 4.5)),
        )
        phi_m = np.where(
            mag >= 5.5,
            c.phi2,
            np.where(mag <= 4.5, c.phi1, c.phi1 + (c.phi2 - c.phi1) * (mag - 4.5)),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_mr = phi_m + np.where(
                dist_jb > c.R2,
                c.dPhiR,
                np.where(
                    dist_jb > c.R1,
                    c.dPhiR * (np.log(dist_jb / c.R1) / np.log(c.R2 / c.R1)),
                    0.0,
                ),
            )
        phi_mrv = phi_mr - np.where(
            v_s30 <= self.V1,
            c.dPhiV,
            np.where(
                v_s30 < self.V2,
                c.dPhiV * (np.log(self.V2 / v_s30) / np.log(self.V2 / self.V1)),
                0.0,
            ),
        )
        ln_std = np.sqrt(phi_mrv * phi_mrv + tau * tau)
        return ln_resp, ln_std


class CampbellBozorgnia2014Nshmp(_NgaWest2NshmpBase):
    r"""Campbell and Bozorgnia (2014) model as implemented in nshmp-lib.

    The ``CampbellBozorgnia_2014`` class of USGS nshmp-lib (commit 44728a7d),
    with the coefficients of ``CB14.csv`` and ``CB14_PRVI.csv`` (see
    :mod:`pygmm.nga_west2_nshmp` for the options, the scenario values, and the
    treatment of missing values). Compared with the published model
    (:class:`~pygmm.campbell_bozorgnia_2014.CampbellBozorgnia2014`),
    nshmp-lib:

    - uses the California model (no regional terms) and requires the
      hypocentral depth (``depth_hyp``, which is computed from the rupture
      geometry if it is *None*, see :mod:`pygmm.nga_west2_nshmp`),
    - uses the :math:`V_{S30}` based reference :math:`Z_{2.5}` (equation 33)
      if ``depth_2_5`` is *None* or NaN,
    - computes the reference rock PGA (:math:`V_{S30}` = 1100 m/sec) with
      :math:`Z_{2.5}` = 0.398 km (or, with ``basin=True``, the reference depth
      for 1100 m/sec), and only if :math:`V_{S30} < k_1` of the period
      (otherwise it is 0),
    - for periods from 0.01 to 0.25 s, uses the PGA (with the reference rock
      PGA of the period) if the spectral acceleration is smaller,
    - computes the standard deviation without the site amplification
      uncertainty of the reference rock PGA (equation 30, with the
      :math:`\phi_{\ln AF}` terms cancelling).

    The nshmp-lib ``Gmm`` ids are (see :attr:`GMM_IDS`):

    ========================  =========================================
    nshmp-lib ``Gmm``         Options
    ========================  =========================================
    ``CB_14``                 (defaults)
    ``CB_14_BASE``            ``epistemic=False``
    ``CB_14_BASIN``           ``basin=True``
    ``CB_14_CYBERSHAKE``      ``basin=True, cybershake=True``
    ``CB_14_PRVI``            ``epistemic=False, prvi=True``
    ========================  =========================================

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario with ``mag``, ``dist_rup``, ``dist_jb``,
        ``dist_x``, ``dip``, ``width`` (or ``depth_bor``), ``depth_tor``,
        ``depth_hyp`` (optional), ``mechanism``, ``v_s30``, and optionally
        ``depth_2_5``
    epistemic : bool, optional
        include the USGS epistemic branches (default *True*)
    basin : bool, optional
        use the USGS deep basin model (default *False*)
    cybershake : bool, optional
        use the CyberShake basin terms (default *False*, requires
        ``basin=True``)
    prvi : bool, optional
        use the PRVI coefficients (default *False*)
    ims : str or sequence of str, optional
        intensity measures to compute (see
        :class:`AbrahamsonSilvaKamai2014Nshmp`). If *None* (default), all
        intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_west2_nshmp import CampbellBozorgnia2014Nshmp
    >>> s = Scenario(
    ...     mag=np.array([6.0, 7.0]), dist_rup=np.array([10.0, 50.0]),
    ...     dist_jb=np.array([8.0, 48.0]), dist_x=np.array([8.0, -48.0]),
    ...     dip=60.0, depth_tor=2.0, depth_bor=15.0, mechanism="RS",
    ...     v_s30=400.0, depth_2_5=np.array([np.nan, 4.0]))
    >>> CampbellBozorgnia2014Nshmp(s, basin=True).spec_accels.shape
    (2, 21)
    """

    NAME = "Campbell & Bozorgnia (2014) (USGS nshmp-lib)"
    ABBREV = "CB14_NSHMP"

    COEFF, COEFF_PRVI = _load_coeffs(
        "campbell_bozorgnia_2014-nshmp.csv", "campbell_bozorgnia_2014-nshmp-prvi.csv"
    )
    PERIODS = COEFF["period"]

    GMM_IDS = {
        "CB_14": dict(),
        "CB_14_BASE": dict(epistemic=False),
        "CB_14_BASIN": dict(basin=True),
        "CB_14_CYBERSHAKE": dict(basin=True, cybershake=True),
        "CB_14_PRVI": dict(epistemic=False, prvi=True),
    }
    OPTIONS = ("epistemic", "basin", "cybershake", "prvi")

    H4 = 1.0
    C = 1.88
    N = 1.18
    PHI_LNAF_SQ = 0.09
    # Reference rock condition of the nonlinear site term
    V_S30_ROCK = 1100.0
    DEPTH_2_5_ROCK = 0.398

    PARAMS = [
        model.NumericParameter("mag", True, 3.3, 8.5),
        model.NumericParameter("dist_rup", True, 0.0, 300.0),
        model.NumericParameter("dist_jb", True, 0.0, 300.0),
        model.NumericParameter("dist_x", True),
        model.NumericParameter("dip", True, 0.0, 90.0),
        model.NumericParameter("width", False),
        model.NumericParameter("depth_bor", False),
        model.NumericParameter("depth_tor", True, 0.0, 20.0),
        model.NumericParameter("depth_hyp", False, 0.0, 20.0),
        model.CategoricalParameter("mechanism", True, MECHANISMS),
        model.NumericParameter("v_s30", True, 150.0, 1500.0),
        model.NumericParameter("depth_2_5", False, 0.0, 10.0),
    ]

    def __init__(
        self,
        scenario: model.Scenario,
        epistemic: bool = True,
        basin: bool = False,
        cybershake: bool = False,
        prvi: bool = False,
        ims=None,
    ):
        """Initialize the model."""
        super().__init__(
            scenario,
            ims,
            epistemic=epistemic,
            basin=basin,
            cybershake=cybershake,
            prvi=prvi,
        )

    @staticmethod
    def _basin_term(c, depth_2_5, cybershake) -> np.ndarray:
        """Basin term of a Z_2.5 (km) -- equation 20."""
        slope = np.where(cybershake, c.slope_cy, c.c16 * c.k3)
        return np.where(
            depth_2_5 <= 1.0,
            c.c14 * (depth_2_5 - 1.0),
            np.where(
                depth_2_5 > 3.0,
                slope * np.exp(-0.75) * (1.0 - np.exp(-0.25 * (depth_2_5 - 3.0))),
                0.0,
            ),
        )

    def _basin_response(self, c, periods, v_s30, depth_2_5, cybershake):
        """Basin response term with the reference depth -- equations 20, 33."""
        depth_ref = np.exp(7.089 - 1.144 * np.log(v_s30))
        term_ref = self._basin_term(c, depth_ref, False)
        term = self._basin_term(c, depth_2_5, cybershake)
        if self._basin:
            in_basin = (periods > 0.5) & (depth_2_5 > BASIN_Z2P5_UPPER)
            scale = (
                np.clip(depth_2_5, BASIN_Z2P5_UPPER, BASIN_Z2P5_LOWER)
                - BASIN_Z2P5_UPPER
            ) / (BASIN_Z2P5_LOWER - BASIN_Z2P5_UPPER)
            scaled = term_ref * (1.0 - scale) + term * scale
            scaled = np.where(cybershake, scaled + CY_CSIM, scaled)
            scaled = np.where(periods == 0.75, scaled * 0.585, scaled)
            term = np.where(in_basin, scaled, term_ref)
        return np.where(np.isnan(depth_2_5), term_ref, term)

    def _calc_mean(self, c, periods, v_s30, depth_2_5, pga_rock, cybershake, v):
        """Median -- equations 1 to 25."""
        mag = v["mag"]
        dist_rup = v["dist_rup"]
        dist_x = v["dist_x"]
        dip = v["dip"]
        depth_tor = v["depth_tor"]
        mech = v["mech"]

        # Magnitude term -- equation 2
        f_mag = (c.c0 + c.c1 * mag) + np.where(
            mag > 6.5,
            c.c2 * (mag - 4.5) + c.c3 * (mag - 5.5) + c.c4 * (mag - 6.5),
            np.where(
                mag > 5.5,
                c.c2 * (mag - 4.5) + c.c3 * (mag - 5.5),
                np.where(mag > 4.5, c.c2 * (mag - 4.5), 0.0),
            ),
        )

        # Distance term -- equation 3
        r = np.sqrt(dist_rup * dist_rup + c.c7 * c.c7)
        f_r = (c.c5 + c.c6 * mag) * np.log(r)

        # Style of faulting (c8 = 0 for reverse) -- equations 4 to 6
        f_flt = np.where(
            mech["NS"] & (mag > 4.5),
            np.where(mag <= 5.5, c.c9 * (mag - 4.5), c.c9),
            0.0,
        )

        # Hanging-wall term -- equations 7 to 16
        on_hw = (dist_x >= 0.0) & (mag > 5.5) & (depth_tor <= 16.66)
        if np.any(on_hw):
            r1 = v["width"] * np.cos(np.radians(dip))
            r2 = 62.0 * mag - 350.0
            with np.errstate(divide="ignore", invalid="ignore"):
                rx_r1 = dist_x / r1
                rx_r2_r1 = (dist_x - r1) / (r2 - r1)
                f1_rx = c.h1 + c.h2 * rx_r1 + c.h3 * (rx_r1 * rx_r1)
                f2_rx = self.H4 + c.h5 * rx_r2_r1 + c.h6 * rx_r2_r1 * rx_r2_r1
                f_hw_rx = np.where(dist_x >= r1, np.maximum(f2_rx, 0.0), f1_rx)
                f_hw_rrup = np.where(
                    dist_rup == 0.0, 1.0, (dist_rup - v["dist_jb"]) / dist_rup
                )
            f_hw_m = (1.0 + c.a2 * (mag - 6.5)) * np.where(mag <= 6.5, mag - 5.5, 1.0)
            f_hw_z = 1.0 - 0.06 * depth_tor
            f_hw_d = (90.0 - dip) / 45.0
            f_hw = np.where(
                on_hw, c.c10 * f_hw_rx * f_hw_rrup * f_hw_m * f_hw_z * f_hw_d, 0.0
            )
        else:
            f_hw = 0.0

        # Shallow site response -- equation 18
        vs_k1 = v_s30 / c.k1
        f_site = np.where(
            v_s30 <= c.k1,
            c.c11 * np.log(vs_k1)
            + c.k2
            * (np.log(pga_rock + self.C * vs_k1**self.N) - np.log(pga_rock + self.C)),
            (c.c11 + c.k2 * self.N) * np.log(vs_k1),
        )

        # Basin response -- equation 20
        f_sed = self._basin_response(c, periods, v_s30, depth_2_5, cybershake)

        # Hypocentral depth -- equations 21 to 23
        depth_hyp = v["depth_hyp"]
        f_hyp = np.where(
            depth_hyp <= 7.0, 0.0, np.where(depth_hyp <= 20.0, depth_hyp - 7.0, 13.0)
        )
        f_hyp = f_hyp * np.where(
            mag <= 5.5,
            c.c17,
            np.where(mag <= 6.5, c.c17 + (c.c18 - c.c17) * (mag - 5.5), c.c18),
        )

        # Fault dip -- equation 24
        f_dip = np.where(
            mag > 5.5, 0.0, np.where(mag > 4.5, c.c19 * (5.5 - mag) * dip, c.c19 * dip)
        )

        # Anelastic attenuation -- equation 25
        f_atn = np.where(dist_rup > 80.0, c.c20 * (dist_rup - 80.0), 0.0)

        return f_mag + f_r + f_flt + f_hw + f_site + f_sed + f_hyp + f_dip + f_atn

    @staticmethod
    def _std_mag_dep(lo, hi, mag):
        return hi + (lo - hi) * (5.5 - mag)

    def _calc(self) -> tuple:
        c = self._c
        c_pga = self._c_pga
        periods = self._periods
        mag = self._column("mag")
        v_s30 = self._column("v_s30")
        depth_2_5 = self._column("depth_2_5", np.nan)
        v = dict(
            mag=mag,
            dist_rup=self._column("dist_rup"),
            dist_jb=self._column("dist_jb"),
            dist_x=self._column("dist_x"),
            dip=self._column("dip"),
            depth_tor=self._column("depth_tor"),
            mech=self._mechanism(),
        )
        if self._scenario["depth_hyp"] is not None:
            v["depth_hyp"] = self._column("depth_hyp")
            on_hw = (v["dist_x"] >= 0.0) & (mag > 5.5) & (v["depth_tor"] <= 16.66)
            v["width"] = self._width() if np.any(on_hw) else np.nan
        else:
            # Faults.hypocentralDepth
            v["width"] = self._width()
            v["depth_hyp"] = (
                v["depth_tor"] + np.sin(np.radians(v["dip"])) * v["width"] / 2.0
            )
        pga_periods = np.zeros(1)

        # Reference rock PGA (V_s30 = 1100 m/sec), only for V_s30 < k1
        ln_pga_rock = self._calc_mean(
            c_pga,
            pga_periods,
            self.V_S30_ROCK,
            self.DEPTH_2_5_ROCK,
            0.0,
            False,
            v,
        )
        pga_rock = np.where(v_s30 < c.k1, np.exp(ln_pga_rock), 0.0)

        ln_resp = self._calc_mean(
            c, periods, v_s30, depth_2_5, pga_rock, self._cybershake_periods(), v
        )

        # Short periods are at least the PGA
        short = (periods >= 0.01) & (periods <= 0.25)
        if np.any(short):
            ln_pga = self._calc_mean(
                c_pga, pga_periods, v_s30, depth_2_5, pga_rock, False, v
            )
            ln_resp = np.where(short, np.maximum(ln_resp, ln_pga), ln_resp)

        # Aleatory variability -- equations 27 to 32
        with np.errstate(divide="ignore", invalid="ignore"):
            vs_k1 = v_s30 / c.k1
            alpha = np.where(
                v_s30 < c.k1,
                c.k2
                * pga_rock
                * (1 / (pga_rock + self.C * vs_k1**self.N) - 1 / (pga_rock + self.C)),
                0.0,
            )

        def mag_dep(name):
            lo = getattr(c, name + "1")
            hi = getattr(c, name + "2")
            lo_pga = getattr(c_pga, name + "1")
            hi_pga = getattr(c_pga, name + "2")
            value = np.where(
                mag <= 4.5,
                lo,
                np.where(mag < 5.5, self._std_mag_dep(lo, hi, mag), hi),
            )
            value_pga = np.where(
                mag <= 4.5,
                lo_pga,
                np.where(mag < 5.5, self._std_mag_dep(lo_pga, hi_pga, mag), hi_pga),
            )
            return value, value_pga

        tau_ln_yb, tau_ln_pgab = mag_dep("tau")
        phi_ln_y, phi_ln_pga = mag_dep("phi")

        alpha_tau = alpha * tau_ln_pgab
        tau_sq = (
            tau_ln_yb * tau_ln_yb
            + alpha_tau * alpha_tau
            + 2.0 * alpha * c.rho * tau_ln_yb * tau_ln_pgab
        )
        phi_ln_yb = np.sqrt(phi_ln_y * phi_ln_y - self.PHI_LNAF_SQ)
        phi_ln_pgab = np.sqrt(phi_ln_pga * phi_ln_pga - self.PHI_LNAF_SQ)
        a_phi_ln_pgab = alpha * phi_ln_pgab
        phi_sq = (
            phi_ln_y * phi_ln_y
            + a_phi_ln_pgab * a_phi_ln_pgab
            + 2.0 * c.rho * phi_ln_yb * a_phi_ln_pgab
        )
        ln_std = np.sqrt(phi_sq + tau_sq)
        return ln_resp, ln_std


class ChiouYoungs2014Nshmp(_NgaWest2NshmpBase):
    r"""Chiou and Youngs (2014) model as implemented in nshmp-lib.

    The ``ChiouYoungs_2014`` class of USGS nshmp-lib (commit 44728a7d), with
    the coefficients of ``CY14.csv`` and ``CY14_PRVI.csv`` (see
    :mod:`pygmm.nga_west2_nshmp` for the options, the scenario values, and the
    treatment of missing values). Compared with the published model
    (:class:`~pygmm.chiou_youngs_2014.ChiouYoungs2014`), nshmp-lib uses the
    California model (no regional terms), has no directivity term, has no
    basin term if ``depth_1_0`` is *None* or NaN, and uses the aleatory
    variability for an inferred :math:`V_{S30}` unless ``vs30_measured=True``.

    The nshmp-lib ``Gmm`` ids are (see :attr:`GMM_IDS`):

    ========================  =========================================
    nshmp-lib ``Gmm``         Options
    ========================  =========================================
    ``CY_14``                 (defaults)
    ``CY_14_BASE``            ``epistemic=False``
    ``CY_14_BASIN``           ``basin=True``
    ``CY_14_CYBERSHAKE``      ``basin=True, cybershake=True``
    ``CY_14_VS30_MEASURED``   ``epistemic=False, vs30_measured=True``
    ``CY_14_PRVI``            ``epistemic=False, prvi=True``
    ========================  =========================================

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario with ``mag``, ``dist_rup``, ``dist_jb``,
        ``dist_x``, ``dip``, ``depth_tor``, ``mechanism``, ``v_s30``, and
        optionally ``depth_1_0``
    epistemic : bool, optional
        include the USGS epistemic branches (default *True*)
    basin : bool, optional
        use the USGS deep basin model (default *False*)
    cybershake : bool, optional
        use the CyberShake basin terms (default *False*, requires
        ``basin=True``)
    vs30_measured : bool, optional
        use the aleatory variability for a measured :math:`V_{S30}` (default
        *False*)
    prvi : bool, optional
        use the PRVI coefficients (default *False*)
    ims : str or sequence of str, optional
        intensity measures to compute (see
        :class:`AbrahamsonSilvaKamai2014Nshmp`). If *None* (default), all
        intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_west2_nshmp import ChiouYoungs2014Nshmp
    >>> s = Scenario(
    ...     mag=np.array([6.0, 7.0]), dist_rup=np.array([10.0, 50.0]),
    ...     dist_jb=np.array([8.0, 48.0]), dist_x=np.array([8.0, -48.0]),
    ...     dip=60.0, depth_tor=2.0, mechanism="RS", v_s30=400.0,
    ...     depth_1_0=np.array([np.nan, 0.6]))
    >>> ChiouYoungs2014Nshmp(s, basin=True).spec_accels.shape
    (2, 21)
    """

    NAME = "Chiou & Youngs (2014) (USGS nshmp-lib)"
    ABBREV = "CY14_NSHMP"

    COEFF, COEFF_PRVI = _load_coeffs(
        "chiou_youngs_2014-nshmp.csv", "chiou_youngs_2014-nshmp-prvi.csv"
    )
    PERIODS = COEFF["period"]

    GMM_IDS = {
        "CY_14": dict(),
        "CY_14_BASE": dict(epistemic=False),
        "CY_14_BASIN": dict(basin=True),
        "CY_14_CYBERSHAKE": dict(basin=True, cybershake=True),
        "CY_14_VS30_MEASURED": dict(epistemic=False, vs30_measured=True),
        "CY_14_PRVI": dict(epistemic=False, prvi=True),
    }
    OPTIONS = ("epistemic", "basin", "cybershake", "vs30_measured", "prvi")

    C2 = 1.06
    C4 = -2.1
    C4A = -0.5
    D_C4 = C4A - C4
    CRB = 50.0
    C11 = 0.0
    PHI6 = 300.0
    A = 571.0**4
    B = 1360.0**4 + A

    PARAMS = [
        model.NumericParameter("mag", True, 3.5, 8.5),
        model.NumericParameter("dist_rup", True, 0.0, 300.0),
        model.NumericParameter("dist_jb", True, 0.0, 300.0),
        model.NumericParameter("dist_x", True),
        model.NumericParameter("dip", True, 0.0, 90.0),
        model.NumericParameter("depth_tor", True, 0.0, 20.0),
        model.CategoricalParameter("mechanism", True, MECHANISMS),
        model.NumericParameter("v_s30", True, 180.0, 1500.0),
        model.NumericParameter("depth_1_0", False, 0.0, 3.0),
    ]

    def __init__(
        self,
        scenario: model.Scenario,
        epistemic: bool = True,
        basin: bool = False,
        cybershake: bool = False,
        vs30_measured: bool = False,
        prvi: bool = False,
        ims=None,
    ):
        """Initialize the model."""
        super().__init__(
            scenario,
            ims,
            epistemic=epistemic,
            basin=basin,
            cybershake=cybershake,
            vs30_measured=vs30_measured,
            prvi=prvi,
        )

    def _calc(self) -> tuple:
        c = self._c
        periods = self._periods
        mag = self._column("mag")
        dist_rup = self._column("dist_rup")
        dist_jb = self._column("dist_jb")
        dist_x = self._column("dist_x")
        dip = self._column("dip")
        depth_tor = self._column("depth_tor")
        v_s30 = self._column("v_s30")
        depth_1_0 = self._column("depth_1_0", np.nan)
        mech = self._mechanism()

        # Reference response (V_s30 = 1130 m/sec) -- equation 11
        # Magnitude scaling
        r1 = (
            c.c1
            + self.C2 * (mag - 6.0)
            + ((self.C2 - c.c3) / c.cn) * np.log(1.0 + np.exp(c.cn * (c.cM - mag)))
        )
        # Near-field magnitude and distance scaling
        r2 = self.C4 * np.log(
            dist_rup + c.c5 * np.cosh(c.c6 * np.maximum(mag - c.cHM, 0.0))
        )
        # Far-field distance scaling
        gamma = c.cgamma1 + c.cgamma2 / np.cosh(np.maximum(mag - c.cgamma3, 0.0))
        r3 = (
            self.D_C4 * np.log(np.sqrt(dist_rup * dist_rup + self.CRB * self.CRB))
            + dist_rup * gamma
        )
        # Other source scaling
        cosh_m = np.cosh(2 * np.maximum(mag - 4.5, 0))
        cos_dip = np.cos(np.radians(dip))
        # Centered Z_tor -- equations 4 and 5
        is_rs = mech["RS"]
        mz_tor = np.where(
            is_rs,
            np.where(mag <= 5.849, 2.704, np.maximum(2.704 - 1.226 * (mag - 5.849), 0)),
            np.where(mag <= 4.970, 2.673, np.maximum(2.673 - 1.136 * (mag - 4.970), 0)),
        )
        d_z_tor = depth_tor - mz_tor * mz_tor
        r4 = (c.c7 + c.c7b / cosh_m) * d_z_tor + (
            self.C11 + c.c11b / cosh_m
        ) * cos_dip * cos_dip
        r4 = r4 + np.where(
            is_rs,
            c.c1a + c.c1c / cosh_m,
            np.where(mech["NS"], c.c1b + c.c1d / cosh_m, 0.0),
        )
        # Hanging-wall effect
        r5 = np.where(
            dist_x >= 0.0,
            c.c9
            * np.cos(np.radians(dip))
            * (c.c9a + (1.0 - c.c9a) * np.tanh(dist_x / c.c9b))
            * (
                1
                - np.sqrt(dist_jb * dist_jb + depth_tor * depth_tor) / (dist_rup + 1.0)
            ),
            0.0,
        )
        sa_ref = np.exp(r1 + r2 + r3 + r4 + r5)

        # Soil nonlinearity
        snl = c.phi2 * (
            np.exp(c.phi3 * (np.minimum(v_s30, 1130.0) - 360.0))
            - np.exp(c.phi3 * (1130.0 - 360.0))
        )

        # Mean -- equation 12
        sl = c.phi1 * np.minimum(np.log(v_s30 / 1130.0), 0.0)
        snl_mod = snl * np.log((sa_ref + c.phi4) / c.phi4)
        # Sediment thickness -- equation 1
        v_s30_pow4 = v_s30 * v_s30 * v_s30 * v_s30
        z1_ref = np.exp(-7.15 / 4 * np.log((v_s30_pow4 + self.A) / self.B))
        d_z1 = np.where(np.isnan(depth_1_0), 0.0, depth_1_0 * 1000.0 - z1_ref)
        cy = self._cybershake_periods()
        rk_depth = np.where(
            cy,
            c.phi5cy * (1.0 - np.exp(-d_z1 / c.phi6cy)) + CY_CSIM,
            c.phi5 * (1.0 - np.exp(-d_z1 / self.PHI6)),
        )
        if self._basin:
            rk_depth = rk_depth * basin_scale(
                periods, depth_1_0, BASIN_Z1P0_UPPER, BASIN_Z1P0_LOWER
            )
        ln_resp = np.log(sa_ref) + sl + snl_mod + rk_depth

        # Aleatory variability
        nl0 = snl * sa_ref / (sa_ref + c.phi4)
        m_test = np.minimum(np.maximum(mag, 5.0), 6.5) - 5.0
        tau = c.tau1 + (c.tau2 - c.tau1) / 1.5 * m_test
        sigma_nl0 = c.sigma1 + (c.sigma2 - c.sigma1) / 1.5 * m_test
        vs_term = 0.7 if self._vs30_measured else c.sigma3
        nl0_sq = (1 + nl0) * (1 + nl0)
        sigma_nl0 = sigma_nl0 * np.sqrt(vs_term + nl0_sq)
        ln_std = np.sqrt(tau * tau * nl0_sq + sigma_nl0 * sigma_nl0)
        return ln_resp, ln_std


class Idriss2014Nshmp(_NgaWest2NshmpBase):
    r"""Idriss (2014) model as implemented in nshmp-lib.

    The ``Idriss_2014`` class of USGS nshmp-lib (commit 44728a7d), with the
    coefficients of ``Idriss14.csv`` (see :mod:`pygmm.nga_west2_nshmp` for the
    options and the scenario values). The large-magnitude coefficients are
    used for magnitudes greater than 6.75, :math:`V_{S30}` is capped at 1200
    m/sec, and the reverse-faulting term is used for reverse faulting. PGV is
    not provided. ``dist_jb`` is only needed for the epistemic uncertainty.

    The nshmp-lib ``Gmm`` ids are (see :attr:`GMM_IDS`):

    ========================  =========================================
    nshmp-lib ``Gmm``         Options
    ========================  =========================================
    ``IDRISS_14``             (defaults)
    ``IDRISS_14_BASE``        ``epistemic=False``
    ========================  =========================================

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario with ``mag``, ``dist_rup``, ``mechanism``,
        ``v_s30``, and ``dist_jb`` (for ``epistemic=True``)
    epistemic : bool, optional
        include the USGS epistemic branches (default *True*)
    ims : str or sequence of str, optional
        intensity measures to compute (see
        :class:`AbrahamsonSilvaKamai2014Nshmp`). If *None* (default), all
        intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_west2_nshmp import Idriss2014Nshmp
    >>> s = Scenario(
    ...     mag=np.array([6.0, 7.0]), dist_rup=np.array([10.0, 50.0]),
    ...     mechanism="RS", v_s30=760.0)
    >>> Idriss2014Nshmp(s, epistemic=False).spec_accels.shape
    (2, 21)
    """

    NAME = "Idriss (2014) (USGS nshmp-lib)"
    ABBREV = "I14_NSHMP"

    COEFF, _ = _load_coeffs("idriss_2014-nshmp.csv")
    PERIODS = COEFF["period"]

    INDEX_PGV = None
    INDEX_PGA = 0
    INDICES_PSA = np.arange(1, 22)

    GMM_IDS = {
        "IDRISS_14": dict(),
        "IDRISS_14_BASE": dict(epistemic=False),
    }
    OPTIONS = ("epistemic",)

    PARAMS = [
        model.NumericParameter("mag", True, 5.0, 8.5),
        model.NumericParameter("dist_rup", True, 0.0, 150.0),
        model.NumericParameter("dist_jb", False),
        model.CategoricalParameter("mechanism", True, MECHANISMS),
        model.NumericParameter("v_s30", True, 450.0, 1500.0),
    ]

    def __init__(self, scenario: model.Scenario, epistemic: bool = True, ims=None):
        """Initialize the model."""
        super().__init__(scenario, ims, epistemic=epistemic)

    def _calc(self) -> tuple:
        c = self._c
        periods = self._periods
        mag = self._column("mag")
        dist_rup = self._column("dist_rup")
        v_s30 = self._column("v_s30")
        is_rs = self._mechanism()["RS"]

        large = mag > 6.75
        a1 = np.where(large, c.a1_hi, c.a1_lo)
        a2 = np.where(large, c.a2_hi, c.a2_lo)
        b1 = np.where(large, c.b1_hi, c.b1_lo)
        b2 = np.where(large, c.b2_hi, c.b2_lo)
        ln_resp = (
            a1
            + a2 * mag
            + c.a3 * (8.5 - mag) * (8.5 - mag)
            - (b1 + b2 * mag) * np.log(dist_rup + 10.0)
            + c.xi * np.log(np.minimum(v_s30, 1200.0))
            + c.gamma * dist_rup
            + np.where(is_rs, c.phi, 0.0)
        )

        # PGA (period 0) uses ln(0.05)
        s1 = 0.035 * np.where(
            periods <= 0.05,
            np.log(0.05),
            np.where(periods < 3.0, np.log(np.maximum(periods, 0.05)), np.log(3.0)),
        )
        s2 = 0.06 * np.where(mag <= 5.0, 5.0, np.where(mag < 7.5, mag, 7.5))
        ln_std = 1.18 + s1 - s2
        return ln_resp, ln_std


class NgaWest2NshmpTree(model.GroundMotionModel):
    r"""USGS 2023 CONUS NSHM logic trees of the NGA-West2 models.

    The weighted logic trees of the nshmp-lib NGA-West2 models used for
    active crust in the 2023 conterminous U.S. NSHM (nshmp-lib
    ``GmmTotalTree``):

    - "conus" (``TOTAL_TREE_CONUS_ACTIVE_CRUST_2023``, the active crust tree
      of the current CONUS NSHM, nshm-conus 6.2.0): ``ASK_14_BASIN``,
      ``BSSA_14_BASIN``, ``CB_14_BASIN``, and ``CY_14_BASIN`` with weights of
      0.25.
    - "los_angeles" (``TOTAL_TREE_CONUS_ACTIVE_CRUST_2023_LOS_ANGELES``):
      ``ASK_14``, ``BSSA_14``, ``CB_14``, and ``CY_14`` with weights of 0.125,
      and the ``_BASIN`` and ``_CYBERSHAKE`` variants with weights of 0.0625.
    - "san_francisco" (``TOTAL_TREE_CONUS_ACTIVE_CRUST_2023_SAN_FRANCISCO``):
      ``ASK_14``, ``BSSA_14``, ``CB_14``, and ``CY_14`` and the ``_BASIN``
      variants with weights of 0.125.

    As in nshmp-lib (``GroundMotions.combine``), all of the branches (models
    and epistemic branches) are collapsed to a single median, :math:`\ln
    \sum_i w_i \exp(\mu_i)`, and standard deviation, :math:`\sqrt{\sum_i w_i
    \sigma_i^2}`. The collapsed values are suitable for comparisons, not for
    hazard calculations, which should use the individual models and
    epistemic branches.

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario with the values of the models (see
        :mod:`pygmm.nga_west2_nshmp`)
    tree : str, optional
        logic tree: "conus" (default), "los_angeles", or "san_francisco"
    ims : str or sequence of str, optional
        intensity measures to compute (see
        :class:`AbrahamsonSilvaKamai2014Nshmp`). If *None* (default), all
        intensity measures are computed.

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_west2_nshmp import NgaWest2NshmpTree
    >>> s = Scenario(
    ...     mag=np.array([6.0, 7.0]), dist_rup=np.array([10.0, 50.0]),
    ...     dist_jb=np.array([8.0, 48.0]), dist_x=np.array([8.0, -48.0]),
    ...     dip=60.0, depth_tor=2.0, depth_bor=15.0, mechanism="RS",
    ...     v_s30=400.0, depth_1_0=0.6, depth_2_5=4.0)
    >>> NgaWest2NshmpTree(s, ims=["pga", "psa_1p000"]).spec_accels.shape
    (2, 1)
    """

    NAME = "NGA-West2 2023 CONUS NSHM logic tree (USGS nshmp-lib)"
    ABBREV = "NGAW2_NSHMP_TREE"

    MODELS = (
        AbrahamsonSilvaKamai2014Nshmp,
        BooreStewartSeyhanAtkinson2014Nshmp,
        CampbellBozorgnia2014Nshmp,
        ChiouYoungs2014Nshmp,
    )

    #: Logic trees as (model, options, weight)
    TREES = {
        "conus": tuple((m, dict(basin=True), 0.25) for m in MODELS),
        "los_angeles": tuple((m, dict(), 0.125) for m in MODELS)
        + tuple((m, dict(basin=True), 0.0625) for m in MODELS)
        + tuple((m, dict(basin=True, cybershake=True), 0.0625) for m in MODELS),
        "san_francisco": tuple((m, dict(), 0.125) for m in MODELS)
        + tuple((m, dict(basin=True), 0.125) for m in MODELS),
    }

    GMM_IDS = {
        "TOTAL_TREE_CONUS_ACTIVE_CRUST_2023": dict(tree="conus"),
        "TOTAL_TREE_CONUS_ACTIVE_CRUST_2023_LOS_ANGELES": dict(tree="los_angeles"),
        "TOTAL_TREE_CONUS_ACTIVE_CRUST_2023_SAN_FRANCISCO": dict(tree="san_francisco"),
    }

    PERIODS = AbrahamsonSilvaKamai2014Nshmp.PERIODS
    INDEX_PGV = 0
    INDEX_PGA = 1
    INDICES_PSA = np.arange(2, 23)

    PARAMS = [
        model.NumericParameter("mag", True),
        model.NumericParameter("dist_rup", True),
        model.NumericParameter("dist_jb", True),
        model.NumericParameter("dist_x", True),
        model.NumericParameter("dip", True),
        model.NumericParameter("width", False),
        model.NumericParameter("depth_bor", False),
        model.NumericParameter("depth_tor", True),
        model.NumericParameter("depth_hyp", False),
        model.CategoricalParameter("mechanism", True, MECHANISMS),
        model.NumericParameter("v_s30", True),
        model.NumericParameter("depth_1_0", False),
        model.NumericParameter("depth_2_5", False),
    ]

    def __init__(self, scenario: model.Scenario, tree: str = "conus", ims=None):
        """Initialize the model."""
        if tree not in self.TREES:
            raise ValueError(
                f"tree must be one of {', '.join(self.TREES)}, not {tree!r}"
            )
        self.tree = tree
        super().__init__(scenario, ims)
        sum_resp = 0.0
        sum_var = 0.0
        for cls, options, weight in self.TREES[tree]:
            assert np.array_equal(cls.PERIODS, self.PERIODS)
            m = cls(self._scenario, ims=ims, **options)
            sum_resp = sum_resp + weight * np.exp(m._ln_resp)
            sum_var = sum_var + weight * m._ln_std * m._ln_std
        self._ln_resp = np.log(sum_resp)
        self._ln_std = np.sqrt(sum_var)
