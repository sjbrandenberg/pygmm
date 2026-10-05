"""NGA-East ground motion model as implemented in USGS nshmp-lib."""

import collections
import functools
import gzip
import os

import numpy as np

from . import model
from .nga_east_usgs_2017 import NgaEastUsgs2017, _load_tables

#: Versions of the model
VERSIONS = ("2018", "2023", "2026")

#: Weights of the 5th, 50th, and 95th percentile branches of the site amplification
EPI_WTS = (0.185, 0.63, 0.185)
#: Standard normal variate of the 95th percentile
Z_SCORE_5_95 = 1.645

#: Weights of the updated EPRI (2013) and panel standard deviation models
SIGMA_WTS = (0.8, 0.2)

#: Reference V_S30 (m/s) of the Chapman and Guo (2021) PSA ratios
VS_REF_CPA = 1000.0
#: Sediment thickness scale (km) of the coastal plain taper
Z_CUT = 0.2

#: Site amplification options of each version (nshmp-lib ``SiteAmp_2018``,
#: ``SiteAmp_2023``, and ``SiteAmp_2026``): standard normal variate of the
#: site amplification branches, if f4 is the average of the Hashash et al. (2017)
#: and (2020) values, the minimum V_S30 (m/s) of the nonlinear term, and if the
#: rock PGA of the nonlinear term is capped at 1 g
SITE_AMP_OPTIONS = {
    "2018": dict(z_score=1.0, f4_mod=False, v_min_nl=150.0, cap_pga_rock=False),
    "2023": dict(z_score=Z_SCORE_5_95, f4_mod=True, v_min_nl=150.0, cap_pga_rock=False),
    "2026": dict(z_score=Z_SCORE_5_95, f4_mod=True, v_min_nl=200.0, cap_pga_rock=True),
}

# Grid of the Chapman and Guo (2021) PSA ratio tables
CPA_DEPTHS = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    + [8.5, 9.5, 10.5, 11.5, 12.5]
)
CPA_MAGS = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.2])
CPA_DISTS = np.array(
    [0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    + [700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1350.0, 1500.0]
)


def _load_coeffs():
    """Coefficients of each period, shared with NgaEastUsgs2017."""
    extra = model.load_data_file("nga_east_2023.csv", 3)
    np.testing.assert_array_equal(extra["period"], NgaEastUsgs2017.PERIODS)
    names = list(NgaEastUsgs2017.COEFF.dtype.names)
    names += [n for n in extra.dtype.names if n != "period"]
    arrays = [NgaEastUsgs2017.COEFF[n] for n in NgaEastUsgs2017.COEFF.dtype.names]
    arrays += [extra[n] for n in extra.dtype.names if n != "period"]
    return np.rec.fromarrays(arrays, names=names)


@functools.lru_cache(maxsize=None)
def _load_cpa_tables():
    """Load the Chapman and Guo (2021) PSA ratio tables.

    Returns
    -------
    ratios : :class:`np.ndarray`
        PSA ratios with shape (depths, magnitudes, distances, periods).
    """
    fname = os.path.join(os.path.dirname(__file__), "data", "chapman_guo_2021.csv.gz")
    with gzip.open(fname, "rt") as fp:
        data = np.loadtxt(fp, delimiter=",", comments="#", skiprows=3)
    shape = (len(CPA_DEPTHS), len(CPA_MAGS), len(CPA_DISTS))
    grid = np.stack(np.meshgrid(CPA_DEPTHS, CPA_MAGS, CPA_DISTS, indexing="ij"), -1)
    assert np.array_equal(data[:, :3], grid.reshape(-1, 3))
    return np.ascontiguousarray(data[:, 3:].reshape(shape + (-1,)))


def _clamped_position(keys, value):
    """Clamped index and fraction, as in nshmp-lib table lookups.

    The index is in [0, len(keys) - 2] and the fraction is in [0, 1].
    """
    index = np.clip(np.searchsorted(keys, value, side="right") - 1, 0, len(keys) - 2)
    lo = keys[index]
    fraction = np.clip((value - lo) / (keys[index + 1] - lo), 0.0, 1.0)
    return index, fraction


def z_site_scale(depth_sed):
    r"""Coastal plain taper of nshmp-lib (``NgaEast.zSiteScale``).

    :math:`(1 - e^{-Z_{sed} / 0.2})^4`, which is 0 for :math:`Z_{sed}` = 0 or
    NaN (not on the coastal plain) and increases to 1 at about 1 km.

    Parameters
    ----------
    depth_sed : array_like
        sediment thickness (km)

    Returns
    -------
    scale : :class:`np.ndarray`
        taper with the shape of `depth_sed`
    """
    depth_sed = np.asarray(depth_sed, dtype=float)
    s = 1.0 - np.exp(-depth_sed / Z_CUT)
    return np.where(np.isnan(depth_sed), 0.0, s * s * s * s)


def cpa_ln_ratio(depth_sed, mag, dist_jb, indices=None):
    """Natural log of the Chapman and Guo (2021) PSA ratio.

    Trilinear interpolation of the PSA ratio tables in sediment thickness,
    magnitude, and distance (nshmp-lib ``ChapmanGuo_2021.cpaPsaRatio``). Values
    beyond the tables are clamped to the table limits. nshmp-lib uses the
    Joyner-Boore distance.

    Parameters
    ----------
    depth_sed, mag, dist_jb : array_like
        sediment thickness (km), moment magnitude, and Joyner-Boore distance
        (km). NaN sediment thickness is used as 0.
    indices : array_like, optional
        coefficient rows (periods) to compute; all periods if *None*

    Returns
    -------
    ln_ratio : :class:`np.ndarray`
        natural log of the ratio with periods along the last axis
    """
    depth_sed, mag, dist_jb = np.broadcast_arrays(
        np.asarray(depth_sed, dtype=float),
        np.asarray(mag, dtype=float),
        np.asarray(dist_jb, dtype=float),
    )
    depth_sed = np.where(np.isnan(depth_sed), 0.0, depth_sed)
    data = _load_cpa_tables()
    if indices is not None:
        data = data[..., indices]
    i, zf = _clamped_position(CPA_DEPTHS, depth_sed)
    j, mf = _clamped_position(CPA_MAGS, mag)
    k, rf = _clamped_position(CPA_DISTS, dist_jb)
    zf, mf, rf = (model.as_column(f) for f in (zf, mf, rf))

    def interp(lo, hi, f):
        return lo + f * (hi - lo)

    z1m1 = interp(data[i, j, k], data[i, j, k + 1], rf)
    z1m2 = interp(data[i, j + 1, k], data[i, j + 1, k + 1], rf)
    z2m1 = interp(data[i + 1, j, k], data[i + 1, j, k + 1], rf)
    z2m2 = interp(data[i + 1, j + 1, k], data[i + 1, j + 1, k + 1], rf)
    z1 = interp(z1m1, z1m2, mf)
    z2 = interp(z2m1, z2m2, mf)
    return np.log(interp(z1, z2, zf))


def median_adjustment(c, v_s30, z_scale):
    r"""2023 NSHM adjustment of the hard-rock median (ln units).

    :math:`(1 - s) [a + b \ln(\min(V_{S30}, 2000) / 1000)]`, with the
    :math:`V_{S30}` term only for :math:`V_{S30} >` 1000 m/s, where :math:`s`
    is the coastal plain taper (:func:`z_site_scale`) and :math:`a` and
    :math:`b` are the ``nga_adj`` and ``vs30_b`` coefficients.

    Parameters
    ----------
    c : :class:`pygmm.model.Coefficients`
        coefficients of the computed periods
    v_s30 : array_like
        :math:`V_{S30}` (m/s) with a trailing axis for the periods
    z_scale : array_like
        coastal plain taper with a trailing axis for the periods

    Returns
    -------
    adj : :class:`np.ndarray`
        adjustment with periods along the last axis
    """
    adj = np.where(
        v_s30 > 1000.0,
        c.nga_adj + c.vs30_b * np.log(np.minimum(v_s30, 2000.0) / 1000.0),
        c.nga_adj,
    )
    return (1.0 - z_scale) * adj


#: Site terms that do not depend on the rock PGA
SiteTerms = collections.namedtuple(
    "SiteTerms", ["f_lin", "var_lin", "f_2", "sigma_f_2", "no_amp"]
)


def site_terms(c, periods, v_s30, version):
    """Site amplification terms that do not depend on the rock PGA.

    The Stewart et al. (2020) linear and Hashash et al. (2020) nonlinear site
    amplification model as implemented in nshmp-lib (``NgaEast.SiteAmp_2018``,
    ``SiteAmp_2023``, and ``SiteAmp_2026``).

    Parameters
    ----------
    c : :class:`pygmm.model.Coefficients`
        coefficients of the computed periods
    periods : :class:`np.ndarray`
        computed periods (s; -1 for PGV and 0 for PGA)
    v_s30 : array_like
        :math:`V_{S30}` (m/s) with a trailing axis for the periods
    version : str
        model version: "2018", "2023", or "2026"

    Returns
    -------
    terms : :class:`SiteTerms`
        linear amplification and its variance (ln units), nonlinear amplification
        and its standard deviation per unit of :math:`\\ln((PGA_r + f_3) / f_3)`,
        and if :math:`V_{S30} \\ge` 3000 m/s, for which there is no amplification
    """
    opts = SITE_AMP_OPTIONS[version]
    v_min = 150.0
    v_max = 3000.0
    v_lin_ref = 760.0
    v_l = 200.0
    v_u = 2000.0

    v_s30 = np.asarray(v_s30, dtype=float)
    no_amp = v_s30 >= v_max
    v_s30 = np.maximum(v_s30, v_min)

    # Vs30-dependent weights of the impedance and gradient f760 models: 0.767
    # for the impedance model at 600 m/s and above, and 0.1 at 400 m/s and below
    vw_1, vw_2 = 600.0, 400.0
    wt_1, wt_2 = 0.767, 0.1
    wt_scale = (wt_1 - wt_2) / (np.log(vw_1) - np.log(vw_2))
    with np.errstate(divide="ignore", invalid="ignore"):
        wt_i = np.select(
            [v_s30 < vw_2, v_s30 < vw_1],
            [wt_2, wt_scale * np.log(v_s30 / vw_2) + wt_2],
            wt_1,
        )
    wt_g = 1.0 - wt_i
    f_760 = c.f760i * wt_i + c.f760g * wt_g
    sigma_760 = c.f760is * wt_i + c.f760gs * wt_g

    with np.errstate(divide="ignore", invalid="ignore"):
        f_2000 = c.c * np.log(c.V2 / v_lin_ref)
        # Interpolated in ln(Vs30) between 2000 and 3000 m/s, where the total
        # linear amplification (f_v + f_760) is zero
        ln_v_u = np.log(v_u)
        f_v_high = f_2000 + (np.log(v_s30) - ln_v_u) * (-f_760 - f_2000) / (
            np.log(v_max) - ln_v_u
        )
        f_v = np.select(
            [v_s30 <= c.V1, v_s30 <= c.V2, v_s30 <= v_u],
            [c.c * np.log(c.V1 / v_lin_ref), c.c * np.log(v_s30 / v_lin_ref), f_2000],
            f_v_high,
        )

        sigma_t = c.sig_l - c.sig_vc
        v_t_low = (v_s30 - v_l) / (c.Vf - v_l)
        v_t_mid = (v_s30 - c.V2) / (v_u - c.V2)
        sigma_v = np.select(
            [v_s30 < c.Vf, v_s30 <= c.V2, v_s30 <= v_u],
            [
                c.sig_l - 2.0 * sigma_t * v_t_low + sigma_t * v_t_low * v_t_low,
                c.sig_vc,
                c.sig_vc + (c.sig_u - c.sig_vc) * v_t_mid * v_t_mid,
            ],
            c.sig_u * (1.0 - np.log(v_s30 / v_u) / np.log(v_max / v_u)),
        )

    f_lin = f_v + f_760
    var_lin = sigma_v * sigma_v + sigma_760 * sigma_760

    # Nonlinear term. nshmp-lib uses a reference velocity of 3000 m/s for the
    # intensity measures ordered at or after the 0.4 s spectral acceleration,
    # which are the periods of 0.4 s and longer (PGA and PGV are before).
    v_ref_nl = np.where(periods >= 0.4, v_max, v_lin_ref)
    v_s30_nl = np.maximum(v_s30, opts["v_min_nl"])
    f_4 = c.f4 * 0.5 + c.f4mod * 0.5 if opts["f4_mod"] else c.f4
    f_2 = np.where(
        v_s30_nl < c.Vc,
        f_4
        * (
            np.exp(c.f5 * (np.minimum(v_s30_nl, v_ref_nl) - 360.0))
            - np.exp(c.f5 * (v_ref_nl - 360.0))
        ),
        0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_f_2 = np.select(
            [v_s30_nl < 300.0, v_s30_nl < 1000.0],
            [
                c.sig_c,
                c.sig_c - c.sig_c / np.log(1000.0 / 300.0) * np.log(v_s30_nl / 300.0),
            ],
            0.0,
        )
    return SiteTerms(f_lin, var_lin, f_2, sigma_f_2, no_amp)


def rock_pga_term(f_3, pga_rock, version):
    """Rock PGA term of the nonlinear site amplification.

    Parameters
    ----------
    f_3 : :class:`np.ndarray`
        f3 coefficient of the computed periods
    pga_rock : array_like
        PGA (g) for :math:`V_{S30}` = 3000 m/s with a trailing axis for the periods
    version : str
        model version: "2018", "2023", or "2026". The 2026 version caps the rock
        PGA at 1 g.

    Returns
    -------
    ln_ref : :class:`np.ndarray`
        :math:`\\ln((PGA_r + f_3) / f_3)`
    """
    if SITE_AMP_OPTIONS[version]["cap_pga_rock"]:
        pga_rock = np.minimum(pga_rock, 1.0)
    return np.log((pga_rock + f_3) / f_3)


def site_amp(terms, ln_ref, with_sigma=True):
    """Site amplification and its standard deviation (ln units).

    Parameters
    ----------
    terms : :class:`SiteTerms`
        terms from :func:`site_terms`
    ln_ref : array_like
        rock PGA term from :func:`rock_pga_term`
    with_sigma : bool, optional
        if the standard deviation is computed

    Returns
    -------
    f_s : :class:`np.ndarray`
        site amplification, which is 0 for :math:`V_{S30} \\ge` 3000 m/s
    sigma_s : :class:`np.ndarray` or None
        standard deviation of the site amplification (*None* if not
        `with_sigma`)
    """
    f_s = np.where(terms.no_amp, 0.0, terms.f_lin + terms.f_2 * ln_ref)
    if not with_sigma:
        return f_s, None
    sigma_f_2_ref = terms.sigma_f_2 * ln_ref
    sigma_s = np.sqrt(terms.var_lin + sigma_f_2_ref * sigma_f_2_ref)
    return f_s, np.where(terms.no_amp, 0.0, sigma_s)


def site_amp_factor(f_s, sigma_s, z_score):
    """Linear site amplification factor of the three-point distribution.

    nshmp-lib (``NgaEast.SiteTerm.apply``) applies the site amplification as
    :math:`\\ln(0.185 e^{\\mu + f_s + z \\sigma_s} + 0.63 e^{\\mu + f_s} + 0.185
    e^{\\mu + f_s - z \\sigma_s})`, or :math:`\\mu` if :math:`f_s` = 0. This
    returns the ratio of the linear ground motion to :math:`e^\\mu`.

    Parameters
    ----------
    f_s, sigma_s : array_like
        site amplification and its standard deviation from :func:`site_amp`
    z_score : float
        standard normal variate of the 5th and 95th percentile branches

    Returns
    -------
    factor : :class:`np.ndarray`
        linear amplification factor
    """
    wt_lo, wt_mid, wt_hi = EPI_WTS
    exp_sigma = np.exp(sigma_s * z_score)
    factor = np.exp(f_s) * (wt_lo * exp_sigma + wt_mid + wt_hi / exp_sigma)
    return np.where(f_s == 0.0, 1.0, factor)


def sigma_epri(c, mag):
    """Updated EPRI (2013) aleatory standard deviation (ln units).

    Parameters
    ----------
    c : :class:`pygmm.model.Coefficients`
        coefficients of the computed periods
    mag : array_like
        moment magnitude with a trailing axis for the periods
    """

    def interp_mag(m5, m6, m7):
        return np.select(
            [mag <= 5.0, mag <= 6.0, mag <= 7.0],
            [
                m5,
                m5 + (mag - 5.0) * (m6 - m5) / (6.0 - 5.0),
                m6 + (mag - 6.0) * (m7 - m6) / (7.0 - 6.0),
            ],
            m7,
        )

    tau = interp_mag(c.tau_M5, c.tau_M6, c.tau_M7)
    phi = interp_mag(c.phi_M5, c.phi_M6, c.phi_M7)
    return np.sqrt(phi * phi + tau * tau)


def sigma_panel(c, mag, v_s30):
    """Panel aleatory standard deviation (ln units).

    Global :math:`\\tau` and :math:`\\phi_{SS}` models, central branch, with the
    Stewart et al. (2019) :math:`\\phi_{S2S}` model.

    Parameters
    ----------
    c : :class:`pygmm.model.Coefficients`
        coefficients of the computed periods
    mag, v_s30 : array_like
        moment magnitude and :math:`V_{S30}` (m/s) with trailing axes for the
        periods
    """
    # tau (Equation 5-1)
    tau = np.select(
        [mag <= 4.5, mag <= 5.0, mag <= 5.5, mag <= 6.5],
        [
            c.t1,
            c.t1 + (c.t2 - c.t1) * (mag - 4.5) / 0.5,
            c.t2 + (c.t3 - c.t2) * (mag - 5.0) / 0.5,
            c.t3 + (c.t4 - c.t3) * (mag - 5.5),
        ],
        c.t4,
    )
    # phi_SS (Equation 5-2)
    phi_ss = np.select(
        [mag <= 5.0, mag <= 6.5],
        [c.ss_a, c.ss_a + (mag - 5.0) * (c.ss_b - c.ss_a) / 1.5],
        c.ss_b,
    )
    # phi_S2S of Stewart et al. (2019)
    v_1, v_2 = 1200.0, 1500.0
    phi_s2s = np.select(
        [v_s30 < v_1, v_s30 < v_2],
        [c.s2s1, c.s2s1 - ((c.s2s1 - c.s2s2) / (v_2 - v_1)) * (v_s30 - v_1)],
        c.s2s2,
    )
    return np.sqrt(tau * tau + phi_ss * phi_ss + phi_s2s * phi_s2s)


def combined_sigma(c, mag, v_s30, sum_mean_wts=1.0):
    """Standard deviation of the combined logic tree of nshmp-lib.

    nshmp-lib (``GroundMotions.combine``) combines the branches of the ground
    motion logic tree (median models times standard deviation models) with the
    square root of the weighted sum of the variances.

    Parameters
    ----------
    c : :class:`pygmm.model.Coefficients`
        coefficients of the computed periods
    mag, v_s30 : array_like
        moment magnitude and :math:`V_{S30}` (m/s) with trailing axes for the
        periods
    sum_mean_wts : array_like
        sum of the weights of the median models (1 up to round-off)
    """
    s_epri = sigma_epri(c, mag)
    s_panel = sigma_panel(c, mag, v_s30)
    wt_epri, wt_panel = SIGMA_WTS
    var = wt_epri * s_epri * s_epri + wt_panel * s_panel * s_panel
    return np.sqrt(sum_mean_wts * var)


class NgaEast(model.GroundMotionModel):
    r"""NGA-East model as implemented in USGS nshmp-lib.

    The NGA-East ground motion model for central and eastern North America
    (Goulet et al., 2017, 2018) with the site amplification, adjustments, and
    logic tree combination of the USGS nshmp-lib library (``NgaEast``), which is
    used for the U.S. Geological Survey national seismic hazard models. The
    median is a composite of the 17 NGA-East (Sammons) median models for
    hard-rock conditions (:math:`V_{S30}` = 3000 m/s), with period-dependent
    weights, and the same tables and interpolation as :class:`NgaEastUsgs2017`.
    Site amplification uses the Stewart et al. (2020) linear and Hashash et al.
    (2020) nonlinear models, which are applied to each median model with a
    three-point distribution of the site amplification (5th, 50th, and 95th
    percentile branches with weights of 0.185, 0.63, and 0.185):
    :math:`\ln(0.185 e^{\mu + f_s + z \sigma_s} + 0.63 e^{\mu + f_s} + 0.185
    e^{\mu + f_s - z \sigma_s})`, or :math:`\mu` for :math:`V_{S30} \ge` 3000
    m/s. :math:`V_{S30} <` 150 m/s is used as 150 m/s.

    The versions differ in:

    - "2018" (``Gmm.NGA_EAST_2018``, 2018 NSHM): :math:`z` = 1 and the
      Hashash et al. (2017) nonlinear coefficient :math:`f_4`. The median is
      the same as :class:`NgaEastUsgs2017`, but the standard deviation is
      combined differently (see below).
    - "2023" (``Gmm.NGA_EAST_2023``, 2023 NSHM, CONUS.2023.R1): :math:`z` =
      1.645, so that the branches represent the 5th and 95th percentiles, and
      :math:`f_4` is the average of the Hashash et al. (2017) and (2020) values
      (``f4`` and ``f4mod``).
    - "2026" (``Gmm.NGA_EAST_2026``, CONUS.2023.R2 and the current CONUS NSHM):
      as 2023, with a minimum :math:`V_{S30}` of 200 m/s in the nonlinear site
      term and the rock PGA of the nonlinear site term capped at 1 g.

    The "2023" and "2026" versions have two variants that depend on the
    sediment thickness of the Atlantic and Gulf coastal plain,
    :math:`Z_{sed}` (``depth_sed``), through the taper :math:`s = (1 -
    e^{-Z_{sed}/0.2})^4`, which is 0 off the coastal plain (``depth_sed`` of
    NaN or *None*) and increases to 1 at about 1 km:

    - ``adjusted=True``: the 2023 NSHM adjustment of the hard-rock median,
      :math:`(1 - s) [a + b \ln(\min(V_{S30}, 2000) / 1000)]` with the
      :math:`V_{S30}` term only for :math:`V_{S30} >` 1000 m/s, is added to the
      hard-rock medians of the 17 models (and, as in nshmp-lib, to their rock
      PGA of the nonlinear site term, using the adjustment of the computed
      period). It reduces the short-period ground motions off the coastal
      plain.
    - ``cpa=True``: coastal plain amplification. For :math:`s >` 0, the natural
      log of the Chapman and Guo (2021) PSA ratio (relative to a 1000 m/s
      reference site, interpolated in :math:`Z_{sed}`, magnitude, and
      Joyner-Boore distance) minus :math:`s` times the site amplification
      :math:`f_s` at 1000 m/s is added to the site-amplified median of each
      model.

    The nshmp-lib ``Gmm`` identifiers are (see :attr:`GMM_IDS`):

    ==========================  =========================================
    nshmp-lib ``Gmm``           Options
    ==========================  =========================================
    ``NGA_EAST_2018``           ``version="2018"``
    ``NGA_EAST_2023``           ``version="2023"``
    ``NGA_EAST_2023_ADJUSTED``  ``version="2023", adjusted=True``
    ``NGA_EAST_2023_CPA``       ``version="2023", cpa=True``
    ``NGA_EAST_2026``           ``version="2026"``
    ``NGA_EAST_2026_ADJUSTED``  ``version="2026", adjusted=True``
    ``NGA_EAST_2026_CPA``       ``version="2026", cpa=True``
    ==========================  =========================================

    ``adjusted=True`` and ``cpa=True`` can be combined, although nshmp-lib has
    no such ``Gmm``. The stable-crust logic tree of the current CONUS NSHM
    (nshm-conus 6.2.0) gives weights of 1/3 to ``NGA_EAST_2026`` and
    ``NGA_EAST_2026_ADJUSTED`` and 1/6 each to the corresponding seed model
    logic trees (``NGA_EAST_SEEDS_2026`` and ``NGA_EAST_SEEDS_2026_ADJUSTED``),
    which are not implemented here.

    nshmp-lib provides a logic tree of 34 ground motions: the 17 median models
    times two aleatory variability models, the updated EPRI (2013) model
    (weight of 0.8) and the panel model (weight of 0.2; global :math:`\tau`
    and :math:`\phi_{SS}`, central branch, with the Stewart et al. (2019)
    :math:`\phi_{S2S}` model). This model gives the combination of
    nshmp-lib's ``GroundMotions.combine``: the median is the weighted mean of
    the linear ground motions, :math:`\ln \sum_i w_i e^{\mu_i}`, and the
    standard deviation is the square root of the weighted mean of the
    variances, :math:`\sqrt{0.8 \sigma_{EPRI}^2 + 0.2 \sigma_{panel}^2}`. In
    contrast, nshmp-haz (and :class:`NgaEastUsgs2017`) uses the weighted mean
    of the standard deviations, which is up to about 0.004 smaller. The
    standard deviation is the same for all versions.

    The model is vectorized. Each scenario value (``mag``, ``dist_rup``,
    ``v_s30``, ``dist_jb``, and ``depth_sed``) can be a scalar or an array, and
    the arrays are broadcast against each other. For a scalar scenario, the
    response and standard deviation have one value per period, as in other
    models. For arrays of N scenarios, they have shape (N, periods), so, for
    example, ``pga`` has shape (N,) and ``spec_accels`` has shape (N, 21).

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario. ``dist_jb`` is required for ``cpa=True`` if
        a ``depth_sed`` is greater than 0. ``depth_sed`` is the thickness of the
        Atlantic and Gulf coastal plain sediments (km); *None* or NaN is a site
        off the coastal plain.
    version : str, optional
        model version: "2018", "2023", or "2026" (default)
    adjusted : bool, optional
        apply the 2023 NSHM adjustment of the hard-rock median (not for "2018")
    cpa : bool, optional
        apply the Chapman and Guo (2021) coastal plain amplification (not for
        "2018")
    ims : str or sequence of str, optional
        intensity measures to compute: "pga", "pgv", spectral periods such as
        "psa_1p000" (1.0 s), "psa_ngawest2_21" (the 21 NGA-West2 comparison
        periods, which are all of the model's periods), and/or "psa_all" (all
        21 periods). Computing only the needed intensity measures is much
        faster for large vectorized scenarios. If *None* (default), all
        intensity measures are computed.

    References
    ----------
    Goulet, C., Bozorgnia, Y., Kuehn, N., Al Atik, L., Youngs, R., Graves, R.,
    and Atkinson, G. (2017). NGA-East ground-motion models for the U.S.
    Geological Survey national seismic hazard maps. PEER Report No. 2017/03,
    Pacific Earthquake Engineering Research Center.

    Goulet, C. A., Bozorgnia, Y., Abrahamson, N., et al. (2018). Central and
    eastern North America ground-motion characterization: NGA-East final
    report. PEER Report No. 2018/08, Pacific Earthquake Engineering Research
    Center.

    Stewart, J., Parker, G., Atkinson, G., Boore, D., Hashash, Y., and Silva,
    W. (2020). Ergodic site amplification model for central and eastern North
    America. Earthquake Spectra, 36(1), 42-68.

    Hashash, Y., Harmon, J., Ilhan, O., Parker, G., and Stewart, J. (2020).
    Nonlinear site amplification model for ergodic seismic hazard analysis in
    central and eastern North America. Earthquake Spectra, 36(1), 69-86.

    Stewart, J., Parker, G., Al Atik, L., et al. (2019). Site-to-site standard
    deviation model for central and eastern North America. UC E-Scholarship,
    https://escholarship.org/uc/item/2sc5g220.

    Chapman, M. C., and Guo, Z. (2021). A response spectral ratio model to
    account for amplification and attenuation effects in the Atlantic and Gulf
    coastal plain. Bulletin of the Seismological Society of America, 111(4),
    1849-1867. https://doi.org/10.1785/0120200322

    USGS nshmp-lib, ``gov.usgs.earthquake.nshmp.gmm.NgaEast`` and
    ``ChapmanGuo_2021``, commit 44728a7d
    (https://code.usgs.gov/ghsc/nshmp/nshmp-lib).

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_east import NgaEast
    >>> s = Scenario(mag=6.5, dist_rup=20.0, v_s30=760.0)
    >>> m = NgaEast(s)
    >>> round(float(m.pga), 4), round(float(m.ln_std_pga), 4)
    (0.4616, 0.6472)
    >>> round(float(NgaEast(s, adjusted=True).pga), 4)
    0.4435
    >>> s = Scenario(
    ...     mag=np.array([5.0, 6.0, 7.0]), dist_rup=np.array([10.0, 50.0, 200.0]),
    ...     dist_jb=np.array([8.0, 49.0, 200.0]), v_s30=400.0,
    ...     depth_sed=np.array([np.nan, 0.5, 2.0]))
    >>> NgaEast(s, version="2023", cpa=True, ims=["pga"]).pga.shape
    (3,)
    >>> m = NgaEast(s, **NgaEast.GMM_IDS["NGA_EAST_2026_ADJUSTED"])
    >>> m.spec_accels.shape
    (3, 21)

    """

    NAME = "NGA-East (USGS nshmp-lib)"
    ABBREV = "NGAE"

    #: Coefficients (sigma, site amplification, adjustments, and model weights)
    COEFF = _load_coeffs()
    PERIODS = COEFF["period"]

    INDEX_PGV = NgaEastUsgs2017.INDEX_PGV
    INDEX_PGA = NgaEastUsgs2017.INDEX_PGA
    INDICES_PSA = NgaEastUsgs2017.INDICES_PSA

    #: Weights of the 17 median models, shape (periods, 17)
    WEIGHTS = NgaEastUsgs2017.WEIGHTS

    #: Model options of the nshmp-lib ``Gmm`` identifiers
    GMM_IDS = {
        "NGA_EAST_2018": dict(version="2018"),
        "NGA_EAST_2023": dict(version="2023"),
        "NGA_EAST_2023_ADJUSTED": dict(version="2023", adjusted=True),
        "NGA_EAST_2023_CPA": dict(version="2023", cpa=True),
        "NGA_EAST_2026": dict(version="2026"),
        "NGA_EAST_2026_ADJUSTED": dict(version="2026", adjusted=True),
        "NGA_EAST_2026_CPA": dict(version="2026", cpa=True),
    }

    # Constraints of nshmp-lib (NgaEast.CONSTRAINTS and Site.ZSED_RANGE)
    LIMITS = dict(
        mag=(4.0, 8.2), dist_rup=(0.0, 1000.0), v_s30=(150.0, 3000.0), depth_sed=(0, 25)
    )

    PARAMS = [
        model.NumericParameter("mag", True, 4.0, 8.2),
        model.NumericParameter("dist_rup", True, 0.0, 1000.0),
        model.NumericParameter("v_s30", True, 150.0, 3000.0),
        model.NumericParameter("dist_jb", False, 0.0, 1500.0),
        model.NumericParameter("depth_sed", False, 0.0, 25.0),
    ]

    def __init__(
        self,
        scenario: model.Scenario,
        version: str = "2026",
        adjusted: bool = False,
        cpa: bool = False,
        ims=None,
    ):
        """Initialize the model."""
        version = str(version)
        if version not in VERSIONS:
            raise ValueError(
                f"version must be one of {', '.join(VERSIONS)}, not {version!r}"
            )
        if version == "2018" and (adjusted or cpa):
            raise ValueError(
                "The adjusted and cpa options are not available for version 2018"
            )
        self.version = version
        self.adjusted = bool(adjusted)
        self.cpa = bool(cpa)
        super().__init__(scenario, ims)
        self._ln_resp = self._calc_ln_resp()
        self._ln_std = self._calc_ln_std()

    def _coeffs(self):
        return model.Coefficients(
            **{
                name: self._coeff_rows(self.COEFF[name])
                for name in self.COEFF.dtype.names
            }
        )

    def _depth_sed(self):
        depth_sed = self._scenario.depth_sed
        return np.nan if depth_sed is None else np.asarray(depth_sed, dtype=float)

    def _calc_ln_resp(self) -> np.ndarray:
        """Calculate the natural logarithm of the response.

        Returns
        -------
        ln_resp : class:`np.array`:
            natural log of the response

        """
        s = self._scenario
        c = self._coeffs()
        periods = self._coeff_rows(self.PERIODS)
        weights = self._coeff_rows(self.WEIGHTS)
        version = self.version
        z_score = SITE_AMP_OPTIONS[version]["z_score"]

        mag, dist_rup = np.broadcast_arrays(
            np.asarray(s.mag, dtype=float), np.asarray(s.dist_rup, dtype=float)
        )
        index, frac_r, frac_m = NgaEastUsgs2017._table_position(dist_rup, mag)

        ln_tables = _load_tables()
        tables = ln_tables if self._indices is None else ln_tables[:, :, self._indices]
        tables_pga = ln_tables[:, :, self.INDEX_PGA]

        v_s30 = model.as_column(np.asarray(s.v_s30, dtype=float))
        terms = site_terms(c, periods, v_s30, version)

        depth_sed = self._depth_sed()
        z_scale = model.as_column(z_site_scale(depth_sed))
        # Adjustment of the hard-rock median
        mu_adj = median_adjustment(c, v_s30, z_scale) if self.adjusted else 0.0
        # Coastal plain amplification
        apply_cpa = self.cpa and np.any(z_scale > 0)
        if apply_cpa:
            if s.dist_jb is None:
                raise ValueError("dist_jb is required for cpa=True with depth_sed")
            on_cpa = z_scale > 0
            dist_jb = np.asarray(s.dist_jb, dtype=float)
            f_cpa = np.where(
                on_cpa, cpa_ln_ratio(depth_sed, s.mag, dist_jb, self._indices), 0.0
            )
            terms_ref = site_terms(c, periods, VS_REF_CPA, version)
            z_scale_cpa = np.where(on_cpa, z_scale, 0.0)

        frac_r_col = model.as_column(frac_r)
        frac_m_col = model.as_column(frac_m)
        # Weighted sum of the linear response of the 17 models
        sum_resp = 0
        for i in range(len(tables)):
            ln_resp_rock = (
                NgaEastUsgs2017._interpolate(tables[i], index, frac_r_col, frac_m_col)
                + mu_adj
            )
            ln_pga_rock = NgaEastUsgs2017._interpolate(
                tables_pga[i], index, frac_r, frac_m
            )
            # As in nshmp-lib, the adjustment of the computed period is applied
            pga_rock = np.exp(model.as_column(ln_pga_rock) + mu_adj)
            ln_ref = rock_pga_term(c.f3, pga_rock, version)
            f_s, sigma_s = site_amp(terms, ln_ref)
            if apply_cpa:
                # Site amplification at the reference V_S30 of the PSA ratios
                f_s_ref, _ = site_amp(terms_ref, ln_ref, with_sigma=False)
                ln_resp_rock = ln_resp_rock + (f_cpa - f_s_ref * z_scale_cpa)
            resp = np.exp(ln_resp_rock) * site_amp_factor(f_s, sigma_s, z_score)
            sum_resp = sum_resp + resp * weights[:, i]

        return np.log(sum_resp)

    def _calc_ln_std(self) -> np.ndarray:
        """Calculate the logarithmic standard deviation.

        Returns
        -------
        ln_std : class:`np.array`:
            natural log standard deviation

        """
        s = self._scenario
        c = self._coeffs()
        mag = model.as_column(np.asarray(s.mag, dtype=float))
        v_s30 = model.as_column(np.asarray(s.v_s30, dtype=float))
        sum_wts = np.sum(self._coeff_rows(self.WEIGHTS), axis=1)
        ln_std = combined_sigma(c, mag, v_s30, sum_wts)
        return np.broadcast_to(ln_std, np.shape(self._ln_resp)).copy()
