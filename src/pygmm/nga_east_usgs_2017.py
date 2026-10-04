"""NGA-East for USGS (2017) ground motion model."""

import functools
import gzip
import os

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from . import model


@functools.lru_cache(maxsize=None)
def _load_tables():
    """Load the median ground-motion tables of the 17 models.

    Returns
    -------
    ln_tables : :class:`np.ndarray`
        natural log of the ground motion with shape (17, distances × magnitudes,
        periods). The distance-magnitude axis is flattened with magnitude varying
        fastest.
    """
    fname = os.path.join(
        os.path.dirname(__file__), "data", "nga_east_usgs_2017-tables.csv.gz"
    )
    with gzip.open(fname, "rt") as fp:
        data = np.loadtxt(fp, delimiter=",", comments="#", skiprows=5)
    model_ids = data[:, 0].astype(int)
    periods = data[:, 1]
    n_models = 17
    n_dists = len(NgaEastUsgs2017.TABLE_DISTS)
    n_mags = len(NgaEastUsgs2017.TABLE_MAGS)
    n_periods = len(NgaEastUsgs2017.PERIODS)
    # Rows are ordered by model, period, and distance
    assert np.array_equal(model_ids, np.repeat(np.arange(1, 18), n_periods * n_dists))
    assert np.array_equal(
        periods, np.tile(np.repeat(NgaEastUsgs2017.PERIODS, n_dists), n_models)
    )
    values = data[:, 3:].reshape(n_models, n_periods, n_dists * n_mags)
    # Periods along the last axis, so a gather by table position gives values with
    # periods along the last axis
    return np.ascontiguousarray(np.log(values).transpose(0, 2, 1))


class NgaEastUsgs2017(model.GroundMotionModel):
    r"""NGA-East for USGS (2017) model.

    The NGA-East ground motion model for central and eastern North America
    (CENA) developed for the U.S. Geological Survey (USGS) national seismic
    hazard maps (Goulet et al., 2017; PEER Report 2017/03). The median model is
    a composite of 17 median models, which are provided as tables of ground
    motion for hard-rock conditions (:math:`V_{S30}` = 3000 m/s) versus rupture
    distance (0 to 1500 km) and moment magnitude (4.0 to 8.2), with
    period-dependent weights. Site amplification uses the Stewart et al. (2017;
    PEER Report 2017/04) linear model and the Hashash et al. (2017; PEER Report
    2017/05) nonlinear model.

    This implementation follows the USGS nshmp-haz implementation
    (``NgaEastUsgs_2017.Usgs17``, ``Gmm.NGA_EAST_USGS``) and reproduces its
    results:

    - Each table is interpolated bilinearly in :math:`\log_{10} R_{rup}` and
      magnitude on the natural log of the ground motion. Distances and
      magnitudes beyond the tables are clamped to the table limits (a distance
      of 0 is used as 1e-5 km).
    - For each of the 17 models, the site amplification (:math:`f_T`) and its
      epistemic uncertainty (:math:`\sigma_T`) are computed from the model's
      hard-rock PGA and are applied with a three-point distribution:
      :math:`\ln(0.185 e^{\mu + f_T + \sigma_T} + 0.63 e^{\mu + f_T} + 0.185
      e^{\mu + f_T - \sigma_T})`. For :math:`V_{S30} \ge` 3000 m/s, there is
      no site amplification; for :math:`V_{S30} <` 150 m/s, 150 m/s is used.
      Between 2000 and 3000 m/s, the linear amplification is interpolated in
      :math:`\ln V_{S30}` to zero at 3000 m/s. Just below 3000 m/s, the
      three-point distribution still uses the standard deviation of
      :math:`f_{760}`, so the median is up to about 3% larger than at 3000 m/s.
      The nonlinear reference velocity
      is 760 m/s for PGA, PGV, and periods less than 0.4 s, and 3000 m/s
      otherwise.
    - The median is the weighted mean of the 17 models' linear ground motions,
      :math:`\ln \sum_i w_i e^{\mu_i}`.
    - The standard deviation is the weighted mean of the standard deviations of
      two aleatory variability models (not of their variances): 0.2 times the
      panel model (global :math:`\tau` and :math:`\phi_{SS}`, central branch,
      with the Stewart et al. (2019) :math:`\phi_{S2S}` model) plus 0.8 times
      the updated EPRI (2013) model.

    The USGS updated seed model logic tree, the individual Sammons and seed
    models, and the Gulf Coastal Plain amplification are not implemented. The
    spectral periods are those supported by nshmp-haz (the 0.025 and 0.04 s
    tables are not used because the site amplification model does not provide
    them).

    The model is vectorized. Each scenario value (``mag``, ``dist_rup``, and
    ``v_s30``) can be a scalar or an array, and the arrays are broadcast
    against each other. For a scalar scenario, the response and standard
    deviation have one value per period, as in other models. For arrays of N
    scenarios, they have shape (N, periods), so, for example, ``pga`` has shape
    (N,) and ``spec_accels`` has shape (N, 21).

    Parameters
    ----------
    scenario : :class:`pygmm.model.Scenario`
        earthquake scenario
    ims : str or sequence of str, optional
        intensity measures to compute: "pga", "pgv", spectral periods such as
        "psa_1p000" (1.0 s), "psa_ngawest2_21" (the 21 NGA-West2 comparison
        periods, which are all of the model's periods), and/or "psa_all" (all
        21 periods). Computing only the needed intensity measures is much
        faster for large vectorized scenarios. If *None* (default), all
        intensity measures are computed.

    References
    ----------
    Goulet, C. A., Bozorgnia, Y., Kuehn, N., Al Atik, L., Youngs, R. R.,
    Graves, R. W., and Atkinson, G. M. (2017). NGA-East ground-motion models
    for the U.S. Geological Survey national seismic hazard maps. PEER Report
    No. 2017/03, Pacific Earthquake Engineering Research Center.

    Stewart, J. P., Parker, G. A., Harmon, J. A., Atkinson, G. M., Boore,
    D. M., Darragh, R. B., Silva, W. J., and Hashash, Y. M. A. (2017). Expert
    panel recommendations for ergodic site amplification in central and
    eastern North America. PEER Report No. 2017/04, Pacific Earthquake
    Engineering Research Center.

    Hashash, Y. M. A., Harmon, J. A., Ilhan, O., Parker, G. A., and Stewart,
    J. P. (2017). Recommendation for ergodic nonlinear site amplification in
    central and eastern North America. PEER Report No. 2017/05, Pacific
    Earthquake Engineering Research Center.

    USGS nshmp-haz, ``gov.usgs.earthquake.nshmp.gmm.NgaEastUsgs_2017``
    (https://github.com/usgs/nshmp-haz).

    Examples
    --------
    >>> import numpy as np
    >>> from pygmm.model import Scenario
    >>> from pygmm.nga_east_usgs_2017 import NgaEastUsgs2017
    >>> s = Scenario(mag=6.5, dist_rup=20.0, v_s30=760.0)
    >>> m = NgaEastUsgs2017(s)
    >>> round(float(m.pga), 4), round(float(m.ln_std_pga), 4)
    (0.4275, 0.6461)
    >>> s = Scenario(
    ...     mag=np.array([5.0, 6.0, 7.0]), dist_rup=np.array([10.0, 50.0, 200.0]),
    ...     v_s30=400.0)
    >>> NgaEastUsgs2017(s, ims=["pga"]).pga.shape
    (3,)
    >>> m = NgaEastUsgs2017(s, ims=["pga", "psa_0p200", "psa_1p000"])
    >>> m.psa_ims
    ['psa_0p200', 'psa_1p000']
    >>> m.spec_accels.shape
    (3, 2)

    """

    NAME = "NGA-East for USGS (2017)"
    ABBREV = "NGAE17"

    # Coefficients (sigma, site amplification, and model weights) for each period
    COEFF = model.load_data_file("nga_east_usgs_2017.csv", 4)
    PERIODS = COEFF["period"]

    INDEX_PGV = 0
    INDEX_PGA = 1
    INDICES_PSA = np.arange(2, 23)

    #: Weights of the 17 median models, shape (periods, 17)
    WEIGHTS = np.asarray(
        structured_to_unstructured(COEFF[[f"w_{i}" for i in range(1, 18)]])
    )

    #: Distances (km) of the tables. The first distance of 0 km is used as 1e-5 km.
    TABLE_DISTS = np.array(
        [1e-5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        + [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 175.0, 200.0]
        + [250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0, 800.0, 1000.0]
        + [1200.0, 1500.0]
    )
    #: Magnitudes of the tables
    TABLE_MAGS = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.8, 8.0, 8.2])

    LIMITS = dict(mag=(4.0, 8.2), dist_rup=(0.0, 1500.0), v_s30=(200.0, 3000.0))

    PARAMS = [
        model.NumericParameter("mag", True, 4.0, 8.2),
        model.NumericParameter("dist_rup", True, 0.0, 1500.0),
        model.NumericParameter("v_s30", True, 200.0, 3000.0),
    ]

    # Weights of the three-point distribution of the site amplification
    SITE_AMP_WTS = (0.185, 0.63, 0.185)
    # Weights of the panel and updated EPRI standard deviation models
    SIGMA_WTS = (0.2, 0.8)

    def __init__(self, scenario: model.Scenario, ims=None):
        """Initialize the model."""
        super().__init__(scenario, ims)
        self._ln_resp = self._calc_ln_resp()
        self._ln_std = self._calc_ln_std()

    @classmethod
    def _table_position(cls, dist_rup, mag):
        """Flattened table index and bin fractions of each scenario.

        Same as ``ClampingTable.position`` in nshmp-haz: the index is clamped to
        [0, len - 2] and the fraction to [0, 1].
        """

        def index_fraction(keys, value):
            index = np.clip(
                np.searchsorted(keys, value, side="right") - 1, 0, len(keys) - 2
            )
            lo = keys[index]
            fraction = np.clip((value - lo) / (keys[index + 1] - lo), 0.0, 1.0)
            return index, fraction

        with np.errstate(divide="ignore", invalid="ignore"):
            # A distance of 0 gives -inf, which clamps to the first distance
            log_dist = np.log10(dist_rup)
            ir, frac_r = index_fraction(np.log10(cls.TABLE_DISTS), log_dist)
        im, frac_m = index_fraction(cls.TABLE_MAGS, mag)
        index = ir * len(cls.TABLE_MAGS) + im
        return index, frac_r, frac_m

    @classmethod
    def _interpolate(cls, table, index, frac_r, frac_m):
        """Bilinear interpolation of a table with periods along the last axis."""
        n_mags = len(cls.TABLE_MAGS)
        c11 = table[index]
        c12 = table[index + 1]
        c21 = table[index + n_mags]
        c22 = table[index + n_mags + 1]
        i1 = c11 + frac_m * (c12 - c11)
        i2 = c21 + frac_m * (c22 - c21)
        return i1 + frac_r * (i2 - i1)

    def _calc_ln_resp(self) -> np.ndarray:
        """Calculate the natural logarithm of the response.

        Returns
        -------
        ln_resp : class:`np.array`:
            natural log of the response

        """
        s = self._scenario
        c = model.Coefficients(
            **{
                name: self._coeff_rows(self.COEFF[name])
                for name in self.COEFF.dtype.names
            }
        )
        periods = self._coeff_rows(self.PERIODS)
        weights = self._coeff_rows(self.WEIGHTS)

        mag, dist_rup = np.broadcast_arrays(
            np.asarray(s.mag, dtype=float), np.asarray(s.dist_rup, dtype=float)
        )
        index, frac_r, frac_m = self._table_position(dist_rup, mag)

        ln_tables = _load_tables()
        tables = ln_tables if self._indices is None else ln_tables[:, :, self._indices]
        tables_pga = ln_tables[:, :, self.INDEX_PGA]

        f_lin, sigma_lin, f_2, sigma_f_2, no_amp = self._calc_site_terms(c, periods)
        # No site amplification (f_T = sigma_T = 0) for V_S30 >= 3000 m/s
        f_lin = np.where(no_amp, 0.0, f_lin)
        var_lin = np.where(no_amp, 0.0, sigma_lin * sigma_lin)
        f_2 = np.where(no_amp, 0.0, f_2)
        sigma_f_2 = np.where(no_amp, 0.0, sigma_f_2)

        frac_r_col = model.as_column(frac_r)
        frac_m_col = model.as_column(frac_m)
        wt_lo, wt_mid, wt_hi = self.SITE_AMP_WTS
        # Weighted sum of the linear response of the 17 models
        sum_resp = 0
        for i in range(len(tables)):
            ln_resp_rock = self._interpolate(tables[i], index, frac_r_col, frac_m_col)
            pga_rock = np.exp(self._interpolate(tables_pga[i], index, frac_r, frac_m))
            # Nonlinear site term of this model
            ln_ref = np.log((model.as_column(pga_rock) + c.f3) / c.f3)
            # Site amplification (f_T) and its standard deviation (sigma_T)
            f_t = f_lin + f_2 * ln_ref
            sigma_f_2_ref = sigma_f_2 * ln_ref
            exp_sigma_t = np.exp(np.sqrt(var_lin + sigma_f_2_ref * sigma_f_2_ref))
            # Three-point distribution of the site amplification:
            # ln(0.185 exp(mu + f_t + sigma_t) + 0.63 exp(mu + f_t)
            #    + 0.185 exp(mu + f_t - sigma_t))
            resp = np.exp(ln_resp_rock + f_t)
            resp *= wt_lo * exp_sigma_t + wt_mid + wt_hi / exp_sigma_t
            resp *= weights[:, i]
            sum_resp = sum_resp + resp

        return np.log(sum_resp)

    def _calc_site_terms(self, c, periods):
        """Site terms that are the same for each of the 17 models.

        The site amplification is from Stewart et al. (2017) and Hashash et al.
        (2017) as implemented in nshmp-haz (``NgaEastUsgs_2017.SiteAmp``).

        Returns
        -------
        f_lin : :class:`np.ndarray`
            linear site amplification (ln units)
        sigma_lin : :class:`np.ndarray`
            standard deviation of the linear site amplification
        f_2 : :class:`np.ndarray`
            nonlinear site amplification per unit of ln((PGA_r + f_3) / f_3)
        sigma_f_2 : :class:`np.ndarray`
            standard deviation of the nonlinear site amplification per unit of
            ln((PGA_r + f_3) / f_3)
        no_amp : :class:`np.ndarray`
            if V_S30 is at least 3000 m/s, for which there is no amplification
        """
        v_min = 150.0
        v_max = 3000.0
        v_lin_ref = 760.0
        v_l = 200.0
        v_u = 2000.0

        v_s30_in = model.as_column(np.asarray(self._scenario.v_s30, dtype=float))
        no_amp = v_s30_in >= v_max
        v_s30 = np.maximum(v_s30_in, v_min)

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
                [
                    c.c * np.log(c.V1 / v_lin_ref),
                    c.c * np.log(v_s30 / v_lin_ref),
                    f_2000,
                ],
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
        sigma_lin = np.sqrt(sigma_v * sigma_v + sigma_760 * sigma_760)

        # Nonlinear reference velocity. nshmp-haz uses 3000 m/s for the
        # intensity measures ordered at or after the 0.4 s spectral acceleration,
        # which are the periods of 0.4 s and longer (PGA and PGV are before).
        v_ref_nl = np.where(periods >= 0.4, v_max, v_lin_ref)
        f_2 = np.where(
            v_s30 < c.Vc,
            c.f4
            * (
                np.exp(c.f5 * (np.minimum(v_s30, v_ref_nl) - 360.0))
                - np.exp(c.f5 * (v_ref_nl - 360.0))
            ),
            0.0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_f_2 = np.select(
                [v_s30 < 300.0, v_s30 < 1000.0],
                [
                    c.sig_c,
                    c.sig_c - c.sig_c / np.log(1000.0 / 300.0) * np.log(v_s30 / 300.0),
                ],
                0.0,
            )
        return f_lin, sigma_lin, f_2, sigma_f_2, no_amp

    def _calc_ln_std(self) -> np.ndarray:
        """Calculate the logarithmic standard deviation.

        Returns
        -------
        ln_std : class:`np.array`:
            natural log standard deviation

        """
        s = self._scenario
        c = model.Coefficients(
            **{
                name: self._coeff_rows(self.COEFF[name])
                for name in self.COEFF.dtype.names
            }
        )
        mag = model.as_column(np.asarray(s.mag, dtype=float))
        v_s30 = model.as_column(np.asarray(s.v_s30, dtype=float))

        # Panel model: tau (Equation 5-1), global model, central branch
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
        # phi_SS (Equation 5-2), global model, central branch
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
        sigma_panel = np.sqrt(tau * tau + phi_ss * phi_ss + phi_s2s * phi_s2s)

        # Updated EPRI (2013) model, interpolated in magnitude
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

        tau_epri = interp_mag(c.tau_M5, c.tau_M6, c.tau_M7)
        phi_epri = interp_mag(c.phi_M5, c.phi_M6, c.phi_M7)
        sigma_epri = np.sqrt(phi_epri * phi_epri + tau_epri * tau_epri)

        wt_panel, wt_epri = self.SIGMA_WTS
        ln_std = sigma_panel * wt_panel + sigma_epri * wt_epri
        return np.broadcast_to(ln_std, np.shape(self._ln_resp)).copy()
