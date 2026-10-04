"""Test the NGA-East for USGS (2017) model."""

import itertools
import math
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from pygmm.model import Scenario
from pygmm.nga_east_usgs_2017 import NgaEastUsgs2017, _load_tables

PERIODS = [0.05, 0.3, 1.0, 2.5]

# Results of nshmp-haz (Gmm.NGA_EAST_USGS), given with 10 decimals
NSHMP = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data", "nga_east_usgs_2017-nshmp.csv.gz"),
    comment="#",
)


def compute(**kwds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NgaEastUsgs2017(Scenario(**kwds))


@pytest.fixture(scope="module")
def nshmp_model():
    return compute(
        mag=NSHMP["mag"].values,
        dist_rup=NSHMP["dist_rup"].values,
        v_s30=NSHMP["v_s30"].values,
    )


def nshmp_columns():
    cols = np.searchsorted(NgaEastUsgs2017.PERIODS, NSHMP["period"].values)
    np.testing.assert_array_equal(NgaEastUsgs2017.PERIODS[cols], NSHMP["period"])
    return cols


def test_nshmp_median(nshmp_model):
    rows = np.arange(len(NSHMP))
    median = np.exp(nshmp_model._ln_resp[rows, nshmp_columns()])
    # The reference values are rounded to 10 decimals
    np.testing.assert_allclose(median, NSHMP["median"], rtol=0, atol=5.1e-11)


def test_nshmp_sigma(nshmp_model):
    rows = np.arange(len(NSHMP))
    sigma = nshmp_model._ln_std[rows, nshmp_columns()]
    np.testing.assert_allclose(sigma, NSHMP["sigma"], rtol=0, atol=5.1e-11)


# Scalar transliteration of nshmp-haz NgaEastUsgs_2017.Usgs17, used to check the
# vectorized model at cases that are not in the nshmp-haz results


def ref_position(keys, value):
    i = int(np.searchsorted(keys, value, side="right")) - 1
    i = min(max(i, 0), len(keys) - 2)
    lo, hi = keys[i], keys[i + 1]
    frac = 0.0 if value < lo else 1.0 if value > hi else (value - lo) / (hi - lo)
    return i, frac


def ref_table(table, mag, dist_rup):
    # table has shape (distances, magnitudes)
    ir, fr = ref_position(
        np.log10(NgaEastUsgs2017.TABLE_DISTS),
        math.log10(dist_rup) if dist_rup > 0 else -math.inf,
    )
    im, fm = ref_position(NgaEastUsgs2017.TABLE_MAGS, mag)
    i1 = table[ir, im] + fm * (table[ir, im + 1] - table[ir, im])
    i2 = table[ir + 1, im] + fm * (table[ir + 1, im + 1] - table[ir + 1, im])
    return i1 + fr * (i2 - i1)


def ref_site_amp(c, period, pga_rock, v_s30):
    if v_s30 >= 3000.0:
        return 0.0, 0.0
    v_s30 = max(v_s30, 150.0)
    wt_scale = (0.767 - 0.1) / (math.log(600.0) - math.log(400.0))
    if v_s30 < 400.0:
        wti = 0.1
    elif v_s30 < 600.0:
        wti = wt_scale * math.log(v_s30 / 400.0) + 0.1
    else:
        wti = 0.767
    wtg = 1.0 - wti
    f760 = c["f760i"] * wti + c["f760g"] * wtg
    f760s = c["f760is"] * wti + c["f760gs"] * wtg
    if v_s30 <= c["V1"]:
        fv = c["c"] * math.log(c["V1"] / 760.0)
    elif v_s30 <= c["V2"]:
        fv = c["c"] * math.log(v_s30 / 760.0)
    elif v_s30 <= 2000.0:
        fv = c["c"] * math.log(c["V2"] / 760.0)
    else:
        f2000 = c["c"] * math.log(c["V2"] / 760.0)
        x1, x2 = math.log(2000.0), math.log(3000.0)
        fv = f2000 + (math.log(v_s30) - x1) * (-f760 - f2000) / (x2 - x1)
    if v_s30 < c["Vf"]:
        st = c["sig_l"] - c["sig_vc"]
        vt = (v_s30 - 200.0) / (c["Vf"] - 200.0)
        fvs = c["sig_l"] - 2.0 * st * vt + st * vt * vt
    elif v_s30 <= c["V2"]:
        fvs = c["sig_vc"]
    elif v_s30 <= 2000.0:
        vt = (v_s30 - c["V2"]) / (2000.0 - c["V2"])
        fvs = c["sig_vc"] + (c["sig_u"] - c["sig_vc"]) * vt * vt
    else:
        fvs = c["sig_u"] * (1.0 - math.log(v_s30 / 2000.0) / math.log(3000.0 / 2000.0))
    f_lin = fv + f760
    s_lin = math.sqrt(fvs**2 + f760s**2)
    # PGV (-1), PGA (0), and periods < 0.4 s are ordered before SA0P4 in nshmp-haz
    v_ref_nl = 3000.0 if period >= 0.4 else 760.0
    rk = math.log((pga_rock + c["f3"]) / c["f3"])
    f_nl = 0.0
    if v_s30 < c["Vc"]:
        f2 = c["f4"] * (
            math.exp(c["f5"] * (min(v_s30, v_ref_nl) - 360.0))
            - math.exp(c["f5"] * (v_ref_nl - 360.0))
        )
        f_nl = f2 * rk
    sf2 = 0.0
    if v_s30 < 300.0:
        sf2 = c["sig_c"]
    elif v_s30 < 1000.0:
        sf2 = c["sig_c"] - c["sig_c"] / math.log(1000.0 / 300.0) * math.log(
            v_s30 / 300.0
        )
    return f_lin + f_nl, math.sqrt(s_lin**2 + (sf2 * rk) ** 2)


def ref_sigma(c, mag, v_s30):
    if mag <= 4.5:
        tau = c["t1"]
    elif mag <= 5.0:
        tau = c["t1"] + (c["t2"] - c["t1"]) * (mag - 4.5) / 0.5
    elif mag <= 5.5:
        tau = c["t2"] + (c["t3"] - c["t2"]) * (mag - 5.0) / 0.5
    elif mag <= 6.5:
        tau = c["t3"] + (c["t4"] - c["t3"]) * (mag - 5.5)
    else:
        tau = c["t4"]
    if mag <= 5.0:
        phi_ss = c["ss_a"]
    elif mag <= 6.5:
        phi_ss = c["ss_a"] + (mag - 5.0) * (c["ss_b"] - c["ss_a"]) / 1.5
    else:
        phi_ss = c["ss_b"]
    if v_s30 < 1200.0:
        phi_s2s = c["s2s1"]
    elif v_s30 < 1500.0:
        phi_s2s = c["s2s1"] - (c["s2s1"] - c["s2s2"]) / 300.0 * (v_s30 - 1200.0)
    else:
        phi_s2s = c["s2s2"]
    panel = math.sqrt(tau**2 + phi_ss**2 + phi_s2s**2)

    def interp(m5, m6, m7):
        if mag <= 5.0:
            return c[m5]
        if mag <= 6.0:
            return c[m5] + (mag - 5.0) * (c[m6] - c[m5])
        if mag <= 7.0:
            return c[m6] + (mag - 6.0) * (c[m7] - c[m6])
        return c[m7]

    tau_e = interp("tau_M5", "tau_M6", "tau_M7")
    phi_e = interp("phi_M5", "phi_M6", "phi_M7")
    return 0.2 * panel + 0.8 * math.sqrt(phi_e**2 + tau_e**2)


def ref_model(mag, dist_rup, v_s30):
    """ln median and sigma at all periods."""
    tables = _load_tables()
    n_r, n_m = len(NgaEastUsgs2017.TABLE_DISTS), len(NgaEastUsgs2017.TABLE_MAGS)
    ln_resps, ln_stds = [], []
    coeff = NgaEastUsgs2017.COEFF
    for j, period in enumerate(NgaEastUsgs2017.PERIODS):
        c = {name: coeff[name][j] for name in coeff.dtype.names}
        total = 0.0
        for i in range(17):
            mu = ref_table(tables[i, :, j].reshape(n_r, n_m), mag, dist_rup)
            pga = math.exp(
                ref_table(
                    tables[i, :, NgaEastUsgs2017.INDEX_PGA].reshape(n_r, n_m),
                    mag,
                    dist_rup,
                )
            )
            f_t, s_t = ref_site_amp(c, period, pga, v_s30)
            mu_amp = mu + f_t
            mu_site = math.log(
                0.185 * math.exp(mu_amp + s_t)
                + 0.63 * math.exp(mu_amp)
                + 0.185 * math.exp(mu_amp - s_t)
            )
            total += math.exp(mu_site) * NgaEastUsgs2017.WEIGHTS[j, i]
        ln_resps.append(math.log(total))
        ln_stds.append(ref_sigma(c, mag, v_s30))
    return np.array(ln_resps), np.array(ln_stds)


# Includes Vs30 below 200 and 150 m/s, at the phi_S2S, f760, and fv breaks,
# between 2000 and 3000 m/s, at and above 3000 m/s, and magnitudes and distances
# at and beyond the table limits
REF_CASES = [
    (3.5, 0.0, 100.0),
    (4.0, 0.5, 150.0),
    (4.7, 1500.0, 185.0),
    (5.2, 2000.0, 250.0),
    (5.75, 7.5, 450.0),
    (6.3, 33.0, 600.0),
    (6.5, 1e-6, 760.0),
    (7.0, 120.0, 1300.0),
    (7.6, 15.0, 1500.0),
    (8.2, 3.0, 1999.0),
    (8.5, 75.0, 2500.0),
    (9.0, 3000.0, 3000.0),
    (6.0, 10.0, 3500.0),
]


@pytest.mark.parametrize("mag,dist_rup,v_s30", REF_CASES)
def test_reference_implementation(mag, dist_rup, v_s30):
    m = compute(mag=mag, dist_rup=dist_rup, v_s30=v_s30)
    ln_resp, ln_std = ref_model(mag, dist_rup, v_s30)
    np.testing.assert_allclose(m._ln_resp, ln_resp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(m._ln_std, ln_std, rtol=1e-12)


def test_vs30_above_3000_has_no_site_term():
    # V_S30 >= 3000 m/s gives the hard-rock response of the tables
    rock = compute(mag=6.0, dist_rup=20.0, v_s30=3000.0)
    above = compute(mag=6.0, dist_rup=20.0, v_s30=5000.0)
    np.testing.assert_array_equal(rock._ln_resp, above._ln_resp)
    # Weighted mean of the tables at a table distance and magnitude
    tables = _load_tables()
    n_m = len(NgaEastUsgs2017.TABLE_MAGS)
    index = 5 * n_m + 4  # 20 km and M 6
    expected = np.log(np.sum(NgaEastUsgs2017.WEIGHTS.T * np.exp(tables[:, index]), 0))
    np.testing.assert_allclose(rock._ln_resp, expected, rtol=1e-14)

    # As in nshmp-haz, the response is not continuous at 3000 m/s. Below 3000 m/s,
    # the linear amplification approaches zero but the three-point distribution
    # still uses the standard deviation of f760.
    near = compute(mag=6.0, dist_rup=20.0, v_s30=2999.999)
    c = NgaEastUsgs2017.COEFF
    sigma_760 = c.f760is * 0.767 + c.f760gs * (1 - 0.767)
    np.testing.assert_allclose(
        near._ln_resp - rock._ln_resp,
        np.log(0.37 * np.cosh(sigma_760) + 0.63),
        atol=1e-5,
    )


def test_vs30_below_150_is_clamped():
    m_150 = compute(mag=6.0, dist_rup=20.0, v_s30=150.0)
    m_100 = compute(mag=6.0, dist_rup=20.0, v_s30=100.0)
    np.testing.assert_array_equal(m_100._ln_resp, m_150._ln_resp)
    # The standard deviation is the same for any V_S30 < 1200 m/s
    np.testing.assert_array_equal(m_100._ln_std, m_150._ln_std)


def test_table_clamping():
    # Magnitudes and distances beyond the tables use the table limits
    np.testing.assert_array_equal(
        compute(mag=9.0, dist_rup=20.0, v_s30=760.0)._ln_resp,
        compute(mag=8.2, dist_rup=20.0, v_s30=760.0)._ln_resp,
    )
    np.testing.assert_array_equal(
        compute(mag=3.0, dist_rup=20.0, v_s30=760.0)._ln_resp,
        compute(mag=4.0, dist_rup=20.0, v_s30=760.0)._ln_resp,
    )
    np.testing.assert_array_equal(
        compute(mag=6.0, dist_rup=3000.0, v_s30=760.0)._ln_resp,
        compute(mag=6.0, dist_rup=1500.0, v_s30=760.0)._ln_resp,
    )
    np.testing.assert_array_equal(
        compute(mag=6.0, dist_rup=0.0, v_s30=760.0)._ln_resp,
        compute(mag=6.0, dist_rup=1e-5, v_s30=760.0)._ln_resp,
    )


def test_out_of_range_values_warn():
    with pytest.warns(UserWarning, match="v_s30"):
        NgaEastUsgs2017(
            Scenario(mag=6.0, dist_rup=20.0, v_s30=np.array([150.0, 760.0]))
        )
    with pytest.warns(UserWarning, match="mag"):
        NgaEastUsgs2017(Scenario(mag=np.array([6.0, 8.5]), dist_rup=20.0, v_s30=760.0))


def test_pgv_units():
    # PGV in cm/s; ~10 cm/s for M 6 at 20 km on a 760 m/s site
    m = compute(mag=6.0, dist_rup=20.0, v_s30=760.0)
    assert 1.0 < m.pgv < 100.0
    assert m.ln_std_pgv > 0


# Vectorized model


MAGS = [4.0, 5.5, 6.75, 8.2, 8.5]
DISTS = [0.0, 10.0, 75.0, 1500.0, 2000.0]
V_S30S = [140.0, 260.0, 1300.0, 2500.0, 3000.0]


def scalar_results(mag, dist_rup, v_s30):
    m = compute(mag=mag, dist_rup=dist_rup, v_s30=v_s30)
    return {
        "pga": m.pga,
        "ln_pga": m.ln_pga,
        "ln_std_pga": m.ln_std_pga,
        "pgv": m.pgv,
        "ln_std_pgv": m.ln_std_pgv,
        "spec_accels": m.spec_accels,
        "ln_stds": m.ln_stds,
        "interp_spec_accels": m.interp_spec_accels(PERIODS),
        "interp_ln_stds": m.interp_ln_stds(PERIODS),
    }


@pytest.fixture(scope="module")
def grid():
    rows = list(itertools.product(MAGS, DISTS, V_S30S))
    mag, dist_rup, v_s30 = (np.array(c) for c in zip(*rows))
    return rows, compute(mag=mag, dist_rup=dist_rup, v_s30=v_s30)


@pytest.mark.parametrize(
    "key",
    [
        "pga",
        "ln_pga",
        "ln_std_pga",
        "pgv",
        "ln_std_pgv",
        "spec_accels",
        "ln_stds",
        "interp_spec_accels",
        "interp_ln_stds",
    ],
)
def test_vectorized_matches_scalar(grid, key):
    rows, m = grid
    if key.startswith("interp"):
        actual = getattr(m, key)(PERIODS)
    else:
        actual = getattr(m, key)
    expected = np.array([scalar_results(*row)[key] for row in rows])
    assert actual.shape == expected.shape
    if key.startswith("interp"):
        # interp1d evaluates 2-D arrays with slightly different floating point round-off
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_vectorized_shapes(grid):
    rows, m = grid
    n = len(rows)
    assert m.pga.shape == (n,)
    assert m.pgv.shape == (n,)
    assert m.ln_std_pga.shape == (n,)
    assert m.spec_accels.shape == (n, 21)
    assert m.ln_stds.shape == (n, 21)
    assert m.interp_spec_accels(PERIODS).shape == (n, len(PERIODS))


def test_scalar_shapes_are_unchanged():
    m = compute(mag=6.5, dist_rup=20.0, v_s30=760.0)
    assert isinstance(m.pga, float)
    assert isinstance(m.ln_pga, float)
    assert isinstance(m.ln_std_pga, float)
    assert isinstance(m.pgv, float)
    assert m.spec_accels.shape == (21,)
    assert m.ln_stds.shape == (21,)
    assert m.interp_spec_accels(PERIODS).shape == (len(PERIODS),)


def test_scalars_broadcast_with_arrays():
    mag = np.array([5.5, 6.5, 7.5])
    dist_rup = np.array([5.0, 20.0, 80.0])
    m = compute(mag=mag, dist_rup=dist_rup, v_s30=400.0)
    expected = [scalar_results(mg, d, 400.0)["pga"] for mg, d in zip(mag, dist_rup)]
    np.testing.assert_array_equal(m.pga, expected)

    # Only V_S30 is an array
    v_s30 = np.array([200.0, 760.0, 2500.0])
    m = compute(mag=6.0, dist_rup=20.0, v_s30=v_s30)
    assert m.spec_accels.shape == (3, 21)
    assert m.ln_stds.shape == (3, 21)
    expected = np.array([scalar_results(6.0, 20.0, v)["spec_accels"] for v in v_s30])
    np.testing.assert_array_equal(m.spec_accels, expected)

    # Only the magnitude is an array, the standard deviation has the full shape
    m = compute(mag=mag, dist_rup=20.0, v_s30=760.0)
    assert m.ln_stds.shape == (3, 21)


def test_multidimensional_inputs():
    mag, dist_rup = np.meshgrid(
        [5.5, 6.5, 7.5], [5.0, 20.0, 80.0, 150.0], indexing="ij"
    )
    m = compute(mag=mag, dist_rup=dist_rup, v_s30=760.0)
    assert m.pga.shape == (3, 4)
    assert m.spec_accels.shape == (3, 4, 21)
    assert m.ln_stds.shape == (3, 4, 21)
    assert m.pga[2, 1] == scalar_results(7.5, 20.0, 760.0)["pga"]


def vector_scenario(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return Scenario(
        mag=rng.uniform(4.0, 8.2, n),
        dist_rup=rng.uniform(0.0, 1500.0, n),
        v_s30=rng.uniform(200.0, 3000.0, n),
    )


@pytest.mark.parametrize("ims", [["pga"], "pga", ("pga",)])
def test_pga_only_matches_default(ims):
    s = vector_scenario()
    full = NgaEastUsgs2017(s)
    pga_only = NgaEastUsgs2017(s, ims=ims)
    np.testing.assert_array_equal(pga_only.pga, full.pga)
    np.testing.assert_array_equal(pga_only.ln_std_pga, full.ln_std_pga)
    # Only one period is computed
    assert pga_only._ln_resp.shape == (500, 1)
    assert pga_only._ln_std.shape == (500, 1)


def test_pga_only_scalar_matches_default():
    s = Scenario(mag=6.5, dist_rup=20.0, v_s30=760.0)
    full = NgaEastUsgs2017(s)
    pga_only = NgaEastUsgs2017(s, ims=["pga"])
    assert pga_only.pga == full.pga
    assert pga_only.ln_std_pga == full.ln_std_pga
    assert isinstance(pga_only.pga, float)


@pytest.mark.parametrize(
    "ims", [["psa_all"], ["psa_ngawest2_21"], ["pga", "pgv", "psa_all"]]
)
def test_psa_matches_default(ims):
    s = vector_scenario()
    full = NgaEastUsgs2017(s)
    m = NgaEastUsgs2017(s, ims=ims)
    np.testing.assert_array_equal(m.periods, full.periods)
    np.testing.assert_array_equal(m.spec_accels, full.spec_accels)
    np.testing.assert_array_equal(m.ln_stds, full.ln_stds)
    np.testing.assert_allclose(
        m.interp_spec_accels(PERIODS), full.interp_spec_accels(PERIODS), rtol=1e-12
    )


def test_selected_periods_match_default():
    s = vector_scenario()
    full = NgaEastUsgs2017(s)
    m = NgaEastUsgs2017(s, ims=["pgv", "psa_0p200", "psa_1p000"])
    assert m.psa_ims == ["psa_0p200", "psa_1p000"]
    cols = np.searchsorted(full.periods, [0.2, 1.0])
    np.testing.assert_array_equal(m.spec_accels, full.spec_accels[:, cols])
    np.testing.assert_array_equal(m.ln_stds, full.ln_stds[:, cols])
    np.testing.assert_array_equal(m.pgv, full.pgv)
    np.testing.assert_array_equal(m.ln_std_pgv, full.ln_std_pgv)
    with pytest.raises(ValueError, match="pga was not computed"):
        m.pga


@pytest.mark.parametrize(
    "ims,match",
    [
        (["pgd"], "does not provide 'pgd'"),
        (["psa_0p040"], "does not provide 'psa_0p040'"),
        (["sa"], "not a valid intensity measure"),
    ],
)
def test_invalid_ims_raise(ims, match):
    with pytest.raises(ValueError, match=match):
        NgaEastUsgs2017(vector_scenario(), ims=ims)


@pytest.mark.parametrize("ims", [None, ["pga"]])
def test_ln_pga(ims):
    s = vector_scenario()
    m = NgaEastUsgs2017(s, ims=ims)
    assert m.ln_pga.shape == (500,)
    np.testing.assert_array_equal(np.exp(m.ln_pga), m.pga)
    np.testing.assert_array_equal(m.ln_pga, NgaEastUsgs2017(s).ln_pga)
