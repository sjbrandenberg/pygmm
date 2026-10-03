import itertools
import warnings

import numpy as np
import pytest

from pygmm import Scenario
from pygmm.afshari_stewart_2016 import AfshariStewart2016


def test_run():
    """Simple test to make sure we can get outputs."""

    s = Scenario(mag=6, dist_rup=50, v_s30=300, mechanism="SS")

    m = AfshariStewart2016(s)

    m.duration

    m.std_err


# Coefficients from Tables 1 to 3 of Afshari and Stewart (2016) for D5-75, D5-95,
# and D20-80. c_5 applies to the basin depth differential in m (see Figure 12).
PAPER = {
    "M_1": [5.35, 5.2, 5.2],
    "M_2": [7.15, 7.4, 7.4],
    "b_0": {
        "NS": [1.555, 2.541, 1.409],
        "RS": [0.7806, 1.612, 0.7729],
        "SS": [1.279, 2.302, 0.8804],
    },
    "b_1": {
        "NS": [4.992, 3.170, 4.778],
        "RS": [7.061, 4.536, 6.579],
        "SS": [5.578, 3.467, 6.188],
    },
    "b_2": [0.9011, 0.9443, 0.7414],
    "b_3": [-1.684, -3.911, -3.164],
    "c_1": [0.1159, 0.3165, 0.0646],
    "c_2": [0.1065, 0.2539, 0.0865],
    "c_3": [0.0682, 0.0932, 0.0373],
    "c_4": [-0.2246, -0.3183, -0.4237],
    "c_5": [0.0006, 0.0006, 0.0005],
    "V_ref": [368.2, 369.9, 369.6],
    "tau_1": [0.28, 0.25, 0.30],
    "tau_2": [0.25, 0.19, 0.19],
    "phi_1": [0.54, 0.43, 0.56],
    "phi_2": [0.41, 0.35, 0.45],
}


def paper_model(mag, dist_rup, v_s30, mechanism, region="california", depth_1_0=None):
    """ln(duration) and standard deviation from Equations 2 to 15 of the paper.

    `depth_1_0` is in km, as in pygmm scenarios.
    """
    ln_dur = []
    std = []
    for i in range(3):

        def coef(name):
            value = PAPER[name]
            return value[mechanism][i] if isinstance(value, dict) else value[i]

        # Source (Equations 3 to 6)
        m_star = 6.0
        if mag <= coef("M_2"):
            stress = np.exp(coef("b_1") + coef("b_2") * (mag - m_star))
        else:
            stress = np.exp(
                coef("b_1")
                + coef("b_2") * (coef("M_2") - m_star)
                + coef("b_3") * (mag - coef("M_2"))
            )
        moment = 10 ** (1.5 * mag + 16.05)
        f_0 = 4.9e6 * 3.2 * (stress / moment) ** (1 / 3)
        f_e = 1 / f_0 if mag > coef("M_1") else coef("b_0")

        # Path (Equation 7)
        r_1, r_2 = 10.0, 50.0
        if dist_rup <= r_1:
            f_p = coef("c_1") * dist_rup
        elif dist_rup <= r_2:
            f_p = coef("c_1") * r_1 + coef("c_2") * (dist_rup - r_1)
        else:
            f_p = (
                coef("c_1") * r_1
                + coef("c_2") * (r_2 - r_1)
                + coef("c_3") * (dist_rup - r_2)
            )

        # Site (Equations 8 to 12)
        v_1 = 600.0
        f_s = coef("c_4") * np.log(min(v_s30, v_1) / coef("V_ref"))
        if depth_1_0 is not None:
            if region == "japan":
                ln_mu = -5.23 / 2 * np.log(
                    (v_s30**2 + 412.39**2) / (1360**2 + 412.39**2)
                ) - np.log(1000)
            else:
                ln_mu = -7.15 / 4 * np.log(
                    (v_s30**4 + 570.94**4) / (1360**4 + 570.94**4)
                ) - np.log(1000)
            dz_1 = 1000 * (depth_1_0 - np.exp(ln_mu))
            f_s += coef("c_5") * min(dz_1, 200.0)

        ln_dur.append(np.log(f_e + f_p) + f_s)

        # Standard deviation (Equations 13 to 15)
        tau = (
            coef("tau_1")
            + (coef("tau_2") - coef("tau_1")) * (np.clip(mag, 6.5, 7.0) - 6.5) / 0.5
        )
        phi = (
            coef("phi_1")
            + (coef("phi_2") - coef("phi_1")) * (np.clip(mag, 5.5, 5.75) - 5.5) / 0.25
        )
        std.append(np.sqrt(tau**2 + phi**2))
    return np.array(ln_dur), np.array(std)


@pytest.mark.parametrize(
    "mag,dist_rup,v_s30,mechanism,region,depth_1_0",
    list(
        itertools.product(
            [4.0, 5.3, 6.0, 7.0, 7.6],
            [0.0, 5.0, 30.0, 120.0],
            [200.0, 400.0, 900.0],
            ["NS", "RS", "SS"],
            ["california", "japan"],
            [None, 0.05, 0.4, 1.5],
        )
    ),
)
def test_matches_paper(mag, dist_rup, v_s30, mechanism, region, depth_1_0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = AfshariStewart2016(
            Scenario(
                mag=mag,
                dist_rup=dist_rup,
                v_s30=v_s30,
                mechanism=mechanism,
                region=region,
                depth_1_0=depth_1_0,
            )
        )
    ln_dur, std = paper_model(mag, dist_rup, v_s30, mechanism, region, depth_1_0)
    np.testing.assert_allclose(m._ln_dur, ln_dur, rtol=1e-10)
    np.testing.assert_allclose(m._std_err, std, rtol=1e-10)


def ln_dur(**kwds):
    s = Scenario(mag=6.5, dist_rup=30.0, mechanism="SS", **kwds)
    return AfshariStewart2016(s)._ln_dur


def test_basin_term_scale():
    # For a site 100 m deeper than the mean depth for its v_s30, the basin term is
    # c_5 * 100 m, and it is capped at c_5 * 200 m for deeper basins
    v_s30 = 300.0
    mean = AfshariStewart2016.calc_depth_1_0(v_s30, "california")
    c_5 = np.array(PAPER["c_5"])
    base = ln_dur(v_s30=v_s30)
    np.testing.assert_allclose(
        ln_dur(v_s30=v_s30, depth_1_0=mean + 0.1) - base, c_5 * 100, rtol=1e-6
    )
    np.testing.assert_allclose(
        ln_dur(v_s30=v_s30, depth_1_0=mean + 1.0) - base, c_5 * 200, rtol=1e-9
    )


def test_region_defaults_to_california():
    kwds = dict(v_s30=300.0, depth_1_0=0.5)
    default = ln_dur(**kwds)
    np.testing.assert_array_equal(default, ln_dur(region="california", **kwds))
    np.testing.assert_array_equal(default, ln_dur(region="global", **kwds))


def test_region_does_not_matter_without_basin_depth():
    np.testing.assert_array_equal(
        ln_dur(v_s30=300.0, region="japan"), ln_dur(v_s30=300.0, region="california")
    )


def test_japan_and_california_basin_terms_differ():
    kwds = dict(v_s30=300.0, depth_1_0=0.2)
    assert np.all(np.abs(ln_dur(region="japan", **kwds) - ln_dur(**kwds)) > 1e-3)
