"""Test the NGA-Subduction USGS (2018) model.

nshmp-haz does not provide reference results for this model, so the model is
compared with a deliberately literal, scalar transcription of
``NgaSubductionUsgs_2018.calc`` and ``calcMean`` in nshmp-haz (Java).
"""

import csv
import itertools
import math
import os
import warnings

import numpy as np
import pytest

from pygmm.model import Scenario
from pygmm.nga_subduction_usgs_2018 import NgaSubductionUsgs2018 as NGASUB18

# The grid includes V_s30 values above the recommended limit (1000 m/sec)
pytestmark = pytest.mark.filterwarnings("ignore:v_s30 .* recommended limit")

# Literal transcription of nshmp-haz
####################################

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "pygmm",
    "data",
    "nga_subduction_usgs_2018.csv",
)


def load_java_coeffs():
    """Coefficients by nshmp-haz IMT name, read with the Java column names."""
    with open(DATA_FILE) as fp:
        lines = [line for line in fp if not line.startswith("#")]
    coeffs = {}
    for row in csv.DictReader(lines):
        period = row.pop("period")
        imt = "PGA" if float(period) == 0 else float(period)
        coeffs[imt] = {k: float(v) for k, v in row.items()}
    return coeffs


JAVA_COEFFS = load_java_coeffs()

VS30_ROCK = 1000.0
N = 1.18
C = 1.88
C4 = 10.0
A3 = 0.1
A5 = 0.0
A9 = 0.4
A10 = 1.73
C1_SLAB = 7.2
D_INT_EPI_LO = -0.3
D_INT_EPI_HI = 0.3
PHI = 0.62
EPI_WTS = [0.2, 0.6, 0.2]
SIGMA_WT = [1.0]


def java_calc_mean(c, slab, pgaRock, Mw, rRup, zTop, vs30):
    # Mw scaling
    C1 = C1_SLAB if slab else c["c1int"]
    a13m = c["a13"] * (10 - Mw) * (10 - Mw)
    fMag = (c["a4"] if Mw <= C1 else A5) * (Mw - C1) + a13m

    # Depth scaling; slab only
    fDepth = c["a11"] * (min(zTop, 100.0) - 60.0) if slab else 0.0

    # Nonlinear site response scaling
    vsS = min(vs30, VS30_ROCK)
    lnVs = math.log(vsS / c["vlin"])
    if vs30 < c["vlin"]:
        fSite = (
            c["a12"] * lnVs
            - c["b"] * math.log(pgaRock + C)
            + c["b"] * math.log(pgaRock + C * math.pow(vsS / c["vlin"], N))
        )
    else:
        fSite = (c["a12"] + c["b"] * N) * lnVs

    dC1term = c["a4"] * (C1_SLAB - c["c1int"]) if slab else 0.0
    return (
        c["a1"]
        + dC1term
        + (c["a2"] + (c["a14"] if slab else 0.0) + A3 * (Mw - 7.8))
        * math.log(rRup + C4 * math.exp((Mw - 6.0) * A9))
        + c["a6"] * rRup
        + (A10 if slab else 0.0)
        + fMag
        + fDepth
        + fSite
    )


def java_weighted_sum_ln(data, weights):
    # Data.weightedSumLn
    total = 0.0
    for d, w in zip(data, weights):
        total += math.exp(d) * w
    return math.log(total)


def java_weighted_sum(data, weights):
    # Data.weightedSum
    total = 0.0
    for d, w in zip(data, weights):
        total += d * w
    return total


def java_calc(imt, slab, include_epi, Mw, rRup, zTop, vs30):
    """Return (ln median, sigma) as computed by nshmp-haz."""
    coeffs = JAVA_COEFFS[imt]
    coeffsPGA = JAVA_COEFFS["PGA"]
    pgaRock = (
        math.exp(java_calc_mean(coeffsPGA, slab, 0.0, Mw, rRup, zTop, VS30_ROCK))
        if vs30 < coeffs["vlin"]
        else 0.0
    )
    mu = java_calc_mean(coeffs, slab, pgaRock, Mw, rRup, zTop, vs30)
    sigma = math.hypot(PHI, coeffs["tau"])

    # mu Cascadia adjustment
    mu += coeffs["adj_slab"] if slab else coeffs["adj_int"]

    if not include_epi:
        return mu, sigma

    epiLo = coeffs["adj_slab_epi_lo"] if slab else D_INT_EPI_LO
    epiHi = coeffs["adj_slab_epi_hi"] if slab else D_INT_EPI_HI
    means = [mu + epiLo, mu, mu + epiHi]
    # MultiScalarGroundMotion
    return java_weighted_sum_ln(means, EPI_WTS), java_weighted_sum([sigma], SIGMA_WT)


def test_coefficient_columns():
    # The data file provides the coefficients used by the Java Coefficients class
    names = set(NGASUB18.COEFF.dtype.names)
    java = (
        "a1 a2 a4 a6 a11 a12 a13 a14 vlin b c1int adj_int adj_slab "
        "adj_slab_epi_lo adj_slab_epi_hi tau"
    ).split()
    assert names == set(java) | {"period"}
    assert NGASUB18.PERIODS[NGASUB18.INDEX_PGA] == 0
    assert len(NGASUB18.INDICES_PSA) == 24
    # Typo fix noted in nshmp-haz: slab epistemic adjustment of 0.3 at 7.5 and 10 s
    c = NGASUB18.COEFF
    np.testing.assert_array_equal(c.adj_slab_epi_hi[c.period >= 7.5], 0.3)


# Event types, magnitudes around C1 (7.2 for slab and c1int, 8.2 to 7.8, for
# interface events), sites around v_lin (865.1 m/sec at short periods, 400 m/sec
# at long periods) and above 1000 m/sec, depths around 100 km, and distances
# including 0
EVENT_TYPES = ["interface", "intraslab"]
MAGS = [5.0, 6.5, 7.2, 7.25, 7.8, 8.0, 8.15, 8.2, 8.5, 9.5]
V_S30S = [150.0, 300.0, 400.0, 760.0, 865.1, 1000.0, 1200.0]
DEPTHS_HYP = [30.0, 100.0, 150.0]
DISTS_RUP = [0.0, 15.0, 100.0, 600.0]
NAMES = ["mag", "dist_rup", "depth_hyp", "v_s30", "event_type"]


def grid_scenario(rows):
    return Scenario(**{n: np.array(c) for n, c in zip(NAMES, zip(*rows))})


@pytest.fixture(scope="module")
def grid():
    rows = list(itertools.product(MAGS, DISTS_RUP, DEPTHS_HYP, V_S30S, EVENT_TYPES))
    return rows, grid_scenario(rows)


@pytest.mark.parametrize("epistemic", [True, False])
def test_matches_nshmp_transcription(grid, epistemic):
    rows, scenario = grid
    m = NGASUB18(scenario, epistemic=epistemic)
    imts = ["PGA"] + list(NGASUB18.PERIODS[NGASUB18.INDICES_PSA])
    expected = np.array(
        [
            [
                java_calc(
                    imt, event_type == "intraslab", epistemic, mag, dist, depth, vs
                )
                for imt in imts
            ]
            for mag, dist, depth, vs, event_type in rows
        ]
    )
    ln_resp = np.column_stack([m.ln_pga, np.log(m.spec_accels)])
    ln_std = np.column_stack([m.ln_std_pga, m.ln_stds])
    np.testing.assert_allclose(ln_resp, expected[..., 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ln_std, expected[..., 1], rtol=1e-12)


def test_pga_ref_excludes_cascadia_adjustment():
    # At V_s30 < V_lin, the reference PGA is the unadjusted rock PGA
    s = Scenario(mag=8.0, dist_rup=20.0, v_s30=200.0, event_type="interface")
    m = NGASUB18(s, epistemic=False)
    assert m.ln_pga == pytest.approx(
        java_calc("PGA", False, False, 8.0, 20.0, np.nan, 200.0)[0], rel=1e-14
    )


# Vectorized scenarios
######################

KEYS = ["pga", "ln_pga", "ln_std_pga", "spec_accels", "ln_stds"]
PERIODS = [0.05, 0.3, 1.0, 2.5]


def scalar_results(*args, **kwds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = NGASUB18(Scenario(**dict(zip(NAMES, args))), **kwds)
    results = {key: getattr(m, key) for key in KEYS}
    results["interp_spec_accels"] = m.interp_spec_accels(PERIODS)
    results["interp_ln_stds"] = m.interp_ln_stds(PERIODS)
    return results


def assert_matches(actual, desired, key):
    assert actual.shape == desired.shape, key
    if key.startswith("interp"):
        # interp1d evaluates 2-D arrays with slightly different floating point round-off
        np.testing.assert_allclose(actual, desired, rtol=1e-12, err_msg=key)
    else:
        np.testing.assert_array_equal(actual, desired, err_msg=key)


@pytest.mark.parametrize("epistemic", [True, False])
def test_vectorized_matches_scalar(grid, epistemic):
    rows, scenario = grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = NGASUB18(scenario, epistemic=epistemic)
    expected = [scalar_results(*row, epistemic=epistemic) for row in rows]
    for key in KEYS + ["interp_spec_accels", "interp_ln_stds"]:
        if key.startswith("interp"):
            actual = getattr(m, key)(PERIODS)
        else:
            actual = getattr(m, key)
        desired = np.array([e[key] for e in expected])
        assert_matches(actual, desired, key)


def test_vectorized_shapes(grid):
    rows, scenario = grid
    m = NGASUB18(scenario)
    n = len(rows)
    assert m.pga.shape == (n,)
    assert m.ln_pga.shape == (n,)
    assert m.ln_std_pga.shape == (n,)
    assert m.spec_accels.shape == (n, 24)
    assert m.ln_stds.shape == (n, 24)
    assert m.interp_spec_accels(PERIODS).shape == (n, len(PERIODS))


def test_scalar_shapes_are_unchanged():
    for event_type in EVENT_TYPES:
        m = NGASUB18(
            Scenario(
                mag=7.0,
                dist_rup=80.0,
                depth_hyp=50.0,
                v_s30=400.0,
                event_type=event_type,
            )
        )
        assert isinstance(m.pga, float)
        assert isinstance(m.ln_pga, float)
        assert isinstance(m.ln_std_pga, float)
        assert m.spec_accels.shape == (24,)
        assert m.ln_stds.shape == (24,)
        assert m.interp_spec_accels(PERIODS).shape == (len(PERIODS),)


def test_scalars_broadcast_with_arrays():
    mag = np.array([6.0, 7.0, 8.0])
    dist_rup = np.array([20.0, 80.0, 200.0])
    m = NGASUB18(
        Scenario(
            mag=mag,
            dist_rup=dist_rup,
            depth_hyp=60.0,
            v_s30=300.0,
            event_type="intraslab",
        )
    )
    for i in range(3):
        expected = scalar_results(mag[i], dist_rup[i], 60.0, 300.0, "intraslab")
        assert m.pga[i] == expected["pga"]
        np.testing.assert_array_equal(m.spec_accels[i], expected["spec_accels"])


def test_mixed_event_types():
    event_type = np.array(["interface", "intraslab", "intraslab", "interface"])
    mag = np.array([9.0, 7.0, 6.0, 8.0])
    depth_hyp = np.array([np.nan, 50.0, 120.0, np.nan])
    m = NGASUB18(
        Scenario(
            mag=mag,
            dist_rup=100.0,
            depth_hyp=depth_hyp,
            v_s30=360.0,
            event_type=event_type,
        )
    )
    for i in range(4):
        expected = scalar_results(mag[i], 100.0, depth_hyp[i], 360.0, event_type[i])
        np.testing.assert_array_equal(m.spec_accels[i], expected["spec_accels"])
        assert m.pga[i] == expected["pga"]
    assert np.all(np.isfinite(m.spec_accels))


def test_multidimensional_inputs():
    mag, dist_rup = np.meshgrid(
        [6.0, 7.0, 8.0], [20.0, 50.0, 100.0, 300.0], indexing="ij"
    )
    m = NGASUB18(
        Scenario(
            mag=mag,
            dist_rup=dist_rup,
            depth_hyp=60.0,
            v_s30=400.0,
            event_type="intraslab",
        )
    )
    assert m.pga.shape == (3, 4)
    assert m.spec_accels.shape == (3, 4, 24)
    assert m.ln_stds.shape == (3, 4, 24)
    assert m.pga[2, 1] == scalar_results(8.0, 50.0, 60.0, 400.0, "intraslab")["pga"]


def test_interface_depth_can_be_none():
    params = dict(
        mag=np.array([7.0, 8.5]), dist_rup=np.array([30.0, 90.0]), v_s30=500.0
    )
    m = NGASUB18(Scenario(event_type="interface", **params))
    expected = NGASUB18(
        Scenario(event_type="interface", depth_hyp=np.array([10.0, 500.0]), **params)
    )
    np.testing.assert_array_equal(m.spec_accels, expected.spec_accels)
    # Interface events do not depend on depth
    assert m.pga[0] == scalar_results(7.0, 30.0, None, 500.0, "interface")["pga"]


@pytest.mark.parametrize(
    "event_type", ["intraslab", np.array(["interface", "intraslab"])]
)
def test_missing_depth_raises(event_type):
    with pytest.raises(ValueError, match="depth_hyp is required"):
        NGASUB18(Scenario(mag=7.0, dist_rup=50.0, v_s30=400.0, event_type=event_type))


def test_depth_limited_to_100_km():
    params = dict(mag=7.0, dist_rup=150.0, v_s30=400.0, event_type="intraslab")
    deep = NGASUB18(Scenario(depth_hyp=np.array([100.0, 200.0, 600.0]), **params))
    assert deep.pga[0] == deep.pga[1] == deep.pga[2]
    shallow = NGASUB18(Scenario(depth_hyp=50.0, **params))
    assert shallow.pga < deep.pga[0]


@pytest.mark.filterwarnings("ignore:Using default value")
def test_invalid_event_types():
    # The event type has no default, so results for invalid entries are NaN
    with pytest.warns(UserWarning, match="2 of 4 values"):
        m = NGASUB18(
            Scenario(
                mag=7.0,
                dist_rup=50.0,
                depth_hyp=60.0,
                v_s30=400.0,
                event_type=np.array(["interface", "deep", "intraslab", "crustal"]),
            )
        )
    assert np.all(np.isnan(m.spec_accels[[1, 3]]))
    assert np.all(np.isnan(m.pga[[1, 3]]))
    assert np.all(np.isfinite(m.spec_accels[[0, 2]]))
    assert m.pga[0] == scalar_results(7.0, 50.0, 60.0, 400.0, "interface")["pga"]
    assert m.pga[2] == scalar_results(7.0, 50.0, 60.0, 400.0, "intraslab")["pga"]
    # Same for a scalar scenario
    assert np.isnan(scalar_results(7.0, 50.0, 60.0, 400.0, "deep")["pga"])


def test_epistemic_branches():
    s = Scenario(
        mag=np.array([7.0, 8.5, 6.5, 7.5]),
        dist_rup=np.array([60.0, 30.0, 100.0, 200.0]),
        depth_hyp=np.array([np.nan, np.nan, 50.0, 90.0]),
        v_s30=np.array([300.0, 760.0, 200.0, 1000.0]),
        event_type=np.array(["interface", "interface", "intraslab", "intraslab"]),
    )
    epi = NGASUB18(s)
    center = NGASUB18(s, epistemic=False)
    assert epi.epistemic and not center.epistemic

    ln_epi = np.column_stack([epi.ln_pga, np.log(epi.spec_accels)])
    ln_center = np.column_stack([center.ln_pga, np.log(center.spec_accels)])
    # By Jensen's inequality, the collapsed median is at least the central median
    assert np.all(ln_epi > ln_center)

    # The offset for interface events does not depend on the period
    offset = math.log(0.2 * math.exp(-0.3) + 0.6 + 0.2 * math.exp(0.3))
    np.testing.assert_allclose(ln_epi[:2] - ln_center[:2], offset, rtol=1e-12)

    # Slab events use the period-dependent adjustments
    c = NGASUB18.COEFF
    offset = np.log(
        0.2 * np.exp(c.adj_slab_epi_lo) + 0.6 + 0.2 * np.exp(c.adj_slab_epi_hi)
    )
    np.testing.assert_allclose(
        ln_epi[2:] - ln_center[2:], np.broadcast_to(offset, (2, 25)), rtol=1e-12
    )

    # The standard deviation is the same for the branches
    np.testing.assert_array_equal(epi.ln_stds, center.ln_stds)
    np.testing.assert_allclose(epi.ln_stds, np.hypot(0.62, c.tau[1:]) * np.ones((4, 1)))


def test_epistemic_positional():
    s = Scenario(mag=7.0, dist_rup=50.0, v_s30=400.0, event_type="interface")
    assert NGASUB18(s, False, ["pga"]).pga == NGASUB18(s, epistemic=False).pga


def vector_scenario(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return Scenario(
        mag=rng.uniform(5.0, 9.5, n),
        dist_rup=rng.uniform(0.0, 500.0, n),
        depth_hyp=rng.uniform(20.0, 150.0, n),
        v_s30=rng.uniform(150.0, 1000.0, n),
        event_type=rng.choice(EVENT_TYPES, n),
    )


@pytest.mark.parametrize("epistemic", [True, False])
@pytest.mark.parametrize("ims", [["pga"], "pga", ("pga",)])
def test_pga_only_matches_default(ims, epistemic):
    s = vector_scenario()
    full = NGASUB18(s, epistemic=epistemic)
    m = NGASUB18(s, epistemic=epistemic, ims=ims)
    for key in ["pga", "ln_pga", "ln_std_pga"]:
        np.testing.assert_array_equal(getattr(m, key), getattr(full, key), err_msg=key)
    # Only one period is computed
    assert m._ln_resp.shape == (500, 1)
    assert m._ln_std.shape == (500, 1)


def test_pga_only_scalar_matches_default():
    s = Scenario(
        mag=7.0, dist_rup=50.0, depth_hyp=70.0, v_s30=300.0, event_type="intraslab"
    )
    m = NGASUB18(s, ims=["pga"])
    assert m.pga == NGASUB18(s).pga
    assert isinstance(m.pga, float)


@pytest.mark.parametrize(
    "ims", [["psa_all"], ["psa_ngawest2_21"], ["pga", "psa_1p000", "psa_0p200"]]
)
def test_psa_matches_default(ims):
    s = vector_scenario()
    full = NGASUB18(s)
    m = NGASUB18(s, ims=ims)
    cols = np.isin(full.periods, m.periods)
    np.testing.assert_array_equal(m.spec_accels, full.spec_accels[:, cols])
    np.testing.assert_array_equal(m.ln_stds, full.ln_stds[:, cols])


@pytest.mark.parametrize("ims", [None, ["pga"]])
def test_ln_pga(ims):
    m = NGASUB18(vector_scenario(), ims=ims)
    assert m.ln_pga.shape == (500,)
    np.testing.assert_array_equal(np.exp(m.ln_pga), m.pga)


def test_pgv_is_not_provided():
    with pytest.raises(ValueError, match="does not provide 'pgv'"):
        NGASUB18(vector_scenario(), ims=["pgv"])
