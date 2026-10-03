"""Test selecting intensity measures with the ``ims`` argument."""

import warnings

import numpy as np
import pytest

import pygmm
from pygmm.model import GroundMotionModel

NGAWEST2_21 = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
NGAWEST2_21 += [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]


def idriss(ims=None, n=300):
    rng = np.random.default_rng(0)
    s = pygmm.Scenario(
        mag=rng.uniform(5.0, 8.0, n),
        dist_rup=rng.uniform(0.0, 150.0, n),
        v_s30=rng.uniform(450.0, 1200.0, n),
        mechanism=rng.choice(["SS", "RS"], n),
    )
    return pygmm.Idriss2014(s, ims=ims)


def bssa14(ims=None, n=300):
    rng = np.random.default_rng(0)
    s = pygmm.Scenario(
        mag=rng.uniform(4.0, 8.0, n),
        dist_jb=rng.uniform(0.0, 200.0, n),
        v_s30=rng.uniform(180.0, 1300.0, n),
        mechanism=rng.choice(["U", "SS", "NS", "RS"], n),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pygmm.BooreStewartSeyhanAtkinson2014(s, ims=ims)


MODELS = {"idriss": (idriss, 22), "bssa14": (bssa14, 105)}


@pytest.fixture(params=sorted(MODELS))
def make(request):
    return MODELS[request.param]


@pytest.mark.parametrize(
    "period,name", [(0.01, "psa_0p010"), (1.0, "psa_1p000"), (10.0, "psa_10p000")]
)
def test_psa_name_and_period(period, name):
    assert GroundMotionModel.psa_name(period) == name
    assert GroundMotionModel.psa_period(name) == period


@pytest.mark.parametrize("name", ["psa_1p0", "psa_1p000", "psa_1", "psa_1p00000"])
def test_psa_period_accepts_any_number_of_decimals(name):
    assert GroundMotionModel.psa_period(name) == 1.0


@pytest.mark.parametrize("name", ["psa_", "psa_abc", "psa_1p0p0", "psa1p000"])
def test_invalid_psa_names_raise(name):
    with pytest.raises(ValueError):
        GroundMotionModel.psa_period(name)


def test_default_computes_all_periods(make):
    factory, n_periods = make
    m = factory()
    assert len(m.periods) == n_periods
    assert len(m.psa_ims) == n_periods
    assert m.spec_accels.shape == (300, n_periods)


def test_ngawest2_21(make):
    factory, _ = make
    full = factory()
    m = factory(ims=["psa_ngawest2_21"])
    np.testing.assert_array_equal(m.periods, NGAWEST2_21)
    assert m.psa_ims == [GroundMotionModel.psa_name(p) for p in NGAWEST2_21]
    cols = np.searchsorted(full.periods, NGAWEST2_21)
    np.testing.assert_array_equal(m.spec_accels, full.spec_accels[:, cols])
    np.testing.assert_array_equal(m.ln_stds, full.ln_stds[:, cols])
    assert m._ln_resp.shape == (300, 21)


def test_psa_all_matches_default(make):
    factory, _ = make
    full = factory()
    m = factory(ims=["psa_all"])
    np.testing.assert_array_equal(m.periods, full.periods)
    np.testing.assert_array_equal(m.spec_accels, full.spec_accels)
    np.testing.assert_array_equal(m.ln_stds, full.ln_stds)


def test_pga_and_ngawest2_21(make):
    factory, _ = make
    full = factory()
    m = factory(ims=["pga", "psa_ngawest2_21"])
    np.testing.assert_array_equal(m.pga, full.pga)
    np.testing.assert_array_equal(m.ln_std_pga, full.ln_std_pga)
    assert m.spec_accels.shape == (300, 21)


def test_single_periods_are_sorted(make):
    factory, _ = make
    full = factory()
    m = factory(ims=["psa_1p000", "psa_0p2", "pga"])
    np.testing.assert_array_equal(m.periods, [0.2, 1.0])
    assert m.psa_ims == ["psa_0p200", "psa_1p000"]
    cols = np.searchsorted(full.periods, [0.2, 1.0])
    np.testing.assert_array_equal(m.spec_accels, full.spec_accels[:, cols])
    np.testing.assert_array_equal(m.pga, full.pga)


def test_single_period_is_one_column(make):
    factory, _ = make
    m = factory(ims=["psa_1p000"])
    assert m._ln_resp.shape == (300, 1)
    assert m.spec_accels.shape == (300, 1)


def test_interpolation_uses_computed_periods(make):
    factory, _ = make
    m = factory(ims=["psa_0p200", "psa_1p000"])
    np.testing.assert_allclose(
        m.interp_spec_accels([0.2, 1.0]), m.spec_accels, rtol=1e-12
    )
    # Outside of the computed periods, interpolation is not possible
    assert np.all(np.isnan(m.interp_spec_accels([0.1])))


def test_duplicate_and_overlapping_ims(make):
    factory, _ = make
    m = factory(ims=["psa_1p000", "psa_ngawest2_21", "psa_1p0"])
    np.testing.assert_array_equal(m.periods, NGAWEST2_21)


def test_missing_period_raises(make):
    factory, _ = make
    with pytest.raises(ValueError, match="Missing periods \\(s\\): 0.123"):
        factory(ims=["psa_0p123"])


@pytest.mark.parametrize("ims", [["psa"], ["sa_1p000"], ["PGA"]])
def test_invalid_ims_raise(make, ims):
    factory, _ = make
    with pytest.raises(ValueError, match="not a valid intensity measure"):
        factory(ims=ims)


@pytest.mark.parametrize("attr", ["periods", "psa_ims", "spec_accels", "ln_stds"])
def test_psa_not_computed_raises(make, attr):
    factory, _ = make
    m = factory(ims=["pga"])
    with pytest.raises(ValueError, match="Spectral accelerations were not computed"):
        getattr(m, attr)


def test_models_without_ims_support_are_unchanged():
    # Models without the ims argument keep all of their periods
    m = pygmm.ChiouYoungs2014(
        pygmm.Scenario(
            mag=6.5, dist_rup=20.0, dist_jb=20.0, dist_x=20.0, v_s30=760.0, dip=90.0
        )
    )
    assert len(m.periods) == len(m.INDICES_PSA)
    assert len(m.psa_ims) == len(m.INDICES_PSA)
