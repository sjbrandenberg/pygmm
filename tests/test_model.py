#!/usr/bin/env python
"""Test model interface using C03 model."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pygmm import Campbell2003 as C03
from pygmm.model import Scenario, as_column, equals, take_periods


@pytest.fixture
def model():
    return C03(Scenario(mag=6.5, dist_rup=20))


def test_ln_std(model):
    assert_array_equal(model._ln_std, model.ln_stds)


def test_scenario():
    s = Scenario(mag=6, dist_jb=20)
    assert_allclose(s.mag, s["mag"])
    assert_allclose(s.dist_jb, s["dist_jb"])


@pytest.mark.parametrize(
    "attr", ["pga", "ln_std_pga", "pgv", "ln_std_pgv", "pgd", "ln_std_pgd"]
)
def test_pga(model, attr):
    with pytest.raises(NotImplementedError):
        getattr(model, attr)


@pytest.mark.parametrize(
    "values",
    [
        np.array(["SS", "RS", "NS", "U"]),
        np.array([["SS", "RS"], ["U", "RS"]]),
        np.array(["SS", "RS", "NS", "U"])[::2],
        np.array(["SS", "RS"], dtype=object),
        ["SS", "RS"],
        "RS",
        np.array([1, 2, 3]),
    ],
)
@pytest.mark.parametrize("option", ["RS", "U", "LONGER", 2])
def test_equals_matches_comparison(values, option):
    assert_array_equal(equals(values, option), np.asarray(values) == option)


def test_take_periods_and_as_column():
    one_d = np.arange(5.0)
    two_d = np.arange(10.0).reshape(2, 5)
    assert take_periods(one_d, 2) == 2.0
    assert_array_equal(take_periods(two_d, 2), [2.0, 7.0])
    assert as_column(3.0).shape == (1,)
    assert as_column(np.zeros(4)).shape == (4, 1)


def test_models_without_ims_are_unchanged(model):
    # Models that do not support ims compute all intensity measures by default
    assert model._indices is None
    assert_array_equal(model.spec_accels, np.exp(model._ln_resp))


def test_ln_pga_not_implemented(model):
    # Campbell (2003) does not provide PGA
    with pytest.raises(NotImplementedError):
        model.ln_pga
