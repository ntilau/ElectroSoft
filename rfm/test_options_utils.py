"""Tests for default options, option merging and utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from idempy import (
    AdaptiveSamplingDefaultOptions,
    DUMDefaultOptions,
    FDVFASDefaultOptions,
    FDVFDefaultOptions,
    FDVFSplitDefaultOptions,
    HamiltonianDefaultOptions,
    SparseHamDefaultOptions,
    TDVFDefaultOptions,
    TDVFSplitDefaultOptions,
    merge_options,
)
from idempy import utils


def test_default_options_have_expected_shape():
    opts = FDVFDefaultOptions(3)
    assert opts["p4poles"].shape == (3, 3)
    assert opts["p4res"].shape == (3, 3)
    assert np.all(opts["p4poles"] == np.eye(3))
    assert np.all(opts["p4res"] == 1.0)
    for factory in (TDVFDefaultOptions, FDVFSplitDefaultOptions,
                    TDVFSplitDefaultOptions, FDVFASDefaultOptions,
                    HamiltonianDefaultOptions):
        value = factory(2)
        assert isinstance(value, dict) and value
    assert DUMDefaultOptions()
    assert SparseHamDefaultOptions()
    assert AdaptiveSamplingDefaultOptions()


def test_merge_options_overrides_only_non_none_fields():
    defaults = {"a": 1, "b": np.array([1.0, 2.0])}
    merged = merge_options({"a": 5, "b": None, "c": 7}, defaults)
    assert merged["a"] == 5
    assert np.allclose(merged["b"], defaults["b"])
    # The default array must not be mutated or shared.
    merged["b"][0] = 99.0
    assert defaults["b"][0] == 1.0
    unchanged = merge_options(None, defaults)
    assert set(unchanged) == {"a", "b"}
    assert unchanged["a"] == defaults["a"]
    assert np.allclose(unchanged["b"], defaults["b"])


def test_merge_options_rejects_non_dict():
    with pytest.raises(TypeError):
        merge_options([("tol", 1e-3)], {"tol": 1e-4})


@pytest.mark.parametrize(
    "func,good,bad",
    [
        (utils.CheckPositiveInteger, "3", "0"),
        (utils.CheckPositiveReal, "1.5", "-1"),
        (utils.CheckPositiveZeroInt, "0", "-2"),
        (utils.CheckPositiveZeroReal, "0", "-0.5"),
        (utils.CheckPowerOfTwo, "8", "7"),
    ],
)
def test_checkers(func, good, bad):
    assert func(good, "h") == ""
    message = func(bad, "h")
    assert isinstance(message, str) and message.startswith("h:")


def test_check_real_range_and_conversion_factors():
    assert utils.CheckRealRangeAB("0.5", "h", 0.0, 1.0) == ""
    assert utils.CheckRealRangeAB("1.1", "h", 0.0, 1.0)
    assert utils.FindConversionFactor("m", "u") == pytest.approx(1e3)
    assert utils.FindConversionFactor("n", "m") == pytest.approx(1e-6)
    assert utils.FindReferenceType(50.0) == "scattering"
    assert utils.FindReferenceType(0.0) == "admittance"


def test_gram_schmidt_removes_projection():
    r = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    v = np.array([1.0, 2.0, 3.0])
    y, h = utils.GramSchmidt(r, v)
    assert y.shape == v.shape
    assert np.linalg.norm(y) == pytest.approx(1.0)
    residual = v - r @ h
    assert np.allclose(residual / np.linalg.norm(residual), y)
    assert abs(np.dot(r[:, 0], y)) < 1e-12
    assert abs(np.dot(r[:, 1], y)) < 1e-12


def test_leg_index_and_name_limits():
    assert utils.LegIndexStr(0, 1, 2) == "S12"
    assert utils.LegIndexStr(1, 2, 3) == "S23"
    assert utils.LimitDatasetNameLength("abcdefghij", 5) == "a...j"
    assert utils.LimitDatasetNameLength("abc", 5) == "abc"
    path = "/very/long/path/to/some/deeply/nested/file.txt"
    shortened = utils.LimitPathLength(path, 20)
    assert len(shortened) <= 20
    assert shortened.startswith("/") and shortened.endswith(".txt")


def test_check_weights_accepts_scalars_and_arrays():
    weights, error = utils.CheckWeights(1.0, 2)
    assert error == ""
    assert weights.shape == (2, 2)
    weights, error = utils.CheckWeights(np.array([1.0, 2.0]), 2)
    assert error == ""
    assert weights.shape == (2, 2)


def test_cut_text_and_must_use_sparse():
    text = ["line%d" % k for k in range(10)]
    assert utils.cut_text(text, 2, 3) == ["line2", "line3", "line4"]
    assert utils.cut_text(text, 9, 5) == ["line9"]
    from idempy import MOD
    full = MOD(isABCD=True, D=np.zeros((2, 2)), A=np.eye(2),
               B=np.ones((2, 2)), C=np.ones((2, 2)))
    assert utils.MustUseSparse(full, 64.0) is False
    sparse_mod = MOD(isABCD=True, issparse=True, D=np.zeros((2, 2)),
                     A=np.eye(2), B=np.ones((2, 2)), C=np.ones((2, 2)))
    assert utils.MustUseSparse(sparse_mod, 64.0) is True
