"""Tests for MATLAB-named helpers added to complete the IdEM port.

The functions in this module correspond one-to-one to MATLAB ``.m`` files
that were not yet covered: option factories/mergers, ``CheckPuzzle``,
``RelativeWeightsComputation``, ``Compute_VF_Error``, ``Multi_Adding``,
``GetAdaptiveFrequencySamples``, ``raw2dat``/``fraw2dat`` and ``CheckIdEM``.
"""

from __future__ import annotations

import numpy as np
import pytest

from idempy import (
    AdaptiveSamplingDefaultOptions,
    CheckIdEM,
    CheckPuzzle,
    Compute_VF_Error,
    DUM,
    DUMDefaultOptions,
    FDVFASDefaultOptions,
    FDVFDefaultOptions,
    FDVFSplitDefaultOptions,
    GetAdaptiveSamplingDefaultOptions,
    GetAdaptiveSamplingOptions,
    GetDUMDefaultOptions,
    GetFDVF_AS_DefaultOptions,
    GetFDVFSplitDefaultOptions,
    GetFDVFDefaultOptions,
    GetHamiltonianDefaultOptions,
    GetShowDUMDefaultOptions,
    GetSparseHamDefaultOptions,
    GetTDVFSplitDefaultOptions,
    GetTDVFDefaultOptions,
    GetAdaptiveFrequencySamples,
    HamiltonianDefaultOptions,
    MergeAdaptiveSamplingOptions,
    MergeFDVF_AS_Options,
    MergeFDVFOptions,
    MergeFDVFSplitOptions,
    MergeHamiltonianOptions,
    MergeShowDUMOptions,
    MergeSparseHamOptions,
    MergeTDVFOptions,
    MergeTDVFSplitOptions,
    MOD,
    Multi_Adding,
    RelativeWeightsComputation,
    ScanForValidDUM,
    ScanForValidMOD,
    ShowDUMDefaultOptions,
    SparseHamDefaultOptions,
    TDVFDefaultOptions,
    TDVFSplitDefaultOptions,
    fraw2dat,
    raw2dat,
)


# ---------------------------------------------------------------------------
# Options factories and merge helpers
# ---------------------------------------------------------------------------


def _assert_option_dicts_equal(first, second):
    assert set(first) == set(second)
    for key in first:
        np.testing.assert_equal(first[key], second[key])


def test_matlab_option_factories_alias_python_defaults():
    _assert_option_dicts_equal(GetDUMDefaultOptions(), DUMDefaultOptions())
    _assert_option_dicts_equal(GetFDVFDefaultOptions(3),
                               FDVFDefaultOptions(3))
    _assert_option_dicts_equal(GetTDVFDefaultOptions(2),
                               TDVFDefaultOptions(2))
    _assert_option_dicts_equal(GetFDVFSplitDefaultOptions(2),
                               FDVFSplitDefaultOptions(2))
    _assert_option_dicts_equal(GetTDVFSplitDefaultOptions(2),
                               TDVFSplitDefaultOptions(2))
    _assert_option_dicts_equal(GetFDVF_AS_DefaultOptions(2),
                               FDVFASDefaultOptions(2))
    _assert_option_dicts_equal(GetHamiltonianDefaultOptions(),
                               HamiltonianDefaultOptions())
    _assert_option_dicts_equal(GetSparseHamDefaultOptions(),
                               SparseHamDefaultOptions())
    _assert_option_dicts_equal(GetAdaptiveSamplingDefaultOptions(),
                               AdaptiveSamplingDefaultOptions())
    _assert_option_dicts_equal(GetAdaptiveSamplingOptions(),
                               AdaptiveSamplingDefaultOptions())
    _assert_option_dicts_equal(GetShowDUMDefaultOptions(),
                               ShowDUMDefaultOptions())


def test_merge_helpers_override_and_infer_nports():
    merged = MergeFDVFOptions({"p4poles": np.eye(3), "tol": 1e-9})
    assert merged["tol"] == 1e-9
    assert merged["p4poles"].shape == (3, 3)
    assert merged["p4res"].shape == (3, 3)
    assert np.all(merged["p4res"] == 1.0)

    merged = MergeFDVF_AS_Options({"Nadd": 6, "p4res": np.ones((4, 4))})
    assert merged["Nadd"] == 6
    assert merged["p4res"].shape == (4, 4)
    assert merged["p4poles"] == "eye"

    for merge in (MergeTDVFOptions, MergeFDVFSplitOptions,
                  MergeTDVFSplitOptions, MergeHamiltonianOptions,
                  MergeSparseHamOptions, MergeAdaptiveSamplingOptions,
                  MergeShowDUMOptions):
        defaults = merge(None)
        assert isinstance(defaults, dict) and defaults
        overridden = merge({"sentinel": 123})
        assert overridden["sentinel"] == 123
        assert defaults["sentinel"] if "sentinel" in defaults else True

    with pytest.raises(TypeError):
        MergeFDVFOptions([("tol", 1e-4)])


# ---------------------------------------------------------------------------
# CheckPuzzle
# ---------------------------------------------------------------------------


def test_checkpuzzle_renumbers_labels():
    is_ok, puzzle = CheckPuzzle(np.array([[2, 5], [5, 2]]))
    assert is_ok
    assert np.array_equal(puzzle, np.array([[1, 2], [2, 1]]))

    is_ok, puzzle = CheckPuzzle(np.zeros((2, 2)))
    assert is_ok
    assert np.array_equal(puzzle, np.zeros((2, 2), dtype=int))


def test_checkpuzzle_respects_pindex():
    pindex = np.array([[1, 0], [0, 1]])
    is_ok, puzzle = CheckPuzzle(np.array([[3, 0], [0, 7]]), pindex)
    assert is_ok
    assert np.array_equal(puzzle, np.array([[1, 0], [0, 2]]))

    is_ok, puzzle = CheckPuzzle(np.array([[3, 1], [0, 7]]), pindex)
    assert not is_ok
    assert np.array_equal(puzzle, np.array([[3, 1], [0, 7]]))


def test_checkpuzzle_rejects_invalid_descriptors():
    cases = [
        np.ones((2, 3)),
        np.array([[1.5, 0.0], [0.0, 1.0]]),
        np.array([[-1, 0], [0, 1]]),
        np.array([[np.nan, 0], [0, 1]]),
        np.array([["a", "b"], ["c", "d"]]),
    ]
    for case in cases:
        is_ok, returned = CheckPuzzle(case)
        assert not is_ok
        assert returned.shape == np.asarray(case).shape

    is_ok, _ = CheckPuzzle(np.ones((2, 2)), pindex=np.ones((3, 3)))
    assert not is_ok


# ---------------------------------------------------------------------------
# Relative weights and vector-fit errors
# ---------------------------------------------------------------------------


def _one_port_dum(values, frequencies=None, pindex=None):
    values = np.asarray(values, dtype=complex)
    if frequencies is None:
        frequencies = np.arange(1, values.size + 1, dtype=float)
    return DUM(
        Nports=1,
        pindex=np.ones((1, 1)) if pindex is None else pindex,
        istime=False, isfreq=True, R0=50.0,
        f=np.asarray(frequencies, dtype=float),
        S=values.reshape(-1, 1, 1),
    )


def test_relative_weights_formula_and_floor():
    dum = _one_port_dum([1.0, 2.0, 3.0, 4.0])
    weights = RelativeWeightsComputation(
        dum, np.array([[4.0]]), np.ones((1, 1)), alpha=1.0,
        threshold=0.5, ir=0, ic=0,
    )
    assert np.allclose(weights, [0.5, 0.5, 1.0 / 3.0, 0.25])

    # alpha scales the weights; the floor uses threshold * MaxResponses.
    squared = RelativeWeightsComputation(
        dum, np.array([[4.0]]), np.ones((1, 1)), alpha=2.0,
        threshold=0.5, ir=0, ic=0,
    )
    assert np.allclose(squared, weights ** 2)


def test_relative_weights_invalid_samples_and_pindex():
    dum = _one_port_dum([1.0, np.nan, 3.0])
    weights = RelativeWeightsComputation(
        dum, np.array([[3.0]]), np.ones((1, 1)), 1.0, 0.0, 0, 0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == 0.0
    assert weights[2] == pytest.approx(1.0 / 3.0)

    disabled = RelativeWeightsComputation(
        dum, np.array([[3.0]]), np.zeros((1, 1)), 1.0, 0.0, 0, 0)
    assert np.all(disabled == 0.0)

    with pytest.raises(ValueError):
        RelativeWeightsComputation(dum, None, np.ones((1, 1)), 1, 0, 1, 0)


def test_compute_vf_error_absolute_and_relative_weights():
    dum = _one_port_dum([1.0, 2.0, 3.0, 4.0])
    exact = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex).reshape(-1, 1, 1)
    errp, err = Compute_VF_Error(
        dum, {"p4poles": "all", "p4res": "all"}, exact,
        np.array([[4.0]]))
    assert errp == pytest.approx(0.0, abs=1e-12)
    assert err == pytest.approx(0.0, abs=1e-12)

    shifted = exact + 0.1
    errp, err = Compute_VF_Error(
        dum,
        {"p4poles": "all", "p4res": "all",
         "Weights": {"type": "absolute", "value": 2.0}},
        shifted, np.array([[4.0]]))
    assert errp == pytest.approx(0.2)
    assert err == pytest.approx(0.1)

    zero = np.zeros_like(exact)
    errp, err = Compute_VF_Error(
        dum,
        {"p4poles": "all", "p4res": "all",
         "Weights": {"type": "relative", "alpha": 1.0, "threshold": 0.5}},
        zero, np.array([[4.0]]))
    # Weights are [0.5, 0.5, 1/3, 0.25], residuals are [1, 2, 3, 4].
    expected = np.sqrt((0.5 ** 2 + 1.0 ** 2 + 1.0 ** 2 + 1.0 ** 2) / 4.0)
    assert errp == pytest.approx(expected)
    assert err == pytest.approx(4.0)


def test_compute_vf_error_shape_validation():
    dum = _one_port_dum([1.0, 2.0])
    with pytest.raises(ValueError):
        Compute_VF_Error(dum, {}, np.zeros((3, 1, 1)), None)
    with pytest.raises(TypeError):
        Compute_VF_Error(dum, [], np.zeros((2, 1, 1)), None)


# ---------------------------------------------------------------------------
# Multi_Adding
# ---------------------------------------------------------------------------


def test_multi_adding_selects_largest_error_frequencies():
    dum = _one_port_dum([0.1, 0.2, 0.9, 0.3], frequencies=[1, 2, 3, 4])
    model = np.zeros((4, 1, 1), dtype=complex)
    cp, rp, added = Multi_Adding(
        dum, model, [], [], {"Nadd": 2, "p4poles": "all"})
    assert cp.shape == (1,)
    assert rp.shape == (0,)
    assert added.shape == (1,)
    omega = 2.0 * np.pi * 3.0
    assert added[0] == pytest.approx(-0.01 * omega + 1j * omega)

    cp2, _, added2 = Multi_Adding(
        dum, model, [], [], {"Nadd": 4, "p4poles": "all"})
    assert added2.shape == (2,)
    assert cp2.shape == (2,)
    # The two largest-error frequencies (3 and 4) are used first.
    chosen = sorted(float(np.imag(pole)) for pole in added2)
    assert chosen == pytest.approx([2.0 * np.pi * 3.0, 2.0 * np.pi * 4.0])


def test_multi_adding_keeps_existing_poles_and_avoids_duplicates():
    dum = _one_port_dum([0.1, 0.2, 0.9, 0.3], frequencies=[1, 2, 3, 4])
    model = np.zeros((4, 1, 1), dtype=complex)
    omega3 = 2.0 * np.pi * 3.0
    existing = np.array([-0.01 * omega3 + 1j * omega3])
    cp, rp, added = Multi_Adding(
        dum, model, existing, [10.0], {"Nadd": 2, "p4poles": "all"})
    assert np.array_equal(cp[:1], existing)
    assert np.array_equal(rp, [10.0])
    # The peak at omega=3 is already represented: use omega=4 instead.
    assert added[0].imag == pytest.approx(2.0 * np.pi * 4.0)

    unchanged_cp, unchanged_rp, none = Multi_Adding(
        dum, model, existing, [10.0], {"Nadd": 0})
    assert np.array_equal(unchanged_cp, existing)
    assert np.array_equal(unchanged_rp, [10.0])
    assert none.size == 0


def test_multi_adding_requires_frequency_data():
    time_dum = DUM(Nports=1, pindex=np.ones((1, 1)), istime=True,
                   R0=50.0, t=np.array([0.0, 1.0]),
                   a=[np.array([1.0, 0.0])], b=[np.array([0.0, 1.0])],
                   Ts=1.0)
    with pytest.raises(ValueError):
        Multi_Adding(time_dum, np.zeros((2, 1, 1)), [], [])


# ---------------------------------------------------------------------------
# Adaptive frequency samples
# ---------------------------------------------------------------------------


def _static_mod(r0, direct):
    return MOD(
        isABCD=True, issparse=False, R0=r0,
        D=np.array([[direct]], dtype=complex),
        A=np.zeros((0, 0)), B=np.zeros((0, 1)), C=np.zeros((1, 0)),
    )


def test_adaptive_samples_passive_scattering():
    om, values, passive, nonpassive, violations = GetAdaptiveFrequencySamples(
        _static_mod(50.0, 0.5),
        AdaptiveSamplingOptions={"Nstart": 5, "Nmax": 8, "tol": 1e-6},
    )
    assert om.ndim == 1 and np.all(np.diff(om) > 0.0)
    assert values.shape == om.shape
    assert np.allclose(values, 0.5)
    assert passive and not nonpassive
    assert np.array_equal(passive[0], np.array([om[0], om[-1]]))
    assert violations.shape == (0, 2)


def test_adaptive_samples_nonpassive_scattering():
    om, values, passive, nonpassive, violations = GetAdaptiveFrequencySamples(
        _static_mod(50.0, 1.2),
        AdaptiveSamplingOptions={"Nstart": 5, "Nmax": 8, "tol": 1e-6},
    )
    assert not passive
    assert len(nonpassive) == 1
    assert np.array_equal(nonpassive[0], np.array([om[0], om[-1]]))
    assert np.allclose(values, 1.2)
    assert violations.shape == (1, 2)
    assert violations[0, 0] == pytest.approx(1.2)
    assert om[0] <= violations[0, 1] <= om[-1]


def test_adaptive_samples_admittance_orientation_and_threshold():
    # Admittance passivity requires min eig(D + D^H) >= Lev.
    _, _, passive, nonpassive, _ = GetAdaptiveFrequencySamples(
        _static_mod(0.0, -0.5),
        AdaptiveSamplingOptions={"Nstart": 5, "Nmax": 8, "tol": 1e-6},
    )
    assert not passive and nonpassive

    _, _, passive, nonpassive, _ = GetAdaptiveFrequencySamples(
        _static_mod(0.0, 0.5),
        AdaptiveSamplingOptions={"Nstart": 5, "Nmax": 8, "tol": 1e-6},
    )
    assert passive and not nonpassive

    # A higher threshold can make an otherwise passive static model suspect.
    _, _, passive, nonpassive, violations = GetAdaptiveFrequencySamples(
        _static_mod(50.0, 0.5), Lev=0.16,
        AdaptiveSamplingOptions={"Nstart": 5, "Nmax": 8, "tol": 1e-6},
    )
    assert not passive and nonpassive
    assert violations[0, 0] == pytest.approx(0.5)


def test_adaptive_samples_bands_cover_axis():
    om, _, passive, nonpassive, _ = GetAdaptiveFrequencySamples(
        _static_mod(50.0, 1.2),
        AdaptiveSamplingOptions={"Nstart": 6, "Nmax": 12, "tol": 1e-6},
    )
    bands = sorted(passive + nonpassive, key=lambda band: band[0])
    assert bands[0][0] == pytest.approx(om[0])
    assert bands[-1][1] == pytest.approx(om[-1])
    for left, right in zip(bands[:-1], bands[1:]):
        assert left[1] == pytest.approx(right[0])


# ---------------------------------------------------------------------------
# Raw-to-DAT conversion
# ---------------------------------------------------------------------------


_RAW = """Title: transient run
Plotname: Transient
Flags: real
No. Variables: 3
No. Points: 4
Variables:
\t0\ttime\ttime
\t1\tV(out)\tvoltage
\t2\tI(vs)\tcurrent
Values:
0 0 0
1e-9 0.1 0.2
2e-9 0.2 0.4
3e-9 0.3 0.6
"""


def test_raw2dat_returns_matrix_and_writes_dat(tmp_path):
    path = tmp_path / "result.raw"
    path.write_text(_RAW)
    matrix = raw2dat(str(path))
    expected = np.column_stack([
        [0.0, 1e-9, 2e-9, 3e-9],
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.2, 0.4, 0.6],
    ])
    assert matrix.shape == (4, 3)
    assert np.allclose(matrix, expected)

    dat = tmp_path / "result.dat"
    returned = raw2dat(str(path), str(dat))
    assert np.allclose(returned, expected)
    assert dat.is_file()
    assert np.allclose(np.loadtxt(dat), expected)


def test_fraw2dat_validates_frequency_axis_and_writes_dat(tmp_path):
    path = tmp_path / "frequency.raw"
    path.write_text(_RAW)
    matrix = fraw2dat(str(path))
    assert matrix.shape == (4, 3)

    dat = tmp_path / "frequency.dat"
    fraw2dat(str(path), str(dat))
    assert dat.is_file()

    bad = tmp_path / "bad.raw"
    bad.write_text(_RAW.replace("1e-9 0.1 0.2", "-1e-9 0.1 0.2"))
    with pytest.raises(ValueError):
        fraw2dat(str(bad))


# ---------------------------------------------------------------------------
# MATLAB-compatible dataset scanning
# ---------------------------------------------------------------------------


def test_scan_for_valid_dum_and_mod_return_matlab_empty_string():
    dum = _one_port_dum([1.0])
    assert ScanForValidDUM({"first": 1, "data": dum}) == "data"
    assert ScanForValidDUM({"first": 1}) == ""

    model = _static_mod(50.0, 0.5)
    assert ScanForValidMOD({"first": 1, "model": model}) == "model"
    assert ScanForValidMOD({}) == ""


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------


def test_check_idem_environment():
    assert CheckIdEM() is True
    assert CheckIdEM(verbose=True) is True
