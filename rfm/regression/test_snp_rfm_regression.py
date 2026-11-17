"""End-to-end regression tests for TouchStone ``.sNp`` -> RFM conversion.

The regression contract has two independent halves:

* **Analytic tests.**  Every fixture is generated from a closed-form circuit in
  ``snp_circuits.py``.  The importer is checked against those equations, not
  against another copy of the importer's output.  The fitted rational model is
  checked against the imported samples.
* **Golden tests.**  ``golden/snp_rfm_golden.npz`` stores the fitted response
  and the re-evaluated model response.  These catch unintended drift in the
  Vector-Fit / passivity algorithms without pretending that a stored baseline
  is an independent correctness oracle.

To refresh the static fixtures run ``python tests/regression/generate_fixtures.py``.
To refresh the golden baseline, after an intentional algorithm change and a
review of the analytic results, run ``python tests/regression/regen_golden.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running this directory directly with pytest, without relying on the
# repository-level conftest (whose module name must not be shadowed).
_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from idempy import DUM2TouchStone, MOD2DUM, TouchStone2DUM  # noqa: E402

from snp_cases import (  # noqa: E402
    CASES,
    DATA_DIR,
    FREQ_HZ,
    GOLDEN_DIR,
    expected_scattering,
    fit_case,
    fixture_path,
    load_dum,
)
from snp_circuits import max_singular_value  # noqa: E402

pytestmark = pytest.mark.regression

CASE_IDS = [case.name for case in CASES]
CASE_BY_NAME = {case.name: case for case in CASES}

# Import tolerances are dominated by TouchStone text precision and, for MA/DB
# fixtures, by the magnitude/angle or dB/angle round trip.
IMPORT_RTOL = 1e-12
IMPORT_ATOL_RI = 1e-13
IMPORT_ATOL_MA_DB = 1e-10

# The fixtures are smooth, low-order circuits; a correct fit should agree with
# the samples much more closely than this.  The bound leaves room for BLAS and
# LAPACK differences between machines while still catching a broken solver.
FIT_RTOL = 1e-8
FIT_ATOL = 1e-9
MODEL_RTOL = 1e-8
MODEL_ATOL = 1e-9

# Golden comparisons should be reproducible on the same platform, but do not
# require bit-for-bit equality after a library update.
GOLDEN_RTOL = 1e-6
GOLDEN_ATOL = 1e-9
GOLDEN_FIT_SLACK = 10.0
GOLDEN_FIT_FLOOR = 1e-9

PASSIVITY_ATOL = 1e-8


def _assert_close(actual, expected, label, *, rtol, atol):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, (
        f"{label}: shape {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol, err_msg=label)


def _assert_golden(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, (
        f"{label}: shape {actual.shape} != golden shape {expected.shape}")
    scale = max(float(np.max(np.abs(expected))) if expected.size else 0.0, 1.0)
    limit = GOLDEN_ATOL + GOLDEN_RTOL * scale
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert error <= limit, f"{label}: max |delta| = {error:.3e} > {limit:.3e}"


@pytest.fixture(scope="session")
def golden():
    archive_path = GOLDEN_DIR / "snp_rfm_golden.npz"
    manifest_path = GOLDEN_DIR / "snp_rfm_manifest.json"
    assert archive_path.is_file(), f"missing golden archive {archive_path}"
    assert manifest_path.is_file(), f"missing golden manifest {manifest_path}"
    archive = np.load(archive_path, allow_pickle=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        np.asarray(manifest["frequency_hz"], dtype=float),
        FREQ_HZ,
        rtol=1e-14,
        atol=0.0,
        err_msg="golden frequency grid does not match the fixture grid",
    )
    return archive, manifest


# ---------------------------------------------------------------------------
# Fixture integrity and the independent analytic oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_fixture_file_is_static_and_nonempty(case):
    path = fixture_path(case)
    assert path.is_file(), f"missing fixture {path}"
    assert path.stat().st_size > 0
    assert path.parent == DATA_DIR / Path(case.filename).parent


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_analytic_fixture_has_expected_passivity(case):
    expected = expected_scattering(case)
    assert expected.shape == (FREQ_HZ.size, case.nports, case.nports)
    singular_values = max_singular_value(expected)
    assert singular_values.shape == (FREQ_HZ.size,)
    assert np.all(np.isfinite(singular_values))

    if case.passive:
        assert np.all(singular_values <= 1.0 + 1e-12), (
            f"{case.name}: fixture is not passive, max sigma = "
            f"{np.max(singular_values):.6f}")
    else:
        assert np.all(singular_values > 1.0 + 1e-6), (
            f"{case.name}: active fixture is not active")


def test_case_definitions_are_unique():
    assert len(CASE_BY_NAME) == len(CASES)
    assert len({case.filename for case in CASES}) == len(CASES)
    for case in CASES:
        assert case.fmt in {"RI", "MA", "DB"}
        assert case.unit in {"Hz", "kHz", "MHz", "GHz"}
        assert case.r0 > 0.0
        assert case.order >= 1


# ---------------------------------------------------------------------------
# TouchStone import against the closed-form oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_touchstone_import_matches_closed_form(case):
    dum = load_dum(case)
    expected = expected_scattering(case)

    assert dum.Nports == case.nports
    assert dum.isfreq is True
    assert dum.istime is False
    assert dum.R0 == pytest.approx(case.r0, rel=1e-15, abs=0.0)
    assert np.asarray(dum.pindex).shape == (case.nports, case.nports)
    assert np.all(np.asarray(dum.pindex) == 1)

    _assert_close(dum.f, FREQ_HZ, f"{case.name} frequency", rtol=1e-12, atol=0.0)
    atol = (IMPORT_ATOL_RI if case.fmt == "RI" else IMPORT_ATOL_MA_DB)
    _assert_close(
        dum.S, expected, f"{case.name} S-parameters",
        rtol=IMPORT_RTOL, atol=atol)
    assert dum.Fmax == pytest.approx(FREQ_HZ[-1], rel=1e-12)


def test_ri_ma_db_fixtures_are_equivalent():
    reference = load_dum(CASE_BY_NAME["series_shunt_ri_hz"])
    for name in ("series_shunt_ma_ghz", "series_shunt_db_ghz"):
        other = load_dum(CASE_BY_NAME[name])
        _assert_close(
            other.f, reference.f, f"{name} frequency",
            rtol=1e-13, atol=0.0)
        _assert_close(
            other.S, reference.S, f"{name} S-parameters",
            rtol=1e-12, atol=1e-12)


def test_reference_resistance_is_read_from_option_line():
    default = load_dum(CASE_BY_NAME["series_shunt_ri_hz"])
    high = load_dum(CASE_BY_NAME["series_shunt_75ohm"])
    assert default.R0 == pytest.approx(50.0)
    assert high.R0 == pytest.approx(75.0)
    # The same physical network normalized to a different R0 must give a
    # different scattering matrix, proving R0 is not silently ignored.
    assert not np.allclose(default.S, high.S)


# ---------------------------------------------------------------------------
# Vector-Fit macromodeling quality, stability, and passivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", [case for case in CASES if case.passive],
    ids=[case.name for case in CASES if case.passive])
def test_passive_snp_fits_and_remains_passive(case):
    result = fit_case(case)
    model = result["model"]
    expected = expected_scattering(case)

    assert model.Order == case.order
    assert model.isResPoles is True
    assert result["error"] <= FIT_RTOL, (
        f"{case.name}: fit error {result['error']:.3e}")

    _assert_close(
        result["fitted"], result["dum"].S,
        f"{case.name} fitted vs imported",
        rtol=FIT_RTOL, atol=FIT_ATOL)
    _assert_close(
        result["fitted"], expected,
        f"{case.name} fitted vs closed form",
        rtol=FIT_RTOL, atol=FIT_ATOL)
    _assert_close(
        result["model_response"], result["dum"].S,
        f"{case.name} MOD2DUM vs imported",
        rtol=MODEL_RTOL, atol=MODEL_ATOL)

    real_poles = np.asarray(model.RealPoles, dtype=float).ravel()
    complex_poles = np.asarray(model.ComplexPoles, dtype=complex).ravel()
    assert np.all(np.isfinite(real_poles))
    assert np.all(np.isfinite(complex_poles))
    assert np.all(real_poles < 0.0), f"{case.name}: unstable real pole"
    assert np.all(complex_poles.real < 0.0), (
        f"{case.name}: unstable complex pole")
    assert np.all(np.isfinite(np.asarray(model.D)))

    assert result["is_passive"] is True
    assert result["model_check"] == 1
    assert result["compensated_check"] == 1
    assert result["compensated_response"] is None
    assert np.all(max_singular_value(result["model_response"])
                  <= 1.0 + PASSIVITY_ATOL)


def test_active_snp_is_detected_and_compensated():
    case = CASE_BY_NAME["active_gain"]
    result = fit_case(case)

    assert result["is_passive"] is False
    assert result["model_check"] == 0
    assert result["compensated_check"] == 1
    assert result["compensated_response"] is not None
    assert np.max(max_singular_value(result["model_response"])) > 1.5
    assert np.all(
        max_singular_value(result["compensated_response"])
        <= 1.0 + PASSIVITY_ATOL)

    # Compensation is allowed to change non-passive data (it clamps the
    # active gain here); it must only restore passivity without destroying
    # the response.
    compensated_gain = np.abs(
        np.asarray(result["compensated_response"])[:, 1, 0])
    assert np.all(compensated_gain <= 1.0 + PASSIVITY_ATOL)
    assert np.all(compensated_gain > 0.5)
    assert np.all(np.isfinite(result["compensated_response"]))


# ---------------------------------------------------------------------------
# TouchStone export round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_import_export_roundtrip(case, tmp_path):
    dum = load_dum(case)
    filename = Path(case.filename).name
    DUM2TouchStone(str(tmp_path), filename, dum)
    restored = TouchStone2DUM(str(tmp_path), filename)

    assert restored.Nports == dum.Nports
    assert restored.R0 == pytest.approx(dum.R0)
    _assert_close(restored.f, dum.f, f"{case.name} roundtrip frequency",
                  rtol=1e-14, atol=0.0)
    _assert_close(restored.S, dum.S, f"{case.name} roundtrip S",
                  rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_snp_to_rfm_to_snp_roundtrip(case, tmp_path):
    result = fit_case(case)
    model_dum = MOD2DUM(result["model"], result["dum"])
    filename = f"model_{Path(case.filename).name}"
    DUM2TouchStone(str(tmp_path), filename, model_dum)
    restored = TouchStone2DUM(str(tmp_path), filename)

    _assert_close(
        restored.S, result["model_response"],
        f"{case.name} RFM export/import",
        rtol=1e-12, atol=1e-12)
    _assert_close(
        restored.S, result["dum"].S,
        f"{case.name} RFM export vs original data",
        rtol=MODEL_RTOL, atol=MODEL_ATOL)


# ---------------------------------------------------------------------------
# Golden snapshot
# ---------------------------------------------------------------------------

def test_golden_covers_every_case(golden):
    archive, manifest = golden
    assert manifest["version"] == 1
    assert set(manifest["cases"]) == set(CASE_BY_NAME)
    available = set(archive.files)
    for case in CASES:
        assert f"happ_{case.name}" in available
        assert f"model_response_{case.name}" in available
        entry = manifest["cases"][case.name]
        assert entry["nports"] == case.nports
        assert entry["reference_resistance"] == pytest.approx(case.r0)
        assert entry["requested_order"] == case.order
        if entry["is_passive"]:
            assert f"compensated_response_{case.name}" not in available
        else:
            assert f"compensated_response_{case.name}" in available


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_golden_snp_to_rfm_snapshot(case, golden):
    archive, manifest = golden
    entry = manifest["cases"][case.name]
    result = fit_case(case)

    assert result["model"].Order == entry["model_order"], (
        f"{case.name}: model order drifted")
    assert result["is_passive"] is entry["is_passive"], (
        f"{case.name}: passivity verdict drifted")
    assert result["model_check"] == entry["model_check"]
    assert result["compensated_check"] == entry["compensated_check"]

    limit = max(GOLDEN_FIT_SLACK * entry["fit_error"], GOLDEN_FIT_FLOOR)
    assert result["error"] <= limit, (
        f"{case.name}: fit error {result['error']:.3e} > {limit:.3e}")

    _assert_golden(
        result["fitted"], archive[f"happ_{case.name}"],
        f"{case.name} fitted response")
    _assert_golden(
        result["model_response"], archive[f"model_response_{case.name}"],
        f"{case.name} model response")

    if entry["is_passive"]:
        assert result["compensated_response"] is None
    else:
        _assert_golden(
            result["compensated_response"],
            archive[f"compensated_response_{case.name}"],
            f"{case.name} compensated response")
