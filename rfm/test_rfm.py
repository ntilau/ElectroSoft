"""Tests for the MOD2RFM rational-function-model exporter."""

from __future__ import annotations

import numpy as np
import pytest

from idempy import MOD, MOD2RFM


def _parse_rfm(path):
    """Parse an RFM text file into header dict and named sections.

    Returns ``(header, sections)`` where ``sections`` maps ``"D"`` to the
    direct-feedthrough entries and each pole label to its residue entries,
    all as lists of ``(row, col, value)`` with one-based indices.
    """
    header = {}
    sections = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("%"):
            row, col, value = line.split(",")
            sections[current].append((int(row), int(col), complex(value)))
            continue
        body = line[1:].strip()
        if not body:
            current = None
            continue
        if body.startswith("Matrix "):
            current = body.split()[-1]
            sections[current] = []
            continue
        if ":" in body:
            key, value = body.split(":", 1)
            header[key.strip()] = value.strip()
    return header, sections


def _rebuild(entries, shape):
    matrix = np.zeros(shape, dtype=complex)
    for row, col, value in entries:
        matrix[row - 1, col - 1] = value
    return matrix


def _extract_poles(sections):
    """Return ordered lists of (pole, residue) for real and complex sections."""
    real, complex_ = [], []
    for label in ("Real poles/residues", "Complex poles/residues (positive-imaginary members)"):
        pass
    # sections keys are matrix names only; poles live in the raw text, so we
    # instead reconstruct pole order from section order by re-parsing here.
    return real, complex_


def _parse_rfm_with_poles(path):
    """Parse header, D matrix, and ordered (pole, residue) lists."""
    header = {}
    d_entries = []
    real_poles, complex_poles = [], []
    current_pole = None
    current_section = None  # None, "D", "real", "complex"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("%"):
            row, col, value = line.split(",")
            entry = (int(row), int(col), complex(value))
            if current_section == "D":
                d_entries.append(entry)
            elif current_section == "real" and current_pole is not None:
                real_poles[-1][1].append(entry)
            elif current_section == "complex" and current_pole is not None:
                complex_poles[-1][1].append(entry)
            continue
        body = line[1:].strip()
        if not body:
            current_section = None
            current_pole = None
            continue
        if body.startswith("Matrix "):
            current_section = "D"
            current_pole = None
            continue
        if body.startswith("pole:"):
            pole = complex(body.split(":", 1)[1].strip())
            if current_section == "real":
                real_poles.append([pole, []])
            else:
                complex_poles.append([pole, []])
            current_pole = pole
            continue
        if body.startswith("Real poles"):
            current_section = "real"
            current_pole = None
            continue
        if body.startswith("Complex poles"):
            current_section = "complex"
            current_pole = None
            continue
        if ":" in body:
            key, value = body.split(":", 1)
            header[key.strip()] = value.strip()
    return header, d_entries, real_poles, complex_poles


def test_mod2rfm_roundtrip_poles_residues(tmp_path):
    mod = MOD(
        isABCD=False, isResPoles=True, issparse=False, R0=50.0,
        name="test_model", units="Hz",
        D=np.array([[1.0 + 0j, 0.1 + 0j], [0.1 + 0j, 1.0 + 0j]]),
        RealPoles=np.array([-100.0, -200.0]),
        ComplexPoles=np.array([-300.0 + 400j]),
        RealResidues=[
            np.array([[0.5 + 0j, 0.0 + 0j], [0.0 + 0j, 0.5 + 0j]]),
            np.array([[0.2 + 0j, 0.0 + 0j], [0.0 + 0j, 0.2 + 0j]]),
        ],
        ComplexResidues=[
            np.array([[0.3 + 0.1j, 0.05 + 0j], [0.05 + 0j, 0.3 - 0.1j]]),
        ],
    )
    status = MOD2RFM(mod, str(tmp_path), "model.rfm")
    assert status == ""
    path = tmp_path / "model.rfm"
    assert path.is_file()

    header, d_entries, real_poles, complex_poles = _parse_rfm_with_poles(path)
    assert header["Model name"] == "test_model"
    assert header["Representation"] == "S"
    assert header["Nports"] == "2"
    assert float(header["R0"]) == 50.0
    assert int(header["Order"]) == 4

    D = _rebuild(d_entries, (2, 2))
    assert np.allclose(D, mod.D, rtol=1e-12, atol=1e-12)

    assert [p for p, _ in real_poles] == [-100.0, -200.0]
    for (_, res), expected in zip(real_poles, mod.RealResidues):
        assert np.allclose(_rebuild(res, (2, 2)), expected, rtol=1e-12, atol=1e-12)

    assert [p for p, _ in complex_poles] == [complex(-300.0, 400.0)]
    for (_, res), expected in zip(complex_poles, mod.ComplexResidues):
        assert np.allclose(_rebuild(res, (2, 2)), expected, rtol=1e-12, atol=1e-12)


def test_mod2rfm_abcd_only_model(tmp_path):
    A = np.array([[-100.0, 0.0], [0.0, -200.0]])
    B = np.eye(2)
    C = np.eye(2) * 0.5
    D = np.eye(2)
    mod = MOD(
        isABCD=True, isResPoles=False, issparse=False, R0=50.0,
        name="abcd_only", units="Hz", D=D, A=A, B=B, C=C,
    )
    status = MOD2RFM(mod, str(tmp_path), "abcd.rfm")
    assert status == ""
    header, d_entries, real_poles, complex_poles = _parse_rfm_with_poles(
        tmp_path / "abcd.rfm")
    assert header["Model name"] == "abcd_only"
    assert int(header["Order"]) == 2
    assert [p for p, _ in real_poles] == [-100.0, -200.0]
    assert complex_poles == []


def test_mod2rfm_admittance_representation_y(tmp_path):
    mod = MOD(
        isABCD=False, isResPoles=True, issparse=False, R0=0.0,
        name="adm_model", units="Hz",
        D=np.zeros((1, 1), dtype=complex),
        RealPoles=np.array([-50.0]),
        ComplexPoles=np.zeros(0, dtype=complex),
        RealResidues=[np.array([[1.0 + 0j]])],
        ComplexResidues=[],
    )
    status = MOD2RFM(mod, str(tmp_path), "adm.rfm")
    assert status == ""
    header, _, _, _ = _parse_rfm_with_poles(tmp_path / "adm.rfm")
    assert header["Representation"] == "Y"
    assert float(header["R0"]) == 0.0


def test_mod2rfm_unnamed_model(tmp_path):
    mod = MOD(
        isABCD=False, isResPoles=True, issparse=False, R0=50.0,
        name="", units="",
        D=np.zeros((1, 1), dtype=complex),
        RealPoles=np.zeros(0),
        ComplexPoles=np.zeros(0, dtype=complex),
        RealResidues=[],
        ComplexResidues=[],
    )
    status = MOD2RFM(mod, str(tmp_path), "unnamed.rfm")
    assert status == ""
    header, _, _, _ = _parse_rfm_with_poles(tmp_path / "unnamed.rfm")
    assert header["Model name"] == "unnamed"


def test_mod2rfm_returns_error_for_incomplete_model(tmp_path):
    broken = MOD()
    status = MOD2RFM(broken, str(tmp_path), "broken.rfm")
    assert status
    assert not (tmp_path / "broken.rfm").exists()
