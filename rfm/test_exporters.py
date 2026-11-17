"""Tests for SPICE/ASTAP/ASCII export and generated test decks."""

from __future__ import annotations

import os

import numpy as np
import pytest

from idempy import (
    DUM,
    MOD,
    MOD2Ascii,
    MOD2Astap,
    MOD2Cir,
    MODFill,
    RandomPassiveDUM,
    TestMODAstap_FD,
    TestMODAstap_TD,
    TestMODCir_FD,
    TestMODCir_TD,
)


def _read(path):
    with open(path, encoding="utf-8") as stream:
        return stream.read()


# ---------------------------------------------------------------------------
# Equivalent-circuit export
# ---------------------------------------------------------------------------


def test_mod2astap_writes_subcircuit(tmp_path):
    _, mod = RandomPassiveDUM(type="frequency", Nports=2, Order=6,
                              Bandwidth=5e8, seed=1)
    MOD2Astap(mod, str(tmp_path), "astapmod")
    path = tmp_path / "astapmod"
    assert path.is_file()
    text = _read(path)
    lines = text.splitlines()
    assert lines[0] == ".subckt astapmod p1 n1 p2 n2"
    assert lines[-1] == ".ends astapmod"
    assert "scattering" in text
    for letter in ("C", "R", "G", "E", "H", "V"):
        assert f"\n{letter}" in text
    assert "nan" not in text.lower()
    # The reference resistance appears in the port interface.
    assert "RS1" in text or "R0" in text


def test_mod2astap_ground_references_and_validation(tmp_path):
    _, admittance = RandomPassiveDUM(type="frequency", Nports=2, Order=6,
                                     Bandwidth=5e8, Mismatch=0, seed=1)
    assert admittance.R0 == 0.0
    MOD2Astap(admittance, str(tmp_path), "adm", {"GroundReferences": 1})
    text = _read(tmp_path / "adm")
    assert text.splitlines()[0] == ".subckt adm p1 p2 0"
    assert "All ports share the common reference node 0" in text
    with pytest.raises(ValueError):
        MOD2Astap(admittance, str(tmp_path), "bad",
                  {"GroundReferences": 2})
    with pytest.raises(ValueError):
        MOD2Astap(admittance, str(tmp_path), "")
    with pytest.raises(ValueError):
        MOD2Astap(admittance, str(tmp_path), "two words")


def test_mod2astap_rejects_invalid_model_data(tmp_path):
    model = MOD(isABCD=True, issparse=False, R0=-1.0,
                D=np.zeros((1, 1)), A=np.zeros((1, 1)),
                B=np.zeros((1, 1)), C=np.zeros((1, 1)))
    with pytest.raises(ValueError):
        MOD2Astap(model, str(tmp_path), "bad")
    cap = MOD(isABCD=True, issparse=False, R0=50.0, Cap=-1.0,
              D=np.zeros((1, 1)), A=np.zeros((1, 1)),
              B=np.zeros((1, 1)), C=np.zeros((1, 1)))
    with pytest.raises(ValueError):
        MOD2Astap(cap, str(tmp_path), "bad")


def test_mod2astap_accepts_poles_residues_only(tmp_path):
    _, full = RandomPassiveDUM(type="frequency", Nports=2, Order=6,
                               Bandwidth=5e8, seed=1)
    residues = MOD(
        isResPoles=True, issparse=False, R0=full.R0,
        RealPoles=np.array(full.RealPoles),
        ComplexPoles=np.array(full.ComplexPoles),
        RealResidues=[np.array(R) for R in full.RealResidues],
        ComplexResidues=[np.array(R) for R in full.ComplexResidues],
        D=np.array(full.D), Cap=full.Cap,
    )
    MOD2Astap(residues, str(tmp_path), "resmod")
    assert "scattering" in _read(tmp_path / "resmod")


def test_mod2cir_extension_and_resistor_types(tmp_path):
    _, admittance = RandomPassiveDUM(type="frequency", Nports=2, Order=6,
                                     Bandwidth=5e8, Mismatch=0, seed=1)
    MOD2Cir(admittance, str(tmp_path), "model")
    assert (tmp_path / "model.cir").is_file()
    MOD2Cir(admittance, str(tmp_path), "named.cir")
    assert (tmp_path / "named.cir").is_file()
    assert not (tmp_path / "named.cir.cir").exists()

    for resistor_type in (1, 2, 3, 4):
        MOD2Cir(admittance, str(tmp_path), f"r{resistor_type}",
                {"ResistorType": resistor_type})
        text = _read(tmp_path / f"r{resistor_type}.cir")
        assert ".subckt" in text and ".ends" in text
    assert "isnoisy=0" in _read(tmp_path / "r2.cir")
    with pytest.raises(ValueError):
        MOD2Cir(admittance, str(tmp_path), "bad", {"ResistorType": 5})
    with pytest.raises(ValueError):
        MOD2Cir(admittance, str(tmp_path), "bad", {"GroundReferences": -1})


# ---------------------------------------------------------------------------
# ASCII state-space dump
# ---------------------------------------------------------------------------


def _parse_ascii(path):
    sections = {}
    current = None
    header = {}
    for line in _read(path).splitlines():
        if line.startswith("%"):
            if line.startswith("% Matrix "):
                current = line.split()[-1]
                sections[current] = []
            elif ":" in line:
                key, value = line[1:].split(":", 1)
                header[key.strip()] = value.strip()
            continue
        if not line.strip():
            continue
        row, col, value = line.split(",")
        sections[current].append((int(row), int(col), complex(value)))
    return header, sections


def test_mod2ascii_roundtrips_matrices(frequency_mod, tmp_path):
    status = MOD2Ascii(frequency_mod, str(tmp_path), "model.txt")
    assert status == ""
    header, sections = _parse_ascii(tmp_path / "model.txt")
    assert header["Model name"] == "RandomPassiveDUM"
    assert header["Representation"] == "S"
    for name in ("A", "B", "C", "D"):
        matrix = np.asarray(getattr(frequency_mod, name))
        entries = sections[name]
        rebuilt = np.zeros(matrix.shape, dtype=complex)
        for row, col, value in entries:
            rebuilt[row - 1, col - 1] = value
        assert np.allclose(rebuilt, matrix, rtol=1e-12, atol=1e-12)
        # One-based, column-major listing: first entry is the first nonzero
        # of the first column.
        first_row = int(np.flatnonzero(matrix[:, 0])[0]) + 1
        assert entries[0][0] == first_row and entries[0][1] == 1


def test_mod2ascii_admittance_and_unnamed(tmp_path):
    _, admittance = RandomPassiveDUM(type="frequency", Nports=2, Order=4,
                                     Bandwidth=1e8, Mismatch=0, seed=2)
    admittance.name = ""
    status = MOD2Ascii(admittance, str(tmp_path), "adm.txt")
    assert status == ""
    header, sections = _parse_ascii(tmp_path / "adm.txt")
    assert header["Model name"] == "unnamed"
    assert header["Representation"] == "Y"
    assert set(sections) == {"A", "B", "C", "D"}


def test_mod2ascii_returns_error_for_incomplete_model(tmp_path):
    broken = MOD()
    status = MOD2Ascii(broken, str(tmp_path), "broken.txt")
    assert status
    assert not (tmp_path / "broken.txt").exists()


# ---------------------------------------------------------------------------
# Generated deck sanity
# ---------------------------------------------------------------------------


def test_testmodcir_decks(tmp_path):
    directory = str(tmp_path)
    path = TestMODCir_FD("PSPICE", directory, "fd", 2, 50.0, "model",
                         1e6, 1e9, 11, 1)
    assert path == os.path.join(directory, "fd.cir")
    text = _read(path)
    assert ".ac lin 11" in text
    assert "VS1" in text and "RS1" in text
    assert '.include "model.cir"' in text

    path = TestMODCir_FD("PSPICE", directory, "fdy", 2, 0.0, "model",
                         1e6, 1e9, 11, 1)
    text = _read(path)
    assert "VEXC1" in text and "VS2" in text

    path = TestMODCir_TD("PSPICE", directory, "td", 2, 50.0, "model",
                         "drive", 1e-10, 1e-8, 1)
    text = _read(path)
    assert ".tran " in text and '.include "drive.lib"' in text
    assert "XSRC" in text

    with pytest.raises(ValueError):
        TestMODCir_FD("PSPICE", directory, "bad", 2, 50.0, "model",
                      -1.0, 1e9, 11, 1)
    with pytest.raises(ValueError):
        TestMODCir_FD("PSPICE", directory, "bad", 2, 50.0, "model",
                      1e9, 1e6, 11, 1)
    with pytest.raises(ValueError):
        TestMODCir_FD("PSPICE", directory, "bad", 2, 50.0, "model",
                      1e6, 1e9, 0, 1)
    with pytest.raises(ValueError):
        TestMODCir_FD("PSPICE", directory, "bad", 2, 50.0, "model",
                      1e6, 1e9, 11, 3)
    with pytest.raises(ValueError):
        TestMODCir_TD("PSPICE", directory, "bad", 2, 50.0, "model",
                      "drive", 0.0, 1e-8, 1)
    with pytest.raises(ValueError):
        TestMODCir_TD("PSPICE", directory, "bad", 2, 50.0, "model",
                      "drive", 1e-10, -1.0, 1)


def test_testmodastap_decks(tmp_path):
    directory = str(tmp_path)
    path = TestMODAstap_FD(directory, "fd", 2, 50.0, "model",
                           1e6, 1e9, 11, 1)
    text = _read(path)
    assert ".ac lin 11" in text
    assert ".print ac V(p1,n1)" in text
    assert '.include "model.cir"' in text

    path = TestMODAstap_TD(directory, "td", 2, 50.0, "model", "drive",
                           1e-10, 1e-8, 2)
    text = _read(path)
    assert ".tran " in text and "XSRC" in text

    with pytest.raises(ValueError):
        TestMODAstap_FD(directory, "bad", 2, 50.0, "model",
                        1e6, 1e9, -1, 1)
    with pytest.raises(ValueError):
        TestMODAstap_TD(directory, "bad", 2, 50.0, "model", "drive",
                        1e-10, 0.0, 1)
