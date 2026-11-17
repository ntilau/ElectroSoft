"""Tests for CST, FDTD, CSDF/CSV and SPICE-deck import helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from idempy import (
    AstapMOD2DUM,
    CST2DUM,
    CSTRead,
    ExtractSubDUM,
    Fastap_xt2subckt,
    FDTD2DUM,
    Fspice_xt2subckt,
    ImportCSDF,
    ImportCSV,
    RandomPassiveDUM,
    SpiceMOD2DUM,
)
from idempy import DUM
from conftest import write_port_tables


# ---------------------------------------------------------------------------
# CST
# ---------------------------------------------------------------------------


def test_cstread_returns_time_and_incident_wave(cst_project):
    directory, root = cst_project
    path = os.path.join(directory, f"{root}_IN01",
                        f"{root}_IN01_PORTS.SIG")
    t, y = CSTRead(path)
    expected_t = np.linspace(0.0, 4e-9, 201)
    assert np.allclose(t, expected_t)
    expected_a = np.exp(-((t - 0.5e-9) / 0.15e-9) ** 2)
    assert np.allclose(y, expected_a)
    for version in (3, 4, 5):
        t_v, y_v = CSTRead(path, CSTversion=version)
        assert np.allclose(t_v, t) and np.allclose(y_v, y)
    with pytest.raises(ValueError):
        CSTRead(path, CSTversion=6)


def test_cst2dum_canonical_layout(cst_project):
    directory, root = cst_project
    dum = CST2DUM(os.path.join(directory, root), 2, 50.0)
    assert dum.istime and not dum.isfreq
    assert dum.Nports == 2
    assert dum.R0 == 50.0
    assert np.array_equal(np.asarray(dum.pindex), np.ones((2, 2)))
    assert len(dum.t) == 201
    assert dum.Ts == pytest.approx(4e-9 / 200)
    assert len(dum.a) == 4 and len(dum.b) == 4
    t = np.linspace(0.0, 4e-9, 201)
    for j in range(2):
        delay = 0.5e-9 + 0.1e-9 * j
        a_expected = np.exp(-((t - delay) / 0.15e-9) ** 2)
        for i in range(2):
            k = j * 2 + i  # column-major (output, input) order
            assert np.allclose(dum.a[k], a_expected)
            amplitude = 0.4 if i == j else 0.2 / (1 + abs(i - j))
            b_expected = amplitude * np.exp(
                -((t - delay - 0.05e-9 * (i + 1)) / (0.15e-9 * 1.4)) ** 2
            )
            assert np.allclose(dum.b[k], b_expected)


def test_cst2dum_reciprocal_and_pindex(cst_project):
    directory, root = cst_project
    base = os.path.join(directory, root)
    reciprocal = CST2DUM(base, 2, 50.0, isreciprocal=1)
    # Exactly one of the two off-diagonal responses is retained.
    assert np.array_equal(
        np.asarray(reciprocal.pindex),
        np.array([[1, 1], [0, 1]]),
    )
    assert len(reciprocal.a) == 3
    partial = CST2DUM(base, 2, 50.0, pindex=np.array([[1, 0], [0, 1]]))
    assert np.array_equal(
        np.asarray(partial.pindex), np.array([[1, 0], [0, 1]])
    )
    assert len(partial.a) == 2


def test_cst2dum_layout_and_validation(cst_project):
    directory, root = cst_project
    base = os.path.join(directory, root)
    explicit = CST2DUM(base, 2, 50.0, layout="source_responses")
    assert np.array_equal(np.asarray(explicit.pindex), np.ones((2, 2)))
    with pytest.raises(ValueError):
        CST2DUM(base, 2, 50.0, layout="voltage_current")
    with pytest.raises(ValueError):
        CST2DUM(base, 0, 50.0)
    with pytest.raises(ValueError):
        CST2DUM(base, 2, -1.0)
    with pytest.raises(ValueError):
        CST2DUM(base, 2, 50.0, pindex=np.ones((3, 3)))
    with pytest.raises(ValueError):
        CST2DUM(base, 2, 50.0, layout="nonsense")
    with pytest.raises(FileNotFoundError):
        CST2DUM(os.path.join(directory, "missing"), 2, 50.0)


def test_cst2dum_admittance_reference(cst_project):
    directory, root = cst_project
    dum = CST2DUM(os.path.join(directory, root), 2, 0.0)
    assert dum.R0 == 0.0
    assert dum.istime and len(dum.b) == 4


# ---------------------------------------------------------------------------
# FDTD
# ---------------------------------------------------------------------------


def test_fdtd2dum_canonical_layout(fdtd_project):
    directory, root = fdtd_project
    dum = FDTD2DUM(directory, root, 2, 50.0)
    assert dum.istime and not dum.isfreq
    assert dum.Nports == 2
    assert np.array_equal(np.asarray(dum.pindex), np.ones((2, 2)))
    assert len(dum.t) == 201
    assert dum.Ts == pytest.approx(4e-9 / 200)
    assert dum.name == root
    t = np.linspace(0.0, 4e-9, 201)
    expected_a = np.exp(-((t - 0.6e-9) / 0.15e-9) ** 2)
    assert np.allclose(dum.a[2], expected_a)  # excitation at port 2
    assert not np.allclose(dum.b[0], dum.b[3])


def test_fdtd2dum_padding_of_short_runs(tmp_path):
    t_long = np.linspace(0.0, 4e-9, 201)
    t_short = np.linspace(0.0, 2e-9, 101)
    folder_1 = tmp_path / "run_IN01"
    folder_2 = tmp_path / "run_IN02"
    folder_1.mkdir()
    folder_2.mkdir()
    a1 = np.exp(-((t_long - 0.5e-9) / 0.2e-9) ** 2)
    a2 = np.exp(-((t_short - 0.5e-9) / 0.2e-9) ** 2)
    np.savetxt(folder_1 / "run_IN01_PORTS.OUT",
               np.column_stack([t_long, a1, 0.5 * a1, 0.2 * a1]),
               delimiter=" ")
    np.savetxt(folder_2 / "run_IN02_PORTS.OUT",
               np.column_stack([t_short, a2, 0.5 * a2, 0.2 * a2]),
               delimiter=" ")
    dum = FDTD2DUM(str(tmp_path), "run", 2, 50.0)
    assert len(dum.t) == 201
    assert dum.Ts == pytest.approx(4e-9 / 200)
    for k in (0, 1):  # pairs (0, 0) and (1, 0) from the long run
        assert np.allclose(dum.a[k], a1)
    for k in (2, 3):  # pairs (0, 1) and (1, 1) from the short run
        assert dum.a[k].shape == dum.t.shape
        assert np.allclose(dum.a[k][:101], a2)
        assert np.all(dum.a[k][101:] == 0.0)


def test_fdtd2dum_errors(fdtd_project):
    directory, root = fdtd_project
    with pytest.raises(FileNotFoundError):
        FDTD2DUM(directory, "absent", 1, 50.0)
    with pytest.raises(ValueError):
        FDTD2DUM(directory, root, 2, 50.0, pindex=np.ones((2, 3)))
    with pytest.raises(ValueError):
        FDTD2DUM(directory, root, 2, 50.0, layout="unknown")
    with pytest.raises(ValueError):
        FDTD2DUM(directory, root, 2, 50.0, layout="voltage_current")


# ---------------------------------------------------------------------------
# CSDF / CSV
# ---------------------------------------------------------------------------

_CSDF = """Title: transient run
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


def test_import_csdf(tmp_path):
    path = tmp_path / "result.csd"
    path.write_text(_CSDF)
    x, y, names = ImportCSDF(str(path))
    assert np.allclose(x, [0.0, 1e-9, 2e-9, 3e-9])
    assert y.shape == (4, 2)
    assert np.allclose(y[:, 0], [0.0, 0.1, 0.2, 0.3])
    assert np.allclose(y[:, 1], [0.0, 0.2, 0.4, 0.6])
    assert names == ["V(out)", "I(vs)"]


def test_import_csdf_without_values_raises(tmp_path):
    path = tmp_path / "empty.csd"
    path.write_text("Title: nothing\nNo. Variables: 1\n")
    with pytest.raises(ValueError):
        ImportCSDF(str(path))


def test_import_csv_with_header(tmp_path):
    path = tmp_path / "result.csv"
    path.write_text("Title: test\ntime,V(out),I(vs)\n"
                    "0,0,0\n1e-9,0.1,0.2\n2e-9,0.2,0.4\n")
    x, y, names = ImportCSV(str(path))
    assert np.allclose(x, [0.0, 1e-9, 2e-9])
    assert y.shape == (3, 2)
    assert np.allclose(y[:, 1], [0.0, 0.2, 0.4])
    assert names == ["V(out)", "I(vs)"]


def test_import_csv_without_header(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("0,1,2\n1,3,4\n2,5,6\n")
    x, y, names = ImportCSV(str(path))
    assert np.allclose(x, [0.0, 1.0, 2.0])
    assert y.shape == (3, 2)
    assert names == ["var2", "var3"]


def test_import_csv_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Title: nothing\n")
    with pytest.raises(ValueError):
        ImportCSV(str(path))


# ---------------------------------------------------------------------------
# PWL subcircuits
# ---------------------------------------------------------------------------


def test_fspice_pwl_subcircuit(tmp_path):
    t = np.array([0.0, 1e-9, 2e-9])
    x = np.array([0.0, 1.0, 0.0])
    path = Fspice_xt2subckt(t, x, str(tmp_path), "drive")
    assert path == str(tmp_path / "drive.lib")
    text = open(path, encoding="utf-8").read().splitlines()
    assert text[0] == ".subckt drive P N"
    assert text[1] == "VX P N PWL("
    assert text[2].startswith("+ ") and "," in text[2]
    assert text[-1] == ".ends"
    assert text[-2] == "+)"


def test_fastap_pwl_subcircuit_and_name_sanitizing(tmp_path):
    path = Fastap_xt2subckt([0.0, 1.0], [0.0, 1.0], str(tmp_path), "1drive")
    assert os.path.basename(path) == "X1drive.lib"
    text = open(path, encoding="utf-8").read()
    assert ".subckt X1drive P N" in text
    assert "VX P N PWL(" in text
    assert "+ 0 0" in text and "+ 1 1" in text


def test_pwl_subcircuit_validation(tmp_path):
    with pytest.raises(ValueError):
        Fspice_xt2subckt([0.0, 1.0], [0.0], str(tmp_path), "bad")
    with pytest.raises(ValueError):
        Fspice_xt2subckt([1.0, 0.0], [0.0, 1.0], str(tmp_path), "bad")
    with pytest.raises(ValueError):
        Fspice_xt2subckt([], [], str(tmp_path), "bad")
    with pytest.raises(ValueError):
        Fspice_xt2subckt([0.0, 1.0], [0.0, np.nan], str(tmp_path), "bad")


# ---------------------------------------------------------------------------
# Deck generation and result loading
# ---------------------------------------------------------------------------


def _write_voltage_result(path, t, voltages, names=None):
    names = names or [f"V(p{i + 1})" for i in range(voltages.shape[1])]
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("time," + ",".join(names) + "\n")
        for row in np.column_stack([t, voltages]):
            stream.write(",".join(f"{value:.17g}" for value in row) + "\n")


def test_spice_mod2dum_norun_and_load(tmp_path):
    dum, _ = RandomPassiveDUM(type="time", Nports=2, Order=6,
                              Bandwidth=5e8, seed=1)
    directory = str(tmp_path)
    assert SpiceMOD2DUM(directory, "model", dum, "norun") == ""
    decks = sorted(name for name in os.listdir(directory)
                   if name.endswith(".cir"))
    libraries = sorted(name for name in os.listdir(directory)
                       if name.endswith(".lib"))
    assert decks == ["model_test_p01.cir", "model_test_p02.cir"]
    assert libraries == ["model_test_p01_src.lib", "model_test_p02_src.lib"]
    text = open(os.path.join(directory, "model_test_p01.cir"),
                encoding="utf-8").read()
    assert '.include "model.cir"' in text
    assert ".tran " in text
    assert "V(p1,n1)" in text

    pairs = [(i, j) for j in range(2) for i in range(2)
             if dum.pindex[i, j]]
    r0 = dum.R0
    for j in range(2):
        a = next(np.asarray(dum.a[k]) for k, (_, jj) in enumerate(pairs)
                 if jj == j)
        voltages = np.column_stack([
            np.sqrt(r0) * (0.1 * (i + 1) * a + (a if i == j else 0.0))
            for i in range(2)
        ])
        _write_voltage_result(
            os.path.join(directory, f"model_test_p{j + 1:02d}.csv"),
            dum.t, voltages,
        )
    loaded = SpiceMOD2DUM(directory, "model", dum, "load")
    assert loaded.istime and not loaded.isfreq
    assert len(loaded.b) == len(pairs)
    for k, (i, j) in enumerate(pairs):
        assert np.allclose(loaded.b[k], 0.1 * (i + 1) * np.asarray(dum.a[k]),
                           atol=1e-12)


def test_spice_mod2dum_admittance_load(tmp_path):
    dum, _ = RandomPassiveDUM(type="time", Nports=2, Order=6,
                              Bandwidth=5e8, seed=1)
    admittance = ExtractSubDUM(dum, [0])
    admittance.R0 = 0.0
    directory = str(tmp_path)
    SpiceMOD2DUM(directory, "adm", admittance, "norun")
    a = np.asarray(admittance.a[0])
    current = 3.0 * a
    _write_voltage_result(os.path.join(directory, "adm_test_p01.csv"),
                          admittance.t, current.reshape(-1, 1),
                          names=["I(vs1)"])
    loaded = SpiceMOD2DUM(directory, "adm", admittance, "load")
    assert loaded.R0 == 0.0
    assert np.allclose(loaded.b[0], current, atol=1e-12)


def test_spice_mod2dum_mode_validation(tmp_path):
    dum = DUM(Nports=1, pindex=np.ones((1, 1)), istime=True,
              R0=50.0, t=np.array([0.0, 1.0]),
              a=[np.array([1.0, 0.0])], b=[np.array([0.0, 1.0])], Ts=1.0)
    with pytest.raises(ValueError):
        SpiceMOD2DUM(str(tmp_path), "x", dum, "invalid")
    with pytest.raises(FileNotFoundError):
        SpiceMOD2DUM(str(tmp_path / "absent"), "x", dum, "norun")
    with pytest.raises(FileNotFoundError):
        SpiceMOD2DUM(str(tmp_path), "x", dum, "load")


def test_astap_mod2dum_norun(tmp_path):
    dum, _ = RandomPassiveDUM(type="frequency", Nports=2, Order=6,
                              Bandwidth=5e8, seed=1)
    directory = str(tmp_path)
    assert AstapMOD2DUM(directory, "model", dum, "norun") == ""
    decks = sorted(name for name in os.listdir(directory)
                   if name.endswith(".cir"))
    assert decks == ["model_astap_p01.cir", "model_astap_p02.cir"]
    text = open(os.path.join(directory, "model_astap_p01.cir"),
                encoding="utf-8").read()
    assert ".ac lin" in text and '.include "model.cir"' in text
    with pytest.raises(ValueError):
        AstapMOD2DUM(directory, "model", dum, "fullrun")
    with pytest.raises(FileNotFoundError):
        AstapMOD2DUM(directory, "model", dum, "load")
