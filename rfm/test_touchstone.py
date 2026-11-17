"""TouchStone import/export tests."""

from __future__ import annotations

import os

import numpy as np
import pytest

from idempy import DUM, DUM2TouchStone, TouchStone2DUM


def _dum_2port():
    f = np.array([0.0, 1e9, 2e9])
    S = np.zeros((3, 2, 2), dtype=complex)
    S[:, 0, 0] = [0.1, 0.2 + 0.1j, 0.3 - 0.2j]
    S[:, 1, 0] = [0.01, 0.02j, 0.03]
    S[:, 0, 1] = [0.04, 0.05, 0.06 - 0.01j]
    S[:, 1, 1] = [-0.2, -0.3 + 0.2j, -0.4]
    return DUM(Nports=2, pindex=np.ones((2, 2)), isfreq=True, R0=50.0,
               f=f, S=S)


def test_touchstone_roundtrip_ri(tmp_path):
    dum = _dum_2port()
    DUM2TouchStone(str(tmp_path), "device.s2p", dum)
    path = tmp_path / "device.s2p"
    assert path.is_file()
    restored = TouchStone2DUM(str(tmp_path), "device.s2p")
    assert restored.Nports == 2
    assert np.allclose(restored.f, dum.f)
    assert restored.R0 == pytest.approx(50.0)
    assert np.allclose(restored.S, dum.S, atol=1e-12)


def test_touchstone_extension_is_added(tmp_path):
    dum = _dum_2port()
    DUM2TouchStone(str(tmp_path), "bare", dum)
    assert (tmp_path / "bare.s2p").is_file()
    restored = TouchStone2DUM(str(tmp_path), "bare.s2p")
    assert restored.Nports == 2


def test_touchstone_nan_entries_survive_roundtrip(tmp_path):
    dum = _dum_2port()
    dum.pindex[1, 0] = 0
    dum.S[:, 1, 0] = np.nan
    DUM2TouchStone(str(tmp_path), "partial.s2p", dum)
    restored = TouchStone2DUM(str(tmp_path), "partial.s2p")
    assert np.all(np.isnan(restored.S[:, 1, 0]))
    assert np.allclose(restored.S[:, 0, 0], dum.S[:, 0, 0])


def test_touchstone_ma_and_db_formats(tmp_path):
    f = [1e9]
    # One-port 0.5 angle 90 deg in MA; 20*log10(0.25) dB, 0 deg in DB.
    ma = tmp_path / "ma.s1p"
    ma.write_text(
        "! comment\n# GHz S MA R 50\n1.0 0.5 90\n", encoding="utf-8"
    )
    dum_ma = TouchStone2DUM(str(tmp_path), "ma.s1p")
    assert dum_ma.R0 == pytest.approx(50.0)
    assert dum_ma.S[0, 0, 0] == pytest.approx(0.5j, abs=1e-12)
    assert dum_ma.f[0] == pytest.approx(1e9)

    db = tmp_path / "db.s1p"
    db.write_text("# Hz S DB R 75\n1e9 -12.041199826559248 0\n",
                  encoding="utf-8")
    dum_db = TouchStone2DUM(str(tmp_path), "db.s1p")
    assert dum_db.R0 == pytest.approx(75.0)
    assert dum_db.S[0, 0, 0] == pytest.approx(0.25, abs=1e-9)


def test_touchstone_rejects_bad_extension_and_formats(tmp_path):
    (tmp_path / "bad.txt").write_text("# Hz S RI\n1 1 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        TouchStone2DUM(str(tmp_path), "bad.txt")

    (tmp_path / "z.s1p").write_text("# Hz Z RI R 50\n1 1 0\n",
                                    encoding="utf-8")
    with pytest.raises(NotImplementedError):
        TouchStone2DUM(str(tmp_path), "z.s1p")


def test_touchstone_y_parameters_are_converted(tmp_path):
    r0 = 50.0
    y11 = 0.01 + 0.005j
    path = tmp_path / "y.s1p"
    path.write_text(
        f"# Hz Y RI R {r0}\n1e9 {y11.real} {y11.imag}\n", encoding="utf-8"
    )
    dum = TouchStone2DUM(str(tmp_path), "y.s1p")
    expected = (1.0 - r0 * y11) / (1.0 + r0 * y11)
    assert dum.S[0, 0, 0] == pytest.approx(expected, rel=1e-12)
    assert dum.R0 == pytest.approx(r0)


def test_touchstone_multiline_records(tmp_path):
    # Four S entries on separate lines per frequency point.
    content = (
        "# GHz S RI R 50\n"
        "1.0 0.1 0.0\n0.2 0.0\n0.3 0.0\n0.4 0.0\n"
    )
    (tmp_path / "wrapped.s2p").write_text(content, encoding="utf-8")
    dum = TouchStone2DUM(str(tmp_path), "wrapped.s2p")
    assert dum.S.shape == (1, 2, 2)
    assert dum.S[0, 0, 0] == pytest.approx(0.1)
    assert dum.S[0, 1, 0] == pytest.approx(0.2)
    assert dum.S[0, 0, 1] == pytest.approx(0.3)
    assert dum.S[0, 1, 1] == pytest.approx(0.4)
