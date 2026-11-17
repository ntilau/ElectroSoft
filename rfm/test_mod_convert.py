"""State-space / poles-residues conversion and model evaluation tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from idempy import (
    ConvertDUMUnits,
    ConvertMODUnits,
    DeNormalizeMOD,
    DUM,
    MOD,
    MOD2DUM,
    MODFill,
    NormalizeMOD,
    RandomPassiveDUM,
    ResPoles2ABC,
    SS2ResPoles,
)


def _state_only(mod):
    return MOD(isABCD=True, issparse=False, R0=mod.R0,
               D=np.array(mod.D), A=np.array(mod.A), B=np.array(mod.B),
               C=np.array(mod.C))


def _residues_only(mod):
    return MOD(
        isResPoles=True, issparse=False, R0=mod.R0, D=np.array(mod.D),
        RealPoles=np.array(mod.RealPoles),
        ComplexPoles=np.array(mod.ComplexPoles),
        RealResidues=[np.array(R) for R in mod.RealResidues],
        ComplexResidues=[np.array(R) for R in mod.ComplexResidues],
    )


def _transfer_residues(mod, s):
    out = np.asarray(mod.D, dtype=complex).copy()
    for pole, residue in zip(mod.RealPoles, mod.RealResidues):
        out += np.asarray(residue) / (s - pole)
    for pole, residue in zip(mod.ComplexPoles, mod.ComplexResidues):
        out += np.asarray(residue) / (s - pole)
        out += np.conj(np.asarray(residue)) / (s - np.conj(pole))
    return out


def test_ss2respoles_to_gilbert_roundtrip():
    rng = np.random.default_rng(7)
    A = np.diag([-1.0, -2.0, -0.5])
    B = rng.standard_normal((3, 2))
    C = rng.standard_normal((2, 3))
    poles, residues = SS2ResPoles(A, B, C)
    assert poles.shape == (3,)
    assert len(residues) == 3
    rp = np.array([-1.0, -2.0, -0.5])
    Rr = []
    for pole in rp:
        index = int(np.argmin(np.abs(poles - pole)))
        Rr.append(residues[index])
    A2, B2, C2, Nr = ResPoles2ABC(rp, np.zeros(0, complex), Rr, [],
                                   tol=0.0)
    # Each 2x2 residue matrix has rank two, hence two states per real pole.
    assert Nr == 6
    assert A2.shape == (6, 6)
    for omega in (0.1, 1.0, 5.0):
        s = 1j * omega
        original = C @ np.linalg.solve(s * np.eye(3) - A, B)
        rebuilt = C2 @ np.linalg.solve(s * np.eye(A2.shape[0]) - A2, B2)
        assert np.allclose(rebuilt, original, atol=1e-8)


def test_ss2respoles_non_normal_state_matrix():
    # Upper-triangular A is strongly non-normal: its right and left
    # eigenvector orderings differ, so unpacking scipy's (w, vl, vr) in the
    # wrong order pairs each pole with the wrong eigenvector and corrupts
    # the residues.
    A = np.array([[-1.0, 50.0], [0.0, -3.0]])
    B = np.array([[1.0], [2.0]])
    C = np.array([[3.0, 4.0]])
    poles, residues = SS2ResPoles(A, B, C)
    assert poles.shape == (2,)
    assert len(residues) == 2
    for omega in (0.1, 1.0, 10.0):
        s = 1j * omega
        reference = C @ np.linalg.solve(s * np.eye(2) - A, B)
        rebuilt = sum(np.asarray(R) / (s - pole)
                      for pole, R in zip(poles, residues))
        assert np.allclose(rebuilt, reference, rtol=1e-9, atol=1e-9)


def test_modfill_state_to_residues_preserves_transfer():
    _, exact = RandomPassiveDUM(type="none", Nports=2, Order=4,
                                Bandwidth=1e8, seed=3)
    state = _state_only(exact)
    filled = MODFill(state)
    assert filled.isABCD and filled.isResPoles
    assert filled.Order == exact.Order
    for omega in (1e7, 1e8, 1e9):
        s = 1j * omega
        reference = (np.asarray(exact.C)
                     @ np.linalg.solve(s * np.eye(exact.Order)
                                       - np.asarray(exact.A),
                                       np.asarray(exact.B))
                     + np.asarray(exact.D))
        assert np.allclose(_transfer_residues(filled, s), reference,
                           rtol=1e-8, atol=1e-8)


def test_modfill_residues_to_state_preserves_transfer():
    _, exact = RandomPassiveDUM(type="none", Nports=2, Order=4,
                                Bandwidth=1e8, seed=3)
    residues = _residues_only(exact)
    filled = MODFill(residues)
    assert filled.isABCD and filled.isResPoles
    for omega in (1e7, 1e8, 1e9):
        s = 1j * omega
        reference = _transfer_residues(exact, s)
        rebuilt = (np.asarray(filled.C)
                   @ np.linalg.solve(s * np.eye(filled.A.shape[0])
                                     - np.asarray(filled.A),
                                     np.asarray(filled.B))
                   + np.asarray(filled.D))
        assert np.allclose(rebuilt, reference, rtol=1e-8, atol=1e-8)


def test_mod2dum_frequency_and_time(frequency_dum, time_dum):
    _, exact = RandomPassiveDUM(type="none", Nports=2, Order=6,
                                Bandwidth=5e8, seed=1)
    template = DUM(Nports=2, pindex=np.ones((2, 2)), isfreq=True,
                   R0=exact.R0, f=frequency_dum.f)
    out = MOD2DUM(exact, template)
    assert out.isfreq and not out.istime
    assert np.allclose(out.S, frequency_dum.S, atol=1e-10)

    template_t = DUM(Nports=2, pindex=np.ones((2, 2)), istime=True,
                     R0=exact.R0, t=time_dum.t, a=time_dum.a,
                     Ts=time_dum.Ts)
    out_t = MOD2DUM(exact, template_t)
    assert out_t.istime
    assert len(out_t.b) == len(time_dum.b)
    for measured, fitted in zip(time_dum.b, out_t.b):
        assert np.allclose(fitted, measured, atol=1e-6)


def test_normalize_and_denormalize_roundtrip():
    _, exact = RandomPassiveDUM(type="none", Nports=2, Order=4,
                                Bandwidth=1e8, seed=3)
    normalized, rho_a, rho_b, rho_c = NormalizeMOD(
        exact, return_constants=True
    )
    assert rho_a > 0 and rho_b > 0 and rho_c > 0
    assert np.max(np.abs(normalized.A)) <= 1.0 + 1e-9
    back = DeNormalizeMOD(normalized)
    for name in ("A", "B", "C", "D"):
        assert np.allclose(np.asarray(getattr(back, name)),
                           np.asarray(getattr(exact, name)),
                           rtol=1e-6, atol=1e-7)
    assert np.allclose(back.RealPoles, exact.RealPoles, rtol=1e-6)
    if exact.ComplexPoles.size:
        assert np.allclose(back.ComplexPoles, exact.ComplexPoles, rtol=1e-6)
        for left, right in zip(back.ComplexResidues, exact.ComplexResidues):
            assert np.allclose(left, right, rtol=1e-6, atol=1e-7)


def test_convert_mod_units():
    _, exact = RandomPassiveDUM(type="none", Nports=1, Order=3,
                                Bandwidth=1e8, seed=2)
    zf, ff = 2.0, 3.0
    converted = ConvertMODUnits(exact, zf, ff)
    assert np.allclose(np.asarray(converted.A), ff * np.asarray(exact.A))
    assert np.allclose(np.asarray(converted.B), ff * np.asarray(exact.B))
    assert np.allclose(np.asarray(converted.C),
                       np.asarray(exact.C) / zf)
    assert np.allclose(np.asarray(converted.D),
                       np.asarray(exact.D) / zf)
    assert np.allclose(converted.RealPoles, ff * exact.RealPoles)
    for left, right in zip(converted.RealResidues, exact.RealResidues):
        assert np.allclose(left, right / zf)


def test_convert_dum_units_scattering_and_admittance():
    f = np.array([0.0, 1.0])
    S = np.ones((2, 1, 1), dtype=complex)
    scattering = DUM(Nports=1, isfreq=True, R0=50.0, f=f, S=S,
                     pindex=np.ones((1, 1)))
    out = ConvertDUMUnits(scattering, 2.0, 3.0, 4.0, 5.0)
    assert out.R0 == pytest.approx(150.0)
    assert np.allclose(out.f, 5.0 * f)
    assert np.allclose(out.S, S)  # scattering data are dimensionless

    admittance = DUM(Nports=1, isfreq=True, R0=0.0, f=f, S=S,
                     pindex=np.ones((1, 1)))
    out_y = ConvertDUMUnits(admittance, 2.0, 3.0, 4.0, 5.0)
    assert np.allclose(out_y.S, S * 4.0 / 3.0)

    t = np.array([0.0, 1.0, 2.0])
    time = DUM(Nports=1, istime=True, R0=50.0, t=t,
               a=[np.ones_like(t)], b=[np.ones_like(t)])
    out_t = ConvertDUMUnits(time, 2.0, 3.0, 4.0, 5.0)
    assert np.allclose(out_t.t, 4.0 * t)
    assert out_t.Ts == pytest.approx(4.0)
    assert np.allclose(out_t.a[0], 2.0)
    assert np.allclose(out_t.b[0], 2.0)
