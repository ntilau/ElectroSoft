"""Tests for the DUM/MOD data model and validators."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from idempy import DUM, MOD
from idempy.data_model import scan_for_valid_dum, scan_for_valid_mod


def test_dum_defaults_build_pindex_and_fmax():
    f = np.array([0.0, 1.0, 2.0])
    S = np.zeros((3, 2, 2), dtype=complex)
    dum = DUM(Nports=2, isfreq=True, f=f, S=S)
    assert dum.pindex.shape == (2, 2)
    assert np.all(dum.pindex == 1)
    assert dum.Fmax == pytest.approx(2.0)
    assert dum.Nfreq == 3
    assert dum.is_valid()
    assert dum.response(1, 0).shape == (3,)


def test_dum_time_defaults_ts_and_validity():
    t = np.linspace(0.0, 1e-9, 11)
    a = [np.ones_like(t)]
    b = [np.zeros_like(t)]
    dum = DUM(Nports=1, istime=True, t=t, a=a, b=b)
    assert dum.Ts == pytest.approx(1e-10)
    assert dum.Ntime == 11
    assert dum.is_valid()


def test_dum_invalid_when_required_parts_missing():
    assert not DUM().is_valid()
    assert not DUM(Nports=2, isfreq=True).is_valid()
    dum = DUM(Nports=1)
    dum.isfreq = True
    assert not dum.is_valid()


def test_set_response_creates_nan_array():
    dum = DUM(Nports=2)
    values = np.array([1.0 + 1j, 2.0 - 1j])
    dum.set_response(0, 1, values)
    assert dum.S.shape == (2, 2, 2)
    assert np.all(np.isnan(dum.S[:, 1, 0]))
    assert np.allclose(dum.S[:, 0, 1], values)


def test_mod_order_and_nports_for_residues():
    mod = MOD(
        isResPoles=True,
        R0=50.0,
        D=np.eye(2),
        RealPoles=np.array([-1.0, -2.0]),
        ComplexPoles=np.array([-1.0 + 2.0j]),
        RealResidues=[np.zeros((2, 2)), np.zeros((2, 2))],
        ComplexResidues=[np.zeros((2, 2))],
    )
    assert mod.Nports == 2
    assert mod.Order == 4
    assert mod.is_valid()


def test_mod_order_sparse_state_space_preserved():
    A = sparse.eye(5, format="csr") * -1.0
    B = sparse.csr_matrix(np.ones((5, 2)))
    C = sparse.csr_matrix(np.ones((2, 5)))
    mod = MOD(isABCD=True, issparse=True, R0=50.0, A=A, B=B, C=C,
              D=np.zeros((2, 2)))
    assert sparse.issparse(mod.A)
    assert mod.Order == 5
    assert mod.Nports == 2
    assert mod.is_valid()


def test_mod_invalid_without_representation():
    assert not MOD(D=np.eye(2)).is_valid()
    assert not MOD(isABCD=True, D=np.eye(2)).is_valid()


def test_scan_for_valid_dum_and_mod():
    dum = DUM(Nports=1, istime=True, t=np.arange(3.0),
              a=[np.arange(3.0)], b=[np.arange(3.0)])
    mod = MOD(isResPoles=True, D=np.zeros((1, 1)),
              RealPoles=np.array([-1.0]), ComplexPoles=np.zeros(0),
              RealResidues=[np.zeros((1, 1))], ComplexResidues=[])
    assert scan_for_valid_dum({"x": 3, "d": dum}) == "d"
    assert scan_for_valid_dum({"x": 3}) is None
    assert scan_for_valid_mod({"m": mod}) == "m"
    assert scan_for_valid_mod({}) is None
