"""Passivity check and compensation tests."""

from __future__ import annotations

import numpy as np
import pytest

from idempy import (
    CheckDataPassivity,
    CheckHamiltonian,
    CheckHamiltonianAllSparse,
    CheckHamiltonianAllSparseAdaptive,
    CheckHamiltonianSparse,
    CheckMODPassivity,
    DUM,
    FixAsymptoticPassivity,
    FlippingPoles,
    MOD,
    RandomPassiveDUM,
)


def _perturbed(mod, factor):
    return MOD(isABCD=True, issparse=False, R0=mod.R0,
               D=np.array(mod.D), A=np.array(mod.A), B=np.array(mod.B),
               C=factor * np.array(mod.C))


def test_passive_model_passes_all_checks(frequency_mod, frequency_dum):
    is_passive, returned = CheckHamiltonian(frequency_mod)
    assert is_passive
    assert returned is not None
    assert CheckMODPassivity(frequency_mod) == 1
    assert CheckDataPassivity(frequency_dum, "frequency") == 1
    assert CheckHamiltonianSparse(frequency_mod)[0]
    assert CheckHamiltonianAllSparse(frequency_mod)[0]
    assert CheckHamiltonianAllSparseAdaptive(frequency_mod)[0]


def test_nonpassive_model_is_compensated(frequency_mod):
    perturbed = _perturbed(frequency_mod, -5.0)
    is_passive, compensated = CheckHamiltonian(perturbed)
    assert not is_passive
    assert compensated.isABCD
    assert compensated.Order == perturbed.Order
    assert CheckMODPassivity(compensated) == 1


def test_direct_term_violation_is_not_compensated(frequency_mod):
    static = MOD(isABCD=True, issparse=False, R0=50.0,
                 D=2.0 * np.eye(2), A=np.zeros((0, 0)),
                 B=np.zeros((0, 2)), C=np.zeros((2, 0)))
    is_passive, returned = CheckHamiltonian(static)
    assert not is_passive
    assert CheckMODPassivity(returned) == 0


def test_admittance_model_and_data(frequency_mod):
    dum, mod = RandomPassiveDUM(
        type="frequency", Nports=2, Order=6, Bandwidth=5e8,
        Mismatch=0, seed=1,
    )
    assert dum.R0 == 0.0
    assert mod.isABCD and mod.isResPoles
    assert CheckHamiltonian(mod)[0]
    assert CheckMODPassivity(mod) == 1
    assert CheckDataPassivity(dum, "frequency") == 1


def test_time_domain_data_passivity(time_dum):
    assert CheckDataPassivity(time_dum, "time") == 1
    assert CheckDataPassivity(time_dum, "invalid-domain") == -1


def test_nonpassive_data_is_detected(frequency_dum):
    bad = DUM(Nports=2, pindex=np.ones((2, 2)), isfreq=True,
              R0=frequency_dum.R0, f=frequency_dum.f,
              S=10.0 * frequency_dum.S)
    assert CheckDataPassivity(bad, "frequency") == 0


def test_check_mod_passivity_invalid_inputs():
    assert CheckMODPassivity(None) == -1
    assert CheckMODPassivity(MOD(D=np.zeros((2, 2)))) == -1


def test_fix_asymptotic_passivity_scattering():
    fixed, dcnew = FixAsymptoticPassivity(np.array([[2.0, 0.0],
                                                    [0.0, 1.0]]), 50.0)
    assert fixed
    assert np.max(np.linalg.svd(dcnew, compute_uv=False)) < 1.0
    untouched, same = FixAsymptoticPassivity(np.array([[0.5, 0.0],
                                                       [0.0, 0.4]]), 50.0)
    assert not untouched
    assert np.allclose(same, [[0.5, 0.0], [0.0, 0.4]])


def test_fix_asymptotic_passivity_admittance():
    fixed, dcnew = FixAsymptoticPassivity(-np.eye(2), 0.0)
    assert fixed
    assert np.min(np.linalg.eigvalsh(0.5 * (dcnew + dcnew.conj().T))) > 0


def test_flipping_poles_relocates_origin_poles(frequency_dum):
    rp = np.array([1e-9, -1e6])
    cp = np.array([-1e6 + 1e6j])
    new_rp, new_cp = FlippingPoles(frequency_dum, rp, cp)
    assert new_rp.shape == rp.shape
    assert new_cp.shape == cp.shape
    assert np.all(new_rp < 0)
    assert abs(new_rp[0]) > 1.0
    assert new_rp[1] == pytest.approx(-1e6)
