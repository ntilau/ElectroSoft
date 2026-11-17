"""Frequency-domain Vector Fitting tests."""

from __future__ import annotations

import numpy as np
import pytest

from idempy import DUM, FDVF, FDVF_AS, FDVF_Split, MOD2DUM


def _evaluate(mod, template, reference):
    out = MOD2DUM(mod, template)
    mask = np.isfinite(reference.S) & np.isfinite(out.S)
    return np.max(np.abs(out.S[mask] - reference.S[mask])) / np.max(
        np.abs(reference.S[mask])
    )


def test_fdvf_fits_random_passive_dataset(frequency_dum):
    bandwidth = 2.0 * np.pi * 5e8
    mod, fitted, err = FDVF(
        frequency_dum, 8, bandwidth, {"tol": 1e-10, "MaxIt": 8}
    )
    assert mod.isResPoles and not mod.isABCD
    assert mod.Nports == 2
    assert mod.Order == 8
    assert err < 1e-6
    assert fitted.shape == frequency_dum.S.shape
    mask = np.isfinite(frequency_dum.S)
    assert np.all(np.isfinite(fitted[mask]))
    assert np.all(np.isnan(fitted[~mask]))
    rel = np.max(np.abs(fitted[mask] - frequency_dum.S[mask])) / np.max(
        np.abs(frequency_dum.S[mask])
    )
    assert rel < 1e-6
    # Every pole must be stable and complex poles are stored once,
    # with strictly positive imaginary part.
    assert np.all(mod.RealPoles < 0)
    assert np.all(np.imag(mod.ComplexPoles) > 0)
    assert np.all(np.real(mod.ComplexPoles) < 0)


def test_fdvf_model_reproduces_data(frequency_dum, frequency_mod):
    mod, _, err = FDVF(frequency_dum, 8, 2.0 * np.pi * 5e8,
                       {"tol": 1e-10, "MaxIt": 8})
    template = DUM(Nports=2, pindex=np.ones((2, 2)), isfreq=True,
                   R0=frequency_dum.R0, f=frequency_dum.f)
    assert _evaluate(mod, template, frequency_dum) < 1e-6
    assert mod.name is not None


def test_fdvf_pole_selector_restricts_fit(frequency_dum):
    options = {
        "tol": 1e-10,
        "MaxIt": 8,
        "p4poles": np.eye(2, dtype=int),
    }
    mod, fitted, err = FDVF(frequency_dum, 8, 2.0 * np.pi * 5e8, options)
    assert isinstance(err, float) and np.isfinite(err)
    assert mod.Order == 8
    # The diagonal responses are the ones used to identify the poles and
    # should be reproduced accurately even though the model order is low.
    for port in range(2):
        reference = frequency_dum.S[:, port, port]
        assert np.allclose(fitted[:, port, port], reference, atol=1e-4)


def test_fdvf_rejects_bad_arguments(frequency_dum):
    with pytest.raises(ValueError):
        FDVF(frequency_dum, 0, 2.0 * np.pi * 5e8)
    with pytest.raises(ValueError):
        FDVF(frequency_dum, 4, -1.0)


def test_fdvf_as_automatic_order_selection(frequency_dum):
    options = {
        "tol": 1e-3,
        "Nmax": 12,
        "Nstart": 4,
        "Nadd": 2,
        "InitialIter": 1,
        "PostAddingIter": 1,
        "FinalIter": 1,
    }
    mod = FDVF_AS(frequency_dum, 2.0 * np.pi * 5e8, options)
    assert mod.isResPoles
    assert 0 < mod.Order <= 12
    template = DUM(Nports=2, pindex=np.ones((2, 2)), isfreq=True,
                   R0=frequency_dum.R0, f=frequency_dum.f)
    assert _evaluate(mod, template, frequency_dum) < 2e-2


def test_fdvf_split_produces_sparse_model(frequency_dum):
    mod = FDVF_Split(frequency_dum, 8, 2.0 * np.pi * 5e8,
                     {"SplitType": "column"})
    assert mod.isResPoles and mod.issparse
    assert mod.Order >= 8
    from scipy import sparse
    assert any(sparse.issparse(R) for R in mod.RealResidues)
    template = DUM(Nports=2, pindex=np.ones((2, 2)), isfreq=True,
                   R0=frequency_dum.R0, f=frequency_dum.f)
    assert _evaluate(mod, template, frequency_dum) < 1e-6
