"""Time-domain Vector Fitting tests."""

from __future__ import annotations

import numpy as np
import pytest

from idempy import DUM, MOD2DUM, TDVF, TDVF_ColumnWise, TDVF_Split

# The TD-VF resampling rate is set by the angular *Bandwidth*.  A value a
# few times wider than the signal content keeps enough samples and gives
# well-conditioned pole relocation for the random test data.
TD_BANDWIDTH = 2.0 * np.pi * 5e9


def test_tdvf_fits_time_domain_dataset(time_dum):
    mod, tapp, Yapp, err = TDVF(time_dum, 8, TD_BANDWIDTH,
                                {"tol": 1e-8, "MaxIt": 6})
    assert mod.isResPoles
    assert mod.Nports == 2
    assert 0 < mod.Order <= 10
    assert err < 1e-3
    assert len(tapp) == len(time_dum.a)
    assert len(Yapp) == len(time_dum.b)
    for k in range(len(Yapp)):
        reference = np.interp(tapp[k], time_dum.t,
                              np.asarray(time_dum.b[k], dtype=float))
        denominator = np.max(np.abs(reference))
        relative = np.max(np.abs(Yapp[k] - reference)) / denominator
        assert relative < 2e-3
    assert np.all(mod.RealPoles < 0)
    assert np.all(np.imag(mod.ComplexPoles) > 0)


def test_tdvf_model_generated_from_time_data(time_dum):
    mod, _, _, _ = TDVF(time_dum, 8, TD_BANDWIDTH,
                        {"tol": 1e-8, "MaxIt": 6})
    template = DUM(Nports=2, pindex=np.ones((2, 2)), istime=True,
                   R0=time_dum.R0, t=time_dum.t, a=time_dum.a,
                   Ts=time_dum.Ts)
    response = MOD2DUM(mod, template)
    assert len(response.b) == len(time_dum.b)
    # Resampling by TD-VF limits accuracy on the original (finer) axis;
    # the waveform shape must nevertheless be preserved.
    for measured, fitted in zip(time_dum.b, response.b):
        denominator = np.max(np.abs(measured))
        assert np.max(np.abs(fitted - measured)) / denominator < 0.2


def test_tdvf_rejects_bad_arguments(time_dum):
    with pytest.raises(ValueError):
        TDVF(time_dum, 0, TD_BANDWIDTH)
    with pytest.raises(ValueError):
        TDVF(time_dum, 4, 0.0)


def test_tdvf_split_and_columnwise_are_sparse(time_dum):
    split = TDVF_Split(time_dum, 8, TD_BANDWIDTH, {"SplitType": "column"})
    assert split.isResPoles and split.issparse
    assert split.Order >= 8
    column = TDVF_ColumnWise(time_dum, 8, TD_BANDWIDTH)
    assert column.isResPoles and column.issparse
    assert column.Order >= 8
    from scipy import sparse
    assert any(sparse.issparse(R) for R in split.RealResidues)
    assert any(sparse.issparse(R) for R in column.RealResidues)
