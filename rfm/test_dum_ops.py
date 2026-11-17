"""Tests for dataset operations (extract, sparsify, extend, split, merge)."""

from __future__ import annotations

import numpy as np
import pytest

from idempy import (
    AddFrequencyData,
    DUM,
    ExtractDUMEnergy,
    ExtractLargestResponses,
    ExtractSubDUM,
    ExtendDUM,
    FDVF,
    ReplicateDUMInput,
    SparsifyConnector,
    SparsifyDUM,
    Split2MOD,
    SplitPorts,
)


def test_extract_sub_dum_frequency(frequency_dum):
    sub = ExtractSubDUM(frequency_dum, [0, 1])
    assert sub.Nports == 2
    assert sub.pindex.shape == (2, 2)
    assert np.allclose(sub.S, frequency_dum.S)
    one = ExtractSubDUM(frequency_dum, [0])
    assert one.Nports == 1
    assert one.S.shape == (len(frequency_dum.f), 1, 1)
    assert np.allclose(one.S[:, 0, 0], frequency_dum.S[:, 0, 0])


def test_extract_sub_dum_time(time_dum):
    sub = ExtractSubDUM(time_dum, [1])
    assert sub.Nports == 1
    assert len(sub.a) == 1 and len(sub.b) == 1
    assert sub.Ts == pytest.approx(time_dum.Ts)
    assert np.allclose(sub.t, time_dum.t)


def test_extract_sub_dum_rejects_bad_ports(frequency_dum):
    with pytest.raises(ValueError):
        ExtractSubDUM(frequency_dum, [])
    with pytest.raises(ValueError):
        ExtractSubDUM(frequency_dum, [0, 5])
    with pytest.raises(ValueError):
        ExtractSubDUM(frequency_dum, [-1])


def test_sparsify_dum_removes_weak_responses(frequency_dum):
    threshold = 0.5
    sparse, used = SparsifyDUM(frequency_dum, threshold=threshold,
                               return_threshold=True)
    assert used == threshold
    assert sparse.Nports == frequency_dum.Nports
    energy = ExtractDUMEnergy(frequency_dum)
    expected = (energy >= threshold).astype(int)
    assert np.array_equal(np.asarray(sparse.pindex), expected)
    removed = np.asarray(sparse.pindex) == 0
    assert np.all(np.isnan(sparse.S[:, removed]))
    assert np.allclose(sparse.S[:, ~removed], frequency_dum.S[:, ~removed])


def test_add_frequency_data_starts_at_dc(time_dum):
    result = AddFrequencyData(time_dum)
    assert result.isfreq
    assert result.f[0] == 0.0
    assert np.all(np.diff(result.f) > 0)
    assert result.S.shape == (len(result.f), time_dum.Nports,
                             time_dum.Nports)
    assert result.R0 == time_dum.R0
    assert result.Fmax == pytest.approx(result.f[-1])


def test_add_frequency_data_padding_options(time_dum):
    unpadded = AddFrequencyData(time_dum, padfactor=-1)
    padded = AddFrequencyData(time_dum, padfactor=0)
    assert len(padded.f) >= len(unpadded.f)
    # A positive padding factor adds another factor of two.
    extra = AddFrequencyData(time_dum, padfactor=1)
    assert len(extra.f) >= 2 * (len(padded.f) - 1)


def test_extend_dum_soft_and_hard(time_dum):
    extended = ExtendDUM(time_dum, Bandwidth=1e8, TimeExtensionFactor=2.0)
    assert len(extended.t) > len(time_dum.t)
    assert extended.t[0] == pytest.approx(time_dum.t[0])
    assert extended.Ts == pytest.approx(time_dum.Ts)
    assert len(extended.a[0]) == len(extended.t)
    assert np.allclose(extended.a[0][:len(time_dum.t)], time_dum.a[0])

    limit = time_dum.t[-1] + 0.5 * (extended.t[-1] - time_dum.t[-1])
    hard = ExtendDUM(time_dum, Bandwidth=1e8, TimeExtensionFactor=2.0,
                     MaxTime=limit, truncation="hard")
    assert hard.t[-1] <= limit
    assert len(hard.t) >= len(time_dum.t)
    with pytest.raises(ValueError):
        ExtendDUM(time_dum, Bandwidth=1e8, TimeExtensionFactor=2.0,
                  MaxTime=time_dum.t[-1] / 2, truncation="hard")


def test_extend_dum_rejects_invalid_bandwidth(time_dum):
    with pytest.raises(ValueError):
        ExtendDUM(time_dum, Bandwidth=0.0)


def test_replicate_dum_input_time(time_dum):
    weights = [1.0, 0.5]
    replicated = ReplicateDUMInput(time_dum, weights)
    assert replicated.Nports == time_dum.Nports
    assert replicated.istime
    # One input/output record per valid (output, input) port pair.
    assert len(replicated.a) == len(time_dum.a)
    assert len(replicated.b) == len(time_dum.b)
    assert replicated.pindex.shape == time_dum.pindex.shape
    for original, extended in zip(time_dum.a, replicated.a):
        assert extended.shape == original.shape
        assert np.allclose(extended[:len(original) // 2], original[:len(original) // 2])
    assert not np.allclose(replicated.a[0], time_dum.a[0])
    # A unit weight list reproduces the original waveforms exactly.
    same = ReplicateDUMInput(time_dum, [1.0])
    for original, copied in zip(time_dum.a, same.a):
        assert np.allclose(original, copied)
    for original, copied in zip(time_dum.b, same.b):
        assert np.allclose(original, copied)


def test_energy_and_largest_responses(frequency_dum):
    energy = ExtractDUMEnergy(frequency_dum)
    assert energy.shape == (2, 2)
    assert np.max(energy) == pytest.approx(1.0)
    mask = ExtractLargestResponses(frequency_dum, 2)
    assert mask.shape == (2, 2)
    assert np.count_nonzero(mask) == 2
    # The selected entries are the two largest energies.
    selected = energy[mask != 0]
    rejected = energy[mask == 0]
    assert np.all(selected >= np.max(rejected))


def test_split_ports_column_and_matrix(frequency_dum):
    parts = SplitPorts(frequency_dum, typ="column", Nlargest=1)
    assert len(parts) == frequency_dum.Nports
    for part in parts:
        assert part["pindex"].shape == (2, 2)
        assert np.count_nonzero(part["p4poles"]) <= 1
        assert np.all((part["p4poles"] != 0) <= (part["pindex"] != 0))
    explicit = np.array([[1, 0], [0, 2]])
    labelled = SplitPorts(frequency_dum, typ=explicit)
    assert len(labelled) == 2
    assert np.count_nonzero(labelled[0]["pindex"]) == 1
    with pytest.raises(ValueError):
        SplitPorts(frequency_dum, typ="nonsense")


def test_split2mod_merges_partial_fits(frequency_dum):
    parts = SplitPorts(frequency_dum, typ="column")
    for part in parts:
        mod, _, _ = FDVF(
            frequency_dum, 6, 2.0 * np.pi * 5e8,
            {"tol": 1e-8, "MaxIt": 6, "p4poles": part["p4poles"]},
        )
        part.update(
            rp=np.asarray(mod.RealPoles),
            cp=np.asarray(mod.ComplexPoles),
            rr=[np.asarray(R) for R in mod.RealResidues],
            cr=[np.asarray(R) for R in mod.ComplexResidues],
            dc=np.asarray(mod.D),
        )
    merged = Split2MOD(parts, frequency_dum.Nports, frequency_dum.R0)
    assert merged.isResPoles and merged.issparse
    assert merged.Nports == frequency_dum.Nports
    assert merged.Order >= 6


def test_sparsify_connector_first_order():
    class Spec:
        pass

    spec = Spec()
    spec.NE = np.arange(1, 10).reshape(3, 3)     # pins 1..9
    spec.FE = np.arange(10, 19).reshape(3, 3)    # pins 10..18
    from idempy import RandomPassiveDUM
    dum = RandomPassiveDUM(type="frequency", Nports=18, Order=4,
                           Bandwidth=1e8, seed=1)[0]
    sparse = SparsifyConnector(dum, spec, CouplingType="firstorder")
    mask = np.asarray(sparse.pindex)
    assert mask.shape == (18, 18)
    # Port 0 (pin at row 0, column 0) couples to itself, its right and lower
    # neighbours at the near and far ends, but not to the opposite corner.
    assert mask[0, 0] == 1
    assert mask[0, 1] == 1 and mask[0, 3] == 1
    assert mask[0, 8] == 0
    assert mask[0, 9] == 1  # far end of its own pin
    second = SparsifyConnector(dum, spec, CouplingType="secondorder")
    assert np.count_nonzero(second.pindex) > np.count_nonzero(mask)
