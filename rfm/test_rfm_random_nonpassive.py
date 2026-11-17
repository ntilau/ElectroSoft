"""End-to-end test: random NON-passive S-parameters -> fit -> passify -> RFM.

Verifies that a random *active* multiport dataset (the "initial SNP") survives
the full IdEM pipeline: Vector-Fit macromodeling, Hamiltonian passivity check
and compensation, RFM export, and re-evaluation.

Because the data is non-passive, the fitted model starts out non-passive and
must be passified by :func:`idempy.passivity.CheckHamiltonian`.  Compensation
clamps the response (e.g. a gain of 2 toward 1), so the re-evaluated model is
checked for *passivity* rather than for a small RMS error against the active
initial data.

The DC (zero-frequency) response is trusted: the measured DC values are
enforced during the Vector-Fit fit (via ``H0_enforced``) and re-imposed after
passification, since Hamiltonian compensation perturbs the model internals and
can destroy the DC match.  The re-evaluated model is therefore checked for a
small DC error as well as passivity.
"""

from __future__ import annotations

import numpy as np

from idempy import (
    CheckDataPassivity,
    CheckHamiltonian,
    CheckMODPassivity,
    FDVF,
    MOD2DUM,
    MOD2RFM,
    RandomPassiveDUM,
)
from idempy.fdvf import _fit_poles, _make_mod, _split_poles_residues
from idempy.options import HamiltonianDefaultOptions
from idempy.residues import SS2ResPoles

# Target largest singular value of the active (non-passive) data:
# 1.01 means the data violates passivity by at most 1% over the unit bound.
_ACTIVE_MAX_SV = 1.01
_MAX_PASSIFY_ITERS = 10
# Bound on the absolute DC error after the full fit + passify + DC-refit cycle.
_MAX_DC_ABS_ERR = 0.05


def _ham_options() -> HamiltonianDefaultOptions:
    """Headless Hamiltonian options (no figures / movies / plots)."""
    opts = HamiltonianDefaultOptions()
    opts.update(
        {
            "NoFigures": 1,
            "MustPlotHamEigs": 0,
            "MustMovie": 0,
            "MustPlotBounds": 0,
            "MustPlotErrorBounds": 0,
            "MustPlotBreakpoints": 0,
            "MustPlotSlopes": 0,
            "MustPlotNewEigs": 0,
            "MustPlotSweep": 0,
            "MustHighlightViolations": 0,
        }
    )
    return opts


def _enforce_dc_after_passify(dum, h0_target, passive_model, ham_opts, maxiters: int = 5):
    """Re-impose the DC constraint after Hamiltonian passification.

    Compensation modifies the model internals (often converting to ABCD form
    and clearing poles/residues), which destroys the DC match achieved during
    the initial fit.  This converts the passified model back to poles/residues,
    refits the residues with an ``H0_enforced`` zero-frequency constraint, and
    re-passifies.  Repeats until the model is passive or *maxiters* is reached.

    Returns the (passive, DC-enforced) model.
    """
    model = passive_model
    for _ in range(maxiters):
        # Re-impose the trusted DC constraint: refit residues with H0_enforced.
        # (Hamiltonian compensation perturbs the internals and can destroy the
        # DC match achieved during the initial fit.)
        if model.isABCD:
            poles_all, res_all = SS2ResPoles(model.A, model.B, model.C, model.D)
            rp, cp, _, _ = _split_poles_residues(poles_all, res_all)
        elif model.isResPoles:
            rp = np.asarray(model.RealPoles, dtype=float).ravel()
            cp = np.asarray(model.ComplexPoles, dtype=complex).ravel()
        else:
            break
        best = _fit_poles(
            dum, rp, cp, {"tol": 1e-6, "MaxIt": 8, "H0_enforced": h0_target}
        )
        model = _make_mod(best, dum)
        # Re-passify: compensation perturbs the internals (and the DC match)
        # again.  CheckMODPassivity is the ground-truth convergence criterion.
        _is_passive, model = CheckHamiltonian(model, Options=ham_opts)
        if CheckMODPassivity(model) == 1:
            break
    return model


def test_random_nonpassive_sparameters_passify_to_rfm(tmp_path):
    # --- 1. Initial random NON-passive S-parameter dataset ------------------
    # Start from a passive model, then scale the response up so it becomes
    # active (singular values of S exceed one).
    dum, _exact = RandomPassiveDUM(
        type="frequency",
        Nports=2,
        Order=6,
        Bandwidth=5e8,
        LossFactor=1.0,
        Mismatch=1.0,
        seed=59,
    )
    assert dum.isfreq and dum.S is not None
    mask = np.isfinite(dum.S)
    # Scale the passive data so its largest singular value is exactly
    # _ACTIVE_MAX_SV (1% over the passive unit bound), making it active but
    # only marginally non-passive.
    max_sv = float(np.max(np.abs(dum.S[mask])))
    gain = _ACTIVE_MAX_SV / max_sv if max_sv > 0 else _ACTIVE_MAX_SV
    gain = max(1.0, gain)  # never scale down: keep data at least as active
    dum.S = gain * dum.S
    assert CheckDataPassivity(dum, "frequency") == 0

    # Trusted DC (zero-frequency) values, used to constrain the fit.
    h0_target = dum.S[0].copy()

    # --- 2. Fit + iteratively passify --------------------------------------
    bandwidth = 2.0 * np.pi * 5e8
    model, _fitted, err = FDVF(
        dum, Order=6, Bandwidth=bandwidth,
        Options={"tol": 1e-6, "MaxIt": 8, "H0_enforced": h0_target},
    )
    assert np.isfinite(err)
    # The fit reproduces the active data (and matches DC), so the fitted model
    # is non-passive.
    assert CheckMODPassivity(model) == 0

    ham_opts = _ham_options()
    passive_model = model
    is_passive = False
    for _ in range(_MAX_PASSIFY_ITERS):
        is_passive, passive_model = CheckHamiltonian(
            passive_model, Options=ham_opts
        )
        if is_passive:
            break
    assert is_passive, "Hamiltonian passification did not converge"

    # --- 3. Re-impose DC after passification -------------------------------
    # Hamiltonian compensation perturbs the model internals and can destroy
    # the DC match.  Refit residues with the DC constraint and re-passify.
    passive_model = _enforce_dc_after_passify(
        dum, h0_target, passive_model, ham_opts
    )
    assert CheckMODPassivity(passive_model) == 1

    # --- 4. Re-evaluate: the passified model must be passive ---------------
    model_dum = MOD2DUM(passive_model, dum)
    assert model_dum.S is not None
    assert np.all(np.isfinite(model_dum.S[mask]))
    assert CheckDataPassivity(model_dum, "frequency") == 1

    # --- 5. DC error must be small (trusted DC performance) ----------------
    om = 2.0 * np.pi * dum.f
    from idempy.fdvf import evaluate_freq_responses

    if passive_model.isResPoles:
        H = evaluate_freq_responses(
            passive_model.RealPoles, passive_model.ComplexPoles,
            passive_model.RealResidues, passive_model.ComplexResidues,
            passive_model.D, om, passive_model.Nports,
        )
    else:  # ABCD (state-space) form: H(s) = C (sI-A)^-1 B + D
        n = passive_model.A.shape[0]
        H = np.zeros((om.size, passive_model.Nports, passive_model.Nports),
                     dtype=complex)
        for k, w in enumerate(om):
            s = 1j * w
            H[k] = (
                passive_model.C
                @ np.linalg.solve(s * np.eye(n) - passive_model.A, passive_model.B)
                + passive_model.D
            )

    dc_idx = 0  # f[0] == 0 (DC)
    max_dc_err = 0.0
    for i in range(dum.Nports):
        for j in range(dum.Nports):
            if mask[dc_idx, i, j]:
                data = complex(dum.S[dc_idx, i, j])
                modv = complex(H[dc_idx, i, j])
                max_dc_err = max(max_dc_err, abs(modv - data))
    assert max_dc_err < _MAX_DC_ABS_ERR, (
        f"DC error too large: {max_dc_err:.4f} (limit {_MAX_DC_ABS_ERR})"
    )

    # --- 6. Export the passified model to RFM ------------------------------
    status = MOD2RFM(passive_model, str(tmp_path), "model.rfm")
    assert status == ""
    assert (tmp_path / "model.rfm").is_file()
