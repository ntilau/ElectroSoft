"""End-to-end test: random S-parameters -> fit -> passify -> RFM.

Verifies that a random passive multiport dataset (the "initial SNP") survives
the full IdEM pipeline: Vector-Fit macromodeling, Hamiltonian passivity check
and compensation, RFM export, and re-evaluation.  The model order is increased
iteratively until the RMS error between the initial SNP and the passified RFM
model is at most 0.5%.  No comparison image is written; only the RMS error and
the exported RFM file are validated.
"""

from __future__ import annotations

import numpy as np

from idempy import (
    CheckHamiltonian,
    FDVF,
    MOD2DUM,
    MOD2RFM,
    RandomPassiveDUM,
)
from idempy.options import HamiltonianDefaultOptions

_TARGET_RMS = 5.0e-3  # 0.5%
_MAX_ORDER_ATTEMPTS = 6


def _rms_error(model_dum, dum, mask) -> float:
    """RMS error between re-evaluated model and initial S, relative to the
    initial S RMS level."""
    ref_rms = float(np.sqrt(np.mean(np.abs(dum.S[mask]) ** 2)))
    if ref_rms == 0.0:
        ref_rms = 1.0
    diff = model_dum.S[mask] - dum.S[mask]
    return float(np.sqrt(np.mean(np.abs(diff) ** 2)) / ref_rms)


def test_random_passive_sparameters_passify_to_rfm(tmp_path):
    # --- 1. Initial random passive S-parameter dataset ("SNP initial") ------
    dum, _exact = RandomPassiveDUM(
        type="frequency",
        Nports=2,
        Order=6,
        Bandwidth=5e8,
        LossFactor=1.0,
        Mismatch=1.0,
        seed=11,
    )
    assert dum.isfreq and dum.S is not None
    mask = np.isfinite(dum.S)

    # --- 2. Iteratively fit + passify until RMS error <= 0.5% -------------
    # The data is generated from a low-order passive model, so a fit at or
    # below its true order is naturally passive and needs no compensation
    # (which would otherwise distort the response).  Start low and increase
    # the order only if a given order cannot be passified or misses the target.
    bandwidth = 2.0 * np.pi * 5e8
    order = 6
    passive_model = None
    rms_err = float("inf")
    succeeded = False
    for attempt in range(_MAX_ORDER_ATTEMPTS):
        model, _fitted, err = FDVF(
            dum, Order=order, Bandwidth=bandwidth,
            Options={"tol": 1e-6, "MaxIt": 8},
        )
        assert np.isfinite(err)

        # Iteratively passify the model (headless, no figures).
        ham_opts = HamiltonianDefaultOptions()
        ham_opts.update(
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
        passive_model = model
        is_passive = False
        for _ in range(10):
            is_passive, passive_model = CheckHamiltonian(
                passive_model, Options=ham_opts
            )
            if is_passive:
                break
        if not is_passive:
            # This order could not be passified; try a higher order.
            order += 4
            continue

        # Re-evaluate on the initial axis and measure RMS error.
        model_dum = MOD2DUM(passive_model, dum)
        assert model_dum.S is not None
        assert np.all(np.isfinite(model_dum.S[mask]))
        rms_err = _rms_error(model_dum, dum, mask)

        if rms_err <= _TARGET_RMS:
            succeeded = True
            break
        order += 4

    assert succeeded, (
        f"RMS error {rms_err * 100:.2f}% exceeds 0.5% target "
        f"after {_MAX_ORDER_ATTEMPTS} order attempts"
    )

    # --- 3. Export the passified model to RFM -------------------------------
    status = MOD2RFM(passive_model, str(tmp_path), "model.rfm")
    assert status == ""
    rfm_path = tmp_path / "model.rfm"
    assert rfm_path.is_file()

    # No comparison image is written: the RMS error and the exported RFM file
    # are the only outputs of this test.
