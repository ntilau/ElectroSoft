#!/usr/bin/env python3
"""Regenerate the committed ``.sNp -> RFM`` golden baseline.

Run this only after an intentional, reviewed change to the conversion
algorithm.  The analytic tests in ``test_snp_rfm_regression.py`` remain the
correctness oracle; the golden files catch unintended output drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_PYTHON_ROOT = _HERE.parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from snp_cases import CASES, FREQ_HZ, GOLDEN_DIR, fit_case  # noqa: E402


def main() -> int:
    arrays = {"case_names": np.asarray([case.name for case in CASES])}
    manifest = {
        "version": 1,
        "frequency_hz": [float(value) for value in FREQ_HZ],
        "cases": {},
    }

    for case in CASES:
        result = fit_case(case)
        arrays[f"happ_{case.name}"] = np.asarray(result["fitted"])
        arrays[f"model_response_{case.name}"] = np.asarray(
            result["model_response"])
        if result["compensated_response"] is not None:
            arrays[f"compensated_response_{case.name}"] = np.asarray(
                result["compensated_response"])

        manifest["cases"][case.name] = {
            "nports": int(case.nports),
            "reference_resistance": float(case.r0),
            "requested_order": int(case.order),
            "model_order": int(result["model"].Order),
            "fit_error": float(result["error"]),
            "is_passive": bool(result["is_passive"]),
            "model_check": int(result["model_check"]),
            "compensated_check": int(result["compensated_check"]),
        }

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_DIR / "snp_rfm_golden.npz", **arrays)
    (GOLDEN_DIR / "snp_rfm_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {GOLDEN_DIR / 'snp_rfm_golden.npz'}")
    print(f"wrote {GOLDEN_DIR / 'snp_rfm_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
