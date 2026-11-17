#!/usr/bin/env python3
"""Regenerate the static ``.sNp`` regression fixtures.

The files are derived only from the closed-form circuits in
``snp_circuits.py``; the idempy package is not used.  Running this script
should leave the committed fixtures unchanged unless the fixture definitions
or analytic networks are intentionally modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from snp_cases import DATA_DIR, write_all_fixtures  # noqa: E402


def main() -> int:
    paths = write_all_fixtures(DATA_DIR)
    for path in paths:
        print(f"wrote {path.relative_to(DATA_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
