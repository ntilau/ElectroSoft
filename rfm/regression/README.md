# `.sNp` → rational-function macromodel regression suite

This directory validates the core IdEM conversion path:

```text
TouchStone .sNp  →  DUM  →  Vector Fitting  →  MOD (RFM poles/residues/ABCD)
                                                │
                                                ├─ Hamiltonian passivity check
                                                ├─ passivity compensation
                                                └─ MOD → DUM → .sNp re-export
```

The suite is self-contained: it needs neither MATLAB nor an external SPICE
solver.  All test data are small, static fixtures generated from closed-form
circuit equations, so a failure points to the Python port rather than to a
missing external tool.

## Run

From the repository's `python/` directory:

```bash
# the regression suite only
python3 -m pytest tests/regression -q

# via the marker (same tests)
python3 -m pytest -m regression -q

# the complete Python test suite
python3 -m pytest -q

# one case, with verbose failure output
python3 -m pytest tests/regression -q -k series_shunt_db_ghz -vv
```

## What is checked

### 1. Fixture integrity and independent analytic oracle

`test_fixture_file_is_static_and_nonempty` ensures the committed `.sNp` files
exist.  `test_analytic_fixture_has_expected_passivity` evaluates the same
closed-form networks used to create the files and verifies that passive
fixtures have largest singular value ≤ 1 and the active fixture has > 1.

The equations live in `snp_circuits.py`.  They deliberately do **not** call
`idempy`; they are an oracle independent of the code under test.

### 2. TouchStone import

For every fixture, `test_touchstone_import_matches_closed_form` checks:

* number of ports, `isfreq`/`istime`, `R0`, `pindex`, and `Fmax`;
* the frequency grid;
* the full complex `S[k, i, j]` tensor against the closed-form values.

`test_ri_ma_db_fixtures_are_equivalent` imports the same physical two-port
from RI/Hz, MA/GHz, and DB/GHz files and requires the resulting `DUM` objects
to agree.  This covers the parser's format, unit, and column-major record
ordering simultaneously.  `test_reference_resistance_is_read_from_option_line`
checks that the `R` option is honored rather than silently defaulted.

### 3. Vector-Fit quality, stability, and passivity

For each passive fixture, `test_passive_snp_fits_and_remains_passive` checks:

* requested and returned model order;
* residue-pole (`isResPoles`) representation;
* Vector-Fit error against the imported samples;
* fitted response vs both imported data and the analytic oracle;
* `MOD2DUM` re-evaluation vs imported data;
* all real and complex poles strictly in the left half-plane;
* finite `D`, Hamiltonian verdict `is_passive=True`, and
  `CheckMODPassivity == 1`;
* largest singular value of the model response ≤ 1 + 10⁻⁸.

`test_active_snp_is_detected_and_compensated` uses the unilateral gain fixture
(`S21 = 2`).  It requires the fit to be detected as non-passive
(`model_check == 0`), requires the compensated model to pass
(`compensated_check == 1`), and checks that the compensated response is finite
and passive.  Compensation is expected to alter the active gain; the test
therefore does not require the compensated response to equal the original
data.

### 4. TouchStone round trips

* `test_import_export_roundtrip` covers `.sNp → DUM → .sNp`.
* `test_snp_to_rfm_to_snp_roundtrip` covers
  `.sNp → DUM → RFM → DUM → .sNp` and checks both the re-exported model
  response and its agreement with the original samples.

### 5. Golden snapshots

`test_golden_snp_to_rfm_snapshot` compares each case's fitted samples and
re-evaluated model response against `golden/snp_rfm_golden.npz`.  It also
requires the model order, passivity verdict, `model_check`, and
`compensated_check` to match `golden/snp_rfm_manifest.json`, and checks that
the fit error has not grown beyond the recorded value.

The analytic tests are the correctness oracle.  The golden files catch
unintended numerical drift in algorithms whose exact output is not expressible
in closed form (Vector Fitting, Hamiltonian compensation).  A golden mismatch
is not automatically a bug — but it must be explained.

## Fixtures

| Case | File | Ports | R0 | Order | Passive | Purpose |
|---|---|---:|---:|---:|---|---|
| `rl_50ohm` | `s1p/rl_50ohm.s1p` | 1 | 50 Ω | 4 | yes | scalar R-L load |
| `series_shunt_ri_hz` | `s2p/series_shunt_ri_hz.s2p` | 2 | 50 Ω | 6 | yes | baseline RI/Hz |
| `series_shunt_ma_ghz` | `s2p/series_shunt_ma_ghz.s2p` | 2 | 50 Ω | 6 | yes | MA/GHz parser path |
| `series_shunt_db_ghz` | `s2p/series_shunt_db_ghz.s2p` | 2 | 50 Ω | 6 | yes | DB/GHz parser path |
| `series_shunt_75ohm` | `s2p/series_shunt_75ohm.s2p` | 2 | 75 Ω | 6 | yes | non-default `R` option |
| `star_3port` | `s3p/star_3port.s3p` | 3 | 50 Ω | 6 | yes | multiport star |
| `star_4port` | `s4p/star_4port.s4p` | 4 | 50 Ω | 6 | yes | multiport star |
| `active_gain` | `s2p/active_gain.s2p` | 2 | 50 Ω | 2 | no | non-passive detection/compensation |

The frequency grid is 41 logarithmically spaced points from 1 MHz to 10 GHz.
The perturbed networks use `R = 5 Ω`, `L = 1 nH`, `C = 1 pF`; the one-port
uses `R = 10 Ω`, `L = 2 nH`.

## Tolerances

| Stage | Tolerance | Rationale |
|---|---|---|
| Import, RI | `rtol=1e-12`, `atol=1e-13` | 17-digit text output, exact binary conversion |
| Import, MA/DB | `rtol=1e-12`, `atol=1e-10` | magnitude/angle or dB/angle round trip |
| Vector Fit | `rtol=1e-8`, `atol=1e-9` | typical errors are ~1e-12; leaves room for LAPACK variation |
| `MOD2DUM` | `rtol=1e-8`, `atol=1e-9` | same |
| Passivity | `max σ ≤ 1 + 1e-8` | numerical margin around the unit circle |
| Round-trip export | `rtol=1e-12`, `atol=1e-12` | ASCII writer/reader precision |
| Golden drift | `rtol=1e-6`, `atol=1e-9` | same-platform reproducibility, not bit equality |
| Golden fit error | `max(10 × recorded, 1e-9)` | reject meaningful degradation |

## Updating fixtures and goldens

The fixtures are deterministic.  Regenerate them only after changing the
analytic circuits or the case table:

```bash
python3 tests/regression/generate_fixtures.py
```

This script imports only `snp_cases.py`/`snp_circuits.py`; it does not use
`idempy` to create the reference data.  Verify the resulting diff before
committing it.

Regenerate the golden baseline only after an intentional, reviewed change to
the conversion algorithms (Vector Fitting, `MOD2DUM`, passivity checking, or
compensation):

```bash
python3 tests/regression/regen_golden.py
```

After regeneration, inspect both the `.npz` (via the manifest and the test
output) and `snp_rfm_manifest.json`, then re-run the analytic tests.  Updating
the golden to make a failing test pass without reviewing the analytic result
defeats the purpose of the suite.  The manifest is the human-readable record:
check model orders, passivity verdicts, and fit errors there before committing.

## Files

| File | Role |
|---|---|
| `snp_circuits.py` | independent closed-form S-parameter equations |
| `snp_cases.py` | fixture definitions, TouchStone writer, shared fit pipeline |
| `generate_fixtures.py` | regenerate `data/` from the analytic circuits |
| `regen_golden.py` | regenerate `golden/` from the current implementation |
| `data/` | committed static `.sNp` fixtures |
| `golden/snp_rfm_golden.npz` | fitted and re-evaluated responses |
| `golden/snp_rfm_manifest.json` | model metadata, passivity flags, fit errors |
| `test_snp_rfm_regression.py` | pytest suite (also bootstraps its local import path) |

## Scope and limitations

This suite validates S-parameter TouchStone input only.  It does not cover
Y-parameter TouchStone, CST/FDTD/SPICE import, time-domain Vector Fitting,
model-order reduction/skimming, or ASTAP/CIR/ASCII export.  Those paths have
separate unit tests under `python/tests/`.  The regression suite intentionally
stays small and deterministic so it can run on every change.
