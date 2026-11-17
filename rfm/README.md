# idempy — Python port of the IdEM toolbox

`idempy` is a Python re-implementation of **IdEM**, the MATLAB toolbox for
linear multiport interconnect macromodeling and model-order reduction written
at Politecnico di Torino (2003–2006).

The pipeline is the same as the original:

```
import measured/simulated data
        -> Vector-Fit macromodeling
        -> passivity check and compensation
        -> model-order reduction / skimming
        -> export to circuit solvers
```

The original source tree lives in `../matlab/`.  Most of its `.m` files are
thin signatures whose implementation is locked inside MATLAB R2006 `.p`
p-code, so this port is an independent re-implementation of the documented
algorithms in NumPy/SciPy rather than a line-by-line translation.  Function
names, data-structure field names and calling conventions follow the original
wherever they are documented.

The numerical core is covered by a self-contained pytest suite.  The
`.sNp` → `DUM` → Vector Fit → `MOD` path additionally has an independent
regression suite built from closed-form circuits and committed golden data.

## Requirements

* Python >= 3.9
* NumPy, SciPy, Matplotlib (installed automatically as dependencies)
* Tkinter for the graphical interface (part of the standard CPython
  distribution, but packaged separately on some Linux systems)

## Installation

From this directory:

```bash
python3 -m pip install -e .
# with the test dependency
python3 -m pip install -e ".[test]"
```

The package can also be used directly from a checkout without installing,
because everything lives under `python/`:

```bash
cd python
python3 -c "import idempy"
```

## Quick start

```python
import numpy as np
import idempy as idem

# 1. Generate a passive two-port dataset and the exact model that made it.
dum, exact = idem.RandomPassiveDUM(
    type="frequency", Nports=2, Order=6, Bandwidth=1e9, seed=1)

# 2. Vector-Fit a reduced model.  Bandwidth is an angular frequency (rad/s).
model, fitted, error = idem.FDVF(
    dum, Order=12, Bandwidth=2 * np.pi * 1e9)
print(f"fit error: {error:.3e}")

# 3. Check and, if necessary, compensate passivity.
is_passive, passive_model = idem.CheckHamiltonian(model)
print(f"passive: {is_passive}")

# 4. Compare model and data, then export.
comparison = idem.MOD2DUM(passive_model, dum)
idem.DUM2TouchStone("results", "example.s2p", dum)
idem.MOD2Astap(passive_model, "results", "example")
idem.MOD2Cir(passive_model, "results", "example")
idem.MOD2Ascii(passive_model, "results", "example.txt")
```

Function signatures accept the original MATLAB parameter names, including
`Options` dictionaries and `RootFigHandle` compatibility arguments.

## Typical `.sNp` → RFM workflow

Starting from a TouchStone measurement or simulation result:

```python
from pathlib import Path
import numpy as np
import idempy as idem

dum = idem.TouchStone2DUM("measurements", "device.s2p")

# Bandwidth is angular frequency (rad/s); Order is the requested model order.
model, fitted, error = idem.FDVF(
    dum, Order=12, Bandwidth=2 * np.pi * dum.Fmax)
print(f"fit error: {error:.3e}")

# The returned model is passive.  If compensation was required, it is
# already applied and the stale poles/residues fields are cleared.
is_passive, model = idem.CheckHamiltonian(model)
print(f"input model was passive: {is_passive}")

# Re-evaluate the macromodel on the imported frequency grid and export it.
model_dum = idem.MOD2DUM(model, dum)
output = Path("results")
output.mkdir(exist_ok=True)
idem.DUM2TouchStone(output, "device_model.s2p", model_dum)
idem.MOD2Astap(model, output, "device_model")
idem.MOD2Cir(model, output, "device_model")
idem.MOD2Ascii(model, output, "device_model.txt")
```

`fitted` and `error` describe the Vector-Fit result on the input samples;
`model_dum` is the same rational model evaluated on the dataset's frequency
grid.  `MOD2Astap`, `MOD2Cir` and `MOD2Ascii` export the model itself, not the
re-sampled dataset.

## Pipeline and module map

| Stage | Module | Main entry points |
| --- | --- | --- |
| Data model | `idempy.data_model` | `DUM`, `MOD`, `scan_for_valid_dum`, `scan_for_valid_mod`, `ScanForValidDUM`, `ScanForValidMOD` |
| Options | `idempy.options` | `*DefaultOptions`, `Get*DefaultOptions`, `Merge*Options`, `merge_options` |
| Vector Fitting | `idempy.vectorfit` | `VectorFit`, `MultiVectorFit`, `VectorFitTime`, `MultiVectorFitTime`, `MultiVectorFitPD`, `RelativeWeightsComputation`, `Compute_VF_Error` |
| Frequency-domain macromodeling | `idempy.fdvf` | `FDVF`, `FDVF_Split`, `FDVF_AS`, `relative_error`, `evaluate_freq_responses` |
| Time-domain macromodeling | `idempy.tdvf` | `TDVF`, `TDVF_Split`, `TDVF_ColumnWise` |
| Model-order reduction | `idempy.skimming` | `AddingSkimming`, `En_Skimming`, `Final_Skimming`, `Multi_Adding` |
| Passivity | `idempy.passivity` | `CheckHamiltonian`, `CheckHamiltonianSparse`, `CheckHamiltonianAllSparse`, `CheckHamiltonianAllSparseAdaptive`, `CheckMODPassivity`, `CheckDataPassivity`, `FixAsymptoticPassivity`, `FlippingPoles` |
| Hamiltonian eigensolvers | `idempy.eigensolver` | `SparseHamEig_All`, `SparseHamEig_Few`, `SparseHamEig_SingleShift`, `GetAdaptiveFrequencySamples` |
| Lyapunov equations | `idempy.lyapunov` | `FullLyap`, `SparseLyap`, `AllSparseLyap` |
| State space ↔ poles/residues | `idempy.residues` | `SS2ResPoles`, `ResPoles2ABC`, `ResPoles2BlockABC` |
| Model conversion / scaling | `idempy.mod_convert` | `MODFill`, `MOD2DUM`, `ConvertDUMUnits`, `ConvertMODUnits`, `NormalizeMOD`, `DeNormalizeMOD` |
| Dataset operations | `idempy.dum_ops` | `ExtractSubDUM`, `ExtendDUM`, `SparsifyDUM`, `SparsifyConnector`, `SplitPorts`, `Split2MOD`, `ExtractDUMEnergy`, `ExtractLargestResponses`, `ReplicateDUMInput`, `AddFrequencyData`, `CheckPuzzle` |
| TouchStone I/O | `idempy.touchstone` | `TouchStone2DUM`, `DUM2TouchStone` |
| SPICE/CSV/CSDF import | `idempy.csdf`, `idempy.spice` | `ImportCSDF`, `ImportCSV`, `raw2dat`, `fraw2dat`, `Fspice_xt2subckt`, `Fastap_xt2subckt`, `SpiceMOD2DUM`, `AstapMOD2DUM` |
| CST / FDTD import | `idempy.cst`, `idempy.fdtd` | `CST2DUM`, `CSTRead`, `FDTD2DUM` |
| Circuit export | `idempy.astap`, `idempy.cir`, `idempy.ascii_export` | `MOD2Astap`, `MOD2Cir`, `MOD2Ascii` |
| Display | `idempy.display` | `DUMInfo`, `MODInfo`, `PlotDUM`, `ShowDUM`, `ShowDUMvsDUM`, `ShowDUMvsMOD` |
| Synthetic data | `idempy.random_data` | `RandomPassiveDUM`, `RandomSS_S`, `RandomSS_Y` |
| Utilities | `idempy.utils` | `CheckIdEM`, numerical and filename helpers |

The MATLAB-named API is re-exported from the top-level package, so both
styles work:

```python
import idempy as idem
idem.FDVF(...)

from idempy import DUM, MOD, FDVF
```

A few Python-only helpers, such as `relative_error` and
`evaluate_freq_responses`, remain available from their submodules.

## Data model

`DUM` is the raw dataset:

```text
Nports, pindex, istime, isfreq, R0, isreciprocal,
t, a, b, Ts,          # time-domain responses
f, S, Fmax,           # frequency-domain responses
name, units
```

* `S` has shape `(Nfreq, Nports, Nports)`; `S[k, i, j]` is the response from
  input port `j` to output port `i`.
* `a`/`b` are Python lists with one waveform per valid `(i, j)` pair, in
  column-major order.
* Unavailable responses have `pindex[i, j] == 0` and are stored as `NaN`.

`MOD` is the macromodel:

```text
isABCD, isResPoles, issparse, R0, D,
A, B, C,                                  # state-space form
RealPoles, ComplexPoles,
RealResidues, ComplexResidues,            # poles/residues form
Cap, name, units
```

`MOD.Order` and `MOD.Nports` are derived properties.  Complex poles are
stored with positive imaginary part only; the conjugate pair is implied.

## Import and export formats

* **TouchStone** `.sNp` input and output.  Input accepts S data in RI, MA or
  DB form, frequencies in Hz/kHz/MHz/GHz, and an arbitrary port count inferred
  from the `.s<N>p` extension; Y data is converted to S.  Z, H and G data are
  rejected, and noise parameters are ignored.  Output is always S data in
  RI form with frequencies in Hz.  Missing (`NaN`) responses are written as
  `nan` so that they survive a round trip.
* **CST Microwave Studio** v3/v4/v5 import (`CST2DUM`).
* **FDTD** output import (`FDTD2DUM`).
* **CSDF / CSV** SPICE results (`ImportCSDF`, `ImportCSV`), including
  PowerSPICE raw-to-DAT conversion (`raw2dat`, `fraw2dat`).
* **SPICE subcircuits** for Fspice/Fastap (`Fspice_xt2subckt`,
  `Fastap_xt2subckt`) and conversion back to `DUM`.
* **ASTAP** and **SPICE** netlists (`MOD2Astap`, `MOD2Cir`), plus a plain
  ASCII state-space dump (`MOD2Ascii`).

## Passivity conventions

The port uses the same two representations as IdEM:

* **Scattering** (`R0 > 0`): passive when the largest singular value of `S`
  is at most `sqrt(Lev)` (normally `Lev = 1`).
* **Admittance** (`R0 == 0`): passive when the Hermitian part of `Y` is
  positive semidefinite at all frequencies.

`CheckHamiltonian` returns `(IsPassive, MODpassive)`.  When a model violates
passivity, the returned model has a modified `C` matrix that restores it;
the poles/residues fields are cleared because their residues would no longer
match the compensated state space.

## Testing and validation

The suite is self-contained: it uses synthetic datasets and never invokes
MATLAB or an external SPICE/ASTAP solver.  Install the test dependency with:

```bash
python3 -m pip install -e ".[test]"
```

### Complete suite

Run all tests from this directory:

```bash
python3 -m pytest tests -q
```

Run one module or one test:

```bash
python3 -m pytest tests/test_fdvf.py -q
python3 -m pytest tests/test_fdvf.py::test_fdvf_model_reproduces_data -q
```

The modules mirror the package areas: data model and options, TouchStone and
other importers, exporters, frequency- and time-domain Vector Fitting,
passivity, model conversion, dataset operations, and display.

### `.sNp` → RFM regression suite

`tests/regression/` validates the end-to-end path

```text
.sNp → DUM → Vector Fit → MOD → passivity check/compensation
                                  → MOD2DUM → .sNp
```

It is deliberately independent of the implementation under test:

* the `.sNp` fixtures are generated from closed-form R-L, series/shunt,
  N-port star, and active-gain networks in `snp_circuits.py`;
* the tests compare imported S-parameters with those analytic expressions, not
  with another run of the importer;
* RI/MA/DB formats, Hz/GHz units, 50 Ω/75 Ω references, 1–4 ports, and a
  non-passive model are represented;
* Vector-Fit accuracy, pole stability, passivity, compensation, and TouchStone
  export/re-import round trips are checked for every fixture;
* a committed golden baseline catches numerical drift in Vector Fitting and
  Hamiltonian compensation.

Run it directly or by marker:

```bash
python3 -m pytest tests/regression -q
python3 -m pytest -m regression -q
```

The fixtures are deterministic and can be regenerated with
`python3 tests/regression/generate_fixtures.py`.  The golden baseline should be
regenerated with `python3 tests/regression/regen_golden.py` only after an
intentional, reviewed algorithm change.  See `tests/regression/README.md` for
the fixture table, tolerances, update policy, and failure diagnostics.

The four original `TestMOD*.m` scripts are represented by the `TestMODCir_*` /
`TestMODAstap_*` wrappers in `idempy.spice`; their solver-backed end-to-end
checks require an external circuit simulator, while the numeric and
file-format checks run without one.

## Continuous integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the full pytest
suite on every push and pull request to `main`.  It also runs
`tests/test_rfm_random_passive.py`, which exercises random S-parameters through
fit → passify → RFM export.  The model order is increased iteratively until
the RMS error between the initial SNP and the passified RFM model is at most
0.5%; only the RMS error and the exported RFM file are validated, and no
comparison image is written.

`tests/test_rfm_random_nonpassive.py` covers the same pipeline starting from
random *non-passive* (active) S-parameters: the fit reproduces the active data,
the fitted model is non-passive, and `CheckHamiltonian` compensation restores
passivity.  The re-evaluated model is checked to be passive and exported to an
RFM file; no RMS-against-active-data target is asserted because compensation
legitimately clamps the response (for example, a gain of 2 toward 1).

## Limitations

* The MATLAB `.p` p-code is unreadable and version-locked; no algorithm body
  was translated literally.  The port follows the documented interfaces and
  established vector-fitting / passivity / model-reduction methods.
* The original `IdEM_setup.m` and `IdEM_start.m` MATLAB path scripts are not
  ported; Python installation replaces them.
* The external ASTAP and CIR solvers are not bundled.  Export tests validate
  netlist structure and coefficients, not simulation results.
* MATLAB returns `''` when no valid dataset is found; the Python
  `scan_for_valid_*` helpers return `None`, while the MATLAB-named
  `ScanForValidDUM`/`ScanForValidMOD` wrappers return `''`.
* Port indices in the Python API are zero-based, unlike MATLAB's one-based
  indices.
