# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ElectroSoft is a Python prototyping framework for computational electromagnetics:
a 2D/3D finite element solver based on the E-field formulation with H(curl)
conforming elements. It supports S-parameter extraction, eigenmode analysis,
electrostatics, incident plane waves, and far-field radiation patterns.

It is the Python port of the FES (Finite Element Software) C++ codes from the
ElectroSoft start-up. The Python import package remains `fes` (and the CLI is
`python -m fes`), so the port mirrors the reference implementation's structure.

## Essential commands

The repo ships a venv at `.venv` (created by `./configure` / `make setup`).
Activate it once so `python`, `pytest`, and `ruff` work directly:

```bash
source .venv/bin/activate
```

Without activating, prefix every command with `.venv/bin/python -m` (the sandbox
has no `python` on PATH; use `python3` or `.venv/bin/python`).

| Task | Command |
|------|---------|
| Setup (venv + editable install) | `make setup`  (same as `./configure`) |
| Run full test suite | `make test`  or  `pytest tests/ -q`  (CI equivalent) |
| Run a single test | `pytest tests/test_drivers.py::test_matlab_mode_dumps_system_matrices -v` |
| Lint | `ruff check fes tools tests mm n2f rfm setup.py` |
| Run CLI (one frequency) | `python -m fes run WR90 +f 1e10 +poly +p 2` |
| Legacy CLI spelling | `python -m fes WR90 1e10 +poly +p 2` |
| Dry-run (plan only, no solve) | `python -m fes run WR90 +f 1e10 --dry-run` |
| List / inspect models | `python -m fes models` / `python -m fes info WR90 [--json]` |
| Build native meshers (Triangle / TetGen) | `python tools/build_triangle.py && python tools/build_tetgen.py` |

Ruff is **not** installed by default in the venv — CI installs it with
`pip install -e . pytest ruff`. If `ruff` is unavailable, install it:
`.venv/bin/python -m pip install ruff`.

CLI exit codes: `0` success, `1` usage error, `2` formulation not ported,
`3` I/O error.

## High-level architecture

**Reference path** (the only numerical path — read `README.md` "Architecture"
before touching it):

- **`fes/cli.py`** (`main()`) parses `+flags` / subcommands (`run`, `sweep`,
  `mesh`, `sparam`, `eigen`, `info`, `models`) via `fes/core/options.py`
  (`parse_argv`, `Options`, `ASSEMBLY_TYPES`), builds a `RunPlan` in
  `fes/api.py` (`plan_run`), resolves the formulation (`effective_assembly`:
  explicit CLI flag wins, else the `#Formula` tag in the `.poly`), then
  `dispatch()` looks the assembly up in `fes.api.DRIVERS` (populated lazily by
  `fes.drivers.install()`). A formulation with no driver raises
  `NotPortedError`; `em_h_qs` raises `FormulationUnavailableError`.
- **Drivers** (`fes/drivers.py`) hand results to `fes/post/` writers
  (VTK, Touchstone `.sNp`).
- **Assemblers** (`fes/core/assemblers/`) plus reference kernels
  (`shape`, `shape3d`, `quadrature`, `element_matrix`, `dof`,
  `quadrature_tables3d`) and the mesh backends.

**Formulation phases** (`fes/api.py` FORMULATION_PHASE):
`em_ez_fd`=2, `em_e_fd`=6, `em_e_eig_tl`=7, `em_e_tl_eig`=7,
`em_e_fd_dd`=9 (not ported), `em_e_fd_nl`=10 (not ported), `em_e_qs`=11.

`em_e_fd` (phase 6) is verified working: the Python assembler reproduces the
FES C++ solver output (loop order, DOF numbering, formatting) and the full test
suite passes (`pytest tests/ -q` → 0 failed). See the `em_e_fd_cpp_match`
memory for the Matrix Market comparison details.

**Mesh pipeline:** models live as `.poly` files in `data/`; `#Formula` selects
the formulation. Triangle meshes 2D and writes an `.npz` cache next to the
`.poly`; TetGen meshes 3D on every run and never caches (matching the FES C++
binary). Native mesher sources are expected under `dep/` and are not vendored as
binaries.

**Bundled modules** (independent of the FEM pipeline, MIT licensed, only core
deps): `mm` (modal matching), `n2f` (near-field → far-field Huygens/Kottler
transforms), `rfm` (Touchstone I/O, Vector Fitting, pole-residue macromodeling,
passivity). Each has its own README and suite under `tests/`.

## Repository conventions

- Angles in radians for numerical solvers; degrees for excitation helpers
  (relevant to `n2f`).
- Surface patches must be closed with outwardly directed normals for
  Huygens/Kottler integrals (`n2f`).
- `.poly` format: TetGen PLC sections plus custom trailing sections; `#Regions`
  (alias `#Solids`) materials match mesh region attributes by line order;
  `#Boundaries` carries the boundary label.
- Do not add per-package license copies — the root `LICENSE` covers ElectroSoft
  and the bundled `mm`, `n2f`, `rfm` packages.
- No `.cursor/` or `.github/copilot-instructions.md` rule files exist in this
  repo.

## CI

`.github/workflows/build.yml` runs on push/PR to `main` across
`ubuntu-latest` × `macos-latest` and Python `3.10` / `3.12` / `3.13`:

1. `actions/checkout@v7` with `submodules: true`
2. `actions/setup-python@v7` with `cache-dependency-path: setup.py`
3. Install: `pip install --upgrade pip`, `pip install -e . pytest ruff`
4. Build native meshers: `python tools/build_triangle.py`,
   `python tools/build_tetgen.py`
5. Lint: `ruff check fes tools tests mm n2f rfm setup.py`
6. Unit tests: `python -m pytest -q`

Tests marked `triangle`/`tetgen` (see `pytest.ini`) are skipped unless the
native shared libraries are built. Mesh caches are gitignored and generated
locally on first run.