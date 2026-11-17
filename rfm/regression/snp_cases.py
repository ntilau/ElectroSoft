"""Fixture definitions and shared pipeline for the ``.sNp`` regression tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from snp_circuits import (
    active_gain_two_port,
    rl_one_port,
    series_shunt_two_port,
    star_network,
)

FREQ_HZ = np.geomspace(1e6, 1e10, 41)
BANDWIDTH_RAD = 2.0 * np.pi * float(FREQ_HZ[-1])
DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

_UNIT_TO_HZ = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


@dataclass(frozen=True)
class SnpCase:
    """One static TouchStone fixture and its expected macromodel fit."""

    name: str
    filename: str
    nports: int
    r0: float
    order: int
    passive: bool
    kind: str
    fmt: str = "RI"
    unit: str = "Hz"
    parameters: Dict[str, float] = field(default_factory=dict)


CASES: Tuple[SnpCase, ...] = (
    SnpCase(
        name="rl_50ohm",
        filename="s1p/rl_50ohm.s1p",
        nports=1,
        r0=50.0,
        order=4,
        passive=True,
        kind="rl",
        parameters={"R": 10.0, "L": 2e-9},
    ),
    SnpCase(
        name="series_shunt_ri_hz",
        filename="s2p/series_shunt_ri_hz.s2p",
        nports=2,
        r0=50.0,
        order=6,
        passive=True,
        kind="series_shunt",
        parameters={"R": 5.0, "L": 1e-9, "C": 1e-12},
    ),
    SnpCase(
        name="series_shunt_ma_ghz",
        filename="s2p/series_shunt_ma_ghz.s2p",
        nports=2,
        r0=50.0,
        order=6,
        passive=True,
        kind="series_shunt",
        fmt="MA",
        unit="GHz",
        parameters={"R": 5.0, "L": 1e-9, "C": 1e-12},
    ),
    SnpCase(
        name="series_shunt_db_ghz",
        filename="s2p/series_shunt_db_ghz.s2p",
        nports=2,
        r0=50.0,
        order=6,
        passive=True,
        kind="series_shunt",
        fmt="DB",
        unit="GHz",
        parameters={"R": 5.0, "L": 1e-9, "C": 1e-12},
    ),
    SnpCase(
        name="series_shunt_75ohm",
        filename="s2p/series_shunt_75ohm.s2p",
        nports=2,
        r0=75.0,
        order=6,
        passive=True,
        kind="series_shunt",
        parameters={"R": 5.0, "L": 1e-9, "C": 1e-12},
    ),
    SnpCase(
        name="star_3port",
        filename="s3p/star_3port.s3p",
        nports=3,
        r0=50.0,
        order=6,
        passive=True,
        kind="star",
        unit="GHz",
        parameters={"R": 5.0, "L": 1e-9, "C": 1e-12},
    ),
    SnpCase(
        name="star_4port",
        filename="s4p/star_4port.s4p",
        nports=4,
        r0=50.0,
        order=6,
        passive=True,
        kind="star",
        parameters={"R": 5.0, "L": 1e-9, "C": 1e-12},
    ),
    SnpCase(
        name="active_gain",
        filename="s2p/active_gain.s2p",
        nports=2,
        r0=50.0,
        order=2,
        passive=False,
        kind="active",
        parameters={"gain": 2.0},
    ),
)


def expected_scattering(case: SnpCase) -> np.ndarray:
    """Return the analytic S-matrix oracle for *case* at ``FREQ_HZ``."""
    if case.kind == "rl":
        return rl_one_port(FREQ_HZ, case.r0, **case.parameters)
    if case.kind == "series_shunt":
        return series_shunt_two_port(FREQ_HZ, case.r0, **case.parameters)
    if case.kind == "star":
        return star_network(FREQ_HZ, case.nports, case.r0, **case.parameters)
    if case.kind == "active":
        return active_gain_two_port(FREQ_HZ, **case.parameters)
    raise ValueError(f"unknown fixture kind {case.kind!r}")


def _format_value(value: complex, fmt: str) -> Tuple[float, float]:
    value = complex(value)
    if fmt == "RI":
        return value.real, value.imag
    magnitude = abs(value)
    angle = float(np.degrees(np.angle(value)))
    if fmt == "MA":
        return magnitude, angle
    if fmt == "DB":
        decibels = -float("inf") if magnitude == 0.0 else 20.0 * np.log10(magnitude)
        return decibels, angle
    raise ValueError(f"unknown TouchStone format {fmt!r}")


def _format_number(value: float) -> str:
    return f"{float(value):.17g}"


def touchstone_text(case: SnpCase) -> str:
    """Render *case* as TouchStone text without using the package's writer."""
    scattering = expected_scattering(case)
    scale = _UNIT_TO_HZ[case.unit]
    frequencies = FREQ_HZ / scale
    records = []
    for index, frequency in enumerate(frequencies):
        fields = [_format_number(frequency)]
        # TouchStone stores S column by column: S11, S21, ..., S12, S22, ...
        for input_port in range(case.nports):
            for output in range(case.nports):
                first, second = _format_value(
                    scattering[index, output, input_port], case.fmt)
                fields.extend([_format_number(first),
                               _format_number(second)])
        records.append(" ".join(fields))
    header = [
        f"! idempy regression fixture: {case.name}",
        "! generated from the closed-form network in snp_circuits.py",
        f"# {case.unit} S {case.fmt} R {_format_number(case.r0)}",
    ]
    return "\n".join(header + records) + "\n"


def write_fixture(case: SnpCase, data_dir: Path = DATA_DIR) -> Path:
    """Write *case* to its fixed path and return that path."""
    path = Path(data_dir) / case.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(touchstone_text(case), encoding="utf-8")
    return path


def write_all_fixtures(data_dir: Path = DATA_DIR) -> Tuple[Path, ...]:
    """Regenerate every static ``.sNp`` fixture."""
    return tuple(write_fixture(case, data_dir) for case in CASES)


def fixture_path(case: SnpCase, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / case.filename


def load_dum(case: SnpCase, data_dir: Path = DATA_DIR):
    """Import *case* through the public ``TouchStone2DUM`` API."""
    from idempy import TouchStone2DUM

    path = fixture_path(case, data_dir)
    return TouchStone2DUM(str(path.parent), path.name)


def fit_case(case: SnpCase, data_dir: Path = DATA_DIR) -> dict:
    """Run the full ``.sNp -> DUM -> RFM`` conversion for *case*.

    Returns a dictionary with the imported data, fitted model, fitted
    responses, error, passivity verdict and re-evaluated model responses.
    """
    from idempy import (
        CheckHamiltonian,
        CheckMODPassivity,
        FDVF,
        MOD2DUM,
    )

    dum = load_dum(case, data_dir)
    model, fitted, error = FDVF(
        dum, Order=case.order, Bandwidth=BANDWIDTH_RAD)
    is_passive, compensated = CheckHamiltonian(model)
    model_dum = MOD2DUM(model, dum)
    compensated_dum = None if is_passive else MOD2DUM(compensated, dum)
    return {
        "dum": dum,
        "model": model,
        "fitted": fitted,
        "error": float(error),
        "is_passive": bool(is_passive),
        "compensated": compensated,
        "model_response": np.asarray(model_dum.S),
        "compensated_response": (None if compensated_dum is None
                                 else np.asarray(compensated_dum.S)),
        "model_check": int(CheckMODPassivity(model)),
        "compensated_check": int(CheckMODPassivity(compensated)),
    }
