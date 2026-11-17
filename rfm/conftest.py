"""Shared fixtures and helpers for the idempy test suite."""

from __future__ import annotations

import os

import numpy as np
import pytest

# Make the package importable when pytest is invoked from the repository root
# without an editable install.
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from idempy import DUM, RandomPassiveDUM  # noqa: E402


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def frequency_case():
    """Random passive two-port dataset and the exact model behind it."""
    return RandomPassiveDUM(
        type="frequency", Nports=2, Order=6, Bandwidth=5e8, seed=1
    )


@pytest.fixture(scope="session")
def time_case():
    """Random passive two-port time-domain dataset and its exact model."""
    return RandomPassiveDUM(
        type="time", Nports=2, Order=6, Bandwidth=5e8, seed=1
    )


@pytest.fixture()
def frequency_dum(frequency_case):
    return frequency_case[0]


@pytest.fixture()
def frequency_mod(frequency_case):
    return frequency_case[1]


@pytest.fixture()
def time_dum(time_case):
    return time_case[0]


def global_relative_error(reference, fitted) -> float:
    """Max-norm relative error between two arrays, ignoring NaN entries."""
    reference = np.asarray(reference)
    fitted = np.asarray(fitted)
    mask = np.isfinite(reference) & np.isfinite(fitted)
    if not np.any(mask):
        return float("nan")
    denominator = float(np.max(np.abs(reference[mask])))
    if denominator == 0.0:
        denominator = 1.0
    return float(np.max(np.abs(fitted[mask] - reference[mask]))) / denominator


def make_siso_dum(seed=1, order=3, bandwidth=1e8, nf=80):
    """Small single-port frequency-domain dataset for fast algorithm tests."""
    return RandomPassiveDUM(
        type="frequency", Nports=1, Order=order, Bandwidth=bandwidth,
        seed=seed,
    )[0]


# ---------------------------------------------------------------------------
# Synthetic CST / FDTD transient runs
# ---------------------------------------------------------------------------


def _gaussian(t, delay, width):
    return np.exp(-((t - delay) / width) ** 2)


def write_port_tables(directory, rootname, nports, extension=".SIG",
                      kind="cst"):
    """Write one canonical per-excitation port table per input port.

    Every file uses the ``source_responses`` layout
    ``time, a_j, b_1, ..., b_N`` with smooth analytic waveforms.
    """
    os.makedirs(directory, exist_ok=True)
    t = np.linspace(0.0, 4e-9, 201)
    width = 0.15e-9
    for j in range(nports):
        delay = 0.5e-9 + 0.1e-9 * j
        a = _gaussian(t, delay, width)
        columns = [t, a]
        for i in range(nports):
            amplitude = 0.4 if i == j else 0.2 / (1 + abs(i - j))
            columns.append(amplitude * _gaussian(
                t, delay + 0.05e-9 * (i + 1), width * 1.4
            ))
        table = np.column_stack(columns)
        folder = os.path.join(directory, f"{rootname}_IN{j + 1:02d}")
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(
            folder, f"{rootname}_IN{j + 1:02d}_PORTS{extension}"
        )
        header = (
            "CST transient port signals\n"
            "time a b1 b2\n"
        ) if kind == "cst" else (
            "FDTD port signals\n"
        )
        np.savetxt(filename, table, header=header, comments="% ",
                   delimiter=" ")
    return rootname


@pytest.fixture()
def cst_project(tmp_path):
    root = write_port_tables(str(tmp_path), "board", 2, ".SIG", "cst")
    return str(tmp_path), root


@pytest.fixture()
def fdtd_project(tmp_path):
    root = write_port_tables(str(tmp_path), "antenna", 2, ".OUT", "fdtd")
    return str(tmp_path), root
