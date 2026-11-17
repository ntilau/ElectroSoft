"""Closed-form S-parameters for the ``.sNp`` regression fixtures.

These functions are the independent oracle for the regression tests: the
fixtures are written from these equations, and the tests recompute the same
values instead of trusting the importer under test.  All networks are built
from positive resistors, inductors and capacitors, so their scattering
matrices are passive except for the deliberately active gain fixture.
"""

from __future__ import annotations

import numpy as np


def rl_one_port(frequency_hz, R0=50.0, R=10.0, L=2e-9):
    """Series R-L load against a real reference resistance ``R0``.

    Returns an array with shape ``(Nfreq, 1, 1)``.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    omega = 2.0 * np.pi * frequency_hz
    impedance = R + 1j * omega * L
    gamma = (impedance - R0) / (impedance + R0)
    return gamma.reshape(-1, 1, 1)


def series_shunt_two_port(frequency_hz, R0=50.0, R=5.0, L=1e-9, C=1e-12):
    """Symmetric two-port: series R-L between the ports, shunt C at each.

    The ABCD chain is ``shunt(C) * series(R-L) * shunt(C)``.  Returns an
    array with shape ``(Nfreq, 2, 2)`` using ``S[:, i, j]`` for the response
    from input port ``j`` to output port ``i``.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    omega = 2.0 * np.pi * frequency_hz
    impedance = R + 1j * omega * L
    admittance = 1j * omega * C

    a = 1.0 + impedance * admittance
    b = impedance
    c = 2.0 * admittance + impedance * admittance ** 2
    d = a
    denominator = a + b / R0 + c * R0 + d

    s11 = (a + b / R0 - c * R0 - d) / denominator
    s21 = 2.0 / denominator
    scattering = np.empty((frequency_hz.size, 2, 2), dtype=complex)
    scattering[:, 0, 0] = s11
    scattering[:, 1, 0] = s21
    scattering[:, 0, 1] = s21
    scattering[:, 1, 1] = s11
    return scattering


def star_network(frequency_hz, Nports=3, R0=50.0, R=5.0, L=1e-9,
                 C=1e-12):
    """Symmetric ``Nports``-port star with a common node.

    Each port has a shunt capacitor to ground and a series R-L branch to the
    common star node.  The port admittance matrix is

    ``Y = jwC I + (I - J/N) / (R + jwL)``

    and ``S = (I - R0 Y) (I + R0 Y)^-1``.  Returns ``(Nfreq, N, N)``.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    nports = int(Nports)
    if nports < 2:
        raise ValueError("a star network needs at least two ports")
    identity = np.eye(nports)
    all_ones = np.ones((nports, nports))
    scattering = np.empty((frequency_hz.size, nports, nports), dtype=complex)

    for index, frequency in enumerate(frequency_hz):
        omega = 2.0 * np.pi * frequency
        impedance = R + 1j * omega * L
        admittance = 1j * omega * C
        port_admittance = (admittance * identity
                           + (identity - all_ones / nports) / impedance)
        lhs = identity + R0 * port_admittance
        rhs = identity - R0 * port_admittance
        # S = rhs @ inv(lhs), computed transposed so we use solve().
        scattering[index] = np.linalg.solve(lhs.T, rhs.T).T

    return scattering


def active_gain_two_port(frequency_hz, gain=2.0):
    """Deliberately non-passive unilateral two-port with constant gain."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    scattering = np.zeros((frequency_hz.size, 2, 2), dtype=complex)
    scattering[:, 1, 0] = complex(gain)
    return scattering


def max_singular_value(scattering):
    """Return the largest singular value at each frequency."""
    values = np.linalg.svd(np.asarray(scattering), compute_uv=False)
    return np.max(values, axis=-1)
