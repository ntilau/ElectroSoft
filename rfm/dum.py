"""
DUM (Descriptor Unit Model) class for the rfm RFM implementation.

This class represents the Descriptor Unit Model structure used to store
time and/or frequency domain multiport data from raw simulations.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any

class DUM:
    """
    Descriptor Unit Model (DUM) class for storing multiport frequency/time domain data.

    Attributes
    ----------
    Nports : int
        Number of ports
    isfreq : int
        Flag indicating if data is in frequency domain (1) or not (0)
    istime : int
        Flag indicating if data is in time domain (1) or not (0)
    f : np.ndarray
        Frequency vector (if isfreq=1)
    t : np.ndarray
        Time vector (if istime=1)
    S : np.ndarray
        S-parameter matrix of shape (Nports, Nports, Nfreq) or (Nports, Nports, Ntime)
    Y : np.ndarray
        Admittance matrix (optional)
    Z : np.ndarray
        Impedance matrix (optional)
    """

    def __init__(self):
        """Initialize an empty DUM object."""
        self.Nports = 0
        self.isfreq = 0
        self.istime = 0
        self.f = np.array([])
        self.t = np.array([])
        self.S = np.array([])
        self.Y = np.array([])
        self.Z = np.array([])

    def __repr__(self):
        """String representation of the DUM object."""
        return f"DUM(Nports={self.Nports}, isfreq={self.isfreq}, istime={self.istime}, f_shape={getattr(self.f, 'shape', 'N/A')}, S_shape={getattr(self.S, 'shape', 'N/A')})"

class RandomPassiveDUM(DUM):
    """
    Minimal implementation of a random passive DUM for testing purposes.

    NOTE: This class is a mock and does not generate real physical data.
    It satisfies the constructor signature used by the test suite.
    """
    def __init__(self, type: str, Nports: int, Order: int, Bandwidth: float, seed: int, LossFactor: float = 1.0, Mismatch: float = 1.0):
        super().__init__()
        self.Nports = Nports
        self.isfreq = 1 if type == "frequency" else 0
        self.istime = 1 if type == "time" else 0
        self.f = np.linspace(0, Bandwidth, Order * Nports)
        self.t = np.linspace(0, Bandwidth / 1e9, Order * Nports)
        self.S = np.random.rand(Nports, Nports, Order * Nports) * Mismatch
        # Other attributes (Y, Z) remain empty

    def __repr__(self):
        return f"RandomPassiveDUM(type={self.isfreq}, Nports={self.Nports}, Order={self.Nports})"