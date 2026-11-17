"""Tests for dataset/model inspection strings and Matplotlib views."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from idempy import (
    DUMInfo,
    MODInfo,
    PlotDUM,
    RandomPassiveDUM,
    ShowDUM,
    ShowDUMvsDUM,
    ShowDUMvsMOD,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_duminfo_frequency(frequency_dum):
    lines = DUMInfo(frequency_dum)
    text = "\n".join(lines)
    assert "Dataset name: RandomPassiveDUM" in text
    assert "Number of ports: 2" in text
    assert "Valid port responses: 4 of 4" in text
    assert "Representation: scattering" in text
    assert "Available data: frequency" in text
    assert "Frequency samples: 120" in text
    assert "Frequency range: 0 to 5e+09" in text
    assert "Reciprocal: no" in text


def test_duminfo_time(time_dum):
    text = "\n".join(DUMInfo(time_dum))
    assert "Available data: time" in text
    assert "Time samples: 5001" in text
    assert "Time excitations: 4" in text
    assert "Time step:" in text
    assert "Time duration:" in text


def test_modinfo(frequency_mod):
    text = "\n".join(MODInfo(frequency_mod))
    assert "Model name: RandomPassiveDUM" in text
    assert "Number of ports: 2" in text
    assert "Model order: 6" in text
    assert "Representation: scattering" in text
    assert "Storage: full" in text
    assert "Representations: state space (ABCD), poles/residues" in text
    assert "Real poles: 2" in text
    assert "Complex pole pairs: 2" in text
    assert "Stability: stable" in text
    assert "State-space A: 6 x 6" in text


def test_modinfo_incomplete_model():
    from idempy import MOD

    text = "\n".join(MODInfo(MOD()))
    assert "Number of ports: 0" in text
    assert "Model order: 0" in text
    assert "Representations: none" in text


def test_plot_dum_time_and_frequency(frequency_dum, time_dum):
    assert len(PlotDUM(time_dum, "time")) == 2
    assert len(PlotDUM(time_dum)) == 2
    assert len(PlotDUM(frequency_dum, "frequency")) == 2
    assert len(PlotDUM(frequency_dum, "all", "frequency")) == 2
    # Mismatched domain returns no figures instead of raising.
    assert PlotDUM(frequency_dum, "time") == []
    assert PlotDUM(time_dum, "frequency") == []


def test_show_dum_with_explicit_domain(frequency_dum, time_dum):
    assert len(ShowDUM(frequency_dum, "frequency")) == 2
    assert len(ShowDUM(time_dum, "time")) == 2


def test_show_dum_vs_dum(frequency_dum, time_dum):
    assert len(ShowDUMvsDUM(frequency_dum, frequency_dum)) == 2
    assert len(ShowDUMvsDUM(time_dum, time_dum)) == 2
    # Incompatible domains produce no plot.
    assert ShowDUMvsDUM(frequency_dum, time_dum) == []


def test_show_dum_vs_mod(frequency_dum, frequency_mod, time_case):
    time_dum, time_mod = time_case
    assert len(ShowDUMvsMOD(frequency_dum, frequency_mod)) == 2
    assert len(ShowDUMvsMOD(time_dum, time_mod)) == 2
    assert len(ShowDUMvsMOD(time_dum, time_mod,
                            TimeExtensionFactor=2.0)) == 2


def test_plot_dum_slide_returns_figures(frequency_dum):
    # Agg's ``show`` is a no-op, so the slideshow path is testable.
    figures = PlotDUM(frequency_dum, "frequency", "slide")
    assert len(figures) == 4
    assert all(figure.axes for figure in figures)
