"""UI-tests for Task #104: dose-rate overlay on SlicePanel time-series plot (RadiaCode .rcspg only)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.panels import SlicePanel


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=20, nc=32, t_step=1.0, source_path=None):
    """Synthetic spectrogram; source_path controls the RadiaCode gate."""
    rng = np.random.RandomState(42)
    counts = rng.poisson(100, size=(ns, nc)).astype(np.int64)
    # calibration: E(ch) = 3.94 + 2.37*ch (like RC-103, simplified)
    cal = Calibration(coeffs=[3.94, 2.37])
    t = np.arange(ns, dtype=np.float64) * t_step
    lt = np.full(ns, t_step, dtype=np.float64)
    return Spectrogram(
        counts=counts, calibration=cal,
        time_offsets_s=t, real_time_s=lt, live_time_s=lt,
        source_path=source_path,
    )


def _curve_empty(item) -> bool:
    x, _ = item.getData()
    return x is None or len(x) == 0


# ---------- gate: dose overlay appears only for .rcspg ----------

def test_dose_overlay_created_for_rcspg(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(source_path="scan.rcspg"))
    x, y = p._dose_curve.getData()
    assert x is not None and len(x) == 20   # length == n_slices

def test_dose_overlay_absent_for_n42(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(source_path="scan.n42"))
    assert _curve_empty(p._dose_curve)

def test_dose_overlay_absent_for_aswf(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(source_path="scan.aswf"))
    assert _curve_empty(p._dose_curve)

def test_dose_overlay_absent_when_source_path_none(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(source_path=None))
    assert _curve_empty(p._dose_curve)


# ---------- right axis label contains 'доз' ----------

def test_dose_axis_label_contains_dose(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(source_path="data.rcspg"))
    label_text = p._dose_axis.labelText
    assert "доз" in label_text.lower()


# ---------- set_dose_overlay toggles visibility ----------

def test_set_dose_overlay_off_hides_curve(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(source_path="data.rcspg"))
    assert not _curve_empty(p._dose_curve)   # visible by default
    p.set_dose_overlay(False)
    assert _curve_empty(p._dose_curve)


def test_set_dose_overlay_on_restores_curve(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(source_path="data.rcspg"))
    p.set_dose_overlay(False)
    p.set_dose_overlay(True)
    x, _ = p._dose_curve.getData()
    assert x is not None and len(x) == 20


# ---------- dose array has same length as n_slices ----------

def test_dose_array_length_matches_n_slices(app):
    ns = 15
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=ns, source_path="meas.rcspg"))
    assert p._dose is not None
    assert len(p._dose) == ns


# ---------- non-rcspg: dose is None ----------

def test_dose_none_for_non_rcspg(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(source_path="file.n42"))
    assert p._dose is None
