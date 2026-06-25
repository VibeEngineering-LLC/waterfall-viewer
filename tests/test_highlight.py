import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.view3d import Waterfall3DView
from awf.ui.panels import HeatmapPanel


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=20, nc=40, t_step=2.0):
    counts = np.random.RandomState(0).poisson(50, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch keV
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


# ---------- 3D ----------
def test_view3d_highlight_off_by_default(app):
    v = Waterfall3DView()
    assert v._highlight_on is False


def test_view3d_highlight_mask_marks_channel(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=20, nc=40))
    v.set_energy_lines([(10.0, "#ff0000", "X-10")])
    mask = v._highlight_mask(v._ch_centers, v._nc)
    assert mask.any()
    assert mask[10]  # центр полосы на энергии 10 кэВ -> канал 10


def test_view3d_highlight_enable_recolors_and_keeps_rays(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=20, nc=40))
    v.set_energy_lines([(10.0, "#ff0000", "X"), (25.0, "#00ff00", "Y")])
    v.set_highlight_enabled(True)
    assert v._highlight_on is True
    assert v._surface is not None
    assert len(v._ray_items) == 2  # лучи энергий не теряются при ре-рендере


def test_view3d_highlight_out_of_range_no_mask(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=20, nc=40))  # энергии 0..39 кэВ
    v.set_energy_lines([(999.0, "#ff0000", "far")])
    mask = v._highlight_mask(v._ch_centers, v._nc)
    assert not mask.any()


# ---------- 2D ----------
def test_heatmap_highlight_off_no_items(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=20, nc=40))
    h.set_energy_lines([(10.0, "#ff0000", "X")])
    assert len(h._hl_items) == 0
    assert h._img.opacity() == 1.0


def test_heatmap_highlight_on_dims_and_marks(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=20, nc=40))
    h.set_energy_lines([(10.0, "#ff0000", "X"), (25.0, "#00ff00", "Y")])
    h.set_highlight_enabled(True)
    assert h._img.opacity() < 1.0   # база приглушена
    assert len(h._hl_items) == 2    # два столбца-маркера


def test_heatmap_highlight_out_of_range_skipped(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=20, nc=40))
    h.set_energy_lines([(999.0, "#ff0000", "far"), (10.0, "#00ff00", "X")])
    h.set_highlight_enabled(True)
    assert len(h._hl_items) == 1    # только энергия в диапазоне


def test_heatmap_highlight_toggle_off_restores(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=20, nc=40))
    h.set_energy_lines([(10.0, "#ff0000", "X")])
    h.set_highlight_enabled(True)
    assert len(h._hl_items) == 1
    h.set_highlight_enabled(False)
    assert len(h._hl_items) == 0
    assert h._img.opacity() == 1.0