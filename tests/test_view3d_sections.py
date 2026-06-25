import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.view3d import Waterfall3DView, SectionControls, PLANE_AXES


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=30, nc=40, t_step=2.0):
    counts = np.random.RandomState(0).poisson(50, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch keV
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_view_constructs_with_planes_hidden(app):
    v = Waterfall3DView()
    assert len(v._planes) == 6  # 2 на ось × 3 оси
    for entry in v._planes.values():
        assert entry["mesh"].visible() is False
        assert entry["line"].visible() is False


def test_set_spectrogram_records_geometry(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=40), max_time=400, max_chan=512)
    assert v._nt == 30 and v._nc == 40
    assert v._height_scale == pytest.approx(0.25 * 40)
    assert v._t_centers[0] == pytest.approx(0.0)
    assert v._t_centers[-1] == pytest.approx(58.0)   # 29*2 s
    assert v._ch_centers[0] == pytest.approx(0.0)
    assert v._ch_centers[-1] == pytest.approx(39.0)


def test_plane_value_real_units(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=40), max_time=400, max_chan=512)
    tval, tunit = v.plane_value("time", 0.5)
    assert tunit == "с" and 26.0 <= tval <= 32.0
    eval_, eunit = v.plane_value("energy", 0.25)
    assert eunit == "кэВ" and 8.0 <= eval_ <= 12.0
    cval, cunit = v.plane_value("counts", 1.0)
    assert "отсч" in cunit and cval == pytest.approx(float(v._sg.counts.max()))


def test_set_plane_visibility_and_profile(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)
    # время/энергия: и плоскость, и профиль-линия видны
    v.set_plane("time", 0, 0.5, True)
    assert v._planes[("time", 0)]["mesh"].visible() is True
    assert v._planes[("time", 0)]["line"].visible() is True
    v.set_plane("energy", 1, 0.7, True)
    assert v._planes[("energy", 1)]["line"].visible() is True
    # отсчёты: горизонтальная плоскость есть, профиль-контур не строится
    v.set_plane("counts", 0, 0.5, True)
    assert v._planes[("counts", 0)]["mesh"].visible() is True
    assert v._planes[("counts", 0)]["line"].visible() is False
    # скрытие
    v.set_plane("time", 0, 0.5, False)
    assert v._planes[("time", 0)]["mesh"].visible() is False


def test_plane_position_is_lod_aware(app):
    # 100 срезов прорежены до 10 бинов — позиция плоскости всё равно в реальных секундах
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=100, nc=50, t_step=1.0), max_time=10, max_chan=512)
    assert v._nt == 10            # прорежено
    lo, _ = v.plane_value("time", 0.0)
    hi, _ = v.plane_value("time", 1.0)
    assert lo < hi
    # верх диапазона близок к реальному максимуму времени (99 с), не к индексу бина (9)
    assert hi > 80.0


def test_section_controls_emits(app):
    sc = SectionControls()
    got = []
    sc.planeChanged.connect(lambda a, s, f, vis: got.append((a, s, vis)))
    sc._rows[("energy", 0)]["check"].setChecked(True)
    sc._rows[("time", 1)]["slider"].setValue(500)
    assert any(a == "energy" and s == 0 and vis is True for a, s, vis in got)
    assert any(a == "time" and s == 1 for a, s, vis in got)


def test_section_controls_emit_all(app):
    sc = SectionControls()
    got = []
    sc.planeChanged.connect(lambda a, s, f, vis: got.append((a, s)))
    sc.emit_all()
    assert len(got) == len(PLANE_AXES) * 2
    sc.set_value_label("time", 0, "12.3 с")
    assert sc._rows[("time", 0)]["label"].text() == "12.3 с"