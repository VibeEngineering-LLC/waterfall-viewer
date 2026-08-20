"""#I18N-1: ретрансляция динамических подписей SlicePanel.

Заголовок панели собирается из данных (номер среза, границы ROI, суммы). Раньше
строка собиралась только в show_*(), поэтому при смене языка оставалась на
прежнем: на EN-интерфейсе висело русское «Выборка: срезы …» (скриншот оператора
2026-08-20). Тест держит единый источник формата _current_header_text().
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui import i18n
from awf.ui.panels import SlicePanel


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


@pytest.fixture(autouse=True)
def _reset_lang():
    i18n.reset_for_tests()
    yield
    i18n.reset_for_tests()


def _make_sg(ns=12, nc=20, t_step=2.0):
    counts = np.random.RandomState(7).poisson(40, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def test_integral_header_retranslates(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg())
    assert "Загружено: срезов" in p._header.text()
    i18n.set_language("en")
    p.retranslate()
    assert "Loaded: slices" in p._header.text()
    assert "Загружено" not in p._header.text()


def test_slice_header_retranslates(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg())
    p.show_time_slice(3)
    assert "Срез времени" in p._header.text()
    i18n.set_language("en")
    p.retranslate()
    assert "Time slice" in p._header.text()
    assert "#3" in p._header.text()


def test_roi_header_retranslates(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg())
    p.show_roi(2, 8, 3, 15)
    assert "Выборка: срезы" in p._header.text()
    before = p._header.text()
    i18n.set_language("en")
    p.retranslate()
    after = p._header.text()
    assert "Sample: slices" in after
    assert "Выборка" not in after
    assert "[2:8]" in after and "[3:15]" in after
    assert before != after


def test_roi_header_survives_language_round_trip(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg())
    p.show_roi(1, 5, 0, 10)
    ru_before = p._header.text()
    i18n.set_language("en")
    p.retranslate()
    i18n.set_language("ru")
    p.retranslate()
    assert p._header.text() == ru_before


def test_empty_panel_header_retranslates(app):
    p = SlicePanel()
    assert p._header.text() == "Файл не загружен"
    i18n.set_language("en")
    p.retranslate()
    assert p._header.text() == "No file loaded"
