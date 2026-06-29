"""UI-тесты Задача #110/#111/#112/#114: поиск пиков на 3D-водопаде + PeaksPanel.

#110: find_peaks на 3D-водопаде, зелёные гребни по гребню рельефа.
#114: FWHM(E)-модель (default_fwhm_model, R=7%@662 кэВ) вместо константы 8.0; σ-сеттер.
#111: PeaksPanel в QDockWidget — заполняется из _found_peaks(), sigmaChanged->set_peak_sigma.
#112: peak_time_mask — гребень только в зоне присутствия пика; при постоянном источнике fallback.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.view3d import Waterfall3DView, PEAK_FWHM_CHANNELS


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


NC = 256
CENTERS = (60, 130, 200)
HEIGHTS = (4000.0, 9000.0, 16000.0)   # строго возрастают
KEV_PER_CH = 2.37
E0 = 3.94


def _target_spectrum():
    """Пологий фон + три гауссианы FWHM=PEAK_FWHM_CHANNELS — целевой интегральный спектр."""
    ch = np.arange(NC, dtype=np.float64)
    spec = 200.0 - 0.2 * ch          # фон, всюду > 0
    sigma = PEAK_FWHM_CHANNELS / 2.355
    for c, h in zip(CENTERS, HEIGHTS):
        spec = spec + h * np.exp(-((ch - c) ** 2) / (2.0 * sigma ** 2))
    return spec


def _make_sg(ns=10):
    """Спектрограмма, сумма срезов которой ≈ три гауссианы на пологом фоне."""
    target = _target_spectrum()
    per_slice = np.round(target / ns).astype(np.int64)
    counts = np.tile(per_slice, (ns, 1))           # (ns, NC), total_spectrum ≈ target
    cal = Calibration(coeffs=[E0, KEV_PER_CH])
    t = np.arange(ns, dtype=np.float64)
    lt = np.ones(ns, dtype=np.float64)
    return Spectrogram(counts=counts, calibration=cal,
                       time_offsets_s=t, real_time_s=lt, live_time_s=lt)


def _loaded(ns=10):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns), max_time=400, max_chan=512)
    return v


def test_peaks_hidden_by_default(app):
    v = _loaded()
    assert v._peak_ridge_items == []


def test_peak_search_marks_three_peaks(app):
    v = _loaded()
    v.set_peak_search(True)
    assert len(v._peak_ridge_items) == 3


def test_found_peak_energies_match_calibration(app):
    v = _loaded()
    expected = sorted(E0 + KEV_PER_CH * c for c in CENTERS)
    got = sorted(v._found_peak_energies())
    assert len(got) == len(expected)
    for e, g in zip(expected, got):
        assert abs(g - e) <= 3.0 * KEV_PER_CH   # в пределах ~3 каналов


def test_peak_search_off_clears_ridges(app):
    v = _loaded()
    v.set_peak_search(True)
    assert v._peak_ridge_items
    v.set_peak_search(False)
    assert v._peak_ridge_items == []


def test_peaks_recomputed_on_new_spectrum(app):
    v = _loaded()
    v.set_peak_search(True)
    assert len(v._peak_ridge_items) == 3
    v.set_spectrogram(_make_sg(), max_time=400, max_chan=512)   # новый объект -> перестройка
    assert len(v._peak_ridge_items) == 3


def test_peak_search_no_spectrum_is_safe(app):
    v = Waterfall3DView()
    v.set_peak_search(True)         # спектр не загружен — не должно падать
    assert v._peak_ridge_items == []


def test_peak_ridge_runs_along_time_on_relief(app):
    from OpenGL.GL import GL_DEPTH_TEST
    v = _loaded()
    v.set_peak_search(True)
    item = v._peak_ridge_items[0]
    pos = np.asarray(item.pos)
    nt, nc = v._nt, v._nc
    assert pos.shape == (nt, 3)                      # точка на каждый бин времени
    # хребет идёт вдоль оси времени: X = индекс-времени − nt/2, Y постоянен (один канал энергии)
    assert pos[:, 0] == pytest.approx(np.arange(nt) - nt / 2.0)
    assert np.allclose(pos[:, 1], pos[0, 1])
    # Z следует за гребнем рельефа на канале пика (+анти-z-fighting подъём)
    jc = int(round(pos[0, 1] + nc / 2.0))
    assert 0 <= jc < nc
    lift = 0.01 * float(v._height_scale)
    assert pos[:, 2] == pytest.approx(v._z_surface[:, jc] + lift)
    # depth-occlusion (#95): за более высоким рельефом линия перекрывается
    assert item._GLGraphicsItem__glOpts.get(GL_DEPTH_TEST) is True


# ===== Задача #114: _found_peaks(), set_peak_sigma() =====

def test_found_peaks_returns_founpeak_objects(app):
    """_found_peaks() возвращает FoundPeak-объекты (не просто энергии)."""
    v = _loaded()
    peaks = v._found_peaks()
    assert len(peaks) == 3
    pk = peaks[0]
    for attr in ('energy', 'channel', 'significance', 'fwhm_channels'):
        assert hasattr(pk, attr)


def test_found_peak_energies_derived_from_found_peaks(app):
    """_found_peak_energies() производная от _found_peaks(): единый источник."""
    v = _loaded()
    energies = v._found_peak_energies()
    peaks = v._found_peaks()
    assert len(energies) == len(peaks)
    for e, pk in zip(energies, peaks):
        assert abs(e - pk.energy) < 1e-9


def test_set_peak_sigma_reduces_peaks(app):
    """Больший σ → не больше пиков (монотонность чувствительности)."""
    v = _loaded()
    n_low = len(v._found_peaks())
    v.set_peak_sigma(5.0)
    n_high = len(v._found_peaks())
    assert n_high <= n_low


def test_set_peak_sigma_rebuilds_ridges(app):
    """set_peak_sigma при включённом поиске пересчитывает гребни.
    σ=1000 заведомо выше значимости синтетических пиков (~330σ max)."""
    v = _loaded()
    v.set_peak_search(True)
    n_before = len(v._peak_ridge_items)
    v.set_peak_sigma(1000.0)
    assert len(v._peak_ridge_items) == 0
    v.set_peak_sigma(3.0)
    assert len(v._peak_ridge_items) == n_before


def test_peak_sigma_default_value(app):
    """Дефолтный порог σ = 3.0."""
    from awf.ui.view3d import _PEAK_SIGMA_DEFAULT
    v = Waterfall3DView()
    assert v._peak_sigma == _PEAK_SIGMA_DEFAULT == 3.0


# ===== Задача #119: поиск пиков ограничен пределом отображения (3000 кэВ) =====

def test_found_peaks_capped_at_max_energy(app):
    """Задача #119: пики выше предела отображения (#78, 3000 кэВ) не возвращаются
    _found_peaks() — ни в гребни 3D, ни в таблицу (#111). Калибровка 20 кэВ/канал кладёт
    пики каналов 60/130/200 на ~1204/2604/4004 кэВ → пик 4004 кэВ обязан быть отсечён."""
    from awf.ui.view3d import _MAX_ENERGY_KEV
    target = _target_spectrum()
    ns = 10
    per_slice = np.round(target / ns).astype(np.int64)
    counts = np.tile(per_slice, (ns, 1))
    cal = Calibration(coeffs=[E0, 20.0])           # 20 кэВ/канал → канал 200 ≈ 4004 кэВ
    t = np.arange(ns, dtype=np.float64)
    lt = np.ones(ns, dtype=np.float64)
    sg = Spectrogram(counts=counts, calibration=cal,
                     time_offsets_s=t, real_time_s=lt, live_time_s=lt)
    v = Waterfall3DView()
    v.set_spectrogram(sg, max_time=400, max_chan=512)
    peaks = v._found_peaks()
    assert peaks, "пики ≤ 3000 кэВ должны остаться"
    assert all(pk.energy <= _MAX_ENERGY_KEV for pk in peaks)
    assert max(pk.energy for pk in peaks) < 3000.0     # пик 4004 кэВ отсечён


# ===== Задача #111: PeaksPanel в MainWindow =====

def test_peaks_panel_exists_in_main_window(app):
    """MainWindow имеет _peaks_panel (PeaksPanel) и _peaks_dock."""
    from awf.ui.main_window import MainWindow
    from awf.ui.peaks_panel import PeaksPanel
    win = MainWindow()
    assert hasattr(win, '_peaks_panel')
    assert isinstance(win._peaks_panel, PeaksPanel)
    assert hasattr(win, '_peaks_dock')
    win.close()


def test_peaks_panel_filled_after_redistribute(app):
    """После _redistribute PeaksPanel заполняется 3 пиками (тест-матрица)."""
    from awf.ui.main_window import MainWindow
    win = MainWindow()
    sg = _make_sg(ns=10)
    win._sg = sg
    win._redistribute()
    n = win._peaks_panel._table.rowCount()
    assert n == 3
    win.close()


def test_peaks_panel_sigma_updates_view3d(app):
    """sigmaChanged из PeaksPanel → view3d._peak_sigma обновляется."""
    from awf.ui.main_window import MainWindow
    win = MainWindow()
    win._peaks_panel.sigmaChanged.emit(4.5)
    assert win._view3d._peak_sigma == pytest.approx(4.5)
    win.close()


def test_peaks_panel_retranslate_on_lang_change(app):
    """retranslate() обновляет заголовки на EN при активном EN-языке."""
    from awf.ui.peaks_panel import PeaksPanel
    from awf.ui import i18n
    panel = PeaksPanel()
    # Прямой вызов retranslate() при EN-языке (не зависит от QSettings/сигналов)
    i18n.reset_for_tests()
    i18n._state['lang'] = i18n.LANG_EN
    panel.retranslate()
    col0 = panel._table.horizontalHeaderItem(0)
    assert col0 is not None and col0.text() == 'Energy, keV'
    i18n.reset_for_tests()


# ===== Задача #112: _draw_ridge_segments =====

def test_draw_ridge_two_segments(app):
    """#112: прерывистая маска → 2 отдельных GLLinePlotItem."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=10), max_time=400, max_chan=512)
    nt = v._nt
    xs = np.arange(nt, dtype=np.float64) - nt / 2.0
    mask = np.zeros(nt, dtype=bool)
    if nt >= 6:
        mask[0:2] = True
        mask[4:6] = True
    n0 = len(v._peak_ridge_items)
    v._draw_ridge_segments(xs, 0.0, np.zeros(nt), mask)
    assert len(v._peak_ridge_items) - n0 == 2


# ===== i18n: EN-переводы строк PeaksPanel =====

def test_i18n_peaks_strings(app):
    """Все строки PeaksPanel/#111 имеют EN-переводы."""
    from awf.ui import i18n
    keys = ['Найденные пики', 'Порог значимости, σ', 'Найдено: ',
            'Энергия, кэВ', 'Канал', 'Значимость', 'Высота', 'Площадь']
    for k in keys:
        en = i18n.TRANSLATIONS[i18n.LANG_EN].get(k)
        assert en and en != k, f"Нет EN-перевода: '{k}'"


# ===== Задача #121: гребни пиков обрезаны секущими плоскостями =====

def test_peak_ridges_clipped_to_time_plane(app):
    """#121: активная плоскость времени (slot 0) обрезает гребни — точки только в [i0, i1]."""
    v = _loaded(ns=20)
    v.set_peak_search(True)
    assert v._peak_ridge_items
    i0 = v._frac_to_index(v._t_centers, 0.5)
    v.set_plane("time", 0, 0.5, True)
    nt = v._nt
    assert v._peak_ridge_items, "гребни не должны исчезнуть полностью"
    for item in v._peak_ridge_items:
        xs_idx = np.asarray(item.pos)[:, 0] + nt / 2.0
        assert xs_idx.min() >= i0 - 0.5     # ни одной точки левее нижней границы среза


def test_peak_ridge_hidden_when_channel_outside_energy_window(app):
    """#121: плоскость энергии (верхняя граница), отсекающая канал пика, скрывает его гребень."""
    v = _loaded(ns=10)
    v.set_peak_search(True)
    n_full = len(v._peak_ridge_items)
    assert n_full == 3
    # верхняя граница окна энергии на доле 0.6 (~канал 154): верхний пик (канал 200) выпадает
    v.set_plane("energy", 1, 0.6, True)
    assert 0 < len(v._peak_ridge_items) < n_full   # часть пиков вне окна -> их гребни скрыты


def test_peak_ridges_gone_when_planes_crossed(app):
    """#121: встречные плоскости времени пересеклись (i0 > i1) -> гребней нет (как и поверхности)."""
    v = _loaded(ns=20)
    v.set_peak_search(True)
    assert v._peak_ridge_items
    v.set_plane("time", 0, 0.9, True)   # нижняя граница высоко
    v.set_plane("time", 1, 0.1, True)   # верхняя граница низко -> окно пусто
    assert v._peak_ridge_items == []


def test_peak_ridges_full_span_without_planes(app):
    """#121-регрессия: без активных плоскостей гребни идут по всей оси времени."""
    v = _loaded(ns=10)
    v.set_peak_search(True)
    nt = v._nt
    for item in v._peak_ridge_items:
        assert np.asarray(item.pos).shape == (nt, 3)   # точка на каждый бин времени