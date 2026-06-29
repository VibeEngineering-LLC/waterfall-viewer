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
from PySide6 import QtCore, QtWidgets

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
    """retranslate() обновляет заголовки на EN при активном EN-языке.
    Задача #124: колонка 0 — чекбокс «Показать»→«Show», энергия на колонке 1."""
    from awf.ui.peaks_panel import PeaksPanel
    from awf.ui import i18n
    panel = PeaksPanel()
    # Прямой вызов retranslate() при EN-языке (не зависит от QSettings/сигналов)
    i18n.reset_for_tests()
    i18n._state['lang'] = i18n.LANG_EN
    panel.retranslate()
    col0 = panel._table.horizontalHeaderItem(panel._COL_CHECK)
    col1 = panel._table.horizontalHeaderItem(panel._COL_ENERGY)
    assert col0 is not None and col0.text() == 'Show'
    assert col1 is not None and col1.text() == 'Energy, keV'
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


# ===== Задача #123: единицы Высоты/Площади + метка временно́го окна =====

def test_peaks_panel_headers_carry_units(app):
    """#123: заголовки Высота/Площадь несут единицы — отсчёты (отсч.)."""
    from awf.ui.peaks_panel import PeaksPanel
    panel = PeaksPanel()
    headers = [panel._table.horizontalHeaderItem(c).text()
               for c in range(panel._table.columnCount())]
    assert "Высота, отсч." in headers
    assert "Площадь, отсч." in headers


def test_peaks_panel_window_info_default(app):
    """#123: до загрузки файла окно поиска = «—»."""
    from awf.ui.peaks_panel import PeaksPanel
    panel = PeaksPanel()
    assert panel._window_label.text() == "Окно поиска: —"


def test_peaks_panel_window_info_set(app):
    """#123: set_window_info(N, T) → метка «весь файл», N срезов, длительность."""
    from awf.ui.peaks_panel import PeaksPanel
    panel = PeaksPanel()
    panel.set_window_info(12, 600.0)         # 600 с = 10.0 мин
    txt = panel._window_label.text()
    assert "весь файл" in txt
    assert "12 срезов" in txt
    assert "10.0 мин" in txt


def test_peaks_panel_duration_units(app):
    """#123: длительность форматируется в с / мин / ч по порогам 60 с и 3600 с."""
    from awf.ui.peaks_panel import PeaksPanel
    panel = PeaksPanel()
    assert panel._format_duration(45.0) == "45 с"
    assert panel._format_duration(90.0) == "1.5 мин"
    assert panel._format_duration(7200.0) == "2.00 ч"


def test_peaks_panel_window_info_localized_en(app):
    """#123: при EN-языке метка окна переведена (Search window / whole file / slices)."""
    from awf.ui.peaks_panel import PeaksPanel
    from awf.ui import i18n
    panel = PeaksPanel()
    panel.set_window_info(8, 120.0)
    i18n.reset_for_tests()
    i18n._state['lang'] = i18n.LANG_EN
    panel.retranslate()
    txt = panel._window_label.text()
    i18n.reset_for_tests()
    assert "Search window" in txt and "whole file" in txt and "slices" in txt


def test_main_window_pushes_window_info(app):
    """#123: после _redistribute панель пиков получает окно (N срезов, T)."""
    from awf.ui.main_window import MainWindow
    from awf.ui import i18n
    win = MainWindow()
    i18n.reset_for_tests()          # RU независимо от сохранённого в QSettings языка
    sg = _make_sg(ns=10)
    win._sg = sg
    win._redistribute()
    txt = win._peaks_panel._window_label.text()
    assert "10 срезов" in txt and "весь файл" in txt
    win.close()
    i18n.reset_for_tests()


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


# ===== Задача #124: чекбоксы видимости + подсветка выбранного пика =====

def _three_peaks():
    """Три FoundPeak с возрастающей значимостью — для панельных тестов без 3D."""
    from awf.analysis.types import FoundPeak
    return [
        FoundPeak(channel=60.0, energy=150.0, height=400.0,
                  fwhm_channels=8.0, significance=10.0, area_estimate=8000.0),
        FoundPeak(channel=130.0, energy=312.0, height=900.0,
                  fwhm_channels=8.0, significance=20.0, area_estimate=18000.0),
        FoundPeak(channel=200.0, energy=478.0, height=1600.0,
                  fwhm_channels=8.0, significance=30.0, area_estimate=32000.0),
    ]


def test_peaks_panel_checkbox_column_default_checked(app):
    """#124: 7 колонок; колонка 0 — чекбокс «Показать», по умолчанию отмечен."""
    from awf.ui.peaks_panel import PeaksPanel
    panel = PeaksPanel()
    assert panel._table.columnCount() == 7
    panel.set_peaks(_three_peaks())
    assert panel._table.rowCount() == 3
    for r in range(panel._table.rowCount()):
        chk = panel._table.item(r, panel._COL_CHECK)
        assert chk is not None
        assert chk.checkState() == QtCore.Qt.Checked
        assert chk.data(QtCore.Qt.UserRole) is not None   # UserRole хранит энергию


def test_checkbox_toggle_emits_visibility(app):
    """#124: снятие галочки → peakVisibilityChanged(E, False); установка → (E, True)."""
    from awf.ui.peaks_panel import PeaksPanel
    panel = PeaksPanel()
    panel.set_peaks(_three_peaks())
    got = []
    panel.peakVisibilityChanged.connect(lambda e, vis: got.append((e, vis)))
    chk = panel._table.item(0, panel._COL_CHECK)
    energy = float(chk.data(QtCore.Qt.UserRole))
    chk.setCheckState(QtCore.Qt.Unchecked)
    assert len(got) == 1
    assert got[0][0] == pytest.approx(energy)
    assert got[0][1] is False
    chk.setCheckState(QtCore.Qt.Checked)
    assert len(got) == 2 and got[1][1] is True


def test_cell_click_emits_peak_selected(app):
    """#124: клик по данные-ячейке строки → peakSelected(E); клик по чекбоксу — нет."""
    from awf.ui.peaks_panel import PeaksPanel
    panel = PeaksPanel()
    panel.set_peaks(_three_peaks())
    got = []
    panel.peakSelected.connect(lambda e: got.append(e))
    eitem = panel._table.item(0, panel._COL_ENERGY)
    energy = float(eitem.data(QtCore.Qt.EditRole))
    panel._on_cell_clicked(0, panel._COL_ENERGY)
    assert got == [pytest.approx(energy)]
    panel._on_cell_clicked(0, panel._COL_CHECK)
    assert len(got) == 1   # клик по чекбоксу не шлёт peakSelected


def test_checkbox_state_preserved_across_rebuild(app):
    """#124: снятая галочка сохраняется при повторном set_peaks тем же набором."""
    from awf.ui.peaks_panel import PeaksPanel
    panel = PeaksPanel()
    panel.set_peaks(_three_peaks())
    for r in range(panel._table.rowCount()):
        chk = panel._table.item(r, panel._COL_CHECK)
        if abs(float(chk.data(QtCore.Qt.UserRole)) - 312.0) < 1e-6:
            chk.setCheckState(QtCore.Qt.Unchecked)   # снимаем у пика 312 кэВ
    panel.set_peaks(_three_peaks())                  # повторная заливка тем же набором
    states = {}
    for r in range(panel._table.rowCount()):
        chk = panel._table.item(r, panel._COL_CHECK)
        states[round(float(chk.data(QtCore.Qt.UserRole)), 3)] = chk.checkState()
    assert states[312.0] == QtCore.Qt.Unchecked
    assert states[150.0] == QtCore.Qt.Checked
    assert states[478.0] == QtCore.Qt.Checked


def test_set_peak_visible_false_drops_one_ridge(app):
    """#124: скрытие пика по энергии убирает ровно один гребень из 3D, возврат — восстанавливает."""
    v = _loaded()
    v.set_peak_search(True)
    n_full = len(v._peak_ridge_items)
    assert n_full == 3
    e = v._found_peak_energies()[0]
    v.set_peak_visible(e, False)
    assert len(v._peak_ridge_items) == n_full - 1
    v.set_peak_visible(e, True)
    assert len(v._peak_ridge_items) == n_full


def test_set_peak_highlight_colors_one_ridge_magenta(app):
    """#124: подсветка по энергии красит ровно один гребень в _PEAK_HILITE_RGBA, прочие зелёные."""
    from awf.ui.view3d import _PEAK_HILITE_RGBA, _PEAK_RAY_RGBA
    v = _loaded()
    v.set_peak_search(True)
    e = v._found_peak_energies()[1]
    v.set_peak_highlight(e)

    def col(it):
        return tuple(round(float(x), 3) for x in np.asarray(it.color).ravel()[:4])
    hi = tuple(round(float(x), 3) for x in _PEAK_HILITE_RGBA)
    base = tuple(round(float(x), 3) for x in _PEAK_RAY_RGBA)
    cols = [col(it) for it in v._peak_ridge_items]
    assert cols.count(hi) == 1
    assert cols.count(base) == len(cols) - 1
    v.set_peak_highlight(None)                       # снятие подсветки → все зелёные
    assert all(col(it) == base for it in v._peak_ridge_items)

# ===== Задача #126: гребень не продлевается на пустое место =====

def _transient_sg(nt=20, c=130, amp=30.0, bg=50.0, half=None):
    """Спектрограмма с ОДНИМ пиком на канале c, присутствующим только в первой
    половине срезов (0..half-1). amp<bg по срезу → строгая peak_time_mask (#112)
    обнуляется колоночным гейтом (сценарий #126), но пик значим в интеграле."""
    if half is None:
        half = nt // 2
    chs = np.arange(NC, dtype=np.float64)
    sigma = PEAK_FWHM_CHANNELS / 2.355
    gauss = np.exp(-((chs - c) ** 2) / (2.0 * sigma ** 2))
    rows = np.full((nt, NC), bg, dtype=np.float64)
    rows[:half] += amp * gauss
    counts = np.round(rows).astype(np.int64)
    cal = Calibration(coeffs=[E0, KEV_PER_CH])
    t = np.arange(nt, dtype=np.float64)
    lt = np.ones(nt, dtype=np.float64)
    return Spectrogram(counts=counts, calibration=cal,
                       time_offsets_s=t, real_time_s=lt, live_time_s=lt)


def test_transient_ridge_not_extended_to_empty(app):
    """#126: пик в первой половине времени; раньше (fallback «весь гребень») линия
    тянулась на ВСЮ ось, включая пустой хвост. Теперь гребень ограничен зоной
    присутствия — не покрывает пустые срезы."""
    v = Waterfall3DView()
    v.set_spectrogram(_transient_sg(nt=20), max_time=400, max_chan=512)
    v.set_peak_search(True)
    assert len(v._peak_ridge_items) == 1, "транзиентный пик найден → один гребень"
    pos = np.asarray(v._peak_ridge_items[0].pos)
    nt = v._nt
    n_pts = pos.shape[0]
    assert 0 < n_pts < nt, f"гребень {n_pts} точек — НЕ вся ось {nt} (баг #126)"
    xs_idx = pos[:, 0] + nt / 2.0
    assert xs_idx.min() <= 2.0, "гребень начинается в зоне присутствия (начало оси)"
    assert xs_idx.max() <= nt * 0.7, "гребень НЕ дотягивается до пустого хвоста"