"""Тесты доработок Группы IV по замечаниям оператора IV-R1..IV-R5."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.zscale import smooth_counts, DEFAULT_SMOOTH
from awf.ui.view3d import Waterfall3DView
from awf.ui.panels import HeatmapPanel, SlicePanel


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=30, nc=50, t_step=2.0):
    counts = np.random.RandomState(0).poisson(50, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])  # E(ch) = ch keV
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


# ---------- IV-R5: не отображать последний канал ----------
def test_trim_drops_last_channel():
    sg = _make_sg(ns=10, nc=40)
    sg2 = sg.trimmed_channels(1)
    assert sg2.n_channels == 39
    assert sg2.n_slices == 10
    assert np.array_equal(sg2.counts, sg.counts[:, :-1])


def test_trim_keeps_calibration_and_time():
    sg = _make_sg(ns=8, nc=20)
    sg2 = sg.trimmed_channels(1)
    # энергии оставшихся каналов не меняются (та же калибровка)
    assert np.array_equal(sg2.energies(), sg.energies()[:-1])
    assert np.array_equal(sg2.time_offsets_s, sg.time_offsets_s)
    assert sg2.t0_iso == sg.t0_iso and sg2.source_path == sg.source_path


def test_trim_zero_returns_self():
    sg = _make_sg(ns=5, nc=10)
    assert sg.trimmed_channels(0) is sg


def test_trim_guard_too_many():
    sg = _make_sg(ns=5, nc=2)
    with pytest.raises(ValueError):
        sg.trimmed_channels(2)   # после обрезки 0 каналов — запрещено


def test_trim_default_is_one():
    sg = _make_sg(ns=5, nc=12)
    assert sg.trimmed_channels().n_channels == 11


# ---------- IV-R4: регулируемое усреднение спектра ----------
def test_smooth_radius0_identity():
    a = np.array([0, 0, 10, 0, 0], dtype=float)
    assert np.allclose(smooth_counts(a, 0), a)
    assert DEFAULT_SMOOTH == 0


def test_smooth_moving_average_value():
    a = np.array([0, 0, 10, 0, 0], dtype=float)
    s = smooth_counts(a, 1)             # окно 3
    assert s[2] == pytest.approx(10.0 / 3.0, rel=1e-5)
    assert s[0] == pytest.approx(0.0)
    assert s.sum() == pytest.approx(10.0, rel=1e-5)   # края нулевые -> сумма сохранена


def test_smooth_reduces_variance():
    rng = np.random.RandomState(1)
    a = rng.poisson(100, size=200).astype(float)
    assert smooth_counts(a, 5).var() < a.var()


def test_smooth_2d_axis_preserves_shape():
    m = np.random.RandomState(2).poisson(50, (6, 8)).astype(float)
    assert smooth_counts(m, 2, axis=1).shape == (6, 8)
    assert smooth_counts(m, 2, axis=0).shape == (6, 8)


def test_smooth_per_time_slice_independent_axis1(app):
    # Задача #90: при включении сглаживания SMA применяется ПО КАНАЛАМ в каждом
    # временно́м срезе (axis=1, как в view3d.py:237) — строки-срезы независимы,
    # смешивания по времени нет. Доказываем декомпозицией: 2D-сглаживание по axis=1
    # == одномерный SMA, применённый к каждому срезу по отдельности.
    m = np.random.RandomState(7).poisson(40, (5, 16)).astype(float)
    s = smooth_counts(m, 2, axis=1)
    for i in range(m.shape[0]):
        assert np.allclose(s[i], smooth_counts(m[i], 2))


def test_smooth_view3d_setter(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg())
    v.set_smoothing(4)
    assert v._smooth == 4
    assert v._surface is not None


def test_smooth_heatmap_setter(app):
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg())
    h.set_smoothing(3)
    assert h._smooth == 3


def test_smooth_slice_setter_uses_raw_cache(app):
    s = SlicePanel()
    s.set_spectrogram(_make_sg())
    assert s._raw_spec is not None         # сырой спектр закэширован
    s.set_smoothing(2)
    assert s._smooth == 2


# ---------- IV-R2 / #80: шкала энергий в «конец по времени», зубцы убраны ----------
def test_energy_scale_has_no_vertical_teeth(app):
    # Задача #80: вертикальные оливковые отрезки-зубцы шкалы энергий убраны — остались только подписи.
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    teeth = [it for it in v._axis_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(teeth) == 0     # зубцов больше нет (только GLTextItem-подписи)


def test_energy_teeth_cleared_when_axes_hidden(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_axis_labels_visible(False)
    teeth = [it for it in v._axis_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(teeth) == 0     # при скрытых осях зубцы тоже убраны


# ---------- IV-R3 + #84: одиночная плоскость режет от своего края до текущей позиции ----------
def test_single_plane_clips_from_edge(app):
    # Задача #84: видима только нижняя плоскость (слот 0) -> режет от минимума оси до позиции
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.3, True)
    i0, i1, j0, j1, z_lo, z_hi, ca = v._clip_windows()
    assert i0 > 0                          # нижняя граница поднялась от минимума
    assert i1 == v._nt - 1                 # верхняя сторона не тронута
    assert not ca


def test_single_plane_slot1_clips_upper(app):
    # Задача #84: видима только верхняя плоскость (слот 1) -> режет от максимума оси до позиции
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("energy", 1, 0.6, True)
    i0, i1, j0, j1, *_ = v._clip_windows()
    assert j0 == 0                         # нижняя сторона не тронута
    assert j1 < v._nc - 1                  # верхняя граница опустилась от максимума


def test_single_counts_plane_activates_clip(app):
    # Задача #84: одиночная counts-плоскость слота 0 включает высотную обрезку снизу
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("counts", 0, 0.2, True)
    *_, z_lo, z_hi, ca = v._clip_windows()
    assert ca and z_lo > -np.inf and z_hi == np.inf


def test_both_planes_clip_window(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.3, True)
    v.set_plane("time", 1, 0.6, True)
    i0, i1, j0, j1, z_lo, z_hi, ca = v._clip_windows()
    assert i0 > 0 and i1 < v._nt - 1       # окно уже полного диапазона
    assert i0 <= i1


def test_both_planes_rebuild_surface(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("energy", 0, 0.25, True)
    v.set_plane("energy", 1, 0.75, True)
    assert v._surface is not None          # обрезанная поверхность пересоздана без ошибок
    j0, j1 = v._clip_windows()[2], v._clip_windows()[3]
    assert j0 > 0 and j1 < v._nc - 1


def test_counts_planes_activate_height_clip(app):
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("counts", 0, 0.2, True)
    v.set_plane("counts", 1, 0.8, True)
    *_, z_lo, z_hi, ca = v._clip_windows()
    assert ca and z_hi > z_lo              # активна высотная обрезка по счёту
    assert v._surface is not None


def test_disable_one_plane_keeps_other_clip(app):
    # Задача #84: снятие слота 1 не убирает обрезку слота 0 — его сторона режется по-прежнему
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.3, True)
    v.set_plane("time", 1, 0.6, True)
    v.set_plane("time", 1, 0.6, False)     # снимаем верхнюю -> остаётся нижняя
    i0, i1, *_ = v._clip_windows()
    assert i0 > 0                          # слот 0 продолжает резать от минимума
    assert i1 == v._nt - 1                 # верхняя сторона восстановлена


def test_crossed_planes_empty_surface(app):
    # Задача #84: встречные плоскости пересеклись -> поверхность убрана (пустое окно, без ошибок)
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.8, True)      # нижняя граница выше...
    v.set_plane("time", 1, 0.2, True)      # ...верхней -> перекрытие
    i0, i1, *_ = v._clip_windows()
    assert i0 > i1                         # окно вырождено
    assert v._surface is None              # поверхность снята


# ---------- IV-R1: серая градиентная схема оформления ----------
def test_app_qss_nonempty():
    from awf.ui.style import APP_QSS
    assert isinstance(APP_QSS, str) and len(APP_QSS) > 100
    assert "qlineargradient" in APP_QSS     # градиентная схема


def test_main_window_applies_style(app):
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    assert "qlineargradient" in w.styleSheet()


# ---------- #97: диалог «Выбор фона» наследует тёмную тему ----------
def test_bg_dialog_theme_covers_radio_and_spinbox():
    """Задача #97: QSS темы покрывает виджеты диалога фона — радиокнопки и кнопки-стрелки
    спинбоксов (без них рисовались дефолтным синим/светлым системным стилем) + фон QDialog."""
    from awf.ui.style import APP_QSS
    assert "QDialog" in APP_QSS                       # тёмный фон диалога
    assert "QRadioButton::indicator" in APP_QSS       # индикатор радиокнопки под тему
    assert "QSpinBox::up-button" in APP_QSS and "QSpinBox::down-button" in APP_QSS
    assert "QSpinBox::up-arrow" in APP_QSS            # стрелки светлым цветом


def test_bg_dialog_inherits_window_theme(app):
    """Задача #97: BackgroundDialog создаётся как дочерний к MainWindow (parent), поэтому
    QSS главного окна (градиентная тема) каскадируется на него при отрисовке."""
    import numpy as np
    from awf.ui.main_window import MainWindow
    from awf.ui.background_dialog import BackgroundDialog
    w = MainWindow()
    dlg = BackgroundDialog(8, np.arange(8.0), w)
    assert dlg.parent() is w                          # источник наследуемого QSS — тема окна
    assert "qlineargradient" in w.styleSheet()
    dlg.deleteLater(); w.close()


# ---------- #116/#117: тёмная тема для таблицы пиков и попапов pyqtgraph ----------
def test_peaks_table_dark_theme_in_qss():
    """Задача #116: панель «Найденные пики» (#111) — QTableWidget; в теме были правила
    QTreeView/QTreeWidget/QListView, но НЕ QTableView/QTableWidget → viewport таблицы
    рисовался дефолтным светлым фоном. Проверяем покрытие таблиц тёмной темой."""
    from awf.ui.style import APP_QSS
    assert "QTableView" in APP_QSS and "QTableWidget" in APP_QSS
    assert "QTableCornerButton::section" in APP_QSS    # угловая кнопка под тему
    assert "gridline-color" in APP_QSS                 # тёмная сетка ячеек


def test_app_level_qss_for_pyqtgraph_popups(app):
    """Задача #117: контекстные меню pyqtgraph (ViewBoxMenu) — popup БЕЗ QWidget-родителя,
    поэтому stylesheet окна до них не каскадирует (рисовались системной светлой темой).
    MainWindow ставит APP_QSS на уровень QApplication — тема достаёт parentless-попапы."""
    from PySide6 import QtWidgets
    from awf.ui.main_window import MainWindow
    from awf.ui.style import APP_QSS
    w = MainWindow()
    inst = QtWidgets.QApplication.instance()
    assert inst is not None and inst.styleSheet() == APP_QSS
    w.close()


def test_peaks_table_viewport_base_dark(app):
    """Задача #116 (остаток): QSS красит ячейки/заголовок, но ПУСТУЮ область viewport ниже
    строк и угловую кнопку Qt заливает из палитры (роль Base) — оставались белыми поверх
    тёмной темы. Панель форсит тёмную Base; проверяем, что палитра таблицы тёмная."""
    from PySide6 import QtGui
    from awf.ui.peaks_panel import PeaksPanel
    p = PeaksPanel()
    base = p._table.palette().color(QtGui.QPalette.Base)
    assert base.red() < 60 and base.green() < 60 and base.blue() < 60
    vp_bg = p._table.viewport().palette().color(p._table.viewport().backgroundRole())
    assert vp_bg.red() < 60 and vp_bg.green() < 60 and vp_bg.blue() < 60


# ---------- #62: тулбар «Вид» и строка статуса — крупнее шрифт и выше ----------
def test_toolbar_and_statusbar_sized_up():
    """Задача #62: QSS задаёт увеличенный шрифт и высоту контролов тулбара и строки статуса
    (были скучены). Целевые правила — потомки QToolBar и сам QStatusBar."""
    from awf.ui.style import APP_QSS
    assert "QToolBar QComboBox" in APP_QSS and "min-height" in APP_QSS
    assert "QToolBar QPushButton" in APP_QSS
    assert "QStatusBar QLabel" in APP_QSS
    # размер шрифта тулбара/статуса поднят выше дефолтного 12px (Задача #93: тулбар → 14px как меню)
    assert "font-size: 14px" in APP_QSS
    assert "font-size: 15px" not in APP_QSS    # Задача #93: укрупнённого 15px тулбара больше нет


def test_toolbar_combo_taller_than_default(app):
    """Задача #62: min-height из QSS реально увеличивает высоту контролов тулбара
    (комбобокс Z-шкалы становится выше неоформленного дефолта)."""
    from PySide6 import QtWidgets
    from awf.ui.style import APP_QSS
    from awf.ui.main_window import MainWindow
    QtWidgets.QApplication.instance().setStyleSheet(APP_QSS)
    w = MainWindow()
    bare = QtWidgets.QComboBox()              # без применённого QSS min-height
    assert w._z_combo.minimumSizeHint().height() >= bare.minimumSizeHint().height()
    # строка статуса принудительно выше дефолта (задаём в коде, #62)
    assert w.statusBar().minimumHeight() >= 28
    w.close()
    bare.deleteLater()


# ---------- #63/#68: координатная сетка на делениях шкал + рамка в 1 клетку ----------
def test_grid_built_with_border(app):
    """Задача #63/#68: линии сетки на делениях обеих осей + 4 ребра рамки поля."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    lines = [it for it in v._grid_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(lines) >= 8     # деления по двум осям (≥2+2) + 4 ребра рамки


def test_grid_cleared_when_axes_hidden(app):
    """Задача #63: сетка снимается вместе с осями."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_axis_labels_visible(False)
    assert len(v._grid_items) == 0


def test_grid_border_extends_half_cell_beyond_data(app):
    """Задача #63/#70: рамка поля отстоит за край данных на полклетки (было 1 клетка)."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    xmin, xmax, ymin, ymax, _z = v._axis_extent()
    xs, ys = [], []
    for it in v._grid_items:
        p = np.asarray(it.pos)
        xs.extend(p[:, 0].tolist()); ys.extend(p[:, 1].tolist())
    assert min(xs) < xmin - 1e-6 and max(xs) > xmax + 1e-6
    assert min(ys) < ymin - 1e-6 and max(ys) > ymax + 1e-6
    # #70: отступ — именно полклетки, не целая. Поле рамки шире данных меньше чем на клетку.
    _dv, wx, _u = v._time_ticks()
    cellx = float(np.median(np.diff(wx)))
    assert (xmin - min(xs)) < 0.75 * cellx     # < клетки (≈0.5), не ≈1.0


# ---------- #65: вертикальная шкала счёта (Z) убрана ----------
def test_vertical_count_scale_removed(app):
    """Задача #65: ни заголовка, ни делений вертикальной (Z/счёт) шкалы."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    texts = [it.text for it in v._axis_items if isinstance(it, gl.GLTextItem)]
    assert all("отсч" not in t for t in texts)   # нет «N, отсч.»/«N, отсч/с»


# ---------- #66: единицы на каждой клетке ----------
def test_axis_cell_labels_carry_units(app):
    """Задача #66: подпись каждой клетки несёт единицу (кэВ для энергии, с для времени)."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    texts = [it.text for it in v._axis_items if isinstance(it, gl.GLTextItem)]
    assert any("кэВ" in t for t in texts)
    assert any(t.endswith(" с") for t in texts)


# ---------- #64: единицы оси времени переключаются (с/мин/ч) ----------
def test_time_unit_switch_changes_labels(app):
    """Задача #64: переключение единицы времени пересобирает подписи в выбранной размерности."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=60, nc=50, t_step=60.0))   # 0..3540 с
    v.set_time_unit("мин")
    texts = [it.text for it in v._axis_items if isinstance(it, gl.GLTextItem)]
    assert v._time_unit == "мин"
    assert any(t.endswith(" мин") for t in texts)


# ---------- #71: шаг оси времени 15 минут для длинных водопадов ----------
def test_time_ticks_15min_step_large_waterfall(app):
    """Задача #71: для длинного водопада деления оси времени идут ровно по 15 минут."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=120, nc=50, t_step=60.0))   # ~2 ч записи
    v.set_time_unit("мин")
    dv, _wx, unit = v._time_ticks()
    assert unit == "мин" and len(dv) >= 3
    assert np.allclose(np.diff(dv), 15.0)


def test_time_ticks_fallback_short_waterfall(app):
    """Задача #71: короткая запись (< 45 мин) -> авто-деления, шаг 15 мин не навязывается."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50, t_step=2.0))     # 0..58 с
    dv, _wx, _u = v._time_ticks()
    assert len(dv) >= 2
    assert not np.allclose(np.diff(dv), 900.0)               # не 15-мин сетка


def test_mainwindow_time_unit_fans_out(app):
    """Задача #64: комбобокс «Время» в тулбаре прокидывает единицу во view3d."""
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    w._view3d.set_spectrogram(_make_sg(ns=30, nc=50))
    w._tunit_combo.setCurrentIndex(1)   # «мин»
    assert w._view3d._time_unit == "мин"
    w.close()


# ---------- #74: переключатель-перебор (клик/колесо) вместо выпадающего списка ----------
def test_cyclebutton_click_cycles(app):
    """Задача #74: клик переключает на следующее значение по кругу + эмитит currentIndexChanged."""
    from awf.ui.cyclebutton import CycleButton
    b = CycleButton()
    for k in ("a", "b", "c"):
        b.addItem(k.upper(), k)
    seen = []
    b.currentIndexChanged.connect(lambda i: seen.append(i))
    assert b.currentIndex() == 0 and b.currentData() == "a"
    b.click()                       # a -> b
    assert b.currentData() == "b" and b.currentText() == "B"
    b.click(); b.click()            # b -> c -> по кругу к a
    assert b.currentIndex() == 0 and b.currentData() == "a"
    assert seen == [1, 2, 0]
    b.deleteLater()


def test_cyclebutton_wheel_scrolls(app):
    """Задача #74: колесо мыши листает вперёд (вверх) и назад (вниз, по кругу)."""
    from PySide6 import QtCore
    from awf.ui.cyclebutton import CycleButton

    class _Wheel:                       # лёгкая замена QWheelEvent (без версионной возни)
        def __init__(self, y): self._y = y
        def angleDelta(self): return QtCore.QPoint(0, self._y)
        def accept(self): pass

    b = CycleButton()
    for k in ("a", "b", "c"):
        b.addItem(k, k)
    b.wheelEvent(_Wheel(120))           # вверх -> вперёд: a -> b
    assert b.currentIndex() == 1
    b.wheelEvent(_Wheel(-120)); b.wheelEvent(_Wheel(-120))   # назад: b -> a -> по кругу к c
    assert b.currentIndex() == 2
    b.deleteLater()


# ---------- #73: шрифт подписей делений осей 3D уменьшен ----------
def test_axis_label_font_reduced(app):
    """Задача #73: подписи делений осей 3D мельче прежних (≤8 pt вместо 10)."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    fonts = [it.font for it in v._axis_items if isinstance(it, gl.GLTextItem)]
    assert fonts
    assert all(f.pointSize() <= 8 for f in fonts)


# ---------- #72: шаг оси энергии 200 кэВ для широкого спектра ----------
def test_energy_ticks_200kev_step_wide_spectrum(app):
    """Задача #72: для широкого спектра деления оси энергии идут ровно по 200 кэВ."""
    counts = np.random.RandomState(1).poisson(50, size=(20, 1024)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 3.0])     # 3 кэВ/канал -> 0..~3069 кэВ
    t = np.arange(20, dtype=np.float64) * 2.0
    sg = Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                     real_time_s=np.full(20, 2.0), live_time_s=np.full(20, 2.0))
    v = Waterfall3DView()
    v.set_spectrogram(sg)
    ev, _wy = v._energy_ticks()
    assert len(ev) >= 3
    assert np.allclose(np.diff(ev), 200.0)


def test_energy_ticks_fallback_narrow_spectrum(app):
    """Задача #72: узкий спектр (десятки кэВ) -> авто-деления, шаг 200 кэВ не навязывается."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))   # 0..49 кэВ
    ev, _wy = v._energy_ticks()
    assert len(ev) >= 2
    assert not np.allclose(np.diff(ev), 200.0)


# ---------- #67/#69: маркеры нуклидов на секущих плоскостях Времени ----------
def test_plane_nuclides_drawn_on_visible_time_plane(app):
    """Задача #67: на видимой плоскости Времени рисуются маркеры выбранных линий нуклидов."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#ff0000", "Cs-137", 0.85)])
    items = [it for it in v._plane_nuclide_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(items) == 1


def test_plane_nuclides_absent_without_visible_plane(app):
    """Задача #67: без видимой плоскости Времени маркеры не рисуются."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_energy_lines([(10.0, "#ff0000", "Cs-137", 0.85)])
    assert len(v._plane_nuclide_items) == 0


def test_plane_nuclide_height_scales_with_intensity(app):
    """Задача #69: высота маркера ∝ интенсивности; ярчайшая линия = полная высота zmax."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#ff0000", "A", 1.0), (30.0, "#00ff00", "A", 0.25)])
    items = [it for it in v._plane_nuclide_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(items) == 2
    zmax = v._axis_extent()[4]
    h_hi = float(items[0].pos[1][2])   # I=1.0 -> полная высота
    h_lo = float(items[1].pos[1][2])   # I=0.25 -> ниже
    assert abs(h_hi - zmax) < 1e-3
    assert h_lo < h_hi and abs(h_lo - 0.25 * zmax) < 1e-2


def test_plane_nuclides_backward_compat_3tuple(app):
    """Задача #67: 3-кортежи (без интенсивности) рисуются на полную высоту, без сбоя."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#ff0000", "A"), (30.0, "#00ff00", "B")])
    items = [it for it in v._plane_nuclide_items if isinstance(it, gl.GLLinePlotItem)]
    assert len(items) == 2
    zmax = v._axis_extent()[4]
    assert all(abs(float(it.pos[1][2]) - zmax) < 1e-3 for it in items)


def test_plane_nuclides_cleared_when_lines_removed(app):
    """Задача #67: снятие выбора нуклидов убирает маркеры с плоскостей."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=30, nc=50))
    v.set_plane("time", 0, 0.5, True)
    v.set_energy_lines([(10.0, "#ff0000", "A", 0.5)])
    assert len(v._plane_nuclide_items) == 1
    v.set_energy_lines([])
    assert len(v._plane_nuclide_items) == 0


# ---------- #75: каркас верхних выпадающих меню ----------
def test_top_menus_skeleton_present(app):
    """Задача #75: в строке меню есть Изотопы/Анализ/Сервис/Помощь/О программе (каркас)."""
    from awf.ui.main_window import MainWindow, SETTINGS_ORG, SETTINGS_APP
    from awf.ui import i18n
    # Задача #106: тест должен быть автономен от QSettings прошлых запусков (выбор языка).
    QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP).remove("interface/language")
    i18n.reset_for_tests()
    w = MainWindow()
    titles = [m.title() for m in w.menuBar().findChildren(QtWidgets.QMenu)]
    for expected in ("Изотопы", "Анализ", "Инструменты", "Сервис", "Помощь", "О программе"):
        assert expected in titles
    assert set(w._menus) == {"isotopes", "analysis", "tools", "service", "help", "about"}
    # остальные пункты пока неактивны (каркас); «Изотопы» наполнено в #79, «Анализ» — в #96,
    # «Сервис» — в #106 (подменю «Язык»), «Инструменты» — в #115 (toggleViewAction доков).
    for key, m in w._menus.items():
        if key in ("isotopes", "analysis", "service", "tools"):
            continue
        acts = m.actions()
        assert acts and all(not a.isEnabled() for a in acts)
    # Задача #106: «Сервис» содержит подменю «Язык» с двумя активными пунктами Русский/English
    service_acts = w._menus["service"].actions()
    assert any(a.text() == "Язык" for a in service_acts)
    # Задача #96: меню «Анализ» наполнено — «Выбор фона…» активен, «Наложение/Вычет» до выбора нет
    analysis_titles = [a.text() for a in w._menus["analysis"].actions()]
    assert "Выбор фона…" in analysis_titles
    assert "Наложение фона" in analysis_titles and "Вычет фона" in analysis_titles
    assert w._act_bg_overlay.isEnabled() is False and w._act_bg_subtract.isEnabled() is False
    w.close()


# ---------- #76: отключение подложки (плоское дно рельефа) ----------
def _floor_sg(n=10):
    """Спектр с явным «дном»: почти всё — нули (база); один столбец — рельеф с разбросом высот
    (10..10^5), чтобы log-шкала дала ненулевой контраст (равные пики дали бы zn=0 у всех)."""
    counts = np.zeros((n, n), dtype=np.int64)
    for k in range(5):
        counts[k, 0] = 10 ** (k + 1)
    cal = Calibration(coeffs=[0.0, 1.0])
    t = np.arange(n, dtype=np.float64) * 2.0
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(n, 2.0), live_time_s=np.full(n, 2.0))


def test_floor_visible_by_default(app):
    """Задача #76: по умолчанию подложка видима — ни одна ячейка не прозрачна."""
    v = Waterfall3DView()
    v.set_spectrogram(_floor_sg())
    assert v._floor_visible is True
    alpha = np.asarray(v._surface._colors)[:, 3]
    assert np.all(alpha > 0.0)


def test_floor_hidden_zeroes_base_cells(app):
    """Задача #76: выключенная подложка делает ячейки дна (zn≈0) прозрачными, рельеф остаётся."""
    v = Waterfall3DView()
    v.set_spectrogram(_floor_sg())
    v.set_floor_visible(False)
    alpha = np.asarray(v._surface._colors)[:, 3]
    assert np.any(alpha == 0.0)        # дно (нули) стало прозрачным
    assert np.any(alpha > 0.0)         # рельеф (угол 5×5) остался видимым
    v.set_floor_visible(True)          # вернули — снова всё видимо
    assert np.all(np.asarray(v._surface._colors)[:, 3] > 0.0)


def test_floor_toolbar_checkbox_drives_view(app):
    """Задача #76: чекбокс «Подложка» тулбара управляет _floor_visible вьюера."""
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    w._view3d.set_spectrogram(_floor_sg())
    assert w._floor_check.isChecked() and w._view3d._floor_visible is True
    w._floor_check.setChecked(False)
    assert w._view3d._floor_visible is False
    w._floor_check.setChecked(True)
    assert w._view3d._floor_visible is True
    w.close()


# ---------- #77: подписи осей на ближней к зрителю стороне ----------
def test_axis_labels_follow_viewer_side(app):
    """Задача #77: подписи делений висят на ближнем к зрителю крае; поворот камеры в другой
    квадрант переносит их на сторону взгляда (не за рельеф)."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=40, nc=60))
    def split():
        texts = [it for it in v._axis_items if isinstance(it, gl.GLTextItem)]
        en = [it for it in texts if "кэВ" in it.text]
        tm = [it for it in texts if "кэВ" not in it.text]
        return tm, en
    tm0, en0 = split()    # дефолт azimuth=-60: время y<0, энергия x>0
    assert tm0 and en0
    assert all(it.pos[1] < 0 for it in tm0)
    assert all(it.pos[0] > 0 for it in en0)
    # поворот в противоположный квадрант: cos(120)<0 -> энергия x<0, sin(120)>0 -> время y>0
    v.setCameraPosition(azimuth=120, elevation=35)
    v._maybe_reorient_labels()
    tm1, en1 = split()
    assert tm1 and en1
    assert all(it.pos[1] > 0 for it in tm1)
    assert all(it.pos[0] < 0 for it in en1)


# ---------- #78: ограничить вывод и сетку 3D порогом 3000 кэВ ----------
def test_energy_axis_capped_at_3000(app):
    """Задача #78: каналы с энергией выше 3000 кэВ не выводятся; деления оси энергии
    тоже не выходят за порог."""
    from awf.ui.view3d import _MAX_ENERGY_KEV
    ns, nch = 20, 1000
    counts = np.random.RandomState(1).poisson(20, size=(ns, nch)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 5.0])   # E(ch)=5·ch -> до ~4995 кэВ, заведомо выше порога
    t = np.arange(ns, dtype=np.float64) * 2.0
    sg = Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                     real_time_s=np.full(ns, 2.0), live_time_s=np.full(ns, 2.0))
    v = Waterfall3DView()
    v.set_spectrogram(sg)
    assert v._ch_centers.size > 0
    assert float(v._ch_centers[-1]) <= _MAX_ENERGY_KEV            # вывод обрезан
    assert v._nc == v._ch_centers.size == v._colors_full.shape[1]
    ev, _wy = v._energy_ticks()
    if ev.size:
        assert float(ev.max()) <= _MAX_ENERGY_KEV                # сетка/деления обрезаны


# ---------- #79: меню «Изотопы» — ссылка на окно нуклидов ----------
def test_isotopes_menu_opens_nuclides_dock(app):
    """Задача #79: пункт «Изотопы» — действующая ссылка (toggleViewAction дока нуклидов),
    переключает видимость окна нуклидов."""
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    act = w._menus["isotopes"].actions()[0]
    assert act.isEnabled() and act.isCheckable()      # больше не заглушка, это переключатель
    assert act is w._ndock.toggleViewAction()         # ведёт именно на док нуклидов
    s0 = act.isChecked()
    act.trigger()
    assert act.isChecked() != s0                      # клик переключил видимость окна
    act.trigger()
    assert act.isChecked() == s0                      # повторный клик вернул
    w.close()


# ---------- #115: меню «Инструменты» — перечень окон-доков ----------
def test_tools_menu_lists_dock_windows(app):
    """Задача #115: меню «Инструменты» перечисляет все окна-доки; каждый пункт —
    toggleViewAction соответствующего дока (checkable, переключает видимость окна)."""
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    tools = w._menus["tools"]
    texts = [a.text() for a in tools.actions()]
    for label in ("Найденные пики", "Срезы / Сечения / Выборки",
                  "Сечения (3D)", "Регулировки отображения"):
        assert label in texts                         # каждое окно перечислено
    # окно нуклидов тоже в списке (тот же QAction, что в «Изотопы», #79)
    assert w._ndock.toggleViewAction() in tools.actions()
    # пункт ведёт на свой док и переключает его видимость
    act = w._peaks_dock.toggleViewAction()
    assert act in tools.actions() and act.isCheckable() and act.isEnabled()
    s0 = act.isChecked()
    act.trigger()
    assert act.isChecked() != s0                      # клик переключил окно
    act.trigger()
    assert act.isChecked() == s0                      # повторный клик вернул
    w.close()


# ---------- #82: жёлтое окно (ROI-выделение) на 2D-карте отключено ----------
def test_heatmap_roi_hidden(app):
    """Задача #82: после загрузки данных жёлтый прямоугольник ROI на 2D-карте не показывается."""
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=30, nc=50))
    assert not h._roi.isVisible()      # жёлтое окно выделения скрыто


# ---------- #87: 2D-карта не «уезжает» — зум/панорама привязаны к окну данных ----------
def test_heatmap_view_limited_to_data(app):
    """Задача #87: после загрузки ViewBox 2D-карты ограничен прямоугольником данных —
    панорама не выходит за [0..cols]×[0..rows], минимальный зум = полное окно."""
    h = HeatmapPanel()
    h.set_spectrogram(_make_sg(ns=30, nc=50))
    lim = h._plot.getViewBox().state["limits"]
    assert lim["xLimits"][0] == 0 and lim["xLimits"][1] == h._disp_cols
    assert lim["yLimits"][0] == 0 and lim["yLimits"][1] == h._disp_rows
    assert lim["xRange"][1] == h._disp_cols   # maxXRange = полное окно (минимальный зум)
    assert lim["yRange"][1] == h._disp_rows


# ---------- #92: рукоятки Регулировок не сбрасывают зум 3D ----------
def test_adjust_setters_preserve_camera_zoom(app):
    """Задача #92: ре-рендер от рукояток (set_smoothing/set_contrast/set_colormap) сохраняет
    положение/масштаб камеры — пользовательский зум не сбрасывается на каждый ход рукоятки.
    Новый спектр (другой объект) — камеру по-прежнему кадрирует под размер данных."""
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=40, nc=60))     # первая загрузка кадрирует камеру
    v.setCameraPosition(distance=12.5)            # пользователь приблизил вручную
    azim0 = v.opts["azimuth"]
    v.set_smoothing(5)                            # рукоятка «Сглаживание» -> ре-рендер того же sg
    assert v.opts["distance"] == 12.5             # зум сохранён
    v.set_contrast(gain=2.0)                      # рукоятки «Усиление/Гамма/Отсечка»
    assert v.opts["distance"] == 12.5
    v.set_colormap("inferno")                     # смена палитры — тоже без сброса
    assert v.opts["distance"] == 12.5
    assert v.opts["azimuth"] == azim0             # азимут/поворот тоже не трогаем
    v.set_spectrogram(_make_sg(ns=80, nc=120))    # новый файл — камеру кадрируем заново
    assert v.opts["distance"] != 12.5


# ---------- #89: графики дока «Срезы/Сечения/Выборки» привязаны к окну данных ----------
def test_slice_views_locked_to_data(app):
    """Задача #89: после загрузки X-домен обоих графиков SlicePanel ограничен экстентом
    данных — спектр по энергии [emin..emax], профиль по времени [tmin..tmax];
    панорама не выходит за домен, минимальный зум = полный домен."""
    sp = SlicePanel()
    sg = _make_sg(ns=30, nc=50, t_step=2.0)   # энергии 0..49 кэВ, время 0..58 с
    sp.set_spectrogram(sg)
    emin, emax = float(sg.energies().min()), float(sg.energies().max())
    tmin, tmax = float(sg.time_offsets_s.min()), float(sg.time_offsets_s.max())
    slim = sp._spectrum_plot.getViewBox().state["limits"]
    assert slim["xLimits"][0] == emin and slim["xLimits"][1] == emax
    assert slim["xRange"][1] == emax - emin            # maxXRange = полный домен энергии
    tlim = sp._series_plot.getViewBox().state["limits"]
    assert tlim["xLimits"][0] == tmin and tlim["xLimits"][1] == tmax
    assert tlim["xRange"][1] == tmax - tmin            # maxXRange = полный домен времени