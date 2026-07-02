"""UI-тесты фона/вычитания (Задача #96): наложение кривой фона на спектр среза (SlicePanel)
и вычет фона из всего водопада (MainWindow -> 3D/2D/срезы), аналитика — на исходных данных."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.panels import SlicePanel
from awf.ui.main_window import MainWindow
from awf.ui.background_dialog import BackgroundDialog


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _make_sg(ns=12, nc=20, t_step=2.0):
    counts = np.random.RandomState(7).poisson(40, size=(ns, nc)).astype(np.int64)
    cal = Calibration(coeffs=[0.0, 1.0])               # E(ch) = ch кэВ
    t = np.arange(ns, dtype=np.float64) * t_step
    return Spectrogram(counts=counts, calibration=cal, time_offsets_s=t,
                       real_time_s=np.full(ns, t_step), live_time_s=np.full(ns, t_step))


def _empty(curve) -> bool:
    x, _ = curve.getData()
    return x is None or len(x) == 0


# ---------- SlicePanel: наложение кривой фона ----------

def test_slice_bg_hidden_until_overlay_enabled(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(nc=10))
    p.set_background(np.ones(10))
    assert _empty(p._bg_curve)                         # фон задан, наложение выкл -> пусто
    p.set_background_overlay(True)
    x, _ = p._bg_curve.getData()
    assert x is not None and len(x) == 10              # появилась кривая по всем каналам


def test_slice_bg_unit_scaling(app):
    # live_time на срез = 2 с -> в counts кривая фона = bg_cps * 2, в cps = bg_cps
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=3, nc=8))
    p.show_time_slice(0)                                # окно среза, lt_total = 2 c
    p.set_background(np.ones(8))
    p.set_background_overlay(True)
    _, y_cps = p._bg_curve.getData()                   # дефолт — cps
    assert np.allclose(y_cps, 1.0)
    p.set_unit_mode("counts")
    _, y_counts = p._bg_curve.getData()
    assert np.allclose(y_counts, 2.0)                  # bg_cps * живое_время_окна (2 c)


def test_slice_bg_cleared_on_none(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(nc=8))
    p.set_background(np.ones(8))
    p.set_background_overlay(True)
    assert not _empty(p._bg_curve)
    p.set_background(None)                              # снять фон -> кривая пуста даже при overlay
    assert _empty(p._bg_curve)


def test_slice_bg_size_mismatch_not_drawn(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(nc=8))
    p.set_background(np.ones(5))                        # длина != числу каналов
    p.set_background_overlay(True)
    assert _empty(p._bg_curve)                          # несогласованный фон не рисуется


# ---------- MainWindow: вычет фона из всего водопада ----------

def test_mainwindow_subtract_reduces_waterfall(app):
    w = MainWindow()
    w._on_loaded(_make_sg(ns=12, nc=20))
    orig = float(w._sg.counts.sum())
    w._bg_cps = w._compute_background(("range", 0, w._sg.n_slices))
    w._on_bg_subtract_toggled(True)                    # вычет включён
    assert float(w._slices._sg.counts.sum()) < orig    # 3D/2D/срезы — на вычтенных данных
    assert float(w._heatmap._sg.counts.sum()) < orig
    # Задача #147 (усиливает #134): фон = весь диапазон -> range-ветка сташит сырой блок
    # _bg_raw той же формы -> поячеечный вычет: остаток — ТОЧНЫЙ ноль в каждой ячейке
    # (раньше знаковый пуассонов остаток, ноль лишь в интеграле по времени).
    assert np.allclose(w._slices._sg.counts, 0.0, atol=1e-9)   # самовычет = ноль поячеечно
    w._on_bg_subtract_toggled(False)                   # выключение -> исходный водопад
    assert float(w._slices._sg.counts.sum()) == pytest.approx(orig)
    w.close()


def test_mainwindow_analytics_stays_on_original(app):
    w = MainWindow()
    w._on_loaded(_make_sg(ns=10, nc=16))
    orig = float(w._sg.counts.sum())
    w._bg_cps = w._compute_background(("range", 0, w._sg.n_slices))
    w._on_bg_subtract_toggled(True)
    assert float(w._sg.counts.sum()) == orig           # _sg (источник аналитики) не меняется
    w.close()


def test_mainwindow_reset_background_on_reload(app):
    w = MainWindow()
    w._on_loaded(_make_sg(ns=10, nc=16))
    w._bg_cps = w._compute_background(("range", 0, w._sg.n_slices))
    w._act_bg_overlay.setEnabled(True)
    w._act_bg_subtract.setEnabled(True)
    w._on_bg_subtract_toggled(True)
    w._on_loaded(_make_sg(ns=8, nc=16))                # новый файл -> фон сброшен
    assert w._bg_cps is None
    assert w._bg_subtract is False and w._bg_overlay is False
    assert not w._act_bg_overlay.isEnabled()
    assert not w._act_bg_subtract.isEnabled()
    w.close()


# ---------- BackgroundDialog: сбор выбора источника ----------

def test_bg_dialog_range_result(app):
    dlg = BackgroundDialog(20, time_offsets=np.arange(20.0))
    dlg._lo.setValue(3); dlg._hi.setValue(10)
    dlg._on_accept()                                   # дефолт — радиокнопка диапазона
    assert dlg.result_spec() == ("range", 3, 10)


def test_bg_dialog_file_result(app):
    dlg = BackgroundDialog(20)
    dlg._path = "C:/tmp/bg.aswf"                        # имитируем выбор файла из «Обзор…»
    dlg._rb_file.setChecked(True)
    dlg._on_accept()
    assert dlg.result_spec() == ("file", "C:/tmp/bg.aswf")


def test_bg_dialog_default_full_range(app):
    dlg = BackgroundDialog(15)
    assert dlg._lo.value() == 0 and dlg._hi.value() == 15   # по умолчанию весь диапазон
    assert dlg.result_spec() is None                        # до accept — None


# ---------- #99: на лог-шкале нулевой фон ВЭ-хвоста маскируется (нет «частокола») ----------

def test_slice_bg_log_masks_nonpositive(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(nc=8))
    bg = np.array([1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0])  # нули = пустой ВЭ-хвост фона
    p.set_background(bg)
    p.set_background_overlay(True)
    p.set_spectrum_log(True)                                 # лог Y -> нули фона в nan (#99)
    _, y = p._bg_curve.getData()
    y = np.asarray(y, dtype=np.float64)
    assert np.isnan(y[1]) and np.isnan(y[3]) and np.isnan(y[4])   # разрывы там, где фон == 0
    assert np.isfinite(y[0]) and np.isfinite(y[2]) and np.isfinite(y[5])
    p.set_spectrum_log(False)                                # линейная -> маскирования нет
    _, y2 = p._bg_curve.getData()
    assert np.isfinite(np.asarray(y2, dtype=np.float64)).all()


# ---------- #100: сброс зума графиков среза и времени ----------

def test_slice_has_reset_zoom_button(app):
    p = SlicePanel()
    assert hasattr(p, "_reset_zoom_btn")
    assert p._reset_zoom_btn.text() == "Сброс зума"


def test_slice_reset_zoom_restores_full_view(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=10, nc=16, t_step=2.0))    # энергии 0..15, время 0..18
    sp_vb = p._spectrum_plot.getViewBox()
    sr_vb = p._series_plot.getViewBox()
    sp_vb.setXRange(5.0, 6.0, padding=0)                     # зум обоих графиков в узкое окно
    sr_vb.setXRange(3.0, 4.0, padding=0)
    p.reset_zoom()
    app.processEvents()
    (sx0, sx1), _ = sp_vb.viewRange()
    (tx0, tx1), _ = sr_vb.viewRange()
    assert sx1 - sx0 > 1.5 and tx1 - tx0 > 1.5              # вид шире узкого зума 1.0


# ---------- #125: сброс зума/смещения по двойному клику ----------

def test_slice_double_click_resets_zoom(app):
    """#125: двойной клик по графику среза/времени → reset_zoom; одиночный — нет."""
    p = SlicePanel()
    calls = []
    p.reset_zoom = lambda: calls.append(1)                  # шпион вместо метода

    class _Ev:
        def __init__(self, dbl):
            self._d = dbl

        def double(self):
            return self._d

        def accept(self):
            pass

    p._on_plot_double_click(_Ev(False))                     # одиночный клик не сбрасывает
    assert calls == []
    # сигнал сцены обоих графиков с дабл-кликом -> reset_zoom (проверка проводки)
    p._spectrum_plot.scene().sigMouseClicked.emit(_Ev(True))
    p._series_plot.scene().sigMouseClicked.emit(_Ev(True))
    assert calls == [1, 1]


# ---------- #98: «фоновая простыня» на 3D-водопаде ----------

def test_view3d_bg_sheet_toggle(app):
    from awf.ui.view3d import Waterfall3DView
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=12, nc=20))
    v.set_background_sheet(np.ones(20))
    assert v._bg_sheet is None                          # фон задан, наложение выкл -> простыни нет
    v.set_background_sheet_visible(True)
    assert v._bg_sheet is not None                      # включили наложение -> простыня построена
    v.set_background_sheet_visible(False)
    assert v._bg_sheet is None                          # выключили -> простыня снята


def test_view3d_bg_sheet_built_like_relief(app):
    # #144: простыня — полноценный рельеф, ТЕМ ЖЕ пайплайном что рельеф-образец:
    # сырое поле фона (при raw-совпадении = counts/lt) -> sg.downsample(method="max")
    # -> обрезка _MAX_ENERGY_KEV -> smooth_counts. В _add_bg_sheet приходит 2D-матрица
    # (nt, nc), равная LOD-max от сырого поля (форма и значения совпадают).
    from awf.ui.view3d import Waterfall3DView
    from awf.ui.zscale import smooth_counts
    from awf.model.background import background_from_range
    v = Waterfall3DView()
    sg = _make_sg(ns=16, nc=24)
    v.set_spectrogram(sg)
    bg = background_from_range(sg, 0, sg.n_slices)
    cap = {}
    v._add_bg_sheet = lambda z: cap.__setitem__("z", np.asarray(z, dtype=float))
    v.set_background_sheet(bg, (sg.counts, sg.live_time_s))     # фон = весь файл (самофон)
    v.set_background_sheet_visible(True)
    lt = np.asarray(sg.live_time_s, dtype=float)
    safe = np.where(lt > 0.0, lt, np.inf)
    field = np.asarray(sg.counts, dtype=float) / safe[:, None]  # unit=cps, raw-совпадение
    lod, _t, _c = sg.downsample(v._max_time, v._max_chan, method="max", data=field)
    exp = smooth_counts(np.asarray(lod, dtype=float), v._smooth, axis=1)[:, :v._nc]
    assert cap["z"].shape == (exp.shape[0], v._nc)
    assert np.allclose(cap["z"], exp, atol=1e-6)


# ---------- #141: фон из файла-источника — сырой блок при совпадающей энергосетке ----------

def test_compute_background_file_same_grid_stashes_raw(app, monkeypatch):
    # #141: файл-фон с той же энергосеткой (тот же файл/прибор) -> сырой блок _bg_raw,
    # фон «лохматый как образец» и в срезах (#139), и на простыне 3D (#140).
    import awf.ui.main_window as mw
    full = _make_sg(ns=8, nc=17)
    w = MainWindow()
    w._sg = full.trimmed_channels(1)
    monkeypatch.setattr(mw, "load_spectrogram", lambda p, **k: full)
    bg = w._compute_background(("file", "dummy.n42"))
    assert w._bg_raw is not None
    cnt, lt = w._bg_raw
    assert cnt.shape == w._sg.counts.shape and lt.size == w._sg.n_slices
    assert bg.size == w._sg.n_channels
    w.close()


def test_compute_background_file_other_grid_raw_none(app, monkeypatch):
    # #141: иная энергосетка файла-фона -> сырой поканальный матч не строим (гладкий bg_cps).
    import awf.ui.main_window as mw
    w = MainWindow()
    w._sg = _make_sg(ns=8, nc=17).trimmed_channels(1)
    w._bg_raw = (np.zeros((2, 2)), np.ones(2))
    monkeypatch.setattr(mw, "load_spectrogram", lambda p, **k: _make_sg(ns=8, nc=23))
    bg = w._compute_background(("file", "dummy.n42"))
    assert w._bg_raw is None
    assert bg.size == w._sg.n_channels
    w.close()


# ---------- #142: тумблеры видимости элементов наложения (тулбар «Вид») ----------

def test_bg_overlay_visibility_toggles(app):
    # #142: «Простыня фона» / «Фон среза» режут видимость по-элементно в режиме наложения.
    w = MainWindow()
    assert not w._bg_sheet_check.isEnabled() and not w._bg_curve_check.isEnabled()
    w._act_bg_overlay.setEnabled(True)
    w._act_bg_overlay.setChecked(True)                  # включили режим наложения
    assert w._bg_sheet_check.isEnabled() and w._bg_curve_check.isEnabled()
    assert w._slices._bg_overlay and w._view3d._bg_sheet_on
    w._bg_sheet_check.setChecked(False)                 # скрыли простыню 3D
    assert not w._view3d._bg_sheet_on and w._slices._bg_overlay
    w._bg_curve_check.setChecked(False)                 # скрыли кривую фона в срезе
    assert not w._slices._bg_overlay
    w._act_bg_overlay.setChecked(False)                 # выключили режим наложения
    assert not w._bg_sheet_check.isEnabled() and not w._bg_curve_check.isEnabled()
    w._bg_sheet_check.setChecked(True); w._bg_curve_check.setChecked(True)
    assert not w._view3d._bg_sheet_on and not w._slices._bg_overlay   # без режима — скрыто
    w.close()


# ---------- #143: тумблер «Простыня образца» (основной 3D-рельеф) ----------

def test_surface_visibility_toggle(app):
    # #143: чекбокс «Простыня образца» скрывает основной GLSurfacePlotItem рельефа.
    w = MainWindow()
    assert w._surface_check.isChecked() and w._surface_check.isEnabled()
    assert w._view3d._surface_on                        # дефолт — видима
    w._sg = _make_sg(ns=10, nc=16)
    w._view3d.set_spectrogram(w._sg)
    assert w._view3d._surface is not None                # рельеф построен
    w._surface_check.setChecked(False)                   # скрыли простыню образца
    assert not w._view3d._surface_on and w._view3d._surface is None
    w._surface_check.setChecked(True)                    # вернули
    assert w._view3d._surface_on and w._view3d._surface is not None
    w.close()


# ---------- #101: нижняя граница оси Y графика спектра зафиксирована ----------

def test_slice_spectrum_y_floor_linear(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=6, nc=12))
    ymin, _ = p._spectrum_plot.getViewBox().state["limits"]["yLimits"]
    assert ymin == 0.0                                  # лин.: 0 прибит к низу (счёт ≥ 0)


def test_slice_spectrum_y_floor_log(app):
    p = SlicePanel()
    sg = _make_sg(ns=6, nc=12)
    p.set_spectrogram(sg)
    p.set_spectrum_log(True)
    _, s, lt = p._raw_spec
    disp = p._spec_to_unit(s, lt)
    # Задача #166: пол = 10-й перцентиль pos (не min) — устойчив к ВЭ-выбросам после ε-нормировки.
    expected = float(np.log10(np.percentile(disp[disp > 0], 10.0)))
    ymin, _ = p._spectrum_plot.getViewBox().state["limits"]["yLimits"]
    assert ymin == pytest.approx(expected)              # лог: пол = log10(percentile10(pos))


# ---------- #128: нижняя граница оси Y графика «Скорость в полосе» зафиксирована ----------

def test_slice_series_y_floor_zero(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=6, nc=12))
    ymin, _ = p._series_plot.getViewBox().state["limits"]["yLimits"]
    assert ymin == 0.0                                  # профиль ≥0: «0» прибит к низу


def test_slice_series_y_floor_survives_reset_zoom(app):
    p = SlicePanel()
    p.set_spectrogram(_make_sg(ns=6, nc=12))
    p.reset_zoom()                                       # #100: autoRange обоих графиков
    ymin, _ = p._series_plot.getViewBox().state["limits"]["yLimits"]
    assert ymin == 0.0                                  # лимит липкий, сброс зума не снимает пол


# ---------- #102: окно «Цветовая палитра» с превью-градиентами ----------

def test_palette_dialog_lists_all_and_selects(app):
    from awf.ui.palette_dialog import PaletteDialog
    from awf.ui.colormaps import COLORMAPS
    dlg = PaletteDialog("insight")
    assert len(dlg._rows) == len(COLORMAPS)             # строка на каждую палитру (15)
    assert dlg._rows["insight"].property("selected") is True
    got = []
    dlg.selected.connect(got.append)
    dlg._on_pick("turbo")                               # выбор палитры кликом
    assert dlg.selected_key() == "turbo" and got == ["turbo"]
    assert dlg._rows["turbo"].property("selected") is True
    assert dlg._rows["insight"].property("selected") is False


# ---------- #145: раздельный стиль простыни образца и фона ----------

def test_sheet_styles_independent(app):
    # #145: стиль простыни образца и фона выбирается раздельно (палитра/однотонный/каркас)
    w = MainWindow()
    w._sg = _make_sg(ns=10, nc=16)
    w._view3d.set_spectrogram(w._sg)
    w._view3d.set_background_sheet(np.ones(w._sg.n_channels), (w._sg.counts, w._sg.live_time_s))
    w._view3d.set_background_sheet_visible(True)
    assert w._view3d._surface_style == "palette" and w._view3d._bg_style == "palette"
    w._bg_style_combo.setCurrentIndex(2)                     # фон -> каркас
    assert w._view3d._bg_sheet.opts["drawEdges"] and not w._view3d._bg_sheet.opts["drawFaces"]
    assert w._view3d._surface_style == "palette"             # образец не тронут
    w._smp_style_combo.setCurrentIndex(1)                    # образец -> однотонный
    assert w._view3d._surface_style == "solid" and w._view3d._surface is not None
    assert w._view3d._bg_style == "wire"                     # фон не тронут
    w._on_reset_display()
    assert w._view3d._surface_style == "palette" and w._view3d._bg_style == "palette"
    w.close()


# ---------- #146: «Подложка» гасит дно обеих простыней ----------

def test_floor_toggle_hides_bg_sheet_floor(app):
    # #146: тумблер «Подложка» (#76) гасит дно и у простыни фона;
    # у «каркаса» рёбра без per-vertex alpha — дно вырезается NaN-ом.
    from awf.ui.view3d import Waterfall3DView
    v = Waterfall3DView()
    v.set_floor_visible(True)                       # #150: дефолт теперь выкл — включаем явно
    v.set_spectrogram(_make_sg(ns=12, nc=20))
    v.set_background_sheet(np.full(20, 1e-9))       # фон ~0 -> вся простыня «дно»
    v.set_background_sheet_visible(True)
    assert v._bg_sheet._meshdata.vertexColors()[:, 3].min() > 0.0   # подложка вкл — видна
    v.set_floor_visible(False)                      # выключили «Подложку»
    assert v._bg_sheet._meshdata.vertexColors()[:, 3].max() == 0.0  # дно фона погашено
    v.set_bg_sheet_style("wire")
    assert np.isnan(v._bg_sheet._meshdata.vertexes()[:, 2]).all()   # каркас: дно вырезано
    v.set_floor_visible(True)                       # вернули — рёбра снова с координатами
    assert not np.isnan(v._bg_sheet._meshdata.vertexes()[:, 2]).any()


# ---------- #147: вычет совпадающих простыней — точный ноль ----------

def test_active_spectrogram_passes_raw_block(app):
    # #147: _active_spectrogram передаёт _bg_raw в subtract_background —
    # при фон=тот же файл вычет поячеечно нулевой (как совпадающие простыни).
    from awf.model.background import background_from_range
    w = MainWindow()
    w._sg = _make_sg(ns=8, nc=12)
    w._bg_cps = background_from_range(w._sg, 0, w._sg.n_slices)
    w._bg_raw = (np.asarray(w._sg.counts, dtype=np.float64),
                 np.asarray(w._sg.live_time_s, dtype=np.float64))
    w._bg_subtract = True
    assert np.allclose(w._active_spectrogram().counts, 0.0, atol=1e-12)
    w._bg_raw = None                                # без сырого блока — прежний путь
    assert (w._active_spectrogram().counts != 0).any()
    w.close()


# ---------- #148: фоновый участок выбирается секущими плоскостями Времени ----------

def test_bg_dialog_plane_range_prefill(app):
    # #148: plane_range задан -> поля срезов предзаполнены ближайшими срезами,
    # кнопка «Из сечений» активна, accept отдаёт полуоткрытый [lo, hi).
    dlg = BackgroundDialog(20, time_offsets=np.arange(20.0) * 2.0,
                           plane_range=(6.0, 24.0))   # срезы t=0,2,..38
    assert dlg._planes_btn.isEnabled()
    assert dlg._lo.value() == 3 and dlg._hi.value() == 13   # 6.0->№3, 24.0->№12 (+1)
    dlg._on_accept()
    assert dlg.result_spec() == ("range", 3, 13)


def test_bg_dialog_no_planes_disabled(app):
    # #148: плоскости Времени не заданы -> кнопка неактивна, дефолт — весь диапазон.
    dlg = BackgroundDialog(20, time_offsets=np.arange(20.0))
    assert not dlg._planes_btn.isEnabled()
    assert dlg._lo.value() == 0 and dlg._hi.value() == 20


def test_view3d_time_plane_range(app):
    # #148: time_plane_range — None без видимых плоскостей Времени; невидимый слот -> край
    # оси (семантика #84); результат отсортирован (t0 <= t1).
    from awf.ui.view3d import Waterfall3DView
    v = Waterfall3DView()
    v.set_spectrogram(_make_sg(ns=12, nc=20))
    assert v.time_plane_range() is None                     # плоскости выключены
    v.set_plane("time", 0, 0.25, True)                      # только нижняя граница
    t0, t1 = v.time_plane_range()
    assert t0 == v.plane_value("time", 0.25)[0]
    assert t1 == float(v._t_centers[-1])                    # слот 1 невидим -> край оси
    v.set_plane("time", 1, 0.1, True)                       # верхний слот НИЖЕ нижнего
    t0, t1 = v.time_plane_range()
    assert t0 <= t1                                          # диапазон отсортирован
    assert t0 == v.plane_value("time", 0.1)[0]


def test_view3d_time_plane_range_lod_bins_cover_raw_edges(app):
    # #151: LOD-бины рельефа шире сырых срезов -> диапазон плоскостей должен возвращать
    # КРАЯ крайних бинов в сырых срезах (не центры), иначе крайний сырой срез выпадает
    # из выборки фона и после вычета остаётся «гребнем» у плоскости сечения.
    from awf.ui.view3d import Waterfall3DView
    v = Waterfall3DView()
    sg = _make_sg(ns=8, nc=12)
    v.set_spectrogram(sg, max_time=4, max_chan=12)          # LOD-бин = 2 сырых среза
    v.set_plane("time", 0, 0.0, True)
    v.set_plane("time", 1, 1.0, True)
    t0, t1 = v.time_plane_range()
    t_raw = np.asarray(sg.time_offsets_s, dtype=np.float64)
    assert t0 == float(t_raw[0])
    assert t1 == float(t_raw[-1])                            # не центр последнего бина


# ---------- #149: фон под-диапазона транслируется на всю шкалу времени ----------

def test_compute_background_range_tiles_raw_to_full_scale(app):
    # #149: под-диапазон [lo:hi) -> _bg_raw полной формы (тайлинг), не (hi-lo, nc);
    # срезы внутри участка — клоны самих себя, снаружи — циклические копии участка.
    w = MainWindow()
    w._sg = _make_sg(ns=12, nc=20)
    lo, hi = 4, 8
    w._compute_background(("range", lo, hi))
    cnt, lt = w._bg_raw
    assert cnt.shape == w._sg.counts.shape and lt.size == w._sg.n_slices
    assert np.array_equal(cnt[lo:hi], w._sg.counts[lo:hi])   # внутри — сами себя
    assert np.array_equal(cnt[0], w._sg.counts[lo])          # (0-4)%4=0 -> начало блока
    assert np.array_equal(cnt[hi], w._sg.counts[lo])         # цикл сразу после участка
    w.close()