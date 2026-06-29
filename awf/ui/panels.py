from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from awf.ui.zscale import (apply_z_scale, DEFAULT_GAIN, DEFAULT_GAMMA, DEFAULT_CLIP,
                           smooth_counts)
from awf.ui.colormaps import get_colormap
from awf.analysis.peakmap import DEFAULT_WINDOWS
from awf.model.dose import dose_rate_series  # Задача #104: мощность дозы (RadiaCode)

# Задача #110: поиск фотопиков перенесён на 3D-водопад (awf/ui/view3d.py) — раньше (#108) маркеры
# рисовались здесь, на спектре среза; оператор попросил вести поиск на 3D-спектрограмме.

class HeatmapPanel(QtWidgets.QWidget):
    """2D-карта Время(ось Y)×Энергия/канал(ось X). Цвет = log(1+counts). Прямоугольная выборка
    (pg.RectROI) задаёт окно [t_lo:t_hi, ch_lo:ch_hi] в ПОЛНЫХ индексах исходной матрицы.
    При завершении перемещения/изменения ROI испускается roiChanged(t_lo,t_hi,ch_lo,ch_hi)."""

    # сигнал несёт ПОЛНЫЕ индексы (не дисплейные): t_lo, t_hi, ch_lo, ch_hi
    roiChanged = QtCore.Signal(int, int, int, int)

    # выше этого числа ячеек карту прорежаем для отображения (защита суточных записей)
    DISPLAY_CELL_CAP = 4_000_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sg = None
        self._disp_counts = None  # последняя дисплейная (возможно прорежённая) матрица counts
        self._unit = "cps"        # единицы карты: counts | cps (Задача #44; дефолт cps — #53)
        self._z_mode = "log"      # текущая Z-шкала контраста (linear/sqrt/log)
        self._gain = DEFAULT_GAIN    # регулировка контраста (Задача 16)
        self._gamma = DEFAULT_GAMMA
        self._clip = DEFAULT_CLIP
        self._cmap_name = "insight"  # палитра карты (Задача 17)
        self._smooth = 0             # радиус усреднения спектра по энергии (Замечание IV-R4)
        self._t_scale = 1.0      # n_slices / disp_rows  (полный индекс = дисплейный * scale)
        self._ch_scale = 1.0     # n_channels / disp_cols
        self._disp_rows = 0
        self._disp_cols = 0
        # подсветка выбранных пиков (Задача 18): карта приглушается, столбцы энергий — ярко
        self._highlight_on = False
        self._energy_lines = []      # list[(energy_keV, color, label)] от NuclidePanel.linesChanged
        self._hl_items = []          # текущие вертикальные маркеры-столбцы
        self._BASE_DIM_OPACITY = 0.45
        # изолинии (Задача 20): контуры по квантильным уровням Z-преобразованной карты
        self._contours_on = False
        self._contour_levels = 5
        self._iso_items = []
        # маркеры секущих плоскостей 3D (Задача #39): Время -> горизонтальная линия (ось Y),
        # Энергия -> вертикальная (ось X); цвета совпадают с осями 3D (бирюза/пурпур)
        self._section_items = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)
        self._plot = self._glw.addPlot()
        self._plot.setLabel("bottom", "Канал (энергия)")
        self._plot.setLabel("left", "Время (срез)")
        self._plot.invertY(True)                 # время сверху вниз
        self._img = pg.ImageItem()
        self._img.setColorMap(get_colormap(self._cmap_name))  # палитра Insight по умолчанию
        self._plot.addItem(self._img)
        self._roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("y", width=2))
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._plot.addItem(self._roi)
        self._roi.setVisible(False)
        self._roi.sigRegionChangeFinished.connect(self._on_roi_finished)

    def set_spectrogram(self, sg) -> None:
        """Построить карту. Для огромных матриц (> DISPLAY_CELL_CAP ячеек) показываем
        прорежённую через sg.downsample версию, но ROI пересчитываем обратно в ПОЛНЫЕ индексы."""
        self._sg = sg
        ns, nc = sg.n_slices, sg.n_channels
        # Задача #44: источник — counts или по-срезовая скорость cps (counts/live_time)
        disp_counts = self._disp_from_source(sg, sg.counts_in_unit(self._unit))
        self._disp_counts = disp_counts
        self._disp_rows, self._disp_cols = disp_counts.shape
        self._t_scale = ns / float(self._disp_rows)
        self._ch_scale = nc / float(self._disp_cols)
        # Z-контраст по выбранной шкале (с усреднением спектра по энергии, IV-R4); row-major =>
        # ось0=строки=Время(Y), ось1=столбцы=Канал(X)
        self._img.setImage(self._scaled_image(), axisOrder="row-major", autoLevels=True)
        # Задача #82: жёлтое окно выделения (ROI) отключено — на 2D-карте его не показываем.
        # Окно по умолчанию (центральная четверть) всё равно задаём, чтобы _emit_roi отдал
        # панели срезов дефолтную «полосу ROI» (магента-кривая там продолжает работать).
        x0 = self._disp_cols * 0.25; y0 = self._disp_rows * 0.25
        w = max(1.0, self._disp_cols * 0.5); h = max(1.0, self._disp_rows * 0.5)
        self._roi.setPos([x0, y0]); self._roi.setSize([w, h])
        self._roi.maxBounds = QtCore.QRectF(0, 0, self._disp_cols, self._disp_rows)
        self._roi.setVisible(False)   # #82: скрыто (жёлтый прямоугольник убран)
        self._plot.setRange(xRange=(0, self._disp_cols), yRange=(0, self._disp_rows), padding=0)
        self._lock_view_to_data()     # Задача #87: карта не «уезжает» за пределы данных
        self._emit_roi()
        self._apply_highlight()  # перерисовать маркеры подсветки под новую геометрию (Задача 18)
        self._apply_contours()   # пересчитать изолинии под новые данные (Задача 20)
        self._clear_section_items()  # сбросить маркеры сечений старого файла (Задача #39)

    def _lock_view_to_data(self) -> None:
        """Задача #87: «привязать минимальный зум к окну» — ограничить ViewBox прямоугольником
        данных. Панорама не выходит за [0..cols]×[0..rows]; максимально отдалённый вид
        (минимальный зум) = полное окно. Карта больше не «уезжает» в пустоту."""
        vb = self._plot.getViewBox()
        if vb is None or self._disp_cols <= 0 or self._disp_rows <= 0:
            return
        vb.setLimits(xMin=0, xMax=self._disp_cols, yMin=0, yMax=self._disp_rows,
                     maxXRange=self._disp_cols, maxYRange=self._disp_rows)

    def _disp_from_source(self, sg, src):
        """Дисплейная матрица из источника src (counts или cps): прорежаем method='max', если
        ячеек больше cap, иначе берём как есть (Задача #44 — единицы задаёт вызывающий)."""
        ns, nc = sg.n_slices, sg.n_channels
        if ns * nc > self.DISPLAY_CELL_CAP:
            import math
            factor = math.sqrt(self.DISPLAY_CELL_CAP / float(ns * nc))
            disp_t = max(1, min(ns, int(ns * factor)))
            disp_c = max(1, min(nc, int(nc * factor)))
            dc, _, _ = sg.downsample(disp_t, disp_c, method="max", data=src)
            return np.asarray(dc, dtype=np.float32)
        return np.asarray(src, dtype=np.float32)

    def set_unit_mode(self, mode: str) -> None:
        """Единицы карты: 'counts' | 'cps' (Задача #44). Пересчитать дисплейную матрицу из нового
        источника и перерисовать; ROI/маркеры сечений сохраняются (autoLevels подхватит масштаб)."""
        self._unit = "cps" if mode == "cps" else "counts"
        if self._sg is not None:
            self._disp_counts = self._disp_from_source(self._sg, self._sg.counts_in_unit(self._unit))
            self._redraw()

    def _roi_full_indices(self):
        """Текущий ROI -> (t_lo, t_hi, ch_lo, ch_hi) в ПОЛНЫХ индексах матрицы, с клиппингом."""
        if self._sg is None:
            return (0, 0, 0, 0)
        pos = self._roi.pos(); size = self._roi.size()
        x0 = float(pos.x()); y0 = float(pos.y())
        x1 = x0 + float(size.x()); y1 = y0 + float(size.y())
        ch_lo = int(np.floor(min(x0, x1) * self._ch_scale))
        ch_hi = int(np.ceil(max(x0, x1) * self._ch_scale))
        t_lo = int(np.floor(min(y0, y1) * self._t_scale))
        t_hi = int(np.ceil(max(y0, y1) * self._t_scale))
        ns, nc = self._sg.n_slices, self._sg.n_channels
        ch_lo = max(0, min(nc - 1, ch_lo)); ch_hi = max(ch_lo + 1, min(nc, ch_hi))
        t_lo = max(0, min(ns - 1, t_lo)); t_hi = max(t_lo + 1, min(ns, t_hi))
        return (t_lo, t_hi, ch_lo, ch_hi)

    def _emit_roi(self) -> None:
        t_lo, t_hi, ch_lo, ch_hi = self._roi_full_indices()
        self.roiChanged.emit(t_lo, t_hi, ch_lo, ch_hi)

    def _on_roi_finished(self) -> None:
        self._emit_roi()

    def current_roi(self):
        """Публичный доступ к текущей выборке в полных индексах."""
        return self._roi_full_indices()

    def set_z_scale(self, mode: str) -> None:
        """Сменить Z-шкалу контраста и перерисовать карту (без перезагрузки данных)."""
        self._z_mode = mode
        self._redraw()

    def set_contrast(self, *, gain: float = None, gamma: float = None,
                     clip=None) -> None:
        """Обновить регулировку контраста (Задача 16) и перерисовать карту.
        Не переданные параметры сохраняют текущее значение."""
        if gain is not None:
            self._gain = float(gain)
        if gamma is not None:
            self._gamma = float(gamma)
        if clip is not None:
            self._clip = (float(clip[0]), float(clip[1]))
        self._redraw()

    def set_colormap(self, name: str) -> None:
        """Сменить палитру карты (Задача 17). LUT применяется к ImageItem, данные не трогаем."""
        self._cmap_name = name
        self._img.setColorMap(get_colormap(name))

    def _scaled_image(self):
        """Дисплейная матрица -> усреднение по энергии (IV-R4) -> Z-шкала контраста."""
        base = smooth_counts(self._disp_counts, self._smooth, axis=1)
        return apply_z_scale(base, self._z_mode, gain=self._gain,
                             gamma=self._gamma, clip=self._clip)

    def set_smoothing(self, radius: int) -> None:
        """Радиус скользящего среднего спектра по энергии (Замечание IV-R4); перерисовать карту."""
        self._smooth = max(0, int(radius))
        self._redraw()

    def _redraw(self) -> None:
        """Перерисовать карту из последней дисплейной матрицы (Z-шкала/контраст/усреднение)."""
        if self._disp_counts is not None:
            self._img.setImage(self._scaled_image(), axisOrder="row-major", autoLevels=True)
            self._apply_contours()   # уровни изолиний зависят от Z-карты -> пересчитать

    def set_energy_lines(self, lines) -> None:
        """Задать энергии-маркеры (energy_keV, color, label) для подсветки столбцов (Задача 18).
        Источник — выбранные линии нуклидов NuclidePanel.linesChanged."""
        self._energy_lines = list(lines) if lines else []
        self._apply_highlight()

    def set_highlight_enabled(self, on: bool) -> None:
        """Режим подсветки (Задача 18): карта приглушается, столбцы выбранных энергий — ярко."""
        self._highlight_on = bool(on)
        self._apply_highlight()

    def _clear_hl_items(self) -> None:
        for it in self._hl_items:
            self._plot.removeItem(it)
        self._hl_items = []

    def _apply_highlight(self) -> None:
        """Приглушить карту и поставить яркие вертикальные маркеры на выбранных энергиях.
        Энергия кэВ -> полный канал (argmin|E-e|) -> дисплейный X через _ch_scale (LOD-aware).
        Активна только когда режим включён И есть выбранные линии в диапазоне спектра."""
        self._clear_hl_items()
        if self._sg is None:
            return
        active = self._highlight_on and bool(self._energy_lines)
        self._img.setOpacity(self._BASE_DIM_OPACITY if active else 1.0)
        energies = np.asarray(self._sg.energies(), dtype=np.float64)
        if not active or energies.size == 0:
            return
        emin = float(energies.min()); emax = float(energies.max())
        for ln_t in self._energy_lines:
            energy, color, label = ln_t[0], ln_t[1], ln_t[2]   # 3- или 4-кортеж (Задача #69)
            e = float(energy)
            if e < emin or e > emax:
                continue   # энергия вне диапазона спектра — столбец не подсвечиваем
            full_ch = int(np.argmin(np.abs(energies - e)))
            x = (full_ch + 0.5) / self._ch_scale     # полный канал -> дисплейный X
            ln = pg.InfiniteLine(
                pos=x, angle=90, movable=False, pen=pg.mkPen(color, width=2),
                label=label, labelOpts={"position": 0.05, "color": color,
                                        "fill": (0, 0, 0, 130), "movable": False})
            self._plot.addItem(ln)
            self._hl_items.append(ln)

    def set_contours_enabled(self, on: bool) -> None:
        """Вкл/выкл изолинии (контурный план, Задача 20). Уровни — по квантилям интенсивности
        Z-преобразованной карты; пересчитываются при каждой перерисовке."""
        self._contours_on = bool(on)
        self._apply_contours()

    def set_contour_levels(self, n: int) -> None:
        """Число уровней изолиний (Задача 20). Перерисовать, если контуры включены."""
        self._contour_levels = max(1, int(n))
        if self._contours_on:
            self._apply_contours()

    def _clear_iso_items(self) -> None:
        for it in self._iso_items:
            self._plot.removeItem(it)
        self._iso_items = []

    def _contour_level_values(self, data) -> np.ndarray:
        """Уровни изолиний по квантилям распределения интенсивности (равномерно «по массе»),
        из Z-преобразованной карты. Дубли и граничный максимум (контур не строится) отброшены.
        Возвращает строго возрастающий вектор длиной <= _contour_levels."""
        a = np.asarray(data, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.empty(0, dtype=np.float64)
        n = max(1, int(self._contour_levels))
        qs = np.linspace(0.5, 0.97, n)
        lv = np.unique(np.round(np.quantile(a, qs), 6))
        # строго внутри диапазона (округлённого так же) — уровни на границе контур не строят,
        # а на константной карте граница совпадает с уровнем -> изолиний нет
        amin = float(np.round(a.min(), 6)); amax = float(np.round(a.max(), 6))
        return lv[(lv > amin) & (lv < amax)]

    def _apply_contours(self) -> None:
        """Построить изолинии поверх карты. Данные транспонируем: ImageItem row-major кладёт
        data[row,col] в (x=col,y=row), а IsocurveItem ставит вершину index(i,j)->(x=i,y=j),
        поэтому контуру отдаём data.T — оси изолиний совпадают с осями карты."""
        self._clear_iso_items()
        if not self._contours_on or self._disp_counts is None:
            return
        data = np.asarray(self._scaled_image(), dtype=np.float64)
        levels = self._contour_level_values(data)
        if levels.size == 0:
            return
        dT = np.ascontiguousarray(data.T)
        lo = float(levels.min()); hi = float(levels.max())
        span = (hi - lo) if hi > lo else 1.0
        for lvl in levels:
            # ярче для верхних уровней (alpha 90->230) — глубина «контурного плана»
            a = int(90 + 140 * (float(lvl) - lo) / span)
            iso = pg.IsocurveItem(data=dT, level=float(lvl),
                                  pen=pg.mkPen((255, 255, 255, a), width=1))
            iso.setZValue(10)
            self._plot.addItem(iso)
            self._iso_items.append(iso)

    def _clear_section_items(self) -> None:
        for it in self._section_items:
            self._plot.removeItem(it)
        self._section_items = []

    def _add_section_lines(self, vals, ref, scale, angle, rgb) -> None:
        """Нарисовать линии-маркеры по списку real-значений vals (None -> пропуск). ref —
        полный массив реальных координат оси, scale — LOD-множитель (полный/дисплейный)."""
        if ref.size == 0:
            return
        for v in vals:
            if v is None:
                continue
            full = int(np.argmin(np.abs(ref - float(v))))
            ln = pg.InfiniteLine(pos=(full + 0.5) / scale, angle=angle, movable=False,
                                 pen=pg.mkPen(rgb, width=2, style=QtCore.Qt.DashLine))
            self._plot.addItem(ln)
            self._section_items.append(ln)

    def set_section_markers(self, t_vals, e_vals) -> None:
        """Позиции видимых секущих плоскостей 3D на 2D-карте (Задача #39): Время -> горизонталь
        (ось Y), Энергия -> вертикаль (ось X). Real (с/кэВ) -> дисплейные коорд. через
        _t_scale/_ch_scale; цвета осей 3D (время=бирюза, энергия=пурпур)."""
        self._clear_section_items()
        if self._sg is None:
            return
        times = np.asarray(self._sg.time_offsets_s, dtype=np.float64)
        energies = np.asarray(self._sg.energies(), dtype=np.float64)
        self._add_section_lines(t_vals, times, self._t_scale, 0, (51, 217, 242))
        self._add_section_lines(e_vals, energies, self._ch_scale, 90, (242, 89, 217))


class SlicePanel(QtWidgets.QWidget):
    """Два графика: верх — спектр (Энергия кэВ → Отсчёты), низ — временной ряд (Время с → Отсчёты).
    Метод show_roi() рисует спектр окна времени и временной ряд энергетической полосы, плюс
    показывает сумму отсчётов в выборке. show_time_slice() рисует спектр одного среза.
    show_energy_window() (Задача 19) рисует временной профиль интенсивности в энергоокне."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sg = None
        self._energies = None
        self._times = None
        self._live = None         # live_time_s по срезам — делитель cps (Задача #44)
        self._unit = "cps"        # единицы графиков: counts | cps (Задача #44; дефолт cps — #53)
        self._spec_log = False    # лог-шкала Y графика спектра (Задача #43)
        self._smooth = 0          # радиус усреднения спектра по энергии (Замечание IV-R4)
        self._raw_spec = None     # (energies, spec_raw, lt_total) последнего спектра (Задача #44)
        self._raw_series = None   # (times, band_raw) кривой полосы ROII (Задача #44)
        self._raw_ewin = None     # (times, series_raw) кривой энергоокна (Задача #44)
        self._series_section_items = []  # маркеры сечений Времени на графике отсчётов (Задача #42)
        self._ewin_active = None  # (e_lo,e_hi) активного энергоокна временного профиля (Задача 19)
        self._bg_cps = None       # поканальный фон cps (Задача #96), длина = n_channels
        self._bg_overlay = False  # наложение кривой фона на спектр среза (Задача #96)
        self._dose = None         # np.ndarray дозы по срезам (Задача #104)
        self._dose_unit = "mSv/h" # единицы дозы (Задача #104): 'mSv/h' | 'uSv/h'
        self._dose_visible = True # флаг показа оверлея дозы (Задача #104)
        layout = QtWidgets.QVBoxLayout(self)
        self._header = QtWidgets.QLabel("Файл не загружен")
        self._header.setWordWrap(True)
        layout.addWidget(self._header)
        # --- энергоокно временного профиля (Задача 19.3): пресет нуклида + ручной ввод границ ---
        ewin_row = QtWidgets.QHBoxLayout()
        ewin_row.addWidget(QtWidgets.QLabel("Энергоокно:"))
        self._ewin_preset = QtWidgets.QComboBox()
        # Задача #94: пункт «— вручную —» убран (был пустышкой — его выбор ничего не делал).
        # Комбо держит только пресеты; «ручное» состояние = пустой выбор (currentIndex == -1).
        for w in DEFAULT_WINDOWS:
            self._ewin_preset.addItem(f"{w.name} ({w.center:.0f} кэВ)")
        self._ewin_preset.setCurrentIndex(-1)   # старт без активного пресета (как прежняя «вручную»)
        ewin_row.addWidget(self._ewin_preset)
        self._ewin_lo = QtWidgets.QDoubleSpinBox()
        self._ewin_hi = QtWidgets.QDoubleSpinBox()
        for sb in (self._ewin_lo, self._ewin_hi):
            sb.setDecimals(0); sb.setRange(0.0, 1.0); sb.setSuffix(" кэВ"); sb.setSingleStep(5.0)
        ewin_row.addWidget(self._ewin_lo)
        ewin_row.addWidget(QtWidgets.QLabel("–"))
        ewin_row.addWidget(self._ewin_hi)
        ewin_row.addStretch(1)
        # Задача #43: лог/лин шкала Y графика спектра среза
        self._log_check = QtWidgets.QCheckBox("лог Y")
        self._log_check.toggled.connect(self.set_spectrum_log)
        ewin_row.addWidget(self._log_check)
        # Задача #100: сброс зума/панорамы обоих графиков (спектр среза + временной ряд)
        self._reset_zoom_btn = QtWidgets.QPushButton("Сброс зума")
        self._reset_zoom_btn.setToolTip("Вернуть полный вид графиков среза и времени")
        self._reset_zoom_btn.clicked.connect(self.reset_zoom)
        ewin_row.addWidget(self._reset_zoom_btn)
        layout.addLayout(ewin_row)
        self._spectrum_plot = pg.PlotWidget()
        self._spectrum_plot.setLabel("bottom", "Энергия, кэВ")
        self._spectrum_plot.setLabel("left", "Отсчёты")
        self._spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._spectrum_plot)
        self._series_plot = pg.PlotWidget()
        self._series_plot.setLabel("bottom", "Время, с")
        self._series_plot.setLabel("left", "Отсчёты в полосе")
        self._series_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._series_plot)
        # Задача #41: кривая спектра среза — бирюза рамки плоскости Времени 3D (51,217,242)
        self._spectrum_curve = self._spectrum_plot.plot(
            [], [], pen=pg.mkPen((51, 217, 242), width=2))
        # Задача #96: кривая фона поверх спектра среза (оранжевый пунктир), в текущих единицах
        self._bg_curve = self._spectrum_plot.plot(
            [], [], pen=pg.mkPen((255, 165, 0), width=1, style=QtCore.Qt.DashLine))
        self._legend = self._series_plot.addLegend(offset=(-10, 10))
        self._series_curve = self._series_plot.plot([], [], pen=pg.mkPen("m", width=1),
                                                    name="полоса ROI")
        # жёлтая кривая — временной профиль энергоокна (Задача 19.2), независим от ROI
        self._ewin_curve = self._series_plot.plot([], [], pen=pg.mkPen("y", width=1),
                                                  name="энергоокно")
        # Задача #104: вторая ось Y (правая) — мощность дозы RadiaCode
        self._dose_vb = pg.ViewBox()
        pi = self._series_plot.getPlotItem()
        pi.scene().addItem(self._dose_vb)
        self._dose_axis = pg.AxisItem("right")
        self._dose_axis.setLabel("Мощность дозы, мЗв/ч")
        pi.layout.addItem(self._dose_axis, 2, 3)
        self._dose_axis.linkToView(self._dose_vb)
        self._dose_vb.setXLink(pi.vb)
        self._dose_curve = pg.PlotDataItem([], [],
            pen=pg.mkPen((255, 170, 0), width=2), name="доза")
        self._dose_vb.addItem(self._dose_curve)
        self._dose_in_legend = False  # #105: записи дозы в легенде ещё нет
        self._dose_axis.hide()
        self._series_plot.getViewBox().sigResized.connect(self._sync_dose_vb)
        # Задача #125: двойной клик по любому из графиков → сброс зума/смещения (reset_zoom).
        self._spectrum_plot.scene().sigMouseClicked.connect(self._on_plot_double_click)
        self._series_plot.scene().sigMouseClicked.connect(self._on_plot_double_click)
        self._nuclide_lines = []  # текущие вертикальные маркеры энергий нуклидов на спектре
        self._ewin_preset.currentIndexChanged.connect(self._on_ewin_preset)
        self._ewin_lo.editingFinished.connect(self._on_ewin_spin)
        self._ewin_hi.editingFinished.connect(self._on_ewin_spin)
        self._apply_unit_labels()    # Задача #53: дефолт cps — подписи осей Y сразу в отсч/с

    def set_spectrogram(self, sg) -> None:
        self._sg = sg
        self._energies = np.asarray(sg.energies(), dtype=np.float64)
        self._times = np.asarray(sg.time_offsets_s, dtype=np.float64)
        self._live = np.asarray(sg.live_time_s, dtype=np.float64)   # делитель cps (Задача #44)
        self._clear_series_sections()   # новые данные -> снять старые маркеры сечений (Задача #42)
        # начальный вид: полный интегральный спектр и полная полоса по времени
        spec = np.asarray(sg.total_spectrum(), dtype=np.float64)
        self._plot_spectrum(self._energies, spec, sg.live_time_total())
        band = np.asarray(sg.band_time_series(0, sg.n_channels), dtype=np.float64)
        self._set_series(self._times, band)
        self._header.setText(
            f"Загружено: срезов {sg.n_slices}, каналов {sg.n_channels}. "
            f"Интегральный спектр и полная полоса.")
        # энергоокно (Задача 19): диапазон спинбоксов = диапазон энергий, дефолт — первое окно
        emin = float(self._energies.min()); emax = float(self._energies.max())
        for sb in (self._ewin_lo, self._ewin_hi):
            sb.blockSignals(True); sb.setRange(emin, emax); sb.blockSignals(False)
        w0 = DEFAULT_WINDOWS[0]
        lo = max(emin, min(emax, float(w0.e_lo)))
        hi = max(emin, min(emax, float(w0.e_hi)))
        if hi <= lo:    # окно вне диапазона спектра -> взять центральную треть
            lo = emin + (emax - emin) / 3.0
            hi = emin + 2.0 * (emax - emin) / 3.0
        self._ewin_lo.blockSignals(True); self._ewin_lo.setValue(lo); self._ewin_lo.blockSignals(False)
        self._ewin_hi.blockSignals(True); self._ewin_hi.setValue(hi); self._ewin_hi.blockSignals(False)
        self.show_energy_window(lo, hi)
        # Задача #104: мощность дозы — только для RadiaCode .rcspg (калибровка RC-103)
        src = getattr(sg, "source_path", None) or ""
        if src.lower().endswith(".rcspg"):
            d_msvh = dose_rate_series(sg, unit="mSv/h")
            if float(d_msvh.max()) < 1.0:
                self._dose = dose_rate_series(sg, unit="uSv/h")
                self._dose_unit = "uSv/h"
            else:
                self._dose = d_msvh
                self._dose_unit = "mSv/h"
        else:
            self._dose = None
            self._dose_unit = "mSv/h"
        self._draw_dose_overlay()
        self._lock_views_to_data()    # Задача #89: графики не «уезжают» за окно данных

    def _lock_views_to_data(self) -> None:
        """Задача #89: привязать X-домен графиков к экстенту данных, чтобы они не «уезжали»
        при зуме/панораме. Верхний график — энергия [emin..emax], нижний — время [tmin..tmax];
        панорама не выходит за домен, минимальный зум (макс. отдаление) = полный домен.
        Y верхнего графика (спектр) — отдельно в _lock_spectrum_y (#101, лог/лин). Y нижнего
        графика (профиль) — здесь (#128): он всегда линейный и ≥0, «0» прибит к низу."""
        if self._energies is not None and self._energies.size:
            emin = float(self._energies.min()); emax = float(self._energies.max())
            vb = self._spectrum_plot.getViewBox()
            if vb is not None and emax > emin:
                vb.setLimits(xMin=emin, xMax=emax, maxXRange=emax - emin)
        if self._times is not None and self._times.size:
            tmin = float(self._times.min()); tmax = float(self._times.max())
            vb = self._series_plot.getViewBox()
            if vb is not None:
                # Задача #128: профиль (полоса ROI + энергоокно) — счёт/cps ≥ 0, шкала лин.;
                # фиксируем нижнюю границу Y=0, чтобы отрицательная зона не показывалась и не
                # «уезжала» при зуме/панораме (верх auto по данным). Дозовый правый ViewBox
                # независим и не затрагивается.
                vb.setLimits(yMin=0.0)
                if tmax > tmin:
                    vb.setLimits(xMin=tmin, xMax=tmax, maxXRange=tmax - tmin)

    def reset_zoom(self) -> None:
        """Задача #100: сброс зума/панорамы графиков среза и времени к полному виду данных.
        autoRange() однократно подгоняет ViewBox под содержимое; X-домен при этом не выходит
        за окно данных (ограничен _lock_views_to_data, Задача #89)."""
        for plot in (self._spectrum_plot, self._series_plot):
            vb = plot.getViewBox()
            if vb is not None:
                vb.autoRange()

    def _on_plot_double_click(self, ev) -> None:
        """Задача #125: двойной клик по графику среза/времени → сброс зума и смещения.
        Сцена pyqtgraph эмитит sigMouseClicked на каждый клик; ev.double() истинно только
        на втором клике дабл-клика. Одиночный клик не трогаем (panning/прочее поведение ViewBox)."""
        if ev.double():
            self.reset_zoom()
            ev.accept()

    def _plot_spectrum(self, energies, spec_raw, lt_total=None) -> None:
        """Кэшировать сырой спектр (+ живое время окна для cps, Задача #44) и отрисовать."""
        e = np.asarray(energies, dtype=np.float64)
        s = np.asarray(spec_raw, dtype=np.float64)
        self._raw_spec = (e, s, lt_total)
        self._render_spectrum()

    def _render_spectrum(self) -> None:
        """Перерисовать кривую спектра из кэша в текущих единицах (Задача #44) и со сглаживанием."""
        if self._raw_spec is None:
            return
        e, s, lt_total = self._raw_spec
        disp = self._spec_to_unit(s, lt_total)
        self._spectrum_curve.setData(e, smooth_counts(disp, self._smooth, axis=-1))
        self._render_background(e, lt_total)   # Задача #96: кривая фона в тех же единицах
        self._lock_spectrum_y()                # Задача #101: зафиксировать нижнюю границу Y

    def _lock_spectrum_y(self) -> None:
        """Задача #101: зафиксировать нижнюю границу оси Y графика спектра, чтобы «0» (лин.) или
        пол данных (лог) не «уезжал» при зуме/панораме. Лин.: yMin=0 (счёт ≥ 0). Лог: yMin —
        минимальное положительное отображаемое значение (вид в лог-координатах = log10)."""
        vb = self._spectrum_plot.getViewBox()
        if vb is None or self._raw_spec is None:
            return
        _, s, lt_total = self._raw_spec
        disp = self._spec_to_unit(s, lt_total)
        disp = np.asarray(smooth_counts(disp, self._smooth, axis=-1), dtype=np.float64)
        if not self._spec_log:
            vb.setLimits(yMin=0.0)
            return
        pos = disp[disp > 0.0]
        floor = float(pos.min()) if pos.size else 1e-3
        vb.setLimits(yMin=float(np.log10(floor)))

    def set_smoothing(self, radius: int) -> None:
        """Радиус скользящего среднего спектра по энергии (Замечание IV-R4); перерисовать кривую."""
        self._smooth = max(0, int(radius))
        self._render_spectrum()

    def _spec_to_unit(self, spec_raw, lt_total):
        """Спектр в текущих единицах (Задача #44): cps = сумма отсчётов / живое время окна."""
        s = np.asarray(spec_raw, dtype=np.float64)
        if self._unit == "cps" and lt_total and float(lt_total) > 0.0:
            return s / float(lt_total)
        return s

    def _series_to_unit(self, series_raw):
        """Временной ряд в текущих единицах (Задача #44): cps = по-срезово counts/live_time."""
        s = np.asarray(series_raw, dtype=np.float64)
        if self._unit == "cps" and self._live is not None:
            lt = np.asarray(self._live, dtype=np.float64)
            safe = np.where(lt > 0.0, lt, np.inf)   # «мёртвый» срез -> 0
            return s / safe[:s.size]
        return s

    def _set_series(self, times, band_raw) -> None:
        """Кэшировать сырой ряд полосы ROI и нарисовать в текущих единицах (Задача #42/#44)."""
        t = np.asarray(times, dtype=np.float64)
        b = np.asarray(band_raw, dtype=np.float64)
        self._raw_series = (t, b)
        self._series_curve.setData(t, self._series_to_unit(b))

    def _set_ewin(self, times, series_raw) -> None:
        """Кэшировать сырой ряд энергоокна и нарисовать в текущих единицах (Задача #44)."""
        t = np.asarray(times, dtype=np.float64)
        s = np.asarray(series_raw, dtype=np.float64)
        self._raw_ewin = (t, s)
        self._ewin_curve.setData(t, self._series_to_unit(s))

    def _sync_dose_vb(self) -> None:
        """Задача #104: синхронизировать геометрию dose ViewBox с основным при ресайзе."""
        self._dose_vb.setGeometry(self._series_plot.getViewBox().sceneBoundingRect())

    def set_dose_overlay(self, on: bool) -> None:
        """Задача #104: показать/скрыть оверлей мощности дозы (только для RadiaCode .rcspg)."""
        self._dose_visible = bool(on)
        self._draw_dose_overlay()

    def _draw_dose_overlay(self) -> None:
        """Задача #104: нарисовать/скрыть кривую дозы и правую ось.
        Задача #105: запись дозы в легенде появляется/исчезает вместе с кривой."""
        dose = self._dose
        if dose is None or not self._dose_visible:
            self._dose_curve.setData([], [])
            self._dose_axis.hide()
            self._set_dose_legend(False)
            return
        unit_lbl = "мкЗв/ч" if self._dose_unit == "uSv/h" else "мЗв/ч"
        self._dose_axis.setLabel("Мощность дозы, " + unit_lbl)
        self._dose_curve.setData(self._times, dose)
        self._dose_vb.autoRange()
        self._dose_axis.show()
        self._set_dose_legend(True, unit_lbl)
        self._sync_dose_vb()

    def _set_dose_legend(self, on: bool, unit_lbl: str = "мЗв/ч") -> None:
        """Задача #105: показать/скрыть запись кривой дозы в легенде графика отсчётов
        (кривая дозы в отдельном ViewBox, легенда сама её не подхватывает)."""
        if on and not self._dose_in_legend:
            self._legend.addItem(self._dose_curve, "мощность дозы, " + unit_lbl)
            self._dose_in_legend = True
        elif not on and self._dose_in_legend:
            self._legend.removeItem(self._dose_curve)
            self._dose_in_legend = False

    def _apply_unit_labels(self) -> None:
        """Подписи осей Y графиков под текущие единицы (Задача #44)."""
        if self._unit == "cps":
            self._spectrum_plot.setLabel("left", "Отсчёты/с")
            self._series_plot.setLabel("left", "Скорость в полосе, отсч/с")
        else:
            self._spectrum_plot.setLabel("left", "Отсчёты")
            self._series_plot.setLabel("left", "Отсчёты в полосе")

    def set_unit_mode(self, mode: str) -> None:
        """Единицы всех графиков среза: 'counts' | 'cps' (Задача #44). Перерисовать из кэша."""
        self._unit = "cps" if mode == "cps" else "counts"
        self._apply_unit_labels()
        self._render_spectrum()
        if self._raw_series is not None:
            self._series_curve.setData(self._raw_series[0], self._series_to_unit(self._raw_series[1]))
        if self._raw_ewin is not None:
            self._ewin_curve.setData(self._raw_ewin[0], self._series_to_unit(self._raw_ewin[1]))

    def set_background(self, bg_cps) -> None:
        """Задача #96: задать поканальный фон (cps) для наложения; None — снять. Перерисовать."""
        self._bg_cps = None if bg_cps is None else np.asarray(bg_cps, dtype=np.float64).ravel()
        self._render_spectrum()

    def set_background_overlay(self, on: bool) -> None:
        """Задача #96: вкл/выкл наложение кривой фона на спектр среза."""
        self._bg_overlay = bool(on)
        self._render_spectrum()

    def _render_background(self, energies, lt_total) -> None:
        """Задача #96: кривая фона в единицах текущего спектра. cps: bg как есть; counts:
        bg*живое_время_окна (то же окно, что у спектра). Длины bg и энергий должны совпадать.
        Задача #99: на лог-шкале нулевой/неположительный фон (ВЭ-хвост) -> nan, кривая рвётся
        по сегментам (connect='finite'). Иначе лог-ось рисует «частокол» вертикальных пунктиров
        до ~1e-10 в каналах, где фон == 0 (сумма отсчётов фонового окна там нулевая)."""
        bg = self._bg_cps
        e = np.asarray(energies, dtype=np.float64)
        if not self._bg_overlay or bg is None or bg.size != e.size:
            self._bg_curve.setData([], [])
            return
        disp = bg if self._unit == "cps" else bg * float(lt_total or 0.0)
        disp = np.asarray(smooth_counts(disp, self._smooth, axis=-1), dtype=np.float64)
        if self._spec_log:
            disp = np.where(disp > 0.0, disp, np.nan)
        self._bg_curve.setData(e, disp, connect="finite")

    def set_spectrum_log(self, on: bool) -> None:
        """Лог/лин шкала Y графика спектра среза (Задача #43)."""
        self._spec_log = bool(on)
        self._spectrum_plot.setLogMode(False, self._spec_log)
        self._render_spectrum()   # Задача #99: пере-маскировать фон под новую шкалу (nan на лог)

    def _clear_series_sections(self) -> None:
        """Снять маркеры сечений Времени с нижнего графика (Задача #42)."""
        for item in self._series_section_items:
            self._series_plot.removeItem(item)
        self._series_section_items = []

    def _draw_series_sections(self, t_vals) -> None:
        """Вертикальные бирюзовые маркеры срезов Времени на графике отсчётов (Задача #42)."""
        self._clear_series_sections()
        for v in t_vals:
            if v is None:
                continue
            ln = pg.InfiniteLine(pos=float(v), angle=90, movable=False,
                                 pen=pg.mkPen((51, 217, 242), width=1, style=QtCore.Qt.DashLine))
            self._series_plot.addItem(ln)
            self._series_section_items.append(ln)

    def set_nuclide_lines(self, lines) -> None:
        """Отметить энергии гамма-линий нуклидов вертикальными линиями на графике спектра.
        lines: итерируемое кортежей (energy_keV: float, color: str, label: str).
        Предыдущие маркеры удаляются; спектральная кривая не трогается."""
        for item in self._nuclide_lines:
            self._spectrum_plot.removeItem(item)
        self._nuclide_lines = []
        for ln_t in lines:
            energy, color, label = ln_t[0], ln_t[1], ln_t[2]   # 3- или 4-кортеж (Задача #69)
            ln = pg.InfiniteLine(
                pos=float(energy), angle=90, movable=False,
                pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
                label=label, labelOpts={"position": 0.92, "color": color,
                                        "fill": (0, 0, 0, 130), "movable": False})
            self._spectrum_plot.addItem(ln)
            self._nuclide_lines.append(ln)

    @QtCore.Slot(int)
    def show_time_slice(self, i: int) -> None:
        if self._sg is None:
            return
        i = max(0, min(self._sg.n_slices - 1, int(i)))
        spec = np.asarray(self._sg.energy_spectrum(i), dtype=np.float64)
        self._plot_spectrum(self._energies, spec, self._sg.live_time_total(i, i + 1))
        t = float(self._times[i]) if self._times is not None and self._times.size > i else 0.0
        self._header.setText(f"Срез времени #{i} (t = {t:.1f} с)")

    @QtCore.Slot(int, int, int, int)
    def show_roi(self, t_lo: int, t_hi: int, ch_lo: int, ch_hi: int) -> None:
        """Спектр = сумма по окну времени [t_lo:t_hi]; временной ряд = полоса каналов [ch_lo:ch_hi];
        заголовок = сумма отсчётов в прямоугольной выборке."""
        if self._sg is None:
            return
        spec = np.asarray(self._sg.sum_spectrum(t_lo, t_hi), dtype=np.float64)
        self._plot_spectrum(self._energies, spec, self._sg.live_time_total(t_lo, t_hi))
        band = np.asarray(self._sg.band_time_series(ch_lo, ch_hi), dtype=np.float64)
        self._set_series(self._times, band)
        total = int(self._sg.roi_sum(t_lo, t_hi, ch_lo, ch_hi))
        e_lo = float(self._energies[ch_lo]) if self._energies is not None else 0.0
        e_hi = float(self._energies[min(ch_hi, self._sg.n_channels) - 1]) if self._energies is not None else 0.0
        self._header.setText(
            f"Выборка: срезы [{t_lo}:{t_hi}], каналы [{ch_lo}:{ch_hi}] "
            f"({e_lo:.0f}–{e_hi:.0f} кэВ). Сумма отсчётов = {total}.")

    def show_energy_window(self, e_lo, e_hi) -> None:
        """Временной профиль интенсивности в энергоокне [e_lo,e_hi] — жёлтая кривая нижнего
        графика (Задача 19.2; пример: 662 кэВ → Cs-137, 1461 кэВ → K-40). На каждом срезе времени
        — сумма отсчётов по каналам окна (gross), длина = число срезов. Кривая ROI не трогается."""
        if self._sg is None:
            return
        lo, hi = sorted((float(e_lo), float(e_hi)))
        if hi <= lo:
            hi = lo + 1.0
        series = np.asarray(self._sg.energy_band_time_series(lo, hi), dtype=np.float64)
        self._set_ewin(self._times, series)
        self._ewin_active = (lo, hi)

    def _time_to_index(self, t) -> int:
        """Реальное время (с) -> ПОЛНЫЙ индекс среза (ближайший)."""
        if self._times is None or self._times.size == 0:
            return 0
        return int(np.argmin(np.abs(self._times - float(t))))

    def _energy_to_index(self, e) -> int:
        """Реальная энергия (кэВ) -> ПОЛНЫЙ индекс канала (ближайший)."""
        if self._energies is None or self._energies.size == 0:
            return 0
        return int(np.argmin(np.abs(self._energies - float(e))))

    def sync_sections(self, t_vals, e_vals) -> None:
        """Синхронизация дока срезов с секущими плоскостями 3D (Задача #38). t_vals/e_vals —
        по 2 слота реальных значений (с/кэВ), скрытая плоскость = None. Обе плоскости Энергии ->
        жёлтый профиль энергоокна; обе Времени -> ROI окна времени (каналы из энергоплоскостей
        или вся ось); одна Времени -> спектр этого среза."""
        if self._sg is None:
            return
        self._draw_series_sections(t_vals)   # бирюзовые метки срезов Времени на графике отсчётов (#42)
        e_both = e_vals[0] is not None and e_vals[1] is not None
        if e_both:
            self.show_energy_window(e_vals[0], e_vals[1])
        t_active = [v for v in t_vals if v is not None]
        if len(t_active) == 2:
            ch = ((self._energy_to_index(min(e_vals)), self._energy_to_index(max(e_vals)) + 1)
                  if e_both else (0, self._sg.n_channels))
            self.show_roi(self._time_to_index(min(t_active)),
                          self._time_to_index(max(t_active)) + 1, ch[0], ch[1])
        elif len(t_active) == 1:
            self.show_time_slice(self._time_to_index(t_active[0]))

    @QtCore.Slot(int)
    def _on_ewin_preset(self, idx: int) -> None:
        """Выбран пресет нуклида: выставить границы спинбоксов (с клиппингом к диапазону спектра)
        и перерисовать временной профиль."""
        # Задача #94: «вручную» убран — пресеты теперь с индекса 0; пустой выбор (idx == -1) = ручное.
        if self._sg is None or idx < 0 or idx >= len(DEFAULT_WINDOWS):
            return
        w = DEFAULT_WINDOWS[idx]
        emin = float(self._energies.min()); emax = float(self._energies.max())
        lo = max(emin, min(emax, float(w.e_lo)))
        hi = max(emin, min(emax, float(w.e_hi)))
        self._ewin_lo.blockSignals(True); self._ewin_lo.setValue(lo); self._ewin_lo.blockSignals(False)
        self._ewin_hi.blockSignals(True); self._ewin_hi.setValue(hi); self._ewin_hi.blockSignals(False)
        self.show_energy_window(lo, hi)

    @QtCore.Slot()
    def _on_ewin_spin(self) -> None:
        """Ручная правка границ энергоокна -> снять активный пресет (пустой выбор, Задача #94)
        и перерисовать профиль."""
        if self._sg is None:
            return
        lo = float(self._ewin_lo.value()); hi = float(self._ewin_hi.value())
        self._ewin_preset.blockSignals(True); self._ewin_preset.setCurrentIndex(-1)
        self._ewin_preset.blockSignals(False)
        self.show_energy_window(lo, hi)