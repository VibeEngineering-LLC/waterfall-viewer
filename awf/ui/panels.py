from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from awf.ui.zscale import apply_z_scale

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
        self._z_mode = "log"      # текущая Z-шкала контраста (linear/sqrt/log)
        self._t_scale = 1.0      # n_slices / disp_rows  (полный индекс = дисплейный * scale)
        self._ch_scale = 1.0     # n_channels / disp_cols
        self._disp_rows = 0
        self._disp_cols = 0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)
        self._plot = self._glw.addPlot()
        self._plot.setLabel("bottom", "Канал (энергия)")
        self._plot.setLabel("left", "Время (срез)")
        self._plot.invertY(True)                 # время сверху вниз
        self._img = pg.ImageItem()
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
        if ns * nc > self.DISPLAY_CELL_CAP:
            # подобрать дисплейный размер так, чтобы ячеек было <= cap, сохраняя пропорцию
            import math
            factor = math.sqrt(self.DISPLAY_CELL_CAP / float(ns * nc))
            disp_t = max(1, min(ns, int(ns * factor)))
            disp_c = max(1, min(nc, int(nc * factor)))
            disp_counts, _, _ = sg.downsample(disp_t, disp_c, method="max")
            disp_counts = np.asarray(disp_counts, dtype=np.float32)
        else:
            disp_counts = np.asarray(sg.counts, dtype=np.float32)
        self._disp_counts = disp_counts
        self._disp_rows, self._disp_cols = disp_counts.shape
        self._t_scale = ns / float(self._disp_rows)
        self._ch_scale = nc / float(self._disp_cols)
        # Z-контраст по выбранной шкале; row-major => ось0=строки=Время(Y), ось1=столбцы=Канал(X)
        img = apply_z_scale(disp_counts, self._z_mode)
        self._img.setImage(img, axisOrder="row-major", autoLevels=True)
        # ROI по умолчанию — центральная четверть карты (в дисплейных координатах)
        x0 = self._disp_cols * 0.25; y0 = self._disp_rows * 0.25
        w = max(1.0, self._disp_cols * 0.5); h = max(1.0, self._disp_rows * 0.5)
        self._roi.setPos([x0, y0]); self._roi.setSize([w, h])
        self._roi.maxBounds = QtCore.QRectF(0, 0, self._disp_cols, self._disp_rows)
        self._roi.setVisible(True)
        self._plot.setRange(xRange=(0, self._disp_cols), yRange=(0, self._disp_rows), padding=0)
        self._emit_roi()

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
        if self._disp_counts is not None:
            img = apply_z_scale(self._disp_counts, self._z_mode)
            self._img.setImage(img, axisOrder="row-major", autoLevels=True)


class SlicePanel(QtWidgets.QWidget):
    """Два графика: верх — спектр (Энергия кэВ → Отсчёты), низ — временной ряд (Время с → Отсчёты).
    Метод show_roi() рисует спектр окна времени и временной ряд энергетической полосы, плюс
    показывает сумму отсчётов в выборке. show_time_slice() рисует спектр одного среза."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sg = None
        self._energies = None
        self._times = None
        layout = QtWidgets.QVBoxLayout(self)
        self._header = QtWidgets.QLabel("Файл не загружен")
        self._header.setWordWrap(True)
        layout.addWidget(self._header)
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
        self._spectrum_curve = self._spectrum_plot.plot([], [], pen=pg.mkPen("c", width=1))
        self._series_curve = self._series_plot.plot([], [], pen=pg.mkPen("m", width=1))

    def set_spectrogram(self, sg) -> None:
        self._sg = sg
        self._energies = np.asarray(sg.energies(), dtype=np.float64)
        self._times = np.asarray(sg.time_offsets_s, dtype=np.float64)
        # начальный вид: полный интегральный спектр и полная полоса по времени
        spec = np.asarray(sg.total_spectrum(), dtype=np.float64)
        self._spectrum_curve.setData(self._energies, spec)
        band = np.asarray(sg.band_time_series(0, sg.n_channels), dtype=np.float64)
        self._series_curve.setData(self._times, band)
        self._header.setText(
            f"Загружено: срезов {sg.n_slices}, каналов {sg.n_channels}. "
            f"Интегральный спектр и полная полоса.")

    @QtCore.Slot(int)
    def show_time_slice(self, i: int) -> None:
        if self._sg is None:
            return
        i = max(0, min(self._sg.n_slices - 1, int(i)))
        spec = np.asarray(self._sg.energy_spectrum(i), dtype=np.float64)
        self._spectrum_curve.setData(self._energies, spec)
        t = float(self._times[i]) if self._times is not None and self._times.size > i else 0.0
        self._header.setText(f"Срез времени #{i} (t = {t:.1f} с)")

    @QtCore.Slot(int, int, int, int)
    def show_roi(self, t_lo: int, t_hi: int, ch_lo: int, ch_hi: int) -> None:
        """Спектр = сумма по окну времени [t_lo:t_hi]; временной ряд = полоса каналов [ch_lo:ch_hi];
        заголовок = сумма отсчётов в прямоугольной выборке."""
        if self._sg is None:
            return
        spec = np.asarray(self._sg.sum_spectrum(t_lo, t_hi), dtype=np.float64)
        self._spectrum_curve.setData(self._energies, spec)
        band = np.asarray(self._sg.band_time_series(ch_lo, ch_hi), dtype=np.float64)
        self._series_curve.setData(self._times, band)
        total = int(self._sg.roi_sum(t_lo, t_hi, ch_lo, ch_hi))
        e_lo = float(self._energies[ch_lo]) if self._energies is not None else 0.0
        e_hi = float(self._energies[min(ch_hi, self._sg.n_channels) - 1]) if self._energies is not None else 0.0
        self._header.setText(
            f"Выборка: срезы [{t_lo}:{t_hi}], каналы [{ch_lo}:{ch_hi}] "
            f"({e_lo:.0f}–{e_hi:.0f} кэВ). Сумма отсчётов = {total}.")
