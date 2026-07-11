"""Вкладка «Прибор» (Задача #DATA-3): анализ неспектральных данных файла — мощность дозы,
температура детектора (ASWF v5), GPS-трек (точки окрашены по cps) + экспорт рядов в CSV.
Чистая сборка CSV — модульная функция build_device_csv() (тестируется без Qt-виджета).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from awf.ui.i18n import tr


def build_device_csv(sg) -> str:
    t_offset_s = np.asarray(sg.time_offsets_s)
    duration_s = np.asarray(sg.real_time_s)
    total = sg.band_time_series(0, sg.n_channels)
    lt = np.asarray(sg.live_time_s)
    cps = total / np.where(lt > 0, lt, np.inf)

    rows = [f"index,t_offset_s,duration_s,cps,dose_rate_usv_h,temperature_c,latitude,longitude"]
    for i in range(sg.n_slices):
        dose = sg.dose_rate_usv_h[i] if sg.dose_rate_usv_h is not None else np.nan
        temp = sg.temperature_c[i] if sg.temperature_c is not None else np.nan
        lat = sg.gps_track[i, 0] if sg.gps_track is not None else np.nan
        lon = sg.gps_track[i, 1] if sg.gps_track is not None else np.nan

        dose_str = "" if not np.isfinite(dose) else f"{dose:.6g}"
        temp_str = "" if not np.isfinite(temp) else f"{temp:.6g}"
        lat_str = "" if not np.isfinite(lat) else f"{lat:.6g}"
        lon_str = "" if not np.isfinite(lon) else f"{lon:.6g}"

        rows.append(
            f"{i},{t_offset_s[i]:.3f},{duration_s[i]:.3f},{cps[i]:.4f},"
            f"{dose_str},{temp_str},{lat_str},{lon_str}"
        )
    return "\n".join(rows)


# масштаб единиц времени (Задача #UI-238): подпись оси -> делитель секунд
_TUNIT_DIV = {"с": 1.0, "мин": 60.0, "ч": 3600.0}
_TUNIT_LBL = {"с": "Время, с", "мин": "Время, мин", "ч": "Время, ч"}


class DeviceDataPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._sg = None
        self._tunit = "с"   # единицы оси времени (Задача #UI-238), синхронизируется с тулбаром

        layout = QtWidgets.QVBoxLayout(self)

        # Верхняя строка
        status_layout = QtWidgets.QHBoxLayout()
        self._status = QtWidgets.QLabel(tr("Нет данных прибора."))
        self._status.setWordWrap(True)
        self._export_btn = QtWidgets.QPushButton(tr("Экспорт CSV…"))
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        status_layout.addWidget(self._status)
        status_layout.addStretch()
        status_layout.addWidget(self._export_btn)
        layout.addLayout(status_layout)

        # Графики
        self._dose_plot = pg.PlotWidget()
        self._dose_plot.setLabel("bottom", tr("Время, с"))
        self._dose_plot.setLabel("left", tr("Мощность дозы, мкЗв/ч"))
        self._dose_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._dose_plot)

        self._temp_plot = pg.PlotWidget()
        self._temp_plot.setLabel("bottom", tr("Время, с"))
        self._temp_plot.setLabel("left", tr("Температура, °C"))
        self._temp_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._temp_plot)

        self._gps_plot = pg.PlotWidget()
        self._gps_plot.setLabel("bottom", tr("Долгота"))
        self._gps_plot.setLabel("left", tr("Широта"))
        self._gps_plot.showGrid(x=True, y=True, alpha=0.3)
        self._gps_plot.setAspectLocked(True)
        layout.addWidget(self._gps_plot)

        # Кривые
        self._dose_curve = self._dose_plot.plot([], [], pen=pg.mkPen((255, 167, 38), width=2))
        self._temp_curve = self._temp_plot.plot([], [], pen=pg.mkPen((38, 198, 218), width=2))
        self._gps_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(0, 0, 0, 0))
        self._gps_plot.addItem(self._gps_scatter)

    def set_spectrogram(self, sg) -> None:
        self._sg = sg
        if sg is None:
            self._dose_curve.setData([], [])
            self._temp_curve.setData([], [])
            self._gps_scatter.setData([])
            self._export_btn.setEnabled(False)
            self._status.setText(tr("Нет данных прибора."))
            return

        div = _TUNIT_DIV.get(self._tunit, 1.0)   # Задача #UI-238: с/мин/ч
        t = np.asarray(sg.time_offsets_s, dtype=np.float64) / div

        # Доза
        dose_data = getattr(sg, "dose_rate_usv_h", None)
        if dose_data is not None and np.isfinite(dose_data).any():
            self._dose_curve.setData(t[np.isfinite(dose_data)], dose_data[np.isfinite(dose_data)])
            has_dose = True
        else:
            self._dose_curve.setData([], [])
            has_dose = False

        # Температура
        temp_data = getattr(sg, "temperature_c", None)
        if temp_data is not None and np.isfinite(temp_data).any():
            self._temp_curve.setData(t[np.isfinite(temp_data)], temp_data[np.isfinite(temp_data)])
            has_temp = True
        else:
            self._temp_curve.setData([], [])
            has_temp = False

        # Задача #UI-237: ось времени всегда от нуля до конца записи (autorange на
        # пустых данных рисовал бессмысленные 0.1..0.9)
        t_max = float(t.max()) if t.size and float(t.max()) > 0.0 else 1.0
        for _p in (self._dose_plot, self._temp_plot):
            _p.getViewBox().setLimits(xMin=0.0, xMax=t_max, maxXRange=t_max)
            _p.setXRange(0.0, t_max, padding=0)

        # GPS
        gps_data = getattr(sg, "gps_track", None)
        if gps_data is not None:
            lat = gps_data[:, 0]
            lon = gps_data[:, 1]
            ok = np.isfinite(lat) & np.isfinite(lon)
            if ok.any():
                total = np.asarray(sg.band_time_series(0, sg.n_channels), dtype=np.float64)
                lt = np.asarray(sg.live_time_s, dtype=np.float64)
                cps = total / np.where(lt > 0, lt, np.inf)
                # нормировка cps точек трека в [0,1]; постоянный cps -> все 0.5
                rng = float(cps[ok].max() - cps[ok].min())
                if rng > 0.0:
                    normed = (cps[ok] - float(cps[ok].min())) / rng
                else:
                    normed = np.full(int(ok.sum()), 0.5)

                spots = []
                for j, i in enumerate(np.nonzero(ok)[0]):
                    x = float(normed[j])
                    color = (int(55 + 200 * x), 60, int(255 - 200 * x), 220)
                    spots.append({"pos": (float(lon[i]), float(lat[i])),
                                  "brush": pg.mkBrush(*color)})
                self._gps_scatter.setData(spots)
                has_gps = True
            else:
                self._gps_scatter.setData([])
                has_gps = False
        else:
            self._gps_scatter.setData([])
            has_gps = False

        # Статус
        status_parts = [
            tr("Доза") + ": " + ("✓" if has_dose else "—"),
            tr("Температура") + ": " + ("✓" if has_temp else "—"),
            tr("GPS") + ": " + ("✓" if has_gps else "—")
        ]
        self._status.setText(" · ".join(status_parts))
        self._export_btn.setEnabled(True)

    def set_time_unit(self, unit: str) -> None:
        """Задача #UI-238: единицы оси времени с/мин/ч — синхронно с кнопкой «Время:» тулбара."""
        if unit not in _TUNIT_DIV:
            return
        self._tunit = unit
        lbl = tr(_TUNIT_LBL[unit])
        self._dose_plot.setLabel("bottom", lbl)
        self._temp_plot.setLabel("bottom", lbl)
        if self._sg is not None:
            self.set_spectrogram(self._sg)   # перерисовка дёшева: ряды короткие

    def _export_csv(self) -> None:
        if self._sg is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Экспорт CSV…"), "device_data.csv", "CSV (*.csv)"
        )
        if not path:
            return
        text = build_device_csv(self._sg)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        self._status.setText(f"{tr('Сохранено')}: {path}")

    def retranslate(self) -> None:
        self._status.setText(tr("Нет данных прибора.") if self._sg is None else self._status.text())
        self._export_btn.setText(tr("Экспорт CSV…"))
        _tlbl = tr(_TUNIT_LBL.get(self._tunit, "Время, с"))   # Задача #UI-238
        self._dose_plot.setLabel("bottom", _tlbl)
        self._dose_plot.setLabel("left", tr("Мощность дозы, мкЗв/ч"))
        self._temp_plot.setLabel("bottom", _tlbl)
        self._temp_plot.setLabel("left", tr("Температура, °C"))
        self._gps_plot.setLabel("bottom", tr("Долгота"))
        self._gps_plot.setLabel("left", tr("Широта"))
