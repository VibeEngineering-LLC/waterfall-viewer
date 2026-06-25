"""Вкладка «Аналитика» (Задача 26): 2D-проекция срезов (PCA/t-SNE/UMAP, Задача 24), точки
раскрашены кластерами (Задача 25). Клик по точке -> переход к соответствующему срезу времени
в панели срезов. Легенда кластеров + выбор методов. Все тяжёлые вычисления — в awf.analysis
(Qt-free); здесь только UI и отрисовка ScatterPlotItem.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from awf.analysis.decomposition import feature_matrix, project, METHODS as PROJ_METHODS
from awf.analysis.cluster import cluster, METHODS as CLUSTER_METHODS

# палитра кластеров (повторяется при переполнении); шум (-1) — серый
_CLUSTER_COLORS = (
    (66, 165, 245), (239, 83, 80), (102, 187, 106), (255, 167, 38),
    (171, 71, 188), (38, 198, 218), (255, 238, 88), (141, 110, 99),
)
_NOISE_COLOR = (150, 150, 150)


def _color_for(label: int):
    if label < 0:
        return _NOISE_COLOR
    return _CLUSTER_COLORS[int(label) % len(_CLUSTER_COLORS)]


class AnalyticsPanel(QtWidgets.QWidget):
    """Проекция срезов в 2D + кластерная раскраска. sliceClicked(i) — индекс среза по клику."""

    sliceClicked = QtCore.Signal(int)

    # выше этого числа срезов авто-пересчёт на загрузке не делаем (ждём кнопку)
    AUTORUN_SLICE_CAP = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sg = None
        self._scatters = []          # список ScatterPlotItem (по одному на кластер)
        layout = QtWidgets.QVBoxLayout(self)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Проекция:"))
        self._proj_combo = QtWidgets.QComboBox()
        for m in PROJ_METHODS:
            self._proj_combo.addItem(m.upper(), m)
        ctrl.addWidget(self._proj_combo)
        ctrl.addWidget(QtWidgets.QLabel("Кластеры:"))
        self._clu_combo = QtWidgets.QComboBox()
        for m in CLUSTER_METHODS:
            self._clu_combo.addItem(m, m)
        ctrl.addWidget(self._clu_combo)
        ctrl.addWidget(QtWidgets.QLabel("k:"))
        self._k_spin = QtWidgets.QSpinBox()
        self._k_spin.setRange(1, 20)
        self._k_spin.setValue(3)
        ctrl.addWidget(self._k_spin)
        self._norm_chk = QtWidgets.QCheckBox("Нормировка")
        self._norm_chk.setChecked(True)
        ctrl.addWidget(self._norm_chk)
        self._run_btn = QtWidgets.QPushButton("Пересчитать")
        self._run_btn.clicked.connect(self._recompute)
        ctrl.addWidget(self._run_btn)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        self._status = QtWidgets.QLabel("Загрузите файл и нажмите «Пересчитать».")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Компонента 1")
        self._plot.setLabel("left", "Компонента 2")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._legend = self._plot.addLegend(offset=(-10, 10))
        layout.addWidget(self._plot)

    def set_spectrogram(self, sg) -> None:
        """Принять данные; авто-пересчёт только для небольших записей (иначе ждём кнопку)."""
        self._sg = sg
        self._clear_scatter()
        if sg is not None and sg.n_slices <= self.AUTORUN_SLICE_CAP:
            self._recompute()
        else:
            n = 0 if sg is None else sg.n_slices
            self._status.setText(
                f"Срезов {n} (> {self.AUTORUN_SLICE_CAP}) — нажмите «Пересчитать» вручную.")

    def _clear_scatter(self) -> None:
        for sc in self._scatters:
            self._plot.removeItem(sc)
        self._scatters = []
        if self._legend is not None:
            self._legend.clear()

    @QtCore.Slot()
    def _recompute(self) -> None:
        """Построить матрицу признаков (спектры срезов), спроецировать в 2D и кластеризовать.
        Недоступные опциональные методы (t-SNE/UMAP/DBSCAN/HDBSCAN) перехватываются с понятным
        сообщением вместо падения."""
        if self._sg is None:
            return
        self._clear_scatter()
        try:
            X = feature_matrix(self._sg, normalize=self._norm_chk.isChecked(), log=True)
            pmethod = self._proj_combo.currentData()
            cmethod = self._clu_combo.currentData()
            k = int(self._k_spin.value())
            proj = project(X, method=pmethod, n_components=2)
            clu = cluster(X, method=cmethod, n_clusters=k)
        except ImportError as exc:
            self._status.setText(f"Метод недоступен (пакет не установлен): {exc}")
            return
        except Exception as exc:  # вырожденные данные и пр. — показать, не падать
            self._status.setText(f"Ошибка: {type(exc).__name__}: {exc}")
            return
        coords = np.asarray(proj.coords, dtype=np.float64)
        labels = np.asarray(clu.labels)
        self._draw(coords, labels)
        n_clu = int(clu.n_clusters)
        msg = (f"Метод {pmethod.upper()} / {cmethod}: точек {coords.shape[0]}, "
               f"кластеров {n_clu}")
        if proj.explained_variance is not None and len(proj.explained_variance) >= 2:
            ev = proj.explained_variance
            msg += f"; дисперсия PCA {ev[0] * 100:.0f}% / {ev[1] * 100:.0f}%"
        self._status.setText(msg)

    def _draw(self, coords: np.ndarray, labels: np.ndarray) -> None:
        """По одному ScatterPlotItem на кластер (для легенды); каждая точка несёт индекс среза."""
        for lab in sorted(set(int(v) for v in labels)):
            mask = labels == lab
            idx = np.nonzero(mask)[0]
            pts = coords[mask]
            color = _color_for(lab)
            name = "шум" if lab < 0 else f"Кластер {lab}"
            scat = pg.ScatterPlotItem(
                x=pts[:, 0], y=pts[:, 1], size=8,
                brush=pg.mkBrush(*color, 200), pen=pg.mkPen(0, 0, 0, 0),
                data=list(idx), name=name)
            scat.sigClicked.connect(self._on_points_clicked)
            self._plot.addItem(scat)
            self._scatters.append(scat)

    def _on_points_clicked(self, scatter, points) -> None:
        """Клик по точке -> индекс среза в sliceClicked (подхватывает панель срезов)."""
        if points is None or len(points) == 0:
            return
        idx = points[0].data()
        if idx is None:
            return
        self.sliceClicked.emit(int(idx))