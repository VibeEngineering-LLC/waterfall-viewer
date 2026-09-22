"""Вкладка «Аналитика» (Задача 26): 2D-проекция срезов (PCA/t-SNE/UMAP, Задача 24), точки
раскрашены кластерами (Задача 25). Клик по точке -> переход к соответствующему срезу времени
в панели срезов. Легенда кластеров + выбор методов. Все тяжёлые вычисления — в awf.analysis
(Qt-free); здесь только UI и отрисовка ScatterPlotItem.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from awf.analysis.decomposition import feature_matrix, project, METHODS as PROJ_METHODS
from awf.analysis.cluster import cluster, METHODS as CLUSTER_METHODS
from awf.ui.i18n import tr

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
    _resultReady = QtCore.Signal(int, object)   # (поколение расчёта, результат) — из фонового потока в GUI

    # выше этого числа срезов авто-пересчёт на загрузке не делаем (ждём кнопку)
    AUTORUN_SLICE_CAP = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sg = None
        self._scatters = []          # список ScatterPlotItem (по одному на кластер)
        self._gen = 0                # Задача #PERF-2: номер расчёта; устаревший результат отбрасывается
        self._busy = False
        self._pending = False        # запрос «Пересчитать», пришедший во время активного расчёта
        self._resultReady.connect(self._on_result)
        layout = QtWidgets.QVBoxLayout(self)

        ctrl = QtWidgets.QHBoxLayout()
        self._proj_label = QtWidgets.QLabel(tr("Проекция:"))
        ctrl.addWidget(self._proj_label)
        self._proj_combo = QtWidgets.QComboBox()
        for m in PROJ_METHODS:
            self._proj_combo.addItem(m.upper(), m)
        ctrl.addWidget(self._proj_combo)
        self._clu_label = QtWidgets.QLabel(tr("Кластеры:"))
        ctrl.addWidget(self._clu_label)
        self._clu_combo = QtWidgets.QComboBox()
        for m in CLUSTER_METHODS:
            self._clu_combo.addItem(m, m)
        ctrl.addWidget(self._clu_combo)
        ctrl.addWidget(QtWidgets.QLabel("k:"))
        self._k_spin = QtWidgets.QSpinBox()
        self._k_spin.setRange(1, 20)
        self._k_spin.setValue(3)
        ctrl.addWidget(self._k_spin)
        self._norm_chk = QtWidgets.QCheckBox(tr("Нормировка"))
        self._norm_chk.setChecked(True)
        ctrl.addWidget(self._norm_chk)
        self._run_btn = QtWidgets.QPushButton(tr("Пересчитать"))
        self._run_btn.clicked.connect(self._recompute)
        ctrl.addWidget(self._run_btn)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        self._status = QtWidgets.QLabel(tr("Загрузите файл и нажмите «Пересчитать»."))
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", tr("Компонента 1"))
        self._plot.setLabel("left", tr("Компонента 2"))
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
                f"{tr('Срезов')} {n} (> {self.AUTORUN_SLICE_CAP}) — "
                f"{tr('нажмите «Пересчитать» вручную.')}")

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
        if self._busy:
            # Задача #PERF-2 (стерильный проход): повторный клик «Пересчитать» до конца предыдущего
            # расчёта плодил потоки без ограничения (2→12 за 10 кликов) и валил процесс. Новый запрос
            # запоминаем и отдаём, как только текущий поток отдаст результат (см. _on_result).
            self._pending = True
            return
        self._clear_scatter()
        self._gen += 1
        self._busy = True
        self._run_btn.setEnabled(False)
        args = (self._gen, self._sg, self._norm_chk.isChecked(), self._proj_combo.currentData(),
                self._clu_combo.currentData(), int(self._k_spin.value()))
        self._status.setText(tr("Расчёт…"))
        # Задача #PERF-2: PCA + k-means на матрице «срезы × каналы» шли в потоке интерфейса и на
        # записи 1468×8191 держали окно «Не отвечает» минутами. Считаем в фоне (numpy отпускает GIL).
        threading.Thread(target=self._work, args=args, daemon=True).start()

    @QtCore.Slot(int, object)
    def _on_result(self, gen: int, res) -> None:
        """GUI-поток: принять результат фонового расчёта и нарисовать (устаревший — отбросить)."""
        if gen != self._gen:
            return
        self._busy = False
        self._run_btn.setEnabled(True)
        if self._pending:
            self._pending = False
            self._recompute()   # запрос, накопленный во время расчёта — запустить сразу следующим
            return
        if res[0] == "import":
            self._status.setText(f"{tr('Метод недоступен (пакет не установлен):')} {res[1]}")
            return
        if res[0] == "error":
            self._status.setText(f"{tr('Ошибка')}: {res[1]}")
            return
        _, pmethod, cmethod, proj, clu = res
        coords = np.asarray(proj.coords, dtype=np.float64)
        labels = np.asarray(clu.labels)
        self._draw(coords, labels)
        n_clu = int(clu.n_clusters)
        msg = (f"{tr('Метод')} {pmethod.upper()} / {cmethod}: "
               f"{tr('точек')} {coords.shape[0]}, {tr('кластеров')} {n_clu}")
        if proj.explained_variance is not None and len(proj.explained_variance) >= 2:
            ev = proj.explained_variance
            msg += f"; {tr('дисперсия PCA')} {ev[0] * 100:.0f}% / {ev[1] * 100:.0f}%"
        self._status.setText(msg)

    def _work(self, gen, sg, norm, pmethod, cmethod, k) -> None:
        """Фоновый поток: считает проекцию и кластеры, результат отдаёт сигналом в GUI-поток."""
        try:
            X = feature_matrix(sg, normalize=norm, log=True)
            res = ("ok", pmethod, cmethod, project(X, method=pmethod, n_components=2),
                   cluster(X, method=cmethod, n_clusters=k))
        except ImportError as exc:
            res = ("import", str(exc))
        except Exception as exc:  # вырожденные данные и пр. — показать, не падать
            res = ("error", f"{type(exc).__name__}: {exc}")
        try:
            self._resultReady.emit(gen, res)
        except RuntimeError:      # панель уничтожена, пока шёл расчёт
            pass

    def wait_idle(self, timeout_s: float = 300.0) -> None:
        """Дождаться конца расчёта, обрабатывая события (для тестов и скриптов)."""
        end = time.monotonic() + timeout_s
        while self._busy and time.monotonic() < end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.005)

    def _draw(self, coords: np.ndarray, labels: np.ndarray) -> None:
        """По одному ScatterPlotItem на кластер (для легенды); каждая точка несёт индекс среза."""
        for lab in sorted(set(int(v) for v in labels)):
            mask = labels == lab
            idx = np.nonzero(mask)[0]
            pts = coords[mask]
            color = _color_for(lab)
            name = tr("шум") if lab < 0 else f"{tr('Кластер')} {lab}"
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

    def retranslate(self) -> None:
        """Задача #169: перерисовать подписи вкладки на текущем языке."""
        self._proj_label.setText(tr("Проекция:"))
        self._clu_label.setText(tr("Кластеры:"))
        self._norm_chk.setText(tr("Нормировка"))
        self._run_btn.setText(tr("Пересчитать"))
        self._plot.setLabel("bottom", tr("Компонента 1"))
        self._plot.setLabel("left", tr("Компонента 2"))
        if self._sg is None:
            self._status.setText(tr("Загрузите файл и нажмите «Пересчитать»."))