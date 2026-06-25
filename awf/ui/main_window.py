from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from awf.io.n42_loader import load_n42
from awf.io.rcspg_loader import load_rcspg
from awf.io.aswf_loader import load_aswf
from awf.io.nuclide_lib import default_library
from awf.ui.view3d import Waterfall3DView, SectionControls
from awf.ui.panels import HeatmapPanel, SlicePanel
from awf.ui.zscale import Z_MODES
from awf.ui.colormaps import COLORMAPS
from awf.ui.nuclide_panel import NuclidePanel

def load_spectrogram(path: str, *, max_slices: int | None = None):
    """Диспетчер загрузчиков по расширению: .aswf -> AtomSpectra, .rcspg -> RadiaCode, иначе -> N42/XML."""
    suffix = Path(path).suffix.lower()
    if suffix == ".aswf":
        return load_aswf(path, max_slices=max_slices)
    if suffix == ".rcspg":
        return load_rcspg(path, max_slices=max_slices)
    return load_n42(path, max_slices=max_slices)

class LoaderThread(QtCore.QThread):
    """Фоновая загрузка спектрограммы, чтобы не блокировать UI. Результат/ошибка — через сигналы."""
    loaded = QtCore.Signal(object)   # несёт Spectrogram
    failed = QtCore.Signal(str)      # текст ошибки

    def __init__(self, path: str, max_slices: int | None = None, parent=None):
        super().__init__(parent)
        self._path = path
        self._max_slices = max_slices

    def run(self) -> None:
        try:
            sg = load_spectrogram(self._path, max_slices=self._max_slices)
            self.loaded.emit(sg)
        except Exception as exc:  # любую ошибку отдать в UI-поток, не падать
            self.failed.emit(f"{type(exc).__name__}: {exc}")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AtomSpectra Waterfall Viewer")
        self.resize(1280, 800)
        self._sg = None
        self._loader = None            # ссылка на текущий поток (чтобы не был собран GC)

        # центральная область: вкладки 3D / 2D
        self._tabs = QtWidgets.QTabWidget()
        self._view3d = Waterfall3DView()
        self._heatmap = HeatmapPanel()
        self._tabs.addTab(self._view3d, "3D Waterfall")
        self._tabs.addTab(self._heatmap, "2D Карта (Время×Энергия)")
        self.setCentralWidget(self._tabs)

        # правый док: срезы/сечения/выборки
        self._slices = SlicePanel()
        dock = QtWidgets.QDockWidget("Срезы / Сечения / Выборки", self)
        dock.setWidget(self._slices)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # связь выборки на карте -> панель срезов
        self._heatmap.roiChanged.connect(self._slices.show_roi)

        # правый док: секущие плоскости 3D (Задача 13) — во вкладке поверх дока срезов
        self._sections = SectionControls()
        self._sdock = QtWidgets.QDockWidget("Сечения (3D)", self)
        self._sdock.setWidget(self._sections)
        self._sdock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._sdock)
        self.tabifyDockWidget(dock, self._sdock)
        dock.raise_()
        self._sections.planeChanged.connect(self._on_plane_changed)

        # левый док: библиотека нуклидов; выбор -> вертикальные маркеры энергий на спектре
        self._nuclides = NuclidePanel(default_library())
        ndock = QtWidgets.QDockWidget("Нуклиды", self)
        ndock.setWidget(self._nuclides)
        ndock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, ndock)
        self._nuclides.linesChanged.connect(self._slices.set_nuclide_lines)
        # те же энергии нуклидов -> вертикальные лучи-маркеры в 3D (Задача 15)
        self._nuclides.linesChanged.connect(self._view3d.set_energy_lines)
        # и -> подсветка столбцов на 2D-карте (Задача 18)
        self._nuclides.linesChanged.connect(self._heatmap.set_energy_lines)

        self._build_menu()
        self._build_toolbar()
        self.statusBar().showMessage("Готов. Файл → Открыть… (Ctrl+O)")

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Файл")
        act_open = QtGui.QAction("Открыть…", self)
        act_open.setShortcut(QtGui.QKeySequence.Open)
        act_open.triggered.connect(self._open_dialog)
        menu.addAction(act_open)
        menu.addSeparator()
        act_quit = QtGui.QAction("Выход", self)
        act_quit.setShortcut(QtGui.QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        menu.addAction(act_quit)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Вид")
        tb.addWidget(QtWidgets.QLabel(" Z-шкала: "))
        self._z_combo = QtWidgets.QComboBox()
        for key, label in Z_MODES:
            self._z_combo.addItem(label, key)
        self._z_combo.setCurrentIndex(2)  # по умолчанию log (как было до переключателя)
        self._z_combo.currentIndexChanged.connect(self._on_z_scale_changed)
        tb.addWidget(self._z_combo)
        tb.addWidget(QtWidgets.QLabel("  Палитра: "))
        self._cmap_combo = QtWidgets.QComboBox()
        for key, label in COLORMAPS:
            self._cmap_combo.addItem(label, key)
        self._cmap_combo.setCurrentIndex(0)  # iZotope Insight по умолчанию (Задача 17)
        self._cmap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        tb.addWidget(self._cmap_combo)
        self._axes_check = QtWidgets.QCheckBox("Оси")  # подписи делений 3D (Задача 14)
        self._axes_check.setChecked(True)
        self._axes_check.toggled.connect(self._on_axes_toggled)
        tb.addWidget(self._axes_check)
        self._hl_check = QtWidgets.QCheckBox("Подсветка")  # подсветка выбранных пиков (Задача 18)
        self._hl_check.setChecked(False)
        self._hl_check.toggled.connect(self._on_highlight_toggled)
        tb.addWidget(self._hl_check)
        self._build_contrast_controls(tb)

    @QtCore.Slot(bool)
    def _on_axes_toggled(self, on: bool) -> None:
        """Переключатель подписей делений осей 3D (Задача 14)."""
        self._view3d.set_axis_labels_visible(on)

    @QtCore.Slot(bool)
    def _on_highlight_toggled(self, on: bool) -> None:
        """Переключатель режима подсветки выбранных пиков (Задача 18): база глушится в 3D и 2D,
        столбцы выбранных энергий нуклидов выделяются."""
        self._view3d.set_highlight_enabled(on)
        self._heatmap.set_highlight_enabled(on)
        self._sections.emit_all()  # 3D-поверхность пересоздана — переразместить плоскости

    @QtCore.Slot()
    def _on_colormap_changed(self) -> None:
        """Переключатель палитры -> единая палитра для 2D-карты и 3D-поверхности."""
        name = self._cmap_combo.currentData()
        self._view3d.set_colormap(name)
        self._heatmap.set_colormap(name)
        self._sections.emit_all()  # 3D-поверхность пересоздана — переразместить плоскости

    def _build_contrast_controls(self, tb) -> None:
        """Слайдеры контраста (Задача 16): усиление, гамма, верхняя отсечка перцентиля.
        Один обработчик задаёт единые параметры и для 2D-карты, и для 3D-поверхности."""
        def _mk(label, lo, hi, val):
            tb.addWidget(QtWidgets.QLabel(f"  {label}: "))
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            s.setMaximumWidth(90)
            s.valueChanged.connect(self._on_contrast_changed)
            tb.addWidget(s)
            return s
        self._gain_slider = _mk("Усиление", 20, 500, 100)   # /100 -> 0.2..5.0
        self._gamma_slider = _mk("Гамма", 20, 300, 100)     # /100 -> 0.2..3.0
        self._clip_slider = _mk("Отсечка%", 80, 100, 100)   # верхний перцентиль (100 = без отсечки)

    @QtCore.Slot()
    def _on_contrast_changed(self) -> None:
        """Слайдеры контраста -> единые параметры контраста для 2D-карты и 3D-поверхности."""
        gain = self._gain_slider.value() / 100.0
        gamma = self._gamma_slider.value() / 100.0
        clip = (0.0, float(self._clip_slider.value()))
        self._heatmap.set_contrast(gain=gain, gamma=gamma, clip=clip)
        self._view3d.set_contrast(gain=gain, gamma=gamma, clip=clip)
        # перестроение поверхности сбрасывает плоскости в исходную геометрию — обновить их
        self._sections.emit_all()

    @QtCore.Slot()
    def _on_z_scale_changed(self) -> None:
        mode = self._z_combo.currentData()
        self._view3d.set_z_scale(mode)
        self._heatmap.set_z_scale(mode)
        # после перестроения поверхности — обновить позиции/подписи секущих плоскостей
        self._sections.emit_all()

    @QtCore.Slot(str, int, float, bool)
    def _on_plane_changed(self, axis: str, slot: int, frac: float, visible: bool) -> None:
        """Слайдер/чекбокс дока «Сечения» -> позиция секущей плоскости в 3D + подпись реального значения."""
        self._view3d.set_plane(axis, slot, frac, visible)
        value, unit = self._view3d.plane_value(axis, frac)
        self._sections.set_value_label(axis, slot, f"{value:.1f} {unit}")

    @QtCore.Slot()
    def _open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Открыть спектрограмму", "",
            "Спектрограммы (*.n42 *.xml *.rcspg *.aswf);;N42 / XML (*.n42 *.xml);;"
            "RadiaCode (*.rcspg);;AtomSpectra (*.aswf);;Все файлы (*)")
        if path:
            self.open_file(path)

    def open_file(self, path: str, max_slices: int | None = None) -> None:
        """Запустить фоновую загрузку файла. UI остаётся отзывчивым."""
        self.statusBar().showMessage(f"Загрузка: {path} …")
        self._loader = LoaderThread(path, max_slices=max_slices, parent=self)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.start()

    @QtCore.Slot(object)
    def _on_loaded(self, sg) -> None:
        self._sg = sg
        # порядок важен: сперва панель срезов получает данные, затем карта — её set_spectrogram
        # испускает roiChanged, который сразу нарисует срез по умолчанию.
        self._view3d.set_spectrogram(sg)
        self._slices.set_spectrogram(sg)
        self._heatmap.set_spectrogram(sg)
        # обновить секущие плоскости под новую геометрию (позиции + реальные подписи)
        self._sections.emit_all()
        total = int(np.asarray(sg.counts).sum(dtype=np.int64))
        t0 = sg.t0_iso if sg.t0_iso else "—"
        src = sg.source_path if sg.source_path else "?"
        self.statusBar().showMessage(
            f"{src} — срезов {sg.n_slices} × каналов {sg.n_channels}; "
            f"t0={t0}; всего отсчётов={total}")

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Ошибка загрузки: {message}")
        QtWidgets.QMessageBox.critical(self, "Ошибка загрузки N42", message)

def main(argv: list[str] | None = None) -> int:
    """Точка входа. Необязательный первый аргумент — путь к файлу N42 для авто-открытия."""
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        candidate = argv[1]
        if Path(candidate).exists():
            win.open_file(candidate)
    return int(app.exec())