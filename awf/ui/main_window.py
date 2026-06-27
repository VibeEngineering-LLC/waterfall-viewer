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
from awf.ui.analytics_panel import AnalyticsPanel
from awf.ui.zscale import Z_MODES
from awf.ui.colormaps import COLORMAPS
from awf.ui.nuclide_panel import NuclidePanel
from awf.ui.knobs import AdjustPanel
from awf.ui.style import APP_QSS

# Задача #40: организация/приложение для QSettings (запоминание расположения окон между
# запусками). На Windows пишется в реестр HKCU\Software\<ORG>\<APP>.
SETTINGS_ORG = "VibeEngineering-LLC"
SETTINGS_APP = "AtomSpectraWaterfallViewer"

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
        self.setStyleSheet(APP_QSS)    # серая градиентная схема оформления (Замечание IV-R1)
        self._sg = None
        self._loader = None            # ссылка на текущий поток (чтобы не был собран GC)

        # центральная область: вкладки 3D / 2D
        self._tabs = QtWidgets.QTabWidget()
        self._view3d = Waterfall3DView()
        self._heatmap = HeatmapPanel()
        self._analytics = AnalyticsPanel()      # вкладка «Аналитика» (Задача 26)
        self._tabs.addTab(self._view3d, "3D Waterfall")
        self._tabs.addTab(self._heatmap, "2D Карта (Время×Энергия)")
        self._tabs.addTab(self._analytics, "Аналитика")
        self.setCentralWidget(self._tabs)

        # правый док: срезы/сечения/выборки
        self._slices = SlicePanel()
        dock = QtWidgets.QDockWidget("Срезы / Сечения / Выборки", self)
        dock.setObjectName("dock_slices")   # Задача #40: имя нужно saveState/restoreState
        dock.setWidget(self._slices)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self._slices_dock = dock

        # связь выборки на карте -> панель срезов
        self._heatmap.roiChanged.connect(self._slices.show_roi)
        # клик по точке проекции в «Аналитике» -> показать соответствующий срез (Задача 26)
        self._analytics.sliceClicked.connect(self._on_analytics_slice)

        # правый док: секущие плоскости 3D (Задача 13) — во вкладке поверх дока срезов
        self._sections = SectionControls()
        self._sdock = QtWidgets.QDockWidget("Сечения (3D)", self)
        self._sdock.setObjectName("dock_sections")   # Задача #40
        self._sdock.setWidget(self._sections)
        self._sdock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._sdock)
        self.tabifyDockWidget(dock, self._sdock)
        dock.raise_()
        self._sections.planeChanged.connect(self._on_plane_changed)

        # левый док: библиотека нуклидов; выбор -> вертикальные маркеры энергий на спектре
        self._nuclides = NuclidePanel(default_library())
        ndock = QtWidgets.QDockWidget("Нуклиды", self)
        ndock.setObjectName("dock_nuclides")   # Задача #40
        ndock.setWidget(self._nuclides)
        ndock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, ndock)

        # Задача #35: правило QSS «QDockWidget > QWidget» красит тело панелей только при
        # WA_StyledBackground — кастомные QWidget-подклассы иначе игнорируют background из
        # таблицы стилей, и откреплённый (floating) док показывает системный светлый фон.
        # Сцены pyqtgraph внутри SlicePanel не затронуты — они потомки панели, не дока.
        for _panel in (self._slices, self._sections, self._nuclides):
            _panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self._nuclides.linesChanged.connect(self._slices.set_nuclide_lines)
        # те же энергии нуклидов -> вертикальные лучи-маркеры в 3D (Задача 15)
        self._nuclides.linesChanged.connect(self._view3d.set_energy_lines)
        # и -> подсветка столбцов на 2D-карте (Задача 18)
        self._nuclides.linesChanged.connect(self._heatmap.set_energy_lines)

        # Задача #55: панель регулировок отображения (рукоятки) в отдельном доке.
        # 6 ручек (усиление/гамма/отсечка/сглаживание/освещение + «Окно t» — ширина выборки по
        # времени, #56) с индивидуальными вкл/выкл и сбросом + общий выключатель (bypass к
        # дефолтам). Старые имена _*_slider оставлены алиасами на сами ручки (Knob имеет
        # QSlider-совместимый API) — внешний код/тесты целы.
        self._adjust = AdjustPanel()
        adock = QtWidgets.QDockWidget("Регулировки отображения", self)
        adock.setObjectName("dock_adjust")          # Задача #40: имя нужно saveState/restoreState
        adock.setWidget(self._adjust)
        adock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, adock)
        self._wire_adjust_panel()

        self._build_menu()
        self._build_toolbar()
        # Задача #62: строка статуса была скучена — задаём минимальную высоту и шрифт в коде
        # (QSS min-height сам бар QStatusBar в Qt не применяет, высота берётся из layout).
        sb = self.statusBar()
        sb.setMinimumHeight(28)
        f = sb.font(); f.setPointSize(max(10, f.pointSize() + 1)); sb.setFont(f)
        sb.showMessage("Готов. Файл → Открыть… (Ctrl+O)")

        # Задача #40: восстановить геометрию окна и раскладку доков/тулбара из прошлого запуска.
        # Вызывается ПОСЛЕ создания всех доков/тулбара (иначе restoreState не к чему применять).
        self._settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._restore_layout()

    def _wire_adjust_panel(self) -> None:
        """Задача #55: алиасы старых имён слайдеров на сами ручки (Knob ≈ QSlider по API),
        кэш применённых значений и подписка на изменения панели регулировок."""
        self._gain_slider = self._adjust.rows["gain"].knob
        self._gamma_slider = self._adjust.rows["gamma"].knob
        self._clip_slider = self._adjust.rows["clip"].knob
        self._smooth_slider = self._adjust.rows["smooth"].knob
        self._light_slider = self._adjust.rows["light"].knob
        self._adj_last = self._adjust.values()
        self._adjust.changed.connect(self._on_adjust_changed)

    def _restore_layout(self) -> None:
        """Применить сохранённые QSettings геометрию/состояние окна, если они есть."""
        geo = self._settings.value("geometry")
        state = self._settings.value("windowState")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)

    def closeEvent(self, event) -> None:
        """Задача #40: сохранить геометрию и раскладку доков/тулбара при закрытии окна."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        super().closeEvent(event)

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
        tb.setObjectName("toolbar_view")   # Задача #40: имя нужно saveState/restoreState
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
        tb.addWidget(QtWidgets.QLabel("  Единицы: "))  # Задача #44: счёт / скорость счёта
        self._unit_combo = QtWidgets.QComboBox()
        self._unit_combo.addItem("отсчёты", "counts")
        self._unit_combo.addItem("отсч/с (cps)", "cps")
        self._unit_combo.setCurrentIndex(1)   # Задача #53: дефолт — cps (connect ниже, сигнал не шлём)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        tb.addWidget(self._unit_combo)
        tb.addWidget(QtWidgets.QLabel("  Время: "))  # Задача #64: единицы оси времени 3D-сетки
        self._tunit_combo = QtWidgets.QComboBox()
        for unit in ("с", "мин", "ч"):
            self._tunit_combo.addItem(unit, unit)
        self._tunit_combo.setCurrentIndex(0)          # дефолт — секунды
        self._tunit_combo.currentIndexChanged.connect(self._on_time_unit_changed)
        tb.addWidget(self._tunit_combo)
        self._axes_check = QtWidgets.QCheckBox("Оси")  # подписи делений 3D (Задача 14)
        self._axes_check.setChecked(True)
        self._axes_check.toggled.connect(self._on_axes_toggled)
        tb.addWidget(self._axes_check)
        self._hl_check = QtWidgets.QCheckBox("Подсветка")  # подсветка выбранных пиков (Задача 18)
        self._hl_check.setChecked(False)
        self._hl_check.toggled.connect(self._on_highlight_toggled)
        tb.addWidget(self._hl_check)
        # Изолинии (Задача 20) ВРЕМЕННО отключены в UI: вернём после реализации поиска,
        # идентификации пиков и подгонки их чистыми гауссианами. Механизм контуров в
        # HeatmapPanel (set_contours_enabled / set_contour_levels) сохранён и покрыт тестами;
        # обработчики _on_contours_toggled / _on_contour_levels_changed оставлены для возврата.
        # Задача #55: регулировки усиление/гамма/отсечка/сглаживание/освещение перенесены из
        # тулбара на отдельную панель-рукоятки (док «Регулировки отображения», см. __init__).
        # Задача #51: вернуть все настройки отображения к значениям по умолчанию
        self._reset_btn = QtWidgets.QPushButton("Сброс")
        self._reset_btn.setToolTip("Вернуть настройки отображения к значениям по умолчанию")
        self._reset_btn.clicked.connect(self._on_reset_display)
        tb.addWidget(self._reset_btn)

    @QtCore.Slot()
    def _on_reset_display(self) -> None:
        """Задача #51: сброс настроек отображения к умолчанию. Каждый контрол ставится в своё
        дефолтное значение; Qt шлёт valueChanged/currentIndexChanged/toggled только при реальном
        изменении, и штатный обработчик применяет дефолт на все панели (2D/3D/срезы)."""
        self._z_combo.setCurrentIndex(2)      # log
        self._cmap_combo.setCurrentIndex(0)   # iZotope Insight
        self._unit_combo.setCurrentIndex(1)   # cps (Задача #53 — дефолт)
        self._tunit_combo.setCurrentIndex(0)  # Задача #64: единицы времени — секунды (дефолт)
        self._axes_check.setChecked(True)     # оси видимы
        self._hl_check.setChecked(False)      # подсветка выкл
        # Задача #55: регулировки (усиление/гамма/отсечка/сглаживание/освещение) живут на
        # панели-рукоятках; сброс к дефолтам + включение всех рядов/общего — одним вызовом.
        self._adjust.reset_all()

    @QtCore.Slot(bool)
    def _on_axes_toggled(self, on: bool) -> None:
        """Переключатель подписей делений осей 3D (Задача 14)."""
        self._view3d.set_axis_labels_visible(on)

    @QtCore.Slot(int)
    def _on_time_unit_changed(self, _idx: int) -> None:
        """Единицы оси времени 3D-сетки: с / мин / ч (Задача #64)."""
        unit = self._tunit_combo.currentData() or "с"
        self._view3d.set_time_unit(unit)

    @QtCore.Slot(int)
    def _on_unit_changed(self, _idx: int) -> None:
        """Глобальные единицы графиков: отсчёты / отсч-в-секунду (Задача #44). Веером на все
        панели; 3D/2D пересчитываются от исходника, поэтому переразмещаем плоскости сечений."""
        mode = self._unit_combo.currentData() or "counts"
        self._view3d.set_unit_mode(mode)
        self._heatmap.set_unit_mode(mode)
        self._slices.set_unit_mode(mode)
        self._sections.emit_all()  # 3D-поверхность пересоздана — переразместить плоскости

    @QtCore.Slot(int)
    def _on_analytics_slice(self, i: int) -> None:
        """Клик по точке проекции (Задача 26) -> показать срез в панели срезов и поднять её док."""
        self._slices.show_time_slice(int(i))
        self._slices_dock.raise_()
        self._slices_dock.show()

    @QtCore.Slot(bool)
    def _on_contours_toggled(self, on: bool) -> None:
        """Переключатель изолиний на 2D-карте (Задача 20)."""
        self._heatmap.set_contours_enabled(on)

    @QtCore.Slot(int)
    def _on_contour_levels_changed(self, n: int) -> None:
        """Число уровней изолиний на 2D-карте (Задача 20)."""
        self._heatmap.set_contour_levels(int(n))

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

    @QtCore.Slot()
    def _on_adjust_changed(self) -> None:
        """Задача #55: единый обработчик панели регулировок. Берёт эффективные значения
        (с учётом per-row и общего bypass) и применяет только изменившиеся группы — стоимость
        на один поворот ручки та же, что у прежних отдельных слайдеров (Задачи 16/IV-R4/#46)."""
        v = self._adjust.values()
        last = self._adj_last
        if (v["gain"] != last["gain"] or v["gamma"] != last["gamma"]
                or v["clip"] != last["clip"]):
            gain, gamma = v["gain"] / 100.0, v["gamma"] / 100.0
            clip = (0.0, float(v["clip"]))
            self._heatmap.set_contrast(gain=gain, gamma=gamma, clip=clip)
            self._view3d.set_contrast(gain=gain, gamma=gamma, clip=clip)
        if v["smooth"] != last["smooth"]:
            r = int(v["smooth"])
            self._view3d.set_smoothing(r)
            self._heatmap.set_smoothing(r)
            self._slices.set_smoothing(r)
        if v["light"] != last["light"]:
            self._view3d.set_light_intensity(v["light"] / 100.0)
        if v["tbin"] != last["tbin"]:                       # Задача #56: ширина выборки по t
            self._view3d.set_time_bins(self._tbin_to_bins(v["tbin"]))
        self._adj_last = v
        self._sections.emit_all()  # 3D-поверхность могла пересоздаться — переразместить плоскости

    @staticmethod
    def _tbin_to_bins(value: int) -> int:
        """Задача #56: значение ручки «Окно t» (v/100 = относит. ширина выборки по времени)
        -> число временны́х бинов max_time = round(400/ширина), клип [16..1600]. Шире выборка
        -> меньше бинов («сжатие» водопада); у́же -> больше («растяжение»). Дефолт 100
        (ширина ×1.00) -> 400 = стандартный вид (bypass)."""
        w = max(1, int(value)) / 100.0
        return max(16, min(1600, int(round(400.0 / w))))

    @QtCore.Slot()
    def _on_z_scale_changed(self) -> None:
        mode = self._z_combo.currentData()
        self._view3d.set_z_scale(mode)
        self._heatmap.set_z_scale(mode)
        # после перестроения поверхности — обновить позиции/подписи секущих плоскостей
        self._sections.emit_all()

    @QtCore.Slot(str, int, float, bool)
    def _on_plane_changed(self, axis: str, slot: int, frac: float, visible: bool) -> None:
        """Слайдер/чекбокс дока «Сечения» -> позиция секущей плоскости в 3D + подпись реального
        значения + синхронизация дока срезов (#38) и 2D-карты (#39) с выбранными сечениями."""
        self._view3d.set_plane(axis, slot, frac, visible)
        value, unit = self._view3d.plane_value(axis, frac)
        self._sections.set_value_label(axis, slot, f"{value:.1f} {unit}")
        state = self._view3d.active_plane_values()
        self._slices.sync_sections(state["time"], state["energy"])
        self._heatmap.set_section_markers(state["time"], state["energy"])

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
        # Замечание IV-R5: последний канал АЦП — мусор (переполнение); отбрасываем его единым
        # местом, чтобы все виды (2D/3D/срезы/ROI/идентификация) были консистентны.
        sg = sg.trimmed_channels(1)
        self._sg = sg
        # порядок важен: сперва панель срезов получает данные, затем карта — её set_spectrogram
        # испускает roiChanged, который сразу нарисует срез по умолчанию.
        self._view3d.set_spectrogram(sg)
        self._slices.set_spectrogram(sg)
        self._heatmap.set_spectrogram(sg)
        self._analytics.set_spectrogram(sg)   # вкладка «Аналитика» (Задача 26)
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
    app.setOrganizationName(SETTINGS_ORG)     # Задача #40: дефолтные имена для QSettings()
    app.setApplicationName(SETTINGS_APP)
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        candidate = argv[1]
        if Path(candidate).exists():
            win.open_file(candidate)
    return int(app.exec())