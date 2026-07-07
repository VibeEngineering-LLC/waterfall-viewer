from __future__ import annotations
import os
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt5')
import sys
from pathlib import Path
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from awf.io.n42_loader import load_n42
from awf.io.rcspg_loader import load_rcspg
from awf.io.aswf_loader import load_aswf
from awf.io.nuclide_lib import default_library
from awf.ui.view3d import Waterfall3DView, SectionControls
from awf.ui.panels import HeatmapPanel, SlicePanel
from awf.analysis.peaks import auto_calibrate_fwhm_model   # Задача #130: модель FWHM(E) для идентификации
from awf.ui.analytics_panel import AnalyticsPanel
from awf.ui.background_dialog import BackgroundDialog   # Задача #96
from awf.model.background import (background_from_range, background_from_spectrogram,
                                  subtract_background,   # #138: прямой поканальный вычет сырых данных
                                  tile_background_block)  # #149: трансляция сырого блока на шкалу
from awf.ui.zscale import Z_MODES
from awf.ui.colormaps import COLORMAPS
from awf.ui.palette_dialog import PaletteDialog
from awf.ui.nuclide_panel import NuclidePanel
from awf.ui.peaks_panel import PeaksPanel   # Задача #111: панель «Найденные пики»
from awf.ui.segments_panel import SegmentsPanel   # Задача #131: панель «Сегментация по времени»
from awf.ui.calibration_dialog import CalibrationDialog   # Задача #215: интерактивная калибровка
from awf.model.spectrogram import Calibration   # Задача #215: заменить калибровку после фита
from awf.analysis.segment import segment_by_time, identify_segments   # Задача #131
from awf.analysis.efficiency import (default_gamma1s, load_efficiency_curve,
                                     apply_efficiency)   # Задача #156
from awf.ui.knobs import AdjustPanel
from awf.ui.cyclebutton import CycleButton   # Задача #74: переключатель-перебор вместо QComboBox
from awf.ui.style import APP_QSS
from awf.ui.edge_bar import EdgeBar   # #228: боковые флажки скрытых доков
from awf.ui import i18n          # Задача #106: переключение языка интерфейса RU↔EN
# PyQtAds: на Windows PyQtAds.pyd требует что Qt5-DLL уже в PATH или добавлены через
# os.add_dll_directory (Python 3.8+). Добавляем путь к Qt5/bin из пакета PyQt5 до импорта.
try:
    import importlib.util as _ilu
    _pyqt5_spec = _ilu.find_spec("PyQt5")
    if _pyqt5_spec and _pyqt5_spec.submodule_search_locations:
        _qt5_bin = os.path.join(list(_pyqt5_spec.submodule_search_locations)[0], "Qt5", "bin")
        if os.path.isdir(_qt5_bin):
            os.add_dll_directory(_qt5_bin)
    from PyQtAds import ads as _QtAds   # CDockManager, CDockWidget, *DockWidgetArea
    _ADS_AVAILABLE = True
except (ImportError, OSError, AttributeError):
    _QtAds = None
    _ADS_AVAILABLE = False
from awf.ui.help_dialogs import show_help, show_about, check_for_updates   # Задача #182/#202
from awf.ui.i18n import tr       # короткий доступ к переводу: tr("Файл") -> "File" / "Файл"
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
    loaded = QtCore.pyqtSignal(object)   # несёт Spectrogram
    failed = QtCore.pyqtSignal(str)      # текст ошибки

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
        # Задача #106: i18n — QSettings и язык СНАЧАЛА, потому что подписи меню/тулбара
        # ниже регистрируются через self._register_i18n(...) уже с tr() применённым.
        self._settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        saved_lang = self._settings.value("interface/language", i18n.DEFAULT, type=str)
        i18n.set_language(saved_lang)
        self._i18n_widgets: list[tuple[object, str]] = []
        i18n.signals.changed.connect(self._on_language_changed)
        self._register_i18n(self.setWindowTitle, "Waterfall Viewer")
        self.resize(1280, 800)
        # Задача #117: тему ставим и на уровне ПРИЛОЖЕНИЯ, не только окна. Контекстные
        # меню pyqtgraph (ViewBoxMenu) — popup БЕЗ QWidget-родителя, поэтому stylesheet
        # окна до них не каскадирует и они рисуются системной светлой темой. app-level
        # QSS достаёт и такие parentless-попапы (меню/субменю/спинбоксы экспорта).
        _app = QtWidgets.QApplication.instance()
        if _app is not None:
            _app.setStyleSheet(APP_QSS)
        self.setStyleSheet(APP_QSS)    # серая градиентная схема оформления (Замечание IV-R1)
        self._sg = None
        self._loader = None            # ссылка на текущий поток (чтобы не был собран GC)
        # Задача #96: фон и вычитание. _bg_cps — поканальная скорость фона (cps), выровненная по
        # энергии текущего файла; _bg_subtract/_bg_overlay — состояние пунктов меню «Анализ».
        self._bg_cps = None
        self._bg_raw = None        # Задача #139: (counts_блок, lt_блок) сырого фонового окна (range-источник)
        self._bg_subtract = False
        self._bg_overlay = False
        # Задача #156: нормализация по эффективности ε(E). Кривая — свойство детектора,
        # переживает загрузку нового файла; по умолчанию — измеренная Гамма-1С.
        self._eff_curve = default_gamma1s()
        self._eff_normalize = False

        # центральная область: вкладки 3D / 2D
        self._tabs = QtWidgets.QTabWidget()
        self._view3d = Waterfall3DView()
        self._heatmap = HeatmapPanel()
        self._analytics = AnalyticsPanel()      # вкладка «Аналитика» (Задача 26)
        # Задача #106: подписи вкладок через i18n-реестр (см. _register_i18n / _retranslate_ui)
        _idx_3d = self._tabs.addTab(self._view3d, "3D Waterfall")
        _idx_2d = self._tabs.addTab(self._heatmap, "2D Карта (Время×Энергия)")
        _idx_an = self._tabs.addTab(self._analytics, "Аналитика")
        self._register_i18n(lambda s: self._tabs.setTabText(_idx_3d, s), "3D Waterfall")
        self._register_i18n(lambda s: self._tabs.setTabText(_idx_2d, s), "2D Карта (Время×Энергия)")
        self._register_i18n(lambda s: self._tabs.setTabText(_idx_an, s), "Аналитика")
        # #228: контейнер с боковыми EdgeBar-полосами
        self._left_edge = EdgeBar("left")
        self._right_edge = EdgeBar("right")
        _cnt = QtWidgets.QWidget()
        _lay = QtWidgets.QHBoxLayout(_cnt)
        _lay.setContentsMargins(0, 0, 0, 0)
        _lay.setSpacing(0)
        _lay.addWidget(self._left_edge)
        _lay.addWidget(self._tabs)
        _lay.addWidget(self._right_edge)

        if _ADS_AVAILABLE:
            self._ads_manager = _QtAds.CDockManager(self)
            self.setCentralWidget(self._ads_manager)
            _center_dock = _QtAds.CDockWidget("__center__")
            _center_dock.setWidget(_cnt)
            _center_dock.setFeature(_QtAds.CDockWidget.NoTab, True)
            self._ads_manager.addDockWidget(_QtAds.CenterDockWidgetArea, _center_dock)
        else:
            self._ads_manager = None
            self.setCentralWidget(_cnt)

        def _make_dock(title, widget, area, obj_name, tab_after=None):
            if _ADS_AVAILABLE:
                d = _QtAds.CDockWidget(tr(title))
                d.setObjectName(obj_name)
                d.setWidget(widget)
                ads_area = (_QtAds.RightDockWidgetArea if area == "right"
                            else _QtAds.LeftDockWidgetArea)
                if tab_after is not None:
                    existing_area = tab_after.dockAreaWidget()
                    self._ads_manager.addDockWidget(
                        _QtAds.CenterDockWidgetArea, d, existing_area)
                else:
                    self._ads_manager.addDockWidget(ads_area, d)
                self._register_i18n(d.setWindowTitle, title)
                return d
            else:
                d = QtWidgets.QDockWidget(tr(title), self)
                d.setObjectName(obj_name)
                d.setWidget(widget)
                qt_area = (QtCore.Qt.RightDockWidgetArea if area == "right"
                           else QtCore.Qt.LeftDockWidgetArea)
                self.addDockWidget(qt_area, d)
                if tab_after is not None:
                    self.tabifyDockWidget(tab_after, d)
                self._register_i18n(d.setWindowTitle, title)
                return d

        self._slices = SlicePanel()
        dock = _make_dock("Срезы Ш Сечения / Выборки", self._slices, "right", "dock_slices")
        self._slices_dock = dock
        self._heatmap.roiChanged.connect(self._slices.show_roi)
        self._analytics.sliceClicked.connect(self._on_analytics_slice)

        self._sections = SectionControls()
        self._sdock = _make_dock("Сечения (3D)", self._sections, "right", "dock_sections",
                                 tab_after=dock)
        if _ADS_AVAILABLE:
            dock.setAsCurrentTab()
        else:
            dock.raise_()
        self._sections.planeChanged.connect(self._on_plane_changed)

        self._nuclides = NuclidePanel(default_library())
        ndock = _make_dock("Библиотека нуклидов", self._nuclides, "left", "dock_nuclide_lib")
        self._nlib_dock = ndock
        nidock = _make_dock("Идентификация по найденным пикам",
                            self._nuclides.ident_widget, "left", "dock_nuclide_ident",
                            tab_after=ndock)
        self._nident_dock = nidock

        for _panel in (self._slices, self._sections, self._nuclides,
                       self._nuclides.ident_widget):
            _panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self._nuclides.linesChanged.connect(self._slices.set_nuclide_lines)
        self._nuclides.linesChanged.connect(self._view3d.set_energy_lines)
        self._nuclides.linesChanged.connect(self._heatmap.set_energy_lines)

        self._adjust = AdjustPanel()
        adock = _make_dock("Регулировки отображения", self._adjust, "left", "dock_adjust")
        self._adock = adock
        self._wire_adjust_panel()

        self._peaks_panel = PeaksPanel()
        self._peaks_panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        pdock = _make_dock("Найденные пики", self._peaks_panel, "left", "dock_peaks",
                   tab_after=adock)
        self._peaks_dock = pdock
        self._peaks_panel.sigmaChanged.connect(self._on_peaks_sigma_changed)
        self._peaks_panel.peakSelected.connect(self._view3d.set_peak_highlight)
        self._peaks_panel.peakVisibilityChanged.connect(self._view3d.set_peak_visible)

        self._segments_panel = SegmentsPanel()
        self._segments_panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        segdock = _make_dock("Сегментация по времени", self._segments_panel, "left",
                     "dock_segments", tab_after=pdock)
        self._segments_dock = segdock
        self._segments_panel.recomputeRequested.connect(self._on_segment_recompute)
        self._segments_panel.nuclideSelected.connect(self._nuclides._check_nuclide)

        if _ADS_AVAILABLE:
            for _d, _side in [
                (dock, "right"), (self._sdock, "right"),
                (ndock, "left"), (nidock, "left"),
                (adock, "left"), (pdock, "left"), (segdock, "left"),
            ]:
                _d.viewToggled.connect(
                    lambda vis, d=_d, s=_side: self._on_dock_visibility(d, vis, s)
                )
        else:
            for _d, _side in [
                (dock, "right"), (self._sdock, "right"),
                (ndock, "left"), (nidock, "left"),
                (adock, "left"), (pdock, "left"), (segdock, "left"),
            ]:
                _d.visibilityChanged.connect(
                    lambda vis, d=_d, s=_side: self._on_dock_visibility(d, vis, s)
                )

        self._build_menu()
        self._build_toolbar()
        # Задача #62: строка статуса была скучена — задаём минимальную высоту и шрифт в коде
        # (QSS min-height сам бар QStatusBar в Qt не применяет, высота берётся из layout).
        sb = self.statusBar()
        sb.setMinimumHeight(28)
        f = sb.font(); f.setPointSize(max(10, f.pointSize() + 1)); sb.setFont(f)
        sb.showMessage(tr("Готов. Файл → Открыть… (Ctrl+O)"))

        # Задача #40: восстановить геометрию окна и раскладку доков/тулбара из прошлого запуска.
        # Вызывается ПОСЛЕ создания всех доков/тулбара (иначе restoreState не к чему применять).
        # Задача #106: self._settings уже создан в начале __init__ (нужен для загрузки языка),
        # повторная инициализация снята.
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

    def _on_dock_visibility(self, dock, visible: bool, side: str) -> None:
        """#228: показать/скрыть кнопку-флажок в EdgeBar при скрытии/показе дока."""
        edge = self._left_edge if side == "left" else self._right_edge
        if visible:
            edge.remove_dock(dock)
        else:
            edge.add_dock(dock)

    def _restore_layout(self) -> None:
        """Применить сохранённые QSettings геометрию/состояние окна, если они есть."""
        geo = self._settings.value("geometry")
        state = self._settings.value("windowState")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)
        # Задача #230: восстановить состояние PyQtAds CDockManager
        if self._ads_manager is not None:
            ads_state = self._settings.value("adsState")
            if ads_state is not None:
                self._ads_manager.restoreState(ads_state)

    def closeEvent(self, event) -> None:
        """Задача #40: сохранить геометрию и раскладку доков/тулбара при закрытии окна."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        # Задача #230: сохранить состояние PyQtAds CDockManager
        if self._ads_manager is not None:
            self._settings.setValue("adsState", self._ads_manager.saveState())
        super().closeEvent(event)

    def _register_i18n(self, setter, ru_key: str) -> None:
        """Задача #106: запомнить виджет в i18n-реестре и сразу применить tr()."""
        self._i18n_widgets.append((setter, ru_key))
        setter(tr(ru_key))

    def _on_language_changed(self, code: str) -> None:
        """Задача #106: реакция на смену языка — сохранить выбор и перерисовать всё зарегистрированное."""
        self._settings.setValue("interface/language", code)
        for setter, key in self._i18n_widgets:
            try:
                setter(tr(key))
            except RuntimeError:
                pass
        # Задача #111: PeaksPanel имеет собственный retranslate() для заголовков колонок
        try:
            self._peaks_panel.retranslate()
        except (AttributeError, RuntimeError):
            pass
        # Задача #131: SegmentsPanel — собственный retranslate() заголовков/кнопки/статуса
        try:
            self._segments_panel.retranslate()
        except (AttributeError, RuntimeError):
            pass
        # Задача #169: остальные панели/контролы с собственным retranslate()
        for name in ("_nuclides", "_analytics", "_heatmap", "_slices",
                     "_sections", "_adjust"):
            panel = getattr(self, name, None)
            try:
                panel.retranslate()
            except (AttributeError, RuntimeError):
                pass

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Файл")
        self._register_i18n(menu.setTitle, "Файл")   # Задача #106
        act_open = QtWidgets.QAction("Открыть…", self)
        act_open.setShortcut(QtGui.QKeySequence.Open)
        act_open.triggered.connect(self._open_dialog)
        self._register_i18n(act_open.setText, "Открыть…")
        menu.addAction(act_open)
        # Задача #216: сохранение в .aswf с текущей калибровкой
        act_save_as = QtWidgets.QAction("Сохранить как…", self)
        act_save_as.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._save_as_aswf)
        self._register_i18n(act_save_as.setText, "Сохранить как…")
        menu.addAction(act_save_as)
        # Задача #217: экспорт агрегированного спектра в BecqMoni/LSRM/InterSpec
        act_export = QtWidgets.QAction("Экспорт спектра…", self)
        act_export.setShortcut(QtGui.QKeySequence("Ctrl+E"))  # #MENU-4
        act_export.triggered.connect(self._export_spectrum)
        self._register_i18n(act_export.setText, "Экспорт спектра…")
        menu.addAction(act_export)
        menu.addSeparator()
        act_quit = QtWidgets.QAction("Выход", self)
        act_quit.setShortcut(QtGui.QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        self._register_i18n(act_quit.setText, "Выход")
        menu.addAction(act_quit)
        self._build_stub_menus()   # Задача #75: каркас верхних меню (наполнение позже)

    def _build_stub_menus(self) -> None:
        """#MENU-2 (после #75): порядок Файл→Вид→Анализ→Калибровка→Сервис→Справка.
        «Изотопы» и «О программе» верхнего уровня убраны — доки нуклидов в «Вид»,
        About/Updates в «Справке» (Windows-конвенция)."""
        bar = self.menuBar()
        self._menus = {}
        spec = [
            ("view", "Вид"),                 # #MENU-2: было «Инструменты»
            ("analysis", "Анализ"),
            ("calibration", "Калибровка"),   # #215: перекалибровка
            ("service", "Сервис"),
            ("help", "Справка"),             # #MENU-2: было «Помощь» + смёржено «О программе»
        ]
        for key, title in spec:
            m = bar.addMenu(title)
            self._register_i18n(m.setTitle, title)   # Задача #106: подпись меню через tr()
            if key == "view":
                self._build_view_menu(m)         # #MENU-2: доки (было _build_tools_menu)
            elif key == "analysis":
                self._build_analysis_menu(m)     # #96: фон и вычитание
            elif key == "calibration":
                self._build_calibration_menu(m)  # #215
            elif key == "service":
                self._build_service_menu(m)      # #106: язык RU/EN
            elif key == "help":
                self._build_help_menu(m)         # #MENU-2: Справка + About + Updates
            self._menus[key] = m

    def _build_calibration_menu(self, m) -> None:
        """Задача #215: пункты меню «Калибровка»."""
        act_fit = QtWidgets.QAction("Калибровка по пикам…", self)
        act_fit.setShortcut(QtGui.QKeySequence("Ctrl+K"))  # #MENU-4
        act_fit.triggered.connect(self._open_calibration_dialog)
        self._register_i18n(act_fit.setText, "Калибровка по пикам…")
        m.addAction(act_fit)

    def _build_analysis_menu(self, m) -> None:
        """Задача #96: пункты «Анализ» — фон и вычитание. «Выбор фона» задаёт поканальный фон
        (диапазон срезов текущего файла или отдельный файл); «Наложение» рисует фон поверх
        спектра среза; «Вычет» вычитает фон из всего водопада (3D/2D/срез), отрицательное -> 0."""
        # #MENU-3: группа «Фон» — подменю. #MENU-4: хоткеи Ctrl+B / Ctrl+Shift+B.
        bg_menu = m.addMenu("Фон")
        self._bg_menu = bg_menu  # держим Python-ref: PySide6 иначе может GC-нуть wrapper подменю
        self._register_i18n(bg_menu.setTitle, "Фон")
        act_sel = QtWidgets.QAction("Выбор фона…", self)
        act_sel.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        act_sel.triggered.connect(self._on_bg_select)
        self._register_i18n(act_sel.setText, "Выбор фона…")
        bg_menu.addAction(act_sel)
        self._act_bg_overlay = QtWidgets.QAction("Наложение фона", self)
        self._act_bg_overlay.setCheckable(True)
        self._act_bg_overlay.setEnabled(False)
        self._act_bg_overlay.toggled.connect(self._on_bg_overlay_toggled)
        self._register_i18n(self._act_bg_overlay.setText, "Наложение фона")
        bg_menu.addAction(self._act_bg_overlay)
        self._act_bg_subtract = QtWidgets.QAction("Вычет фона", self)
        self._act_bg_subtract.setShortcut(QtGui.QKeySequence("Ctrl+Shift+B"))
        self._act_bg_subtract.setCheckable(True)
        self._act_bg_subtract.setEnabled(False)
        self._act_bg_subtract.toggled.connect(self._on_bg_subtract_toggled)
        self._register_i18n(self._act_bg_subtract.setText, "Вычет фона")
        bg_menu.addAction(self._act_bg_subtract)
        m.addSeparator()
        # Задача #104: оверлей мощности дозы (только RadiaCode .rcspg, калибровка RC-103)
        self._act_dose = QtWidgets.QAction("Мощность дозы (RadiaCode)", self)
        self._act_dose.setCheckable(True)
        self._act_dose.setChecked(False)
        self._act_dose.setEnabled(False)
        self._register_i18n(self._act_dose.setText, "Мощность дозы (RadiaCode)")
        self._register_i18n(self._act_dose.setToolTip, "Калибровка дозы доступна только для RadiaCode (.rcspg)")
        self._act_dose.toggled.connect(self._on_dose_toggled)
        m.addAction(self._act_dose)
        m.addSeparator()
        # #MENU-3: группа «Пики» — подменю. #MENU-4: Ctrl+F / Ctrl+T.
        peaks_menu = m.addMenu("Пики")
        self._peaks_menu = peaks_menu
        self._register_i18n(peaks_menu.setTitle, "Пики")
        # #110: поиск фотопиков (Mariscotti + Currie) на 3D-спектрограмме.
        self._act_peaks = QtWidgets.QAction("Поиск пиков", self)
        self._act_peaks.setShortcut(QtGui.QKeySequence("Ctrl+F"))
        self._act_peaks.setCheckable(True)
        self._act_peaks.setChecked(False)
        self._register_i18n(self._act_peaks.setText, "Поиск пиков")
        self._register_i18n(self._act_peaks.setToolTip,
                            "Отметить найденные фотопики на 3D-спектрограмме")
        self._act_peaks.toggled.connect(self._on_peaks_toggled)
        peaks_menu.addAction(self._act_peaks)
        # #131: авто-сегментация по времени + посегментная идентификация.
        self._act_segments = QtWidgets.QAction("Сегментация по времени…", self)
        self._act_segments.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        self._register_i18n(self._act_segments.setText, "Сегментация по времени…")
        self._register_i18n(self._act_segments.setToolTip,
                            "Разбить запись по времени и идентифицировать нуклиды в каждом сегменте")
        self._act_segments.triggered.connect(self._on_segments_action)
        peaks_menu.addAction(self._act_segments)
        m.addSeparator()
        # #MENU-3: группа «Эффективность» — подменю. #156: ε(E)-нормализация.
        eff_menu = m.addMenu("Эффективность")
        self._eff_menu = eff_menu
        self._register_i18n(eff_menu.setTitle, "Эффективность")
        self._act_eff_norm = QtWidgets.QAction("Нормализация по эффективности", self)
        self._act_eff_norm.setCheckable(True)
        self._act_eff_norm.setChecked(False)
        self._register_i18n(self._act_eff_norm.setText, "Нормализация по эффективности")
        self._register_i18n(self._act_eff_norm.setToolTip,
                            "Умножить отсчёты каналов на ε_ref/ε(E) — компенсация падения "
                            "эффективности фотопика с энергией")
        self._act_eff_norm.toggled.connect(self._on_eff_norm_toggled)
        eff_menu.addAction(self._act_eff_norm)
        act_eff_load = QtWidgets.QAction("Загрузить кривую эффективности…", self)
        self._register_i18n(act_eff_load.setText, "Загрузить кривую эффективности…")
        act_eff_load.triggered.connect(self._on_eff_load)
        eff_menu.addAction(act_eff_load)
        # #MENU-5: имя текущей кривой ε(E) переехало из меню в статусбар (QLabel).
        self._update_eff_info()

    def _build_service_menu(self, m) -> None:
        """Задача #106: «Сервис» → подменю «Язык» с пунктами Русский / English (QActionGroup,
        эксклюзивный выбор, отмеченный текущий язык). Смена пункта — i18n.set_language(code)."""
        lang_menu = m.addMenu("Язык")
        self._register_i18n(lang_menu.setTitle, "Язык")
        group = QtWidgets.QActionGroup(self)
        group.setExclusive(True)
        cur = i18n.current_language()
        for code, ru_label in ((i18n.LANG_RU, "Русский"), (i18n.LANG_EN, "English")):
            act = QtWidgets.QAction(ru_label, self)
            act.setCheckable(True)
            act.setChecked(code == cur)
            act.triggered.connect(lambda _checked, c=code: i18n.set_language(c))
            self._register_i18n(act.setText, ru_label)
            group.addAction(act)
            lang_menu.addAction(act)

    def _build_view_menu(self, m) -> None:
        """#MENU-2 (было _build_tools_menu, #115): меню «Вид» — все окна-доки.
        toggleViewAction на каждый док. Нуклидные доки #173 добавлены сюда единожды
        (дубли из бывшего меню «Изотопы» устранены)."""
        for dock, label in (
            (self._peaks_dock, "Найденные пики"),
            (self._segments_dock, "Сегментация по времени"),   # Задача #131
            (self._slices_dock, "Срезы / Сечения / Выборки"),
            (self._sdock, "Сечения (3D)"),
            (self._adock, "Регулировки отображения"),
        ):
            act = dock.toggleViewAction()
            self._register_i18n(act.setText, label)
            m.addAction(act)
        m.addSeparator()
        # #173: два дока нуклидов (переехали из бывшего меню «Изотопы»)
        act_nlib = self._nlib_dock.toggleViewAction()
        self._register_i18n(act_nlib.setText, "Библиотека нуклидов")
        m.addAction(act_nlib)
        act_nident = self._nident_dock.toggleViewAction()
        self._register_i18n(act_nident.setText, "Идентификация по найденным пикам")
        m.addAction(act_nident)

    def _build_help_menu(self, m) -> None:
        """#MENU-2: «Справка» — было «Помощь» + смёржено «О программе»
        (Windows-конвенция: About + Updates в Help)."""
        act_help = QtWidgets.QAction("Справка…", self)
        act_help.setShortcut(QtGui.QKeySequence("F1"))
        act_help.triggered.connect(lambda: show_help(self))
        self._register_i18n(act_help.setText, "Справка…")
        m.addAction(act_help)
        m.addSeparator()
        act_upd = QtWidgets.QAction("Проверить обновления", self)
        act_upd.triggered.connect(lambda: check_for_updates(self))
        self._register_i18n(act_upd.setText, "Проверить обновления")
        m.addAction(act_upd)
        act_about = QtWidgets.QAction("О программе…", self)
        act_about.triggered.connect(lambda: show_about(self))
        self._register_i18n(act_about.setText, "О программе…")
        m.addAction(act_about)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Вид")
        tb.setObjectName("toolbar_view")   # Задача #40: имя нужно saveState/restoreState
        self._register_i18n(tb.setWindowTitle, "Вид")   # Задача #106
        _lab_z = QtWidgets.QLabel(" Z-шкала: ")
        self._register_i18n(_lab_z.setText, " Z-шкала: ")
        tb.addWidget(_lab_z)
        self._z_combo = CycleButton()   # Задача #74: клик = следующее значение, колесо = листать
        for _zi, (key, label) in enumerate(Z_MODES):   # Задача #169: tr + ретранслейт
            self._z_combo.addItem(tr(label), key)
            self._register_i18n(
                lambda s, i=_zi: self._z_combo.setItemText(i, s), label)
        self._z_combo.setCurrentIndex(2)  # по умолчанию log (как было до переключателя)
        self._z_combo.currentIndexChanged.connect(self._on_z_scale_changed)
        tb.addWidget(self._z_combo)
        _lab_pal = QtWidgets.QLabel(tr("  Палитра: "))   # Задача #169
        self._register_i18n(_lab_pal.setText, "  Палитра: ")
        tb.addWidget(_lab_pal)
        self._cmap_name = "jet"                            # Задача #102/#177: текущий ключ (дефолт — Jet)
        self._cmap_btn = QtWidgets.QToolButton()          # кнопка открывает окно «Цветовая палитра»
        self._cmap_btn.setText("Jet")
        self._register_i18n(self._cmap_btn.setToolTip, "Выбрать цветовую палитру")
        self._cmap_btn.clicked.connect(self._open_palette_dialog)
        tb.addWidget(self._cmap_btn)
        _lab_u = QtWidgets.QLabel(tr("  Единицы: "))  # Задача #44: счёт / скорость счёта
        self._register_i18n(_lab_u.setText, "  Единицы: ")
        tb.addWidget(_lab_u)
        self._unit_combo = CycleButton()   # Задача #74: клик = следующее значение, колесо = листать
        for _ui, (label, key) in enumerate((("отсчёты", "counts"), ("отсч/с (cps)", "cps"))):
            self._unit_combo.addItem(tr(label), key)
            self._register_i18n(
                lambda s, i=_ui: self._unit_combo.setItemText(i, s), label)
        self._unit_combo.setCurrentIndex(1)   # Задача #53: дефолт — cps (connect ниже, сигнал не шлём)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        tb.addWidget(self._unit_combo)
        _lab_t = QtWidgets.QLabel(tr("  Время: "))  # Задача #64: единицы оси времени 3D-сетки
        self._register_i18n(_lab_t.setText, "  Время: ")   # Задача #169
        tb.addWidget(_lab_t)
        self._tunit_combo = CycleButton()   # Задача #74: клик = следующее значение, колесо = листать
        for _ti, unit in enumerate(("с", "мин", "ч")):
            self._tunit_combo.addItem(tr(unit), unit)
            self._register_i18n(
                lambda s, i=_ti: self._tunit_combo.setItemText(i, s), unit)
        self._tunit_combo.setCurrentIndex(0)          # дефолт — секунды
        self._tunit_combo.currentIndexChanged.connect(self._on_time_unit_changed)
        tb.addWidget(self._tunit_combo)
        self._axes_check = QtWidgets.QCheckBox(tr("Оси"))  # подписи делений 3D (Задача 14)
        self._axes_check.setChecked(True)
        self._axes_check.toggled.connect(self._on_axes_toggled)
        self._register_i18n(self._axes_check.setText, "Оси")   # Задача #169
        tb.addWidget(self._axes_check)
        self._hl_check = QtWidgets.QCheckBox(tr("Подсветка"))  # подсветка выбранных пиков (Задача 18)
        self._hl_check.setChecked(False)
        self._hl_check.toggled.connect(self._on_highlight_toggled)
        self._register_i18n(self._hl_check.setText, "Подсветка")   # Задача #169
        tb.addWidget(self._hl_check)
        self._floor_check = QtWidgets.QCheckBox(tr("Подложка"))  # Задача #76: дно рельефа (фиолет. прямоугольник)
        self._floor_check.setChecked(False)   # Задача #150: по умолчанию подложка выключена
        self._register_i18n(self._floor_check.setText, "Подложка")   # Задача #169
        self._register_i18n(self._floor_check.setToolTip,
                            "Показать/скрыть подложку (плоское дно рельефа)")
        self._floor_check.toggled.connect(self._on_floor_toggled)
        tb.addWidget(self._floor_check)
        # Задача #143: тумблер «Простыня образца» — основной 3D-рельеф спектрограммы.
        self._surface_check = QtWidgets.QCheckBox("Простыня образца")
        self._surface_check.setChecked(True)
        self._register_i18n(self._surface_check.setToolTip,
                            "Показать/скрыть простыню образца (основной 3D-рельеф)")
        self._surface_check.toggled.connect(self._on_surface_toggled)
        self._register_i18n(self._surface_check.setText, "Простыня образца")
        tb.addWidget(self._surface_check)
        # Задача #142: по-элементная видимость наложения фона (активны в режиме «Наложение фона»)
        self._bg_sheet_check = QtWidgets.QCheckBox("Простыня фона")
        self._bg_sheet_check.setChecked(True)
        self._bg_sheet_check.setEnabled(False)
        self._register_i18n(self._bg_sheet_check.setToolTip,
                            "Показать/скрыть простыню фона на 3D-спектрограмме")
        self._bg_sheet_check.toggled.connect(lambda _on: self._apply_bg_overlay_visibility())
        self._register_i18n(self._bg_sheet_check.setText, "Простыня фона")
        tb.addWidget(self._bg_sheet_check)
        self._bg_curve_check = QtWidgets.QCheckBox("Фон среза")
        self._bg_curve_check.setChecked(True)
        self._bg_curve_check.setEnabled(False)
        self._register_i18n(self._bg_curve_check.setToolTip,
                            "Показать/скрыть кривую фона в окне среза")
        self._bg_curve_check.toggled.connect(lambda _on: self._apply_bg_overlay_visibility())
        self._register_i18n(self._bg_curve_check.setText, "Фон среза")
        tb.addWidget(self._bg_curve_check)
        # Задача #145: раздельный стиль простыни образца и фона (палитра/однотонный/каркас)
        _lab_ss = QtWidgets.QLabel("  Стиль обр.: ")
        self._register_i18n(_lab_ss.setText, "  Стиль обр.: ")
        tb.addWidget(_lab_ss)
        self._smp_style_combo = CycleButton()
        for _si, (label, key) in enumerate(
                (("Палитра", "palette"), ("Однотонный", "solid"), ("Каркас", "wire"))):
            self._smp_style_combo.addItem(tr(label), key)   # Задача #169
            self._register_i18n(
                lambda s, i=_si: self._smp_style_combo.setItemText(i, s), label)
        self._register_i18n(self._smp_style_combo.setToolTip, "Стиль простыни образца")
        self._smp_style_combo.currentIndexChanged.connect(self._on_surface_style_changed)
        tb.addWidget(self._smp_style_combo)
        _lab_bs = QtWidgets.QLabel("  Стиль фона: ")
        self._register_i18n(_lab_bs.setText, "  Стиль фона: ")
        tb.addWidget(_lab_bs)
        self._bg_style_combo = CycleButton()
        for _bi, (label, key) in enumerate(
                (("Палитра", "palette"), ("Однотонный", "solid"), ("Каркас", "wire"))):
            self._bg_style_combo.addItem(tr(label), key)   # Задача #169
            self._register_i18n(
                lambda s, i=_bi: self._bg_style_combo.setItemText(i, s), label)
        self._register_i18n(self._bg_style_combo.setToolTip, "Стиль простыни фона")
        self._bg_style_combo.currentIndexChanged.connect(self._on_bg_style_changed)
        tb.addWidget(self._bg_style_combo)
        # Изолинии (Задача 20) ВРЕМЕННО отключены в UI: вернём после реализации поиска,
        # идентификации пиков и подгонки их чистыми гауссианами. Механизм контуров в
        # HeatmapPanel (set_contours_enabled / set_contour_levels) сохранён и покрыт тестами;
        # обработчики _on_contours_toggled / _on_contour_levels_changed оставлены для возврата.
        # Задача #55: регулировки усиление/гамма/отсечка/сглаживание/освещение перенесены из
        # тулбара на отдельную панель-рукоятки (док «Регулировки отображения», см. __init__).
        # Задача #51: вернуть все настройки отображения к значениям по умолчанию
        self._reset_btn = QtWidgets.QPushButton(tr("Сброс"))
        self._register_i18n(self._reset_btn.setText, "Сброс")   # Задача #169
        self._register_i18n(self._reset_btn.setToolTip,
                            "Вернуть настройки отображения к значениям по умолчанию")
        self._reset_btn.clicked.connect(self._on_reset_display)
        tb.addWidget(self._reset_btn)
        self._apply_colormap("jet")            # #178-fix2: применить палитру Jet к view3d/heatmap при старте

    @QtCore.pyqtSlot()
    def _on_reset_display(self) -> None:
        """Задача #51: сброс настроек отображения к умолчанию. Каждый контрол ставится в своё
        дефолтное значение; Qt шлёт valueChanged/currentIndexChanged/toggled только при реальном
        изменении, и штатный обработчик применяет дефолт на все панели (2D/3D/срезы)."""
        self._z_combo.setCurrentIndex(2)      # log
        self._apply_colormap("jet")            # Задача #102/#177: палитра -> Jet (дефолт)
        self._unit_combo.setCurrentIndex(1)   # cps (Задача #53 — дефолт)
        self._tunit_combo.setCurrentIndex(0)  # Задача #64: единицы времени — секунды (дефолт)
        self._axes_check.setChecked(True)     # оси видимы
        self._hl_check.setChecked(False)      # подсветка выкл
        self._floor_check.setChecked(False)   # Задача #150: подложка скрыта (дефолт; было вкл #76)
        self._surface_check.setChecked(True)  # Задача #143: простыня образца видима (дефолт)
        self._bg_sheet_check.setChecked(True)  # Задача #142: элементы наложения видимы (дефолт)
        self._bg_curve_check.setChecked(True)
        self._smp_style_combo.setCurrentIndex(0)  # Задача #145: стили простыней — палитра (дефолт)
        self._bg_style_combo.setCurrentIndex(0)
        # Задача #55: регулировки (усиление/гамма/отсечка/сглаживание/освещение) живут на
        # панели-рукоятках; сброс к дефолтам + включение всех рядов/общего — одним вызовом.
        self._adjust.reset_all()

    @QtCore.pyqtSlot(bool)
    def _on_axes_toggled(self, on: bool) -> None:
        """Переключатель подписей делений осей 3D (Задача 14)."""
        self._view3d.set_axis_labels_visible(on)

    @QtCore.pyqtSlot(bool)
    def _on_floor_toggled(self, on: bool) -> None:
        """Задача #76/#222: показать/скрыть подложку (3D: дно рельефа; 2D: нижний диапазон LUT)."""
        self._view3d.set_floor_visible(on)
        self._heatmap.set_floor_visible(on)

    @QtCore.pyqtSlot(bool)
    def _on_surface_toggled(self, on: bool) -> None:
        """Задача #143: показать/скрыть простыню образца (основной 3D-рельеф)."""
        self._view3d.set_surface_visible(on)

    @QtCore.pyqtSlot(int)
    def _on_surface_style_changed(self, _idx: int) -> None:
        """Задача #145: стиль простыни образца (палитра/однотонный/каркас)."""
        self._view3d.set_surface_style(self._smp_style_combo.currentData() or "palette")

    @QtCore.pyqtSlot(int)
    def _on_bg_style_changed(self, _idx: int) -> None:
        """Задача #145: стиль простыни фона (палитра/однотонный/каркас)."""
        self._view3d.set_bg_sheet_style(self._bg_style_combo.currentData() or "palette")

    @QtCore.pyqtSlot(int)
    def _on_time_unit_changed(self, _idx: int) -> None:
        """Единицы оси времени 3D-сетки и 2D-карты: с / мин / ч (Задача #64/#207)."""
        unit = self._tunit_combo.currentData() or "с"
        self._view3d.set_time_unit(unit)
        self._heatmap.set_time_unit(unit)

    @QtCore.pyqtSlot(int)
    def _on_unit_changed(self, _idx: int) -> None:
        """Глобальные единицы графиков: отсчёты / отсч-в-секунду (Задача #44). Веером на все
        панели; 3D/2D пересчитываются от исходника, поэтому переразмещаем плоскости сечений."""
        mode = self._unit_combo.currentData() or "counts"
        self._view3d.set_unit_mode(mode)
        self._heatmap.set_unit_mode(mode)
        self._slices.set_unit_mode(mode)
        self._sections.emit_all()  # 3D-поверхность пересоздана — переразместить плоскости

    @QtCore.pyqtSlot(int)
    def _on_analytics_slice(self, i: int) -> None:
        """Клик по точке проекции (Задача 26) -> показать срез в панели срезов и поднять её док."""
        self._slices.show_time_slice(int(i))
        self._slices_dock.raise_()
        self._slices_dock.show()

    @QtCore.pyqtSlot()
    def _on_bg_select(self) -> None:
        """Задача #96: открыть диалог выбора фона; вычислить поканальный фон (cps) и применить."""
        if self._sg is None:
            self.statusBar().showMessage(tr("Сначала откройте файл, затем выбирайте фон."))
            return
        # Задача #148: диапазон между секущими плоскостями Времени предзаполняет поля срезов
        dlg = BackgroundDialog(self._sg.n_slices, self._sg.time_offsets_s, self,
                               plane_range=self._view3d.time_plane_range())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            bg = self._compute_background(dlg.result_spec())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("Выбор фона"), f"{type(exc).__name__}: {exc}")
            return
        self._bg_cps = bg
        self._slices.set_background(bg, self._bg_raw)   # Задача #139: сырой блок для «лохматого» оверлея
        self._view3d.set_background_sheet(bg, self._bg_raw)   # #98/#140: простыня из сырого окна
        self._act_bg_overlay.setEnabled(True)
        self._act_bg_subtract.setEnabled(True)
        self._redistribute()
        self.statusBar().showMessage(tr("Фон выбран. Доступны «Наложение» и «Вычет»."))

    def _compute_background(self, spec):
        """Задача #96: спецификация диалога -> поканальный фон (cps), выровненный по текущему файлу.
        Задача #139: для range-источника сташим сырой блок отсчётов+живого времени (self._bg_raw),
        по нему оверлей окна срезов строит фон, «лохматый как образец» (background_window_like)."""
        if spec and spec[0] == "range":
            lo = max(0, int(spec[1])); hi = min(self._sg.n_slices, int(spec[2]))
            # Задача #149: сырые срезы участка транслируются (копируются) на всю шкалу времени;
            # фаза lo — срезы внутри участка клонируют сами себя (самовычет участка = 0, #147).
            self._bg_raw = tile_background_block(
                self._sg.counts[lo:hi],
                np.asarray(self._sg.live_time_s, dtype=np.float64)[lo:hi],
                self._sg.n_slices, phase=lo)
            return background_from_range(self._sg, spec[1], spec[2])
        if spec and spec[0] == "file":
            bg_sg = load_spectrogram(spec[1]).trimmed_channels(1)
            # Задача #141: энергосетка файла-фона совпадает с целью (тот же прибор/файл) ->
            # сырой блок годен поканально, фон строится «лохматым как образец» и в срезах (#139),
            # и на простыне 3D (#140). Иная сетка -> сырой матч не строим (гладкий bg_cps).
            e_bg = np.asarray(bg_sg.energies(), dtype=np.float64)
            e_t = np.asarray(self._sg.energies(), dtype=np.float64)
            if e_bg.size == e_t.size and np.allclose(e_bg, e_t):
                # Задача #149: файл-фон другой длины по времени -> транслируем на шкалу цели
                self._bg_raw = tile_background_block(
                    bg_sg.counts, bg_sg.live_time_s, self._sg.n_slices)
            else:
                self._bg_raw = None
            return background_from_spectrogram(bg_sg, self._sg)
        raise ValueError("неизвестный источник фона")

    @QtCore.pyqtSlot(bool)
    def _on_bg_overlay_toggled(self, on: bool) -> None:
        """Задача #96: наложение фона на спектр среза (панель срезов) и «простыню» на 3D (#98).
        Задача #142: тумблеры тулбара активны только в режиме наложения."""
        self._bg_overlay = bool(on)
        self._bg_sheet_check.setEnabled(self._bg_overlay)
        self._bg_curve_check.setEnabled(self._bg_overlay)
        self._apply_bg_overlay_visibility()

    def _apply_bg_overlay_visibility(self) -> None:
        """Задача #142: элемент наложения виден, когда включён режим «Наложение фона»
        И его тумблер на тулбаре («Простыня фона» — 3D, «Фон среза» — окно среза)."""
        self._slices.set_background_overlay(
            self._bg_overlay and self._bg_curve_check.isChecked())
        self._view3d.set_background_sheet_visible(
            self._bg_overlay and self._bg_sheet_check.isChecked())   # Задача #98

    @QtCore.pyqtSlot(bool)
    def _on_bg_subtract_toggled(self, on: bool) -> None:
        """Задача #96: вычет фона из всего водопада (3D/2D/срез); отрицательное -> 0."""
        self._bg_subtract = bool(on)
        self._redistribute()

    @QtCore.pyqtSlot(bool)
    def _on_eff_norm_toggled(self, on: bool) -> None:
        """Задача #156: вкл/выкл нормализации водопада по эффективности регистрации ε(E)."""
        self._eff_normalize = bool(on)
        self._redistribute()

    def _on_eff_load(self) -> None:
        """Задача #156: загрузка кривой эффективности из файла (.efr/.efa LSRM, .json,
        двухколоночный текст). Ошибка парсинга — предупреждение, кривая не меняется."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Загрузить кривую эффективности…"), "",
            tr("Кривые эффективности (*.efr *.efa *.csv *.txt *.json);;Все файлы (*)"))
        if not path:
            return
        try:
            self._eff_curve = load_efficiency_curve(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, tr("Кривая эффективности"),
                tr("Не удалось загрузить кривую эффективности:") + f"\n{exc}")
            return
        self._update_eff_info()
        if self._eff_normalize:
            self._redistribute()

    def _update_eff_info(self) -> None:
        """#156 + #MENU-5: имя текущей кривой ε(E) — постоянный QLabel в статусбаре
        (было disabled-пункт в меню «Анализ»)."""
        lbl = getattr(self, "_eff_info_label", None)
        if lbl is None:
            lbl = QtWidgets.QLabel(self)
            self.statusBar().addPermanentWidget(lbl)
            self._eff_info_label = lbl
        lbl.setText(tr("Кривая:") + f" {self._eff_curve.name}")

    @QtCore.pyqtSlot(bool)
    def _on_dose_toggled(self, on: bool) -> None:
        """Задача #104: переключатель оверлея мощности дозы (RadiaCode .rcspg)."""
        self._slices.set_dose_overlay(on)

    @QtCore.pyqtSlot(bool)
    def _on_peaks_toggled(self, on: bool) -> None:
        """Задача #110/#111: переключатель поиска фотопиков на 3D-водопаде.
        При включении также заполняет панель «Найденные пики» (#111)."""
        self._view3d.set_peak_search(on)
        self._refresh_peaks_panel()

    @QtCore.pyqtSlot(float)
    def _on_peaks_sigma_changed(self, sigma: float) -> None:
        """Задача #111/#114: изменение σ из PeaksPanel → пересчитать view3d + обновить панель."""
        self._view3d.set_peak_sigma(sigma)
        self._refresh_peaks_panel()

    def _refresh_peaks_panel(self) -> None:
        """Задача #111: обновить PeaksPanel результатами из view3d._found_peaks().
        Задача #123: попутно показать временно́е окно поиска (весь файл: N срезов, T).
        Задача #127: те же найденные пики скормить модулю идентификации нуклидов."""
        peaks = self._view3d._found_peaks()
        self._peaks_panel.set_peaks(peaks)
        sg = self._sg
        # Задача #130: та же авто-модель FWHM(E) (#120), что и у детектора пиков, — окно
        # матчинга идентификации по РЕАЛЬНОЙ ширине детектора, а не по грубому дефолту.
        fwhm_model = None
        if sg is not None:
            counts = np.asarray(sg.total_spectrum(), dtype=np.float64)
            energies = np.asarray(sg.energies(), dtype=np.float64)
            fwhm_model = auto_calibrate_fwhm_model(counts, energies)
        self._nuclides.show_candidates(peaks, fwhm_model=fwhm_model)
        if sg is not None:
            total = float(np.asarray(sg.real_time_s, dtype=np.float64).sum())
            self._peaks_panel.set_window_info(int(sg.n_slices), total)
        else:
            self._peaks_panel.set_window_info(None, None)

    def _open_calibration_dialog(self) -> None:
        """Задача #215: диалог перекалибровки, пресеты + импорт найденных пиков."""
        if self._sg is None:
            QtWidgets.QMessageBox.information(self, tr("Калибровка по пикам"),
                                              tr("Сначала откройте файл спектрограммы."))
            return
        coeffs = list(self._sg.calibration.coeffs)
        n_ch = int(self._sg.counts.shape[1])
        found = self._view3d._found_peaks() or []
        dlg = CalibrationDialog(current_coeffs=coeffs, n_channels=n_ch,
                                found_peaks=list(found), parent=self)
        dlg.calibrationApplied.connect(self._apply_new_calibration)
        dlg.exec()

    def _apply_new_calibration(self, new_coeffs) -> None:
        """Задача #215: заменить Calibration и пере-отрисовать всё окно."""
        if self._sg is None:
            return
        self._sg.calibration = Calibration(
            coeffs=np.asarray(list(new_coeffs), dtype=np.float64))
        self._redistribute(reset=False)
        self._refresh_peaks_panel()

    def _save_as_aswf(self) -> None:
        """Задача #216: сохранить спектрограмму в один .aswf с текущей калибровкой."""
        if self._sg is None:
            QtWidgets.QMessageBox.information(self, tr("Сохранить как"),
                                              tr("Сначала откройте файл спектрограммы."))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Сохранить как"), "", "AtomSpectra waterfall (*.aswf)")
        if not path:
            return
        self._do_save_as_aswf(path)

    def _do_save_as_aswf(self, path: str) -> None:
        """Задача #216: физическая запись .aswf + статус-бар."""
        if not path.lower().endswith(".aswf"):
            path += ".aswf"
        from awf.io.aswf_single_writer import write_aswf
        try:
            write_aswf(path, self._sg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, tr("Сохранить как"),
                                           tr("Ошибка сохранения: ") + str(e))
            return
        self.statusBar().showMessage(tr("Сохранено: ") + path, 5000)

    def _export_spectrum(self) -> None:
        """Задача #217: диалог выбора формата экспорта агрегированного спектра."""
        if self._sg is None:
            QtWidgets.QMessageBox.information(self, tr("Экспорт спектра"),
                                              tr("Сначала откройте файл спектрограммы."))
            return
        flt = "BecqMoni XML (*.xml);;LSRM (*.spe);;InterSpec/ANSI N42 (*.n42)"
        path, sel = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Экспорт спектра"), "", flt)
        if not path:
            return
        fmt = self._detect_export_fmt(path, sel)
        if not path.lower().endswith("." + fmt):
            path += "." + fmt
        self._do_export_spectrum(path, fmt)

    @staticmethod
    def _detect_export_fmt(path: str, sel: str) -> str:
        """Задача #226: BecqMoni XML + LSRM бинарный SPE + InterSpec N42."""
        low = path.lower()
        if low.endswith(".xml"):
            return "xml"
        if low.endswith(".spe"):
            return "spe"
        if low.endswith(".n42"):
            return "n42"
        if "xml" in sel.lower() or "becqmoni" in sel.lower():
            return "xml"
        if "spe" in sel.lower() or "lsrm" in sel.lower():
            return "spe"
        return "n42"

    def _do_export_spectrum(self, path: str, fmt: str) -> None:
        """Задача #217: агрегировать спектрограмму → 1D-спектр, вызвать writer."""
        sg = self._sg
        assert sg is not None
        spec = np.asarray(sg.total_spectrum()).astype(np.int64)
        lt = float(np.asarray(sg.live_time_s, dtype=np.float64).sum())
        rt = float(np.asarray(sg.real_time_s, dtype=np.float64).sum()) if sg.real_time_s is not None else lt
        cal = list(sg.calibration.coeffs) if sg.calibration is not None else None
        try:
            self._write_spectrum(path, fmt, spec, lt, rt, cal)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, tr("Экспорт спектра"),
                                           tr("Ошибка экспорта: ") + str(e))
            return
        self.statusBar().showMessage(tr("Экспорт: ") + path, 5000)

    @staticmethod
    def _write_spectrum(path, fmt, spec, lt, rt, cal):
        """Задача #226: диспетчер записи в xml (BecqMoni) / spe (LSRM бинарный) / n42."""
        if fmt == "xml":
            from awf.io.becqmoni_writer import write_becqmoni_xml
            write_becqmoni_xml(path, spec, live_time_s=lt, real_time_s=rt, calibration=cal)
        elif fmt == "spe":
            from awf.io.spe_writer import write_spe
            write_spe(path, spec, live_time_s=lt, real_time_s=rt, calibration=cal)
        else:
            from awf.io.n42_writer import write_n42
            write_n42(path, spec, live_time_s=lt, real_time_s=rt, calibration=cal)

    @QtCore.pyqtSlot(float)
    def _on_segment_recompute(self, pen_factor: float = 2.0) -> None:
        """Задача #131: пересчитать сегментацию записи по времени + посегментную ID нуклидов.
        Модель FWHM(E) строится раз по суммарному спектру (разрешение от времени не зависит)
        и переиспользуется во всех сегментах."""
        sg = self._sg
        if sg is None:
            self._segments_panel.clear_segments()
            return
        counts = np.asarray(sg.total_spectrum(), dtype=np.float64)
        energies = np.asarray(sg.energies(), dtype=np.float64)
        fwhm_model = auto_calibrate_fwhm_model(counts, energies)
        segs = segment_by_time(sg, pen_factor=float(pen_factor))
        sidents = identify_segments(sg, self._nuclides.library(), segs, fwhm_model=fwhm_model)
        self._segments_panel.set_segments(sidents)
        self._view3d.set_segment_bounds(segs)   # Задача #172: границы для посегментного сглаживания

    def _on_segments_action(self) -> None:
        """Задача #131: пункт меню «Сегментация по времени…» — показать док и пересчитать."""
        self._segments_dock.show()
        self._segments_dock.raise_()
        self._segments_panel._on_recompute()   # эмит recomputeRequested(pen_factor)

    def _analysis_spectrogram(self):
        """Задача #158: аналитическая спектрограмма — вычет фона (#96/#147) БЕЗ нормализации
        по эффективности (#156). Поиск пиков и идентификация работают ТОЛЬКО по ней:
        Currie-порог предполагает пуассонову статистику (var=N), а после умножения на
        ε_ref/ε(E) дисперсия растёт как f² — на реальном файле это дало 10 ложных пиков
        >1700 кэВ; identify_peaks сам вносит ε-поправку в интенсивности линий (#130),
        нормализация входа = двойная коррекция (Th-232 терял все матчи, K-40 падал
        ниже порога 0.30)."""
        sg = self._sg
        if self._bg_subtract and self._bg_cps is not None and sg is not None:
            sg = subtract_background(sg, self._bg_cps, self._bg_raw)
        return sg

    def _active_spectrogram(self):
        """Задача #96: активная спектрограмма для ОТОБРАЖЕНИЯ 3D/2D/срезов — с вычтенным
        фоном, если вычет включён и фон задан. Задача #147: сырой блок _bg_raw — поячеечный
        вычет. Задача #156: нормализация ε(E) — ПОСЛЕ вычета фона (фон и образец сняты одним
        детектором, вычитать надо в сырых отсчётах). Задача #158: нормализация — только
        отображение; анализ (пики/ID) берёт _analysis_spectrogram()."""
        sg = self._analysis_spectrogram()
        if self._eff_normalize and sg is not None:
            sg = apply_efficiency(sg, self._eff_curve)
        return sg

    def _redistribute(self, *, reset: bool = False) -> None:
        """Задача #96: раздать активную спектрограмму в 3D/2D/срезы (аналитика — всегда на
        исходных данных). Порядок как в _on_loaded: срезы -> карта (она шлёт roiChanged).

        Задача #138 (решение оператора: «вычитай сырые данные поканально, ничего не усредняй»):
        3D, 2D и срезы получают ОДНУ И ТУ ЖЕ активную спектрограмму — прямой знаковый поканальный
        вычет сырых данных (subtract_background, #134). Посегментное усреднение (#137) и гейт 3σ
        (#136) из проводки 3D/2D убраны: способ отображения везде одинаков, вычет — как есть.

        Задача #161: reset=True — только для НОВОГО файла (_on_loaded), полный сброс окна срезов
        (set_spectrogram). Для toggle фона/нормализации (bg-select/subtract/eff-norm/eff-load)
        reset=False (дефолт) — update_spectrogram() сохраняет текущий вид (срез/ROI/интеграл) и
        зум; раньше любой такой toggle «ломал» окно срезов, перескакивая на интегральный спектр."""
        sg = self._active_spectrogram()               # #138: поканальный вычет #134 (или сырые данные)
        if sg is None:
            return
        # Задача #158: view3d получает и аналитический источник (без ε-нормализации) —
        # для _found_peaks (пики/ID) и пуассоновой маски гребней (#152 _z_counts_int).
        self._view3d.set_spectrogram(sg, analysis_sg=self._analysis_spectrogram())
        if reset:
            self._slices.set_spectrogram(sg)
        else:
            self._slices.update_spectrogram(sg)
        self._heatmap.set_spectrogram(sg)
        self._sections.emit_all()
        self._refresh_peaks_panel()   # Задача #111: обновить панель пиков после ре-рендера

    def _reset_background(self) -> None:
        """Задача #96: сброс фона при загрузке нового файла — снять кривую/вычет и обесточить
        пункты меню «Наложение/Вычет фона» (на новых данных старый фон бессмыслен)."""
        self._bg_cps = None
        self._bg_raw = None                # Задача #139: снять сырой фоновый блок
        self._bg_subtract = False
        self._bg_overlay = False
        self._slices.set_background(None)
        self._slices.set_background_overlay(False)
        self._view3d.set_background_sheet(None)            # Задача #98: снять «простыню» фона
        self._view3d.set_background_sheet_visible(False)
        for act in (getattr(self, "_act_bg_overlay", None), getattr(self, "_act_bg_subtract", None)):
            if act is None:
                continue
            act.blockSignals(True)
            act.setChecked(False)
            act.setEnabled(False)
            act.blockSignals(False)
        # Задача #142: тумблеры видимости наложения — в дефолт (показывать) и погасить
        for chk in (getattr(self, "_bg_sheet_check", None), getattr(self, "_bg_curve_check", None)):
            if chk is None:
                continue
            chk.blockSignals(True)
            chk.setChecked(True)
            chk.setEnabled(False)
            chk.blockSignals(False)

    @QtCore.pyqtSlot(bool)
    def _on_contours_toggled(self, on: bool) -> None:
        """Переключатель изолиний на 2D-карте (Задача 20)."""
        self._heatmap.set_contours_enabled(on)

    @QtCore.pyqtSlot(int)
    def _on_contour_levels_changed(self, n: int) -> None:
        """Число уровней изолиний на 2D-карте (Задача 20)."""
        self._heatmap.set_contour_levels(int(n))

    @QtCore.pyqtSlot(bool)
    def _on_highlight_toggled(self, on: bool) -> None:
        """Переключатель режима подсветки выбранных пиков (Задача 18): база глушится в 3D и 2D,
        столбцы выбранных энергий нуклидов выделяются."""
        self._view3d.set_highlight_enabled(on)
        self._heatmap.set_highlight_enabled(on)
        self._sections.emit_all()  # 3D-поверхность пересоздана — переразместить плоскости

    @QtCore.pyqtSlot()
    def _open_palette_dialog(self) -> None:
        """Задача #102: окно «Цветовая палитра» с превью-градиентами; выбор применяется живьём."""
        dlg = PaletteDialog(self._cmap_name, self)
        dlg.selected.connect(self._apply_colormap)
        dlg.exec()

    def _apply_colormap(self, name: str) -> None:
        """Задача #102/#17: единая палитра для 2D-карты и 3D-поверхности (+переразместить сечения)."""
        self._cmap_name = name
        label = next((l for k, l, _ in COLORMAPS if k == name), name)
        self._cmap_btn.setText(label)
        self._view3d.set_colormap(name)
        self._heatmap.set_colormap(name)
        self._sections.emit_all()  # 3D-поверхность пересоздана — переразместить плоскости

    @QtCore.pyqtSlot()
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
        if v["tsmooth"] != last.get("tsmooth", 0) or v["tsmooth_by_seg"] != last.get("tsmooth_by_seg", 0):
            self._view3d.set_t_smoothing(int(v["tsmooth"]), bool(v["tsmooth_by_seg"]))
            # Задача #174: первое включение «по сегм.» → авто-сегментация если ещё не было
            if int(v["tsmooth"]) > 0 and v["tsmooth_by_seg"] and not self._view3d.has_segments:
                self._on_segment_recompute()
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

    @QtCore.pyqtSlot()
    def _on_z_scale_changed(self) -> None:
        mode = self._z_combo.currentData()
        self._view3d.set_z_scale(mode)
        self._heatmap.set_z_scale(mode)
        # после перестроения поверхности — обновить позиции/подписи секущих плоскостей
        self._sections.emit_all()

    @QtCore.pyqtSlot(str, int, float, bool)
    def _on_plane_changed(self, axis: str, slot: int, frac: float, visible: bool) -> None:
        """Слайдер/чекбокс дока «Сечения» -> позиция секущей плоскости в 3D + подпись реального
        значения + синхронизация дока срезов (#38) и 2D-карты (#39) с выбранными сечениями."""
        self._view3d.set_plane(axis, slot, frac, visible)
        value, unit = self._view3d.plane_value(axis, frac)
        self._sections.set_value_label(axis, slot, f"{value:.1f} {tr(unit)}")
        state = self._view3d.active_plane_values()
        self._slices.sync_sections(state["time"], state["energy"])
        self._heatmap.set_section_markers(state["time"], state["energy"])

    @QtCore.pyqtSlot()
    def _open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Открыть спектрограмму"), "",
            tr("Спектрограммы (*.n42 *.xml *.rcspg *.aswf);;N42 / XML (*.n42 *.xml);;"
               "RadiaCode (*.rcspg);;AtomSpectra (*.aswf);;Все файлы (*)"))
        if path:
            self.open_file(path)

    def open_file(self, path: str, max_slices: int | None = None) -> None:
        """Запустить фоновую загрузку файла. UI остаётся отзывчивым."""
        self.statusBar().showMessage(f"{tr('Загрузка')}: {path} …")
        self._loader = LoaderThread(path, max_slices=max_slices, parent=self)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.start()

    @QtCore.pyqtSlot(object)
    def _on_loaded(self, sg) -> None:
        # Замечание IV-R5: последний канал АЦП — мусор (переполнение); отбрасываем его единым
        # местом, чтобы все виды (2D/3D/срезы/ROI/идентификация) были консистентны.
        sg = sg.trimmed_channels(1)
        self._sg = sg
        # порядок важен: сперва панель срезов получает данные, затем карта — её set_spectrogram
        # испускает roiChanged, который сразу нарисует срез по умолчанию.
        self._reset_background()              # Задача #96: новый файл -> сбросить фон/вычет
        # Задача #104: пункт «Мощность дозы» — только для RadiaCode .rcspg
        src = getattr(sg, "source_path", None) or ""
        is_rcspg = src.lower().endswith(".rcspg")
        act_dose = getattr(self, "_act_dose", None)
        if act_dose is not None:
            act_dose.setEnabled(is_rcspg)
            act_dose.blockSignals(True)
            act_dose.setChecked(is_rcspg)   # включаем по умолчанию для RadiaCode
            act_dose.blockSignals(False)
        self._analytics.set_spectrogram(sg)   # «Аналитика» (Задача 26) — всегда на исходных данных
        # 3D/2D/срезы + секущие плоскости под новую геометрию (через активную спектрограмму)
        self._redistribute(reset=True)   # Задача #161: новый файл -> полный сброс окна срезов
        # Задача #104: после redistribute явно синхронизировать видимость дозы
        self._slices.set_dose_overlay(is_rcspg)
        # #178-fix1b: при загрузке нового файла авто-запустить сегментацию если tsmooth>0 и by_seg
        adj = self._adjust.values()
        if int(adj.get("tsmooth", 0)) > 0 and adj.get("tsmooth_by_seg", 0):
            self._on_segment_recompute()
        total = int(np.asarray(sg.counts).sum(dtype=np.int64))
        t0 = sg.t0_iso if sg.t0_iso else "—"
        src = sg.source_path if sg.source_path else "?"
        self.statusBar().showMessage(
            f"{src} — {tr('срезов')} {sg.n_slices} × {tr('каналов')} {sg.n_channels}; "
            f"t0={t0}; {tr('всего отсчётов')}={total}")

    @QtCore.pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"{tr('Ошибка загрузки')}: {message}")
        QtWidgets.QMessageBox.critical(self, tr("Ошибка загрузки"), message)

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
