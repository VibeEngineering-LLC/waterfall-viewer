SYSTEM / OUTPUT CONTRACT (читай дословно):
- Сгенерируй ОДИН файл Python: содержимое модуля `awf/ui/main_window.py`.
- Выводи ТОЛЬКО сырой код Python. БЕЗ markdown-ограждений (```), БЕЗ пояснений, БЕЗ текста до/после.
- Первая строка вывода — строковый docstring модуля (тройные кавычки).
- Комментарии и docstrings — на русском. Имена идентификаторов — на английском.
- Зависимости: numpy, PySide6 (QtCore, QtGui, QtWidgets), стандартная библиотека (sys, pathlib).
  Внутренние импорты: awf.io.n42_loader, awf.ui.view3d, awf.ui.panels. НЕ импортируй pyqtgraph напрямую.
- Python 3.12, `from __future__ import annotations`, type hints.
- НЕ оставляй TODO/pass-заглушек. Каждый метод полностью реализован.

ИМПОРТЫ В НАЧАЛЕ:
  from __future__ import annotations
  import sys
  from pathlib import Path
  import numpy as np
  from PySide6 import QtCore, QtGui, QtWidgets
  from awf.io.n42_loader import load_n42
  from awf.ui.view3d import Waterfall3DView
  from awf.ui.panels import HeatmapPanel, SlicePanel

НАЗНАЧЕНИЕ:
Главное окно приложения-вьюера. Несёт 3D-поверхность, 2D-карту и панель срезов; грузит файл N42 в
ФОНОВОМ потоке (QThread), чтобы суточные записи не морозили интерфейс. Связывает выборку (ROI) на
2D-карте с панелью срезов.

КОНТРАКТ С ОСТАЛЬНЫМИ МОДУЛЯМИ (вызывать ТОЛЬКО это; других имён НЕ выдумывать):
  load_n42(path, *, max_slices=None) -> Spectrogram        (может бросить исключение)
  Waterfall3DView() ; .set_spectrogram(sg) ; .clear_surface()
  HeatmapPanel() ; .set_spectrogram(sg) ; сигнал .roiChanged(int,int,int,int) ; .current_roi()
  SlicePanel() ; .set_spectrogram(sg) ; слот .show_roi(int,int,int,int) ; слот .show_time_slice(int)
  Spectrogram: .n_slices, .n_channels, .t0_iso (str|None), .source_path (str|None), .counts (2D ndarray)

КЛАСС-ПОТОК ЗАГРУЗКИ:

  class LoaderThread(QtCore.QThread):
      """Фоновая загрузка N42, чтобы не блокировать UI. Результат/ошибка — через сигналы."""
      loaded = QtCore.Signal(object)   # несёт Spectrogram
      failed = QtCore.Signal(str)      # текст ошибки

      def __init__(self, path: str, max_slices: int | None = None, parent=None):
          super().__init__(parent)
          self._path = path
          self._max_slices = max_slices

      def run(self) -> None:
          try:
              sg = load_n42(self._path, max_slices=self._max_slices)
              self.loaded.emit(sg)
          except Exception as exc:  # любую ошибку отдать в UI-поток, не падать
              self.failed.emit(f"{type(exc).__name__}: {exc}")

ГЛАВНОЕ ОКНО:

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

          self._build_menu()
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

      @QtCore.Slot()
      def _open_dialog(self) -> None:
          path, _ = QtWidgets.QFileDialog.getOpenFileName(
              self, "Открыть файл N42", "", "N42 / XML (*.n42 *.xml);;Все файлы (*)")
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

ВАЖНО ПО API PySide6 (соблюдать дословно):
- QtGui.QAction (НЕ QtWidgets.QAction в PySide6 6.x).
- QtGui.QKeySequence.Open / .Quit — стандартные комбинации.
- QThread: переопределяется run(); запуск через .start(); сигналы emit из run() доставляются в
  UI-поток через очередь автоматически.
- app.exec() (НЕ exec_()).
- НЕ создавать вложенный QApplication, если экземпляр уже существует (использовать .instance()).
