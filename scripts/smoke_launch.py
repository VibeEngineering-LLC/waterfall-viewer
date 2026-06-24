"""Smoke: полный путь `python -m awf <file>` — show() + реальная фоновая загрузка (QThread)."""
import sys
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from awf.ui.main_window import MainWindow

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
sample = Path(__file__).resolve().parent.parent / "sample_data" / "waterfall_sample.n42"
win = MainWindow()
win.show()
win.open_file(str(sample))                       # реальный асинхронный путь через LoaderThread
QtCore.QTimer.singleShot(4000, app.quit)         # даём потоку загрузиться, затем выходим
app.exec()
msg = win.statusBar().currentMessage()
assert win._sg is not None, "фоновая загрузка не завершилась за отведённое время"
assert "всего отсчётов" in msg, f"статус-бар не заполнен после async-загрузки: {msg!r}"
assert win._view3d._surface is not None, "3D-поверхность не построена после async-загрузки"
print("OK launch (async QThread):", msg)
