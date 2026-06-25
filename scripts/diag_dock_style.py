"""Диагностика тёмного фона откреплённого дока (Задача #35).
Строит реальный MainWindow, делает каждый QDockWidget плавающим и считает долю
пикселей фона #2b2d31 против системно-светлых. Запуск:
  QT_QPA_PLATFORM=offscreen PYTHONPATH=<root> python scripts/diag_dock_style.py
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6 import QtGui, QtWidgets
from awf.ui.main_window import MainWindow

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
mw = MainWindow()
mw.show()
app.processEvents(); app.processEvents()

target = np.array([49, 45, 43])  # #2b2d31 в порядке B,G,R
ok = True
for dock in mw.findChildren(QtWidgets.QDockWidget):
    dock.setFloating(True)
    dock.resize(320, 480)
    app.processEvents(); app.processEvents()
    img = dock.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape((h, w, 4))
    dist = np.abs(buf[:, :, :3].astype(int) - target).sum(axis=2)
    dark = int((dist <= 6).sum())
    light = int((buf[:, :, :3].astype(int).sum(axis=2) >= 700).sum())
    verdict = "DARK-OK" if (dark > 500 and light < dark) else "STILL-LIGHT"
    ok = ok and verdict == "DARK-OK"
    print(f"{dock.windowTitle()!r:40} {w}x{h} dark={dark} light={light} -> {verdict}")
print("OVERALL:", "PASS" if ok else "FAIL")