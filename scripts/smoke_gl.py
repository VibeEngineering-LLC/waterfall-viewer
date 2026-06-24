"""Проба OpenGL-стека: создаётся ли GL-контекст и GLSurfacePlotItem на этой машине.
Запуск: .venv\\Scripts\\python.exe scripts\\smoke_gl.py  (печатает OK или трассу ошибки)."""
import sys
import numpy as np
from PySide6 import QtWidgets
import pyqtgraph.opengl as gl

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
view = gl.GLViewWidget()
z = np.random.default_rng(0).random((20, 30)).astype(np.float32)
surf = gl.GLSurfacePlotItem(
    x=np.arange(20, dtype=np.float32),
    y=np.arange(30, dtype=np.float32),
    z=z, shader="heightColor", computeNormals=False)
view.addItem(surf)
view.resize(320, 240)
print("OK: GLViewWidget + GLSurfacePlotItem созданы без ошибок")
