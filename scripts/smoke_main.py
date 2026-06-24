"""Smoke: главное окно строится, _on_loaded интегрирует 3D/2D/срезы, статус-бар заполняется."""
import sys
from pathlib import Path
from PySide6 import QtWidgets
from awf.io.n42_loader import load_n42
from awf.ui.main_window import MainWindow

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
sample = Path(__file__).resolve().parent.parent / "sample_data" / "waterfall_sample.n42"
sg = load_n42(sample)
win = MainWindow()
win._on_loaded(sg)  # синхронно имитируем завершение фоновой загрузки
assert win._view3d._surface is not None, "3D-поверхность не построена"
assert "всего отсчётов" in win.statusBar().currentMessage(), "статус-бар не заполнен"
assert "Выборка" in win._slices._header.text(), "roiChanged -> show_roi не сработал"
print("OK main_window:", win.statusBar().currentMessage())
