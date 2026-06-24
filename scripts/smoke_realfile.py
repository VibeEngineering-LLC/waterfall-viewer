"""Smoke на РЕАЛЬНОМ файле оператора: загрузка + GL-рендер по ракурсам + ROI + срезы."""
import sys, faulthandler
from pathlib import Path
from PySide6 import QtWidgets
from awf.ui.main_window import MainWindow, load_spectrogram

faulthandler.enable()
path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "sample_data" / "waterfall_sample.n42")
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
sg = load_spectrogram(path)
print(f"загружено: срезов={sg.n_slices} каналов={sg.n_channels}")
win = MainWindow(); win.show(); win._on_loaded(sg)
def pump(n=6):
    for _ in range(n): app.processEvents()
pump()
for az, el, dist in [(-60, 35, 200), (45, 30, 50), (120, 80, 400), (0, 5, 120)]:
    win._view3d.setCameraPosition(distance=dist, elevation=el, azimuth=az)
    assert win._view3d.grabFramebuffer().width() > 0; pump(2)
win.resize(900, 600); pump(); win._view3d.grabFramebuffer(); pump()
win._tabs.setCurrentIndex(1); pump()
for p, s in [([5, 5], [40, 40]), ([1, 1], [9999, 9999])]:
    win._heatmap._roi.setPos(p); win._heatmap._roi.setSize(s); win._heatmap._on_roi_finished(); pump()
# Z-шкала: прогнать все режимы через комбобокс (он прокидывает в 2D-карту и 3D)
for idx in (0, 1, 2):  # linear, sqrt, log
    win._z_combo.setCurrentIndex(idx); pump()
    assert win._heatmap._z_mode == win._z_combo.currentData()
    assert win._view3d._z_mode == win._z_combo.currentData()
    win._tabs.setCurrentIndex(0); win._view3d.grabFramebuffer(); pump()
print(f"OK realfile: без краша (Z-шкала проверена: {[m for m,_ in __import__('awf.ui.zscale', fromlist=['Z_MODES']).Z_MODES]})")
