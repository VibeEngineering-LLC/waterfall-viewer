"""Smoke: принудительный GL-рендер (grabFramebuffer) по ракурсам + resize + вкладки + ROI."""
import sys, faulthandler
from pathlib import Path
from PySide6 import QtWidgets
from awf.io.n42_loader import load_n42
from awf.ui.main_window import MainWindow

faulthandler.enable()
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
sample = Path(__file__).resolve().parent.parent / "sample_data" / "waterfall_sample.n42"
win = MainWindow(); win.show(); win._on_loaded(load_n42(sample))
def pump(n=6):
    for _ in range(n): app.processEvents()
pump()
for az, el, dist in [(-60, 35, 200), (45, 30, 50), (120, 80, 400), (0, 5, 120)]:
    win._view3d.setCameraPosition(distance=dist, elevation=el, azimuth=az)
    img = win._view3d.grabFramebuffer(); pump(2)                    # принудительный paintGL
    assert img.width() > 0
win.resize(900, 600); pump(); win._view3d.grabFramebuffer(); pump()   # resize -> resizeGL
win._tabs.setCurrentIndex(1); pump()
for p, s in [([5, 5], [40, 40]), ([900, 9000], [300, 300])]:
    win._heatmap._roi.setPos(p); win._heatmap._roi.setSize(s); win._heatmap._on_roi_finished(); pump()
print("OK interact2: без краша")
