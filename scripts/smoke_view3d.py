"""Smoke: 3D-вьюер строится и принимает реальный образец без ошибок (без show())."""
import sys
from pathlib import Path
from PySide6 import QtWidgets
from awf.io.n42_loader import load_n42
from awf.ui.view3d import Waterfall3DView

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
sample = Path(__file__).resolve().parent.parent / "sample_data" / "waterfall_sample.n42"
sg = load_n42(sample)
view = Waterfall3DView()
view.set_spectrogram(sg)
assert view._surface is not None, "поверхность не создана"
view.clear_surface()
assert view._surface is None
print(f"OK view3d: срезов={sg.n_slices}, каналов={sg.n_channels}, поверхность создана и очищена")
