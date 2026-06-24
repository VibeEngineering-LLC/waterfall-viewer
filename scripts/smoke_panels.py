"""Smoke: 2D-карта и панель срезов строятся, ROI даёт корректные индексы, слоты не падают."""
import sys
from pathlib import Path
from PySide6 import QtWidgets
from awf.io.n42_loader import load_n42
from awf.ui.panels import HeatmapPanel, SlicePanel

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
sample = Path(__file__).resolve().parent.parent / "sample_data" / "waterfall_sample.n42"
sg = load_n42(sample)
captured = []
hm = HeatmapPanel(); hm.roiChanged.connect(lambda *a: captured.append(a)); hm.set_spectrogram(sg)
t_lo, t_hi, ch_lo, ch_hi = hm.current_roi()
assert 0 <= t_lo < t_hi <= sg.n_slices and 0 <= ch_lo < ch_hi <= sg.n_channels, (t_lo, t_hi, ch_lo, ch_hi)
assert captured, "сигнал roiChanged не пришёл при set_spectrogram"
sp = SlicePanel(); sp.set_spectrogram(sg); sp.show_time_slice(0)
sp.show_roi(0, sg.n_slices, 0, sg.n_channels)
assert str(sg.roi_sum(0, sg.n_slices, 0, sg.n_channels)) in sp._header.text()
print(f"OK panels: ROI={hm.current_roi()}, header='{sp._header.text()}'")
