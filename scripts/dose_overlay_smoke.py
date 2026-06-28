"""End-to-end smoke for #104 dose overlay on a real RadiaCode .rcspg file.
Loads file, builds SlicePanel offscreen, checks the _dose curve the UI will draw.
Usage: python dose_overlay_smoke.py <path.rcspg>"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6 import QtWidgets
from awf.io.rcspg_loader import load_rcspg
from awf.ui.panels import SlicePanel

def main(path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sg = load_rcspg(path)
    panel = SlicePanel()
    panel.set_spectrogram(sg)
    d = panel._dose
    assert d is not None, "dose is None for .rcspg source"
    assert len(d) == sg.n_slices, (len(d), sg.n_slices)
    xd, yd = panel._dose_curve.getData()
    print("n_slices        :", sg.n_slices)
    print("dose unit       :", panel._dose_unit)
    print("dose min/med/max : %.4g / %.4g / %.4g" % (np.nanmin(d), np.nanmedian(d), np.nanmax(d)))
    print("peak slice idx  :", int(np.nanargmax(d)))
    print("curve points    :", 0 if xd is None else len(xd))
    print("dose axis shown :", panel._dose_axis.isVisible())
    print("OK")

if __name__ == "__main__":
    main(sys.argv[1])