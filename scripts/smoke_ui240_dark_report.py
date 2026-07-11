"""Smoke #UI-240: поле отчёта целостности под тёмной темой (RGB центра < 128 = не белый)."""
import sys
from PySide6 import QtWidgets
from awf.ui.style import APP_QSS
from awf.ui.integrity_dialog import IntegrityDialog

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
app.setStyleSheet(APP_QSS)
dlg = IntegrityDialog("=== test ===\nline\nline\nline", None)
dlg.resize(640, 480)
dlg.show()
app.processEvents()
pm = dlg._text_edit.grab()
c = pm.toImage().pixelColor(pm.width() // 2, pm.height() // 2)
r, g, b = c.red(), c.green(), c.blue()
print(f"center pixel RGB = ({r},{g},{b})  size={pm.width()}x{pm.height()}")
dark = max(r, g, b) < 128
print("DARK_OK" if dark else "STILL_LIGHT")
sys.exit(0 if dark else 1)
