"""Задача #102: модальное окно «Цветовая палитра» — список палитр с превью-градиентами,
выбор кликом, подсветка текущей. Сигнал selected(key) — для живого применения палитры к 2D/3D."""
from __future__ import annotations
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from awf.ui.colormaps import COLORMAPS, get_colormap

_DIALOG_QSS = """
QDialog { background: #1e2024; }
QScrollArea { border: none; background: #1e2024; }
QScrollArea > QWidget > QWidget { background: #1e2024; }
QFrame#paletteRow { background: #1e2024; border: 1px solid transparent; border-radius: 6px; }
QFrame#paletteRow:hover { background: #26292e; }
QFrame#paletteRow[selected="true"] { border: 1px solid #4a7d6f; background: #232a28; }
QLabel#paletteName { color: #e6e8ea; font-weight: bold; font-size: 13px; }
QLabel#paletteDesc { color: #8a8f96; font-size: 11px; }
"""


def _gradient_pixmap(name: str, w: int = 160, h: int = 22, radius: int = 5) -> QtGui.QPixmap:
    """Горизонтальный градиент палитры -> скруглённый QPixmap для превью в строке окна."""
    lut = get_colormap(name).getLookupTable(0.0, 1.0, w, alpha=False)        # (w, 3) uint8
    arr = np.ascontiguousarray(np.repeat(lut[np.newaxis, :, :], h, axis=0))  # (h, w, 3)
    img = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    out = QtGui.QPixmap(w, h)
    out.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(out)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    path = QtGui.QPainterPath()
    path.addRoundedRect(0.0, 0.0, float(w), float(h), float(radius), float(radius))
    p.setClipPath(path)
    p.drawImage(0, 0, img)
    p.end()
    return out


class _PaletteRow(QtWidgets.QFrame):
    """Строка окна: превью-градиент + название + описание; клик выбирает палитру."""
    clicked = QtCore.Signal(str)

    def __init__(self, key: str, label: str, desc: str, parent=None) -> None:
        super().__init__(parent)
        self._key = key
        self.setObjectName("paletteRow")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(12)
        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(160, 22)
        swatch.setPixmap(_gradient_pixmap(key))
        lay.addWidget(swatch)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(0)
        name = QtWidgets.QLabel(label); name.setObjectName("paletteName")
        sub = QtWidgets.QLabel(desc); sub.setObjectName("paletteDesc")
        col.addWidget(name)
        col.addWidget(sub)
        lay.addLayout(col)
        lay.addStretch(1)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, ev) -> None:
        self.clicked.emit(self._key)
        super().mousePressEvent(ev)


class PaletteDialog(QtWidgets.QDialog):
    """Окно выбора палитры (#102). current — ключ текущей палитры (подсвечивается)."""
    selected = QtCore.Signal(str)

    def __init__(self, current: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Цветовая палитра")
        self.setModal(True)
        self.setMinimumSize(420, 520)
        self.setStyleSheet(_DIALOG_QSS)
        self._rows: dict[str, _PaletteRow] = {}
        self._current = current
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._build_scroll())
        self._mark(current)

    def _build_scroll(self) -> QtWidgets.QScrollArea:
        host = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(host)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(4)
        for key, label, desc in COLORMAPS:
            row = _PaletteRow(key, label, desc)
            row.clicked.connect(self._on_pick)
            self._rows[key] = row
            col.addWidget(row)
        col.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        return scroll

    @QtCore.Slot(str)
    def _on_pick(self, key: str) -> None:
        self._mark(key)
        self.selected.emit(key)

    def _mark(self, key: str) -> None:
        self._current = key
        for k, row in self._rows.items():
            row.set_selected(k == key)

    def selected_key(self) -> str:
        return self._current