from __future__ import annotations
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QFontMetrics
from PyQt5.QtCore import Qt, QRect


class VerticalTabButton(QPushButton):
    def __init__(self, dock: QtWidgets.QDockWidget, side: str, parent=None):
        super().__init__(parent)
        self._dock = dock
        self._side = side
        self.setFixedWidth(24)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(dock.windowTitle())
        self.clicked.connect(self._restore_dock)
        dock.windowTitleChanged.connect(self.update)

    def _restore_dock(self):
        self._dock.show()
        self._dock.raise_()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor("#3d4044") if self.underMouse() else QColor("#2b2d31")
        painter.fillRect(self.rect(), color)
        accent = QColor("#6a8faf")
        if self._side == 'left':
            painter.fillRect(0, 0, 2, self.height(), accent)
        else:
            painter.fillRect(self.width() - 2, 0, 2, self.height(), accent)
        painter.setPen(QColor("#d8dade"))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        angle = -90 if self._side == 'left' else 90
        painter.rotate(angle)
        fm = QFontMetrics(painter.font())
        text = self._dock.windowTitle()
        max_len = self.height() - 8
        text = fm.elidedText(text, Qt.ElideRight, max_len)
        rect = QRect(-self.height() // 2, -self.width() // 2, self.height(), self.width())
        painter.drawText(rect, Qt.AlignCenter, text)
        painter.restore()


class EdgeBar(QWidget):
    def __init__(self, side: str, parent=None):
        super().__init__(parent)
        self._side = side
        self.setFixedWidth(24)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._buttons: dict[QtWidgets.QDockWidget, VerticalTabButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)
        layout.addStretch()
        self.setVisible(False)

    def add_dock(self, dock: QtWidgets.QDockWidget) -> None:
        if dock in self._buttons:
            return
        btn = VerticalTabButton(dock, self._side, self)
        self.layout().insertWidget(self.layout().count() - 1, btn)
        self._buttons[dock] = btn
        self.setVisible(True)

    def remove_dock(self, dock: QtWidgets.QDockWidget) -> None:
        btn = self._buttons.pop(dock, None)
        if btn is None:
            return
        self.layout().removeWidget(btn)
        btn.deleteLater()
        if not self._buttons:
            self.setVisible(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#2b2d31"))