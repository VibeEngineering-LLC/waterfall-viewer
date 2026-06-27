"""CycleButton — компактный переключатель «перебором» вместо выпадающего списка (Задача #74).

Кнопка показывает текущее значение; клик — следующее по кругу, колесо мыши — вперёд/назад.
API совместим с используемой частью QComboBox: addItem(label, data), currentData(),
currentIndex(), setCurrentIndex(i), сигнал currentIndexChanged(int) (эмитится только при
реальном изменении индекса — как у QComboBox, чтобы не ломать reset-логику тулбара)."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class CycleButton(QtWidgets.QPushButton):
    currentIndexChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: list = []
        self._data: list = []
        self._index = -1
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.clicked.connect(lambda: self._advance(1))

    # ---- наполнение (QComboBox-совместимое) ----
    def addItem(self, label, data=None) -> None:
        self._labels.append(str(label))
        self._data.append(data)
        if self._index < 0:
            self._index = 0
            self.setText(self._labels[0])
        self._refit()

    def _refit(self) -> None:
        fm = self.fontMetrics()
        w = max((fm.horizontalAdvance(s) for s in self._labels), default=0)
        self.setMinimumWidth(w + 40)

    def count(self) -> int:
        return len(self._labels)

    def itemData(self, i: int):
        return self._data[i] if 0 <= i < len(self._data) else None

    def itemText(self, i: int) -> str:
        return self._labels[i] if 0 <= i < len(self._labels) else ""

    # ---- доступ к текущему значению (QComboBox-совместимый) ----
    def currentIndex(self) -> int:
        return self._index

    def currentData(self):
        return self._data[self._index] if 0 <= self._index < len(self._data) else None

    def currentText(self) -> str:
        return self._labels[self._index] if 0 <= self._index < len(self._labels) else ""

    def setCurrentIndex(self, i: int) -> None:
        n = len(self._labels)
        if n == 0:
            return
        i = int(i) % n
        if i == self._index:
            return
        self._index = i
        self.setText(self._labels[i])
        self.currentIndexChanged.emit(i)

    # ---- перебор: клик (вперёд) и колесо мыши (вперёд/назад) ----
    def _advance(self, step: int) -> None:
        n = len(self._labels)
        if n:
            self.setCurrentIndex((self._index + step) % n)

    def wheelEvent(self, e) -> None:
        self._advance(1 if e.angleDelta().y() > 0 else -1)
        e.accept()
