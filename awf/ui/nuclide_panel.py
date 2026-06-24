from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets

COLORS = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324")

class NuclidePanel(QtWidgets.QWidget):
    linesChanged = QtCore.Signal(object)  # list[tuple[float, str, str]]

    def __init__(self, nuclides=None, parent=None):
        super().__init__(parent)
        self._nuclides = []

        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Библиотека нуклидов"))

        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Мин. интенсивность, %:"))
        self._min_int = QtWidgets.QDoubleSpinBox()
        self._min_int.setRange(0, 100)
        self._min_int.setSingleStep(1.0)
        self._min_int.setValue(5.0)
        self._min_int.setDecimals(1)
        filter_layout.addWidget(self._min_int)

        self._only_used = QtWidgets.QCheckBox("только основные линии")
        self._only_used.setChecked(True)
        filter_layout.addWidget(self._only_used)

        layout.addLayout(filter_layout)

        self._list = QtWidgets.QListWidget()
        self._list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self._list)

        btn_layout = QtWidgets.QHBoxLayout()
        self._btn_none = QtWidgets.QPushButton("Снять все")
        btn_layout.addWidget(self._btn_none)
        layout.addLayout(btn_layout)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

        self._min_int.valueChanged.connect(self._recompute)
        self._only_used.stateChanged.connect(self._recompute)
        self._list.itemChanged.connect(self._on_item_changed)
        self._btn_none.clicked.connect(self.clear_selection)

        if nuclides is not None:
            self.set_library(nuclides)

    def set_library(self, nuclides) -> None:
        self._nuclides = list(nuclides)
        self._list.blockSignals(True)
        self._list.clear()
        for i, nuclide in enumerate(nuclides):
            item = QtWidgets.QListWidgetItem(nuclide.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            color = COLORS[i % len(COLORS)]
            item.setForeground(QtGui.QColor(color))
            item.setData(QtCore.Qt.UserRole, i)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._recompute()

    def _on_item_changed(self, item) -> None:
        self._recompute()

    def clear_selection(self) -> None:
        self._list.blockSignals(True)
        for row in range(self._list.count()):
            self._list.item(row).setCheckState(QtCore.Qt.Unchecked)
        self._list.blockSignals(False)
        self._recompute()

    def _recompute(self) -> None:
        lines = self._collect_lines()
        checked = sum(1 for row in range(self._list.count())
                      if self._list.item(row).checkState() == QtCore.Qt.Checked)
        self._status.setText(f"выбрано нуклидов: {checked}, линий: {len(lines)}")
        self.linesChanged.emit(lines)

    def _collect_lines(self):
        min_int = self._min_int.value()
        only_used = self._only_used.isChecked()
        lines = []
        checked = 0
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.checkState() != QtCore.Qt.Checked:
                continue
            checked += 1
            i = item.data(QtCore.Qt.UserRole)
            nuc = self._nuclides[i]
            color = COLORS[i % len(COLORS)]
            for ln in nuc.major_lines(min_intensity=min_int, only_used=only_used):
                label = nuc.name
                lines.append((float(ln.energy), color, label))
        return lines

    def selected_lines(self) -> list:
        return self._collect_lines()
