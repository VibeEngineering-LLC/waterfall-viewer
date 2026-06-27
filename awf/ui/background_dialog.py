"""
Диалог выбора источника фона (Задача #96): диапазон срезов текущего измерения ЛИБО отдельный
файл. Результат — через .result_spec(): ("range", t_lo, t_hi) или ("file", path); None если
отменён. Сам расчёт фона делает вызывающий (MainWindow) — диалог только собирает выбор.
"""
from __future__ import annotations

from PySide6 import QtWidgets


class BackgroundDialog(QtWidgets.QDialog):
    def __init__(self, n_slices: int, time_offsets=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор фона")
        self._n = int(n_slices)
        self._times = time_offsets
        self._path = None
        self._spec = None
        root = QtWidgets.QVBoxLayout(self)
        # --- источник 1: диапазон срезов текущего измерения ---
        self._rb_range = QtWidgets.QRadioButton("Из текущего измерения (диапазон срезов)")
        self._rb_range.setChecked(True)
        root.addWidget(self._rb_range)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("срезы:"))
        self._lo = QtWidgets.QSpinBox(); self._lo.setRange(0, max(0, self._n - 1))
        self._hi = QtWidgets.QSpinBox(); self._hi.setRange(1, self._n)
        self._lo.setValue(0); self._hi.setValue(self._n)
        self._lo.valueChanged.connect(self._update_hint)
        self._hi.valueChanged.connect(self._update_hint)
        row.addWidget(self._lo); row.addWidget(QtWidgets.QLabel("–")); row.addWidget(self._hi)
        row.addStretch(1)
        root.addLayout(row)
        self._hint = QtWidgets.QLabel("")
        root.addWidget(self._hint)
        # --- источник 2: отдельный файл ---
        self._rb_file = QtWidgets.QRadioButton("Из файла (.aswf / .rcspg / .n42)")
        root.addWidget(self._rb_file)
        frow = QtWidgets.QHBoxLayout()
        self._path_lbl = QtWidgets.QLabel("файл не выбран")
        self._browse = QtWidgets.QPushButton("Обзор…")
        self._browse.clicked.connect(self._on_browse)
        frow.addWidget(self._path_lbl, 1); frow.addWidget(self._browse)
        root.addLayout(frow)
        # --- OK / Cancel ---
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        self._update_hint()

    def _update_hint(self) -> None:
        lo = self._lo.value(); hi = self._hi.value()
        n_sel = max(0, hi - lo)
        t = self._times
        if t is not None and len(t) >= self._n and self._n > 0:
            a = float(t[max(0, min(self._n - 1, lo))])
            b = float(t[max(0, min(self._n - 1, hi - 1))])
            self._hint.setText(f"диапазон времени ≈ {a:.1f}–{b:.1f} с  ({n_sel} срезов)")
        else:
            self._hint.setText(f"{n_sel} срезов")

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Файл фона", "",
            "Спектрограммы (*.n42 *.xml *.rcspg *.aswf);;Все файлы (*)")
        if path:
            self._path = path
            self._path_lbl.setText(path)
            self._rb_file.setChecked(True)

    def _on_accept(self) -> None:
        if self._rb_file.isChecked():
            if not self._path:
                QtWidgets.QMessageBox.warning(self, "Выбор фона", "Файл фона не выбран.")
                return
            self._spec = ("file", self._path)
        else:
            lo = self._lo.value(); hi = self._hi.value()
            if hi <= lo:
                QtWidgets.QMessageBox.warning(self, "Выбор фона", "Пустой диапазон срезов.")
                return
            self._spec = ("range", lo, hi)
        self.accept()

    def result_spec(self):
        """Выбранный источник фона или None (отменён): ('range',lo,hi) | ('file',path)."""
        return self._spec