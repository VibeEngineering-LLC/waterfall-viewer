"""Задача #215 — диалог интерактивной калибровки E(channel).

Пользователь заполняет таблицу пар (канал ↔ истинная энергия), выбирает степень
полинома, жмёт «Фит» → RMSE в статусе, «Применить» → сигнал calibrationApplied(list)
с новыми коэффициентами (прямой порядок c0..cN, совместимо с Calibration).

Дополнительно:
- Presets combo: набор эталонных энергий по нуклиду (Th-232, Cs-137, Co-60 и т.п.);
  выбор пресета вставляет строки с пустым каналом — пользователь дозаполняет.
- «Из найденных пиков…» — если у MainWindow есть найденные пики (Задача #111),
  предзаполняет строки (канал = pk.channel, «истинная E» = pk.energy — редактируется).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from awf.analysis.calibration_fit import (
    PRESETS,
    compute_rmse,
    fit_calibration,
    format_coeffs,
)
from awf.ui.i18n import tr


class CalibrationDialog(QtWidgets.QDialog):
    """Диалог перекалибровки. Возвращает список коэффициентов через сигнал."""

    calibrationApplied = QtCore.pyqtSignal(list)

    _COL_ENABLED = 0   # Задача #223: чекбокс вкл/откл строки
    _COL_CHANNEL = 1
    _COL_E_CUR = 2
    _COL_E_TRUE = 3

    def __init__(self, current_coeffs, n_channels=8192, found_peaks=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Калибровка по пикам"))
        self._current_coeffs = list(current_coeffs)
        self._n_channels = int(n_channels)
        self._found_peaks = found_peaks or []
        self._last_coeffs = None
        self._active_nuclides: set = set(PRESETS.keys())  # Задача #224

        root = QtWidgets.QVBoxLayout(self)

        cur_lbl = QtWidgets.QLabel(
            f"{tr('Текущая калибровка')}: {format_coeffs(self._current_coeffs)}"
        )
        cur_lbl.setWordWrap(True)
        cur_lbl.setStyleSheet("color: #a0a0a0;")
        root.addWidget(cur_lbl)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel(tr("Пресет:")))
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.addItem(tr("— выберите —"))
        for name in PRESETS.keys():
            self._preset_combo.addItem(name)
        preset_row.addWidget(self._preset_combo)
        self._preset_btn = QtWidgets.QPushButton(tr("Добавить строки"))
        self._preset_btn.setToolTip(tr(
            "Добавить строки с истинными энергиями пресета (канал заполните сами)"
        ))
        self._preset_btn.clicked.connect(self._on_add_preset)
        preset_row.addWidget(self._preset_btn)
        preset_row.addStretch(1)
        self._from_peaks_btn = QtWidgets.QPushButton(tr("Из найденных пиков…"))
        self._from_peaks_btn.setEnabled(len(self._found_peaks) > 0)
        self._from_peaks_btn.setToolTip(tr(
            "Взять каналы и энергии из окна «Найденные пики»; истинную энергию впишите"
        ))
        self._from_peaks_btn.clicked.connect(self._on_from_peaks)
        preset_row.addWidget(self._from_peaks_btn)
        root.addLayout(preset_row)

        nuclide_row = QtWidgets.QHBoxLayout()   # Задача #224
        nuclide_row.addWidget(QtWidgets.QLabel(tr("Нуклиды:")))
        self._nuclides_btn = QtWidgets.QPushButton("")
        self._nuclides_btn.clicked.connect(self._on_nuclides_select)
        nuclide_row.addWidget(self._nuclides_btn)
        self._pick_e_btn = QtWidgets.QPushButton(tr("Подобрать E ▼"))
        self._pick_e_btn.setToolTip(tr("Ближайшие линии активных нуклидов для выбранной строки"))
        self._pick_e_btn.clicked.connect(self._on_pick_energy)
        nuclide_row.addStretch(1)
        nuclide_row.addWidget(self._pick_e_btn)
        root.addLayout(nuclide_row)
        self._update_nuclides_btn()

        self._table = QtWidgets.QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels([
            "", tr("Канал"), tr("E тек., кэВ"), tr("E истинная, кэВ"),
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            self._COL_ENABLED, QtWidgets.QHeaderView.Fixed
        )
        self._table.setColumnWidth(self._COL_ENABLED, 28)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.itemChanged.connect(self._on_table_changed)
        root.addWidget(self._table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton(tr("+ строка"))
        add_btn.clicked.connect(lambda: self._add_row("", ""))
        rm_btn = QtWidgets.QPushButton(tr("− удалить"))
        rm_btn.clicked.connect(self._remove_selected)
        clr_btn = QtWidgets.QPushButton(tr("Очистить"))
        clr_btn.clicked.connect(lambda: self._table.setRowCount(0))
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        btn_row.addWidget(clr_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(QtWidgets.QLabel(tr("Степень:")))
        self._deg_box = QtWidgets.QSpinBox()
        self._deg_box.setRange(1, 4)
        self._deg_box.setValue(2)
        btn_row.addWidget(self._deg_box)
        root.addLayout(btn_row)

        fit_row = QtWidgets.QHBoxLayout()
        self._fit_btn = QtWidgets.QPushButton(tr("Фит"))
        self._fit_btn.clicked.connect(self._on_fit)
        fit_row.addWidget(self._fit_btn)
        self._status_lbl = QtWidgets.QLabel("")
        self._status_lbl.setWordWrap(True)
        fit_row.addWidget(self._status_lbl, 1)
        root.addLayout(fit_row)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel
        )
        self._apply_btn = bb.button(QtWidgets.QDialogButtonBox.Apply)
        self._apply_btn.setText(tr("Применить"))
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _update_nuclides_btn(self) -> None:
        """Задача #224: надпись кнопки нуклидов — показывает N/total."""
        n, t = len(self._active_nuclides), len(PRESETS)
        self._nuclides_btn.setText(f"{tr('Нуклиды')}: {n}/{t} ▼")

    def _on_nuclides_select(self) -> None:
        """Задача #224: чекбокс-меню выбора нуклидов для Подобрать E."""
        menu = QtWidgets.QMenu(self)
        for name in PRESETS:
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(name in self._active_nuclides)
            act.toggled.connect(
                lambda checked, n=name: (
                    self._active_nuclides.add(n) if checked
                    else self._active_nuclides.discard(n)
                )
            )
        menu.exec(self._nuclides_btn.mapToGlobal(self._nuclides_btn.rect().bottomLeft()))
        self._update_nuclides_btn()

    def _add_row(self, ch_text, e_true_text):
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.blockSignals(True)
        chk_item = QtWidgets.QTableWidgetItem()  # Задача #223
        chk_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        chk_item.setCheckState(QtCore.Qt.Checked)
        self._table.setItem(r, self._COL_ENABLED, chk_item)
        ch_item = QtWidgets.QTableWidgetItem(str(ch_text))
        ch_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._table.setItem(r, self._COL_CHANNEL, ch_item)
        e_cur_item = QtWidgets.QTableWidgetItem("")
        e_cur_item.setFlags(e_cur_item.flags() & ~QtCore.Qt.ItemIsEditable)
        e_cur_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        e_cur_item.setForeground(QtGui.QColor("#a0a0a0"))
        self._table.setItem(r, self._COL_E_CUR, e_cur_item)
        et_item = QtWidgets.QTableWidgetItem(str(e_true_text))
        et_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._table.setItem(r, self._COL_E_TRUE, et_item)
        self._table.blockSignals(False)
        self._recompute_e_current(r)

    def _on_pick_energy(self) -> None:
        """Задача #224: меню ближайших линий активных нуклидов для текущей строки."""
        row = self._table.currentRow()
        if row < 0:
            return
        e_cur_it = self._table.item(row, self._COL_E_CUR)
        try:
            e_ref = float(e_cur_it.text().replace(",", ".")) if e_cur_it else 0.0
        except ValueError:
            e_ref = 0.0
        cands = sorted(
            (abs(e - e_ref), n, e)
            for n in sorted(self._active_nuclides) for e in PRESETS.get(n, [])
        )
        if not cands:
            return
        menu = QtWidgets.QMenu(self)
        for _d, nname, e in cands[:25]:
            menu.addAction(f"{nname}  {e:.2f} кэВ").setData(e)
        chosen = menu.exec(self._pick_e_btn.mapToGlobal(self._pick_e_btn.rect().bottomLeft()))
        if chosen is not None and self._table.item(row, self._COL_E_TRUE) is not None:
            self._table.item(row, self._COL_E_TRUE).setText(f"{chosen.data():.2f}")

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _on_add_preset(self):
        idx = self._preset_combo.currentIndex()
        if idx <= 0:
            return
        name = self._preset_combo.currentText()
        energies = PRESETS.get(name, [])
        for e in energies:
            self._add_row("", f"{e:.2f}")

    def _on_from_peaks(self):
        for pk in self._found_peaks:
            ch = float(getattr(pk, "channel", 0.0))
            e_cur = float(getattr(pk, "energy", 0.0))
            self._add_row(f"{ch:.1f}", f"{e_cur:.2f}")

    def _recompute_e_current(self, row):
        ch_item = self._table.item(row, self._COL_CHANNEL)
        e_cur_item = self._table.item(row, self._COL_E_CUR)
        if ch_item is None or e_cur_item is None:
            return
        try:
            ch = float(ch_item.text().replace(",", "."))
        except (TypeError, ValueError):
            e_cur_item.setText("")
            return
        import numpy.polynomial.polynomial as P
        e = float(P.polyval(ch, np.asarray(self._current_coeffs, dtype=np.float64)))
        e_cur_item.setText(f"{e:.2f}")

    def _on_table_changed(self, item):
        if item.column() == self._COL_CHANNEL:
            self._recompute_e_current(item.row())
        elif item.column() == self._COL_ENABLED:  # Задача #223
            self._update_row_dimming(item.row())

    def _update_row_dimming(self, row: int) -> None:
        """Задача #223: затемнить/осветлить строку в зависимости от чекбокса."""
        chk_it = self._table.item(row, self._COL_ENABLED)
        enabled = chk_it is None or chk_it.checkState() == QtCore.Qt.Checked
        clr = QtGui.QColor("#c0c0c0" if enabled else "#555555")
        for col in (self._COL_CHANNEL, self._COL_E_TRUE):
            it = self._table.item(row, col)
            if it is not None:
                it.setForeground(QtGui.QBrush(clr))

    def _collect_pairs(self):
        pairs = []
        for r in range(self._table.rowCount()):
            chk_it = self._table.item(r, self._COL_ENABLED)  # Задача #223
            if chk_it is not None and chk_it.checkState() != QtCore.Qt.Checked:
                continue
            ch_it = self._table.item(r, self._COL_CHANNEL)
            et_it = self._table.item(r, self._COL_E_TRUE)
            if ch_it is None or et_it is None:
                continue
            try:
                ch = float(ch_it.text().replace(",", "."))
                et = float(et_it.text().replace(",", "."))
            except (TypeError, ValueError):
                continue
            if ch <= 0 or et <= 0:
                continue
            pairs.append((ch, et))
        return pairs

    def _on_fit(self):
        pairs = self._collect_pairs()
        if len(pairs) < 2:
            self._status_lbl.setText(tr("Нужно >=2 корректных пар."))
            self._status_lbl.setStyleSheet("color: #d08a8a;")
            self._apply_btn.setEnabled(False)
            self._last_coeffs = None
            return
        try:
            coeffs = fit_calibration(pairs, deg=int(self._deg_box.value()))
            rmse = compute_rmse(coeffs, pairs)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._status_lbl.setText(f"{tr('Ошибка фита')}: {exc}")
            self._status_lbl.setStyleSheet("color: #d08a8a;")
            self._apply_btn.setEnabled(False)
            self._last_coeffs = None
            return
        self._last_coeffs = coeffs
        self._status_lbl.setText(
            f"{tr('RMSE')} = {rmse:.3f} {tr('кэВ')} · {format_coeffs(coeffs)}"
        )
        self._status_lbl.setStyleSheet("color: #a0c8a0;")
        self._apply_btn.setEnabled(True)

    def _on_apply(self):
        if self._last_coeffs is None:
            self._on_fit()
        if self._last_coeffs is None:
            return
        self.calibrationApplied.emit(list(self._last_coeffs))
        self.accept()

    def _last_fitted_coeffs(self):
        return None if self._last_coeffs is None else list(self._last_coeffs)
