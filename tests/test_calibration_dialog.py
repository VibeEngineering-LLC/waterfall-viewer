"""Тесты #215 — интеграция CalibrationDialog: пресеты, фит, сигнал."""
from __future__ import annotations

import pytest

from PySide6 import QtWidgets

from awf.ui.calibration_dialog import CalibrationDialog


@pytest.fixture(scope="module")
def _qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_dialog_construction(_qapp):
    dlg = CalibrationDialog(current_coeffs=[0.0, 0.35, 1.0e-5])
    assert dlg.windowTitle()
    assert dlg._table.rowCount() == 0


def test_add_row_and_e_current(_qapp):
    dlg = CalibrationDialog(current_coeffs=[0.0, 0.35, 0.0])
    dlg._add_row("1000", "355.0")
    assert dlg._table.rowCount() == 1
    e_cur = dlg._table.item(0, dlg._COL_E_CUR).text()
    assert e_cur.replace(",", ".").startswith("350")


def test_preset_adds_rows(_qapp):
    dlg = CalibrationDialog(current_coeffs=[0.0, 0.35, 0.0])
    dlg._preset_combo.setCurrentText("Cs-137")
    dlg._on_add_preset()
    assert dlg._table.rowCount() == 1
    et = dlg._table.item(0, dlg._COL_E_TRUE).text()
    assert et.startswith("661")


def test_fit_signal_and_apply(_qapp):
    dlg = CalibrationDialog(current_coeffs=[0.0, 0.35, 0.0])
    # синтетика: E = 5 + 0.3*ch + 1e-5*ch^2
    for ch, e_true in [(100, 5 + 0.3*100 + 1e-5*100**2),
                       (500, 5 + 0.3*500 + 1e-5*500**2),
                       (2000, 5 + 0.3*2000 + 1e-5*2000**2),
                       (5000, 5 + 0.3*5000 + 1e-5*5000**2)]:
        dlg._add_row(f"{ch}", f"{e_true:.4f}")
    dlg._deg_box.setValue(2)
    captured = []
    dlg.calibrationApplied.connect(lambda c: captured.append(list(c)))
    dlg._on_fit()
    assert dlg._last_coeffs is not None
    assert dlg._apply_btn.isEnabled()
    dlg._on_apply()
    assert len(captured) == 1
    coeffs = captured[0]
    assert len(coeffs) == 3
    assert abs(coeffs[0] - 5.0) < 1e-3
    assert abs(coeffs[1] - 0.3) < 1e-4
    assert abs(coeffs[2] - 1e-5) < 1e-7


def test_fit_too_few_pairs_disables_apply(_qapp):
    dlg = CalibrationDialog(current_coeffs=[0.0, 0.35, 0.0])
    dlg._add_row("100", "35")
    dlg._on_fit()
    assert dlg._last_coeffs is None
    assert not dlg._apply_btn.isEnabled()


def test_from_peaks_populates(_qapp):
    class _Pk:
        def __init__(self, ch, e):
            self.channel = ch
            self.energy = e
    peaks = [_Pk(660.0, 231.0), _Pk(1900.0, 665.0), _Pk(7500.0, 2625.0)]
    dlg = CalibrationDialog(current_coeffs=[0.0, 0.35, 0.0], found_peaks=peaks)
    assert dlg._from_peaks_btn.isEnabled()
    dlg._on_from_peaks()
    assert dlg._table.rowCount() == 3
