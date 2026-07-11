from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.integrity_dialog import IntegrityDialog, build_integrity_text


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_sg(report=None):
    counts = np.full((3, 8), 1, dtype=np.uint16)
    calibration = Calibration(coeffs=np.array([0.0, 1.0]))
    t = np.array([0.0, 60.0, 120.0])
    rt = np.full(3, 60.0)
    lt = rt.copy()
    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=t,
        real_time_s=rt,
        live_time_s=lt,
        t0_iso="2026-07-11T10:00:00Z",
        source_path="test.aswf",
        integrity_report=report,
        temperature_c=np.array([20.0, 21.0, 22.0])
    )


def test_text_ok_report():
    report = {
        "algo": "crc32",
        "checked": 3,
        "ok": 3,
        "bad": 0,
        "bad_rows": [],
        "status": "ok",
        "version": 5,
        "seg_seq": 7,
        "total_at_open": 100
    }
    sg = _make_sg(report)
    text = build_integrity_text(sg)
    assert "CRC32: OK" in text
    assert "test.aswf" in text
    assert "seg_seq: 7" in text
    assert "total_at_open: 100" in text
    assert "5" in text
    assert "Всего отсчётов" in text
    assert ": 24" in text
    assert "Температура: ✓" in text
    assert "GPS: —" in text


def test_text_corrupt_report():
    report = {
        "algo": "crc32",
        "checked": 3,
        "ok": 1,
        "bad": 2,
        "bad_rows": [0, 2],
        "status": "corrupt",
        "version": 5,
        "seg_seq": 7,
        "total_at_open": 100
    }
    sg = _make_sg(report)
    text = build_integrity_text(sg)
    assert "ПОВРЕЖДЕНО" in text
    assert "0, 2" in text


def test_text_no_report():
    sg = _make_sg(None)
    text = build_integrity_text(sg)
    assert "недоступен" in text


def test_text_duration():
    report = {
        "algo": "crc32",
        "checked": 3,
        "ok": 3,
        "bad": 0,
        "bad_rows": [],
        "status": "ok",
        "version": 5,
        "seg_seq": 7,
        "total_at_open": 100
    }
    sg = _make_sg(report)
    text = build_integrity_text(sg)
    assert "180" in text


def test_dialog_smoke(qapp):
    report = {
        "algo": "crc32",
        "checked": 3,
        "ok": 3,
        "bad": 0,
        "bad_rows": [],
        "status": "ok",
        "version": 5,
        "seg_seq": 7,
        "total_at_open": 100
    }
    sg = _make_sg(report)
    text = build_integrity_text(sg)
    dlg = IntegrityDialog(text, None)
    assert dlg._text_edit.toPlainText()
    assert "===" in dlg._text_edit.toPlainText()
    assert dlg.windowTitle()
