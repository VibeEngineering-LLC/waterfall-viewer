from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.device_data_panel import DeviceDataPanel, build_device_csv
from awf.ui.panels import HeatmapPanel


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_sg(n=4, with_dose=True, with_temp=True, with_gps=True, gps_partial=False):
    counts = np.full((n, 16), 2, dtype=np.uint16)
    cal = Calibration(coeffs=np.array([0.0, 1.0]))
    t = np.arange(n, dtype=np.float64) * 60.0
    rt = np.full(n, 60.0)
    lt = rt.copy()
    dose = np.linspace(0.1, 0.4, n) if with_dose else None
    temp = np.linspace(20.0, 23.0, n) if with_temp else None
    gps = np.column_stack([np.linspace(55.0, 55.1, n), np.linspace(37.0, 37.1, n)]) if with_gps else None
    if gps_partial and gps is not None:
        gps[0, :] = np.nan
    return Spectrogram(
        counts=counts,
        calibration=cal,
        time_offsets_s=t,
        real_time_s=rt,
        live_time_s=lt,
        dose_rate_usv_h=dose,
        gps_track=gps,
        temperature_c=temp,
    )


def test_csv_header_and_rows():
    sg = _make_sg(n=4)
    csv_lines = build_device_csv(sg).splitlines()
    assert csv_lines[0] == "index,t_offset_s,duration_s,cps,dose_rate_usv_h,temperature_c,latitude,longitude"
    assert len(csv_lines) == 5


def test_csv_values():
    sg = _make_sg(n=2)
    csv_lines = build_device_csv(sg).splitlines()
    fields = csv_lines[2].split(",")
    assert fields[0] == "1"
    assert fields[1] == "60.000"
    assert fields[3] == "0.5333"
    assert fields[4] == "0.4"
    assert fields[6] == "55.1"


def test_csv_missing_arrays_empty_cells():
    sg = _make_sg(with_dose=False, with_temp=False, with_gps=False)
    csv_lines = build_device_csv(sg).splitlines()
    fields = csv_lines[2].split(",")
    assert fields[4] == ""
    assert fields[5] == ""
    assert fields[6] == ""
    assert fields[7] == ""


def test_panel_smoke_full(qapp):
    p = DeviceDataPanel()
    sg = _make_sg()
    p.set_spectrogram(sg)
    status_text = p._status.text()
    assert status_text.count("✓") == 3
    assert p._export_btn.isEnabled()


def test_panel_smoke_none(qapp):
    p = DeviceDataPanel()
    p.set_spectrogram(None)
    assert not p._export_btn.isEnabled()


def test_panel_gps_partial_no_crash(qapp):
    p = DeviceDataPanel()
    sg = _make_sg(gps_partial=True)
    p.set_spectrogram(sg)
    status_text = p._status.text()
    assert "GPS: ✓" in status_text


def test_panel_no_gps_dash(qapp):
    p = DeviceDataPanel()
    sg = _make_sg(with_gps=False)
    p.set_spectrogram(sg)
    status_text = p._status.text()
    assert "GPS: —" in status_text


def test_panel_time_axis_from_zero(qapp):
    """Задача #UI-237: ось X от 0 до конца записи, даже на пустом графике дозы."""
    p = DeviceDataPanel()
    p.set_spectrogram(_make_sg(n=4, with_dose=False))
    for plot in (p._dose_plot, p._temp_plot):
        x_lo, x_hi = plot.getViewBox().viewRange()[0]
        assert x_lo == pytest.approx(0.0)
        assert x_hi == pytest.approx(180.0)


def test_panel_time_unit_switch(qapp):
    """Задача #UI-238: set_time_unit('ч') масштабирует ось времени в часы."""
    p = DeviceDataPanel()
    p.set_spectrogram(_make_sg(n=4))
    p.set_time_unit("ч")
    x = p._temp_curve.getData()[0]
    assert float(np.max(x)) == pytest.approx(180.0 / 3600.0)


def test_heatmap_temp_overlay_toggle(qapp):
    """Задача #UI-239: оверлей температуры на 2D-карте включается/выключается."""
    hm = HeatmapPanel()
    hm.set_spectrogram(_make_sg(n=4))
    assert not hm._temp_curve_item.isVisible()
    hm.set_temp_overlay(True)
    assert hm._temp_curve_item.isVisible()
    assert len(hm._temp_curve_item.getData()[0]) == 4
    hm.set_temp_overlay(False)
    assert not hm._temp_curve_item.isVisible()


def test_heatmap_temp_overlay_no_data(qapp):
    """Задача #UI-239: без температуры в файле оверлей не включается."""
    hm = HeatmapPanel()
    hm.set_spectrogram(_make_sg(n=4, with_temp=False))
    hm.set_temp_overlay(True)
    assert not hm._temp_curve_item.isVisible()
