from __future__ import annotations

import struct
import json
import zlib
import numpy as np
import pytest
from awf.io.aswf_loader import load_aswf


def _build_aswf_v5(tmp_path, n_rows=3, corrupt_row=None, channels=8192, temp_nan=False) -> str:
    header_reserve = 4096
    row_stride = 16410
    baseline_size = channels * 4

    json_header = {
        "saved_rows": 0,
        "saved_at": 0,
        "format": "atomspectra-waterfall",
        "version": 5,
        "channels": channels,
        "dtype": "uint16",
        "byte_order": "little",
        "row_stride": row_stride,
        "row_fields": [
            {"name": "spectrum", "dtype": "uint16", "channels": channels, "offset": 0},
            {"name": "duration", "dtype": "uint16", "unit": "sec", "offset": 16384},
            {"name": "timestamp", "dtype": "uint32", "unit": "unix_sec", "offset": 16386},
            {"name": "latitude", "dtype": "float32", "unit": "deg", "offset": 16390},
            {"name": "longitude", "dtype": "float32", "unit": "deg", "offset": 16394},
            {"name": "dose_rate", "dtype": "float32", "unit": "usv_h", "offset": 16398},
            {"name": "temperature", "dtype": "float32", "unit": "celsius", "offset": 16402},
            {"name": "crc32", "dtype": "uint32", "algo": "crc32", "covers": 16406, "offset": 16406}
        ],
        "baseline": {"dtype": "uint32", "channels": channels, "byte_order": "little"},
        "seg_seq": 22222,
        "total_at_open": 555555,
        "interval_sec": 60,
        "started_at": 1751000000,
        "calibration": [0.0, 0.123, 0.0]
    }

    json_bytes = json.dumps(json_header).encode("utf-8")
    header_bytes = json_bytes.ljust(header_reserve, b" ")

    baseline = np.arange(channels, dtype="<u4").tobytes()

    payload = bytearray()
    started_at = 1751000000
    for i in range(n_rows):
        row = bytearray(row_stride)
        value = i + 1
        spectrum = np.full(channels, value, dtype="<u2")
        spectrum[100] = 1000 + i
        row[0:16384] = spectrum.tobytes()
        struct.pack_into("<H", row, 16384, 60)
        struct.pack_into("<I", row, 16386, started_at + i * 60)
        struct.pack_into("<f", row, 16390, float("nan"))
        struct.pack_into("<f", row, 16394, float("nan"))
        dose = 0.15 + i * 0.01
        struct.pack_into("<f", row, 16398, dose)
        if temp_nan:
            struct.pack_into("<f", row, 16402, float("nan"))
        else:
            temperature = 23.5 + 0.5 * i
            struct.pack_into("<f", row, 16402, temperature)
        crc_data = bytes(row[:16406])
        crc = zlib.crc32(crc_data) & 0xFFFFFFFF
        struct.pack_into("<I", row, 16406, crc)
        if corrupt_row is not None and i == corrupt_row:
            row[0] ^= 0xFF
        payload.extend(row)

    path = tmp_path / "test.aswf"
    with open(path, "wb") as f:
        f.write(b"ASWF")
        f.write(struct.pack("<I", header_reserve))
        f.write(header_bytes)
        f.write(baseline)
        f.write(payload)

    return str(path)


def test_v5_shape_from_filesize(tmp_path):
    path = _build_aswf_v5(tmp_path, n_rows=3, channels=8192)
    sg = load_aswf(path)
    assert sg.counts.shape == (3, 8192)
    assert sg.counts.dtype == np.uint16


def test_v5_spectrum_values(tmp_path):
    path = _build_aswf_v5(tmp_path, n_rows=3, channels=8192)
    sg = load_aswf(path)
    assert sg.counts[0, 100] == 1000
    assert sg.counts[2, 0] == 3


def test_v5_temperature_series(tmp_path):
    path = _build_aswf_v5(tmp_path, n_rows=3, channels=8192)
    sg = load_aswf(path)
    assert sg.temperature_c.tolist() == pytest.approx([23.5, 24.0, 24.5])


def test_v5_temperature_nan(tmp_path):
    path = _build_aswf_v5(tmp_path, n_rows=3, channels=8192, temp_nan=True)
    sg = load_aswf(path)
    assert sg.temperature_c is not None
    assert np.isnan(sg.temperature_c).all()


def test_v5_dose_and_time(tmp_path):
    path = _build_aswf_v5(tmp_path, n_rows=3, channels=8192)
    sg = load_aswf(path)
    assert sg.dose_rate_usv_h.tolist() == pytest.approx([0.15, 0.16, 0.17])
    assert sg.time_offsets_s.tolist() == [0, 60, 120]


def test_v5_integrity_ok(tmp_path):
    path = _build_aswf_v5(tmp_path, n_rows=3, channels=8192)
    sg = load_aswf(path)
    report = sg.integrity_report
    assert report["status"] == "ok"
    assert report["checked"] == 3
    assert report["bad"] == 0
    assert report["seg_seq"] == 22222
    assert report["total_at_open"] == 555555
    assert report["version"] == 5


def test_v5_integrity_corrupt(tmp_path):
    path = _build_aswf_v5(tmp_path, n_rows=3, channels=8192, corrupt_row=1)
    sg = load_aswf(path)
    report = sg.integrity_report
    assert sg.counts.shape == (3, 8192)
    assert report["status"] == "corrupt"
    assert report["bad"] == 1
    assert report["bad_rows"] == [1]
