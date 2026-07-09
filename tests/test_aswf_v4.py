from __future__ import annotations

import struct
import json
import zlib
import numpy as np
import pytest
from awf.io.aswf_loader import load_aswf


def _build_aswf_v4(tmp_path, n_rows=3, corrupt_row=None, channels=8192) -> str:
    header_reserve = 4096
    row_stride = 16406
    baseline_size = channels * 4

    # JSON header
    json_header = {
        "saved_rows": 0,
        "saved_at": 0,
        "format": "atomspectra-waterfall",
        "version": 4,
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
            {"name": "crc32", "dtype": "uint32", "algo": "crc32", "covers": 16402, "offset": 16402}
        ],
        "baseline": {"dtype": "uint32", "channels": channels, "byte_order": "little"},
        "seg_seq": 12345,
        "total_at_open": 987654,
        "interval_sec": 60,
        "started_at": 1751000000,
        "calibration": [0.0, 0.123, 0.0]
    }

    json_bytes = json.dumps(json_header).encode("utf-8")
    header_bytes = json_bytes.ljust(header_reserve, b" ")

    # Baseline section
    baseline = np.arange(channels, dtype="<u4").tobytes()

    # Payload rows
    payload = bytearray()
    started_at = 1751000000
    for i in range(n_rows):
        row = bytearray(row_stride)
        value = i + 1
        spectrum = np.full(channels, value, dtype="<u2")
        spectrum[100] = 1000 + i  # маркерный канал для test_v4_spectrum_values
        row[0:16384] = spectrum.tobytes()

        # duration
        struct.pack_into("<H", row, 16384, 60)

        # timestamp
        struct.pack_into("<I", row, 16386, started_at + i * 60)

        # latitude and longitude
        struct.pack_into("<f", row, 16390, float("nan"))
        struct.pack_into("<f", row, 16394, float("nan"))

        # dose_rate
        dose = 0.15 + i * 0.01
        struct.pack_into("<f", row, 16398, dose)

        # crc32
        crc_data = bytes(row[:16402])
        crc = zlib.crc32(crc_data) & 0xFFFFFFFF
        struct.pack_into("<I", row, 16402, crc)

        if corrupt_row is not None and i == corrupt_row:
            # Corrupt one byte in spectrum data
            row[0] ^= 0xFF

        payload.extend(row)

    # Build file
    path = tmp_path / "test.aswf"
    with open(path, "wb") as f:
        f.write(b"ASWF")
        f.write(struct.pack("<I", header_reserve))
        f.write(header_bytes)
        f.write(baseline)
        f.write(payload)

    return str(path)


def test_v4_shape_from_filesize(tmp_path):
    path = _build_aswf_v4(tmp_path, n_rows=3)
    sg = load_aswf(path)
    assert sg.n_slices == 3
    assert sg.n_channels == 8192
    assert sg.counts.dtype == np.uint16


def test_v4_spectrum_values(tmp_path):
    path = _build_aswf_v4(tmp_path, n_rows=3)
    sg = load_aswf(path)
    # spectrum in row 0, channel 100 should be 1000
    assert sg.counts[0, 100] == 1000
    # spectrum in row 2, channel 0 should be 3
    assert sg.counts[2, 0] == 3


def test_v4_baseline_section(tmp_path):
    path = _build_aswf_v4(tmp_path, n_rows=3)
    sg = load_aswf(path)
    assert sg.baseline is not None
    assert sg.baseline.shape == (8192,)
    assert sg.baseline[10] == 10


def test_v4_dose_and_time(tmp_path):
    path = _build_aswf_v4(tmp_path, n_rows=3)
    sg = load_aswf(path)
    expected_dose = [0.15, 0.16, 0.17]
    assert sg.dose_rate_usv_h.tolist() == pytest.approx(expected_dose)
    assert sg.time_offsets_s.tolist() == [0, 60, 120]


def test_v4_integrity_ok(tmp_path):
    path = _build_aswf_v4(tmp_path, n_rows=3)
    sg = load_aswf(path)
    report = sg.integrity_report
    assert report["status"] == "ok"
    assert report["checked"] == 3
    assert report["bad"] == 0
    assert report["seg_seq"] == 12345
    assert report["total_at_open"] == 987654
    assert report["version"] == 4


def test_v4_integrity_corrupt(tmp_path):
    path = _build_aswf_v4(tmp_path, n_rows=3, corrupt_row=1)
    sg = load_aswf(path)
    assert sg.counts.shape == (3, 8192)
    report = sg.integrity_report
    assert report["status"] == "corrupt"
    assert report["bad"] == 1
    assert report["bad_rows"] == [1]
