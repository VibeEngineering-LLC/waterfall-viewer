from __future__ import annotations
import struct
import json
from pathlib import Path
import numpy as np
import pytest
from awf.io.aswf_loader import load_aswf, _epoch_s_to_iso


def _build_aswf(tmp_path, header: dict, rows: list[list[int]], header_len: int = 4096) -> str:
    body = json.dumps(header).encode("utf-8")
    if len(body) < header_len:
        body = body + b"\x00" * (header_len - len(body))
    magic = b"ASWF"
    len_field = struct.pack("<I", header_len)
    data = b"".join(np.array(row, dtype="<u2").tobytes() for row in rows)
    path = tmp_path / "synthetic.aswf"
    with open(path, "wb") as f:
        f.write(magic + len_field + body + data)
    return str(path)


def _header(**over):
    default = {
        "format": "aswf",
        "version": 1,
        "channels": 4,
        "interval_sec": 10,
        "calibration": [1.0, 2.0, 0.0],
        "started_at": 1700000000,
        "saved_rows": 3,
        "serial": "TEST",
    }
    default.update(over)
    return default


def _rows():
    return [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_basic_shape_and_dtype(tmp_path):
    header = _header()
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    assert sg.n_slices == 3
    assert sg.n_channels == 4
    assert sg.counts.dtype == np.uint16


def test_counts_values(tmp_path):
    header = _header()
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    np.testing.assert_array_equal(sg.counts[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(sg.counts[2], [8, 9, 10, 11])


def test_counts_writable(tmp_path):
    header = _header()
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    assert sg.counts.flags.writeable is True


def test_calibration(tmp_path):
    header = _header()
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    np.testing.assert_allclose(sg.calibration.coeffs, [1.0, 2.0, 0.0])
    en = sg.energies()
    assert en[0] == pytest.approx(1.0)
    assert en[1] == pytest.approx(3.0)


def test_time_axes(tmp_path):
    header = _header()
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    np.testing.assert_allclose(sg.time_offsets_s, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(sg.real_time_s, [10.0, 10.0, 10.0])
    np.testing.assert_allclose(sg.live_time_s, [10.0, 10.0, 10.0])


def test_t0_iso(tmp_path):
    header = _header()
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    assert sg.t0_iso == "2023-11-14T22:13:20Z"
    assert _epoch_s_to_iso(None) is None
    assert _epoch_s_to_iso(1700000000) == "2023-11-14T22:13:20Z"


def test_max_slices(tmp_path):
    header = _header()
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path, max_slices=2)
    assert sg.n_slices == 2
    assert sg.n_channels == 4


def test_bad_magic_raises(tmp_path):
    path = tmp_path / "bad.aswf"
    with open(path, "wb") as f:
        f.write(b"XXXX" + struct.pack("<I", 4096) + b"\x00" * 4096)
    with pytest.raises(ValueError):
        load_aswf(str(path))


def test_saved_rows_caps_file(tmp_path):
    header = _header(saved_rows=2)
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    assert sg.n_slices == 2


def test_saved_rows_absent_infers_from_size(tmp_path):
    header = _header()
    header.pop("saved_rows", None)
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    assert sg.n_slices == 3


def test_default_calibration_when_missing(tmp_path):
    header = _header()
    header.pop("calibration", None)
    rows = _rows()
    path = _build_aswf(tmp_path, header, rows)
    sg = load_aswf(path)
    assert sg.energies()[0] == pytest.approx(0.0)
    assert sg.energies()[1] == pytest.approx(1.0)


# Задача #180: v2 — row_stride > n_channels*2, per-row row_time uint16 после отсчётов, saved_rows=0.
def _build_aswf_v2(tmp_path, header, rows_counts, rows_time_s, header_len=4096):
    body = json.dumps(header).encode("utf-8")
    if len(body) < header_len:
        body += b"\x00" * (header_len - len(body))
    parts = []
    for cnts, t in zip(rows_counts, rows_time_s):
        parts.append(np.array(cnts, dtype="<u2").tobytes())
        parts.append(struct.pack("<H", int(t)))
    path = tmp_path / "synthetic_v2.aswf"
    with open(path, "wb") as f:
        f.write(b"ASWF" + struct.pack("<I", header_len) + body + b"".join(parts))
    return str(path)


def _header_v2(**over):
    d = {"format": "atomspectra-waterfall", "version": 2, "channels": 4, "interval_sec": 5,
         "calibration": [0.0, 1.0], "started_at": 1700000000, "saved_rows": 0, "serial": "TESTV2",
         "row_stride": 4 * 2 + 2, "row_time": {"dtype": "uint16", "unit": "sec", "offset": 4 * 2}}
    d.update(over)
    return d


def test_v2_saved_rows_zero_falls_back_to_size(tmp_path):
    path = _build_aswf_v2(tmp_path, _header_v2(),
                          [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [60, 60, 60])
    sg = load_aswf(path)
    assert sg.n_slices == 3 and sg.n_channels == 4
    np.testing.assert_array_equal(sg.counts[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(sg.counts[-1], [9, 10, 11, 12])


def test_v2_row_time_per_row(tmp_path):
    path = _build_aswf_v2(tmp_path, _header_v2(), [[0, 0, 0, 0]] * 3, [30, 45, 90])
    sg = load_aswf(path)
    np.testing.assert_allclose(sg.real_time_s, [30.0, 45.0, 90.0])
    np.testing.assert_allclose(sg.time_offsets_s, [0.0, 30.0, 75.0])
    np.testing.assert_allclose(sg.live_time_s, [30.0, 45.0, 90.0])


# ===== Задача #197: v3 тесты =====

def _build_aswf_v3(tmp_path, header, rows_data, baseline_counts=None,
                   header_len=4096, fname="synthetic_v3.aswf"):
    """Собрать бинарный v3-файл. rows_data — list[dict] с ключами из row_fields."""
    body = json.dumps(header).encode("utf-8")
    body = body + b"\x00" * max(0, header_len - len(body))
    bl_bytes = b""
    if baseline_counts is not None:
        bl_bytes = b"".join(struct.pack("<I", int(v)) for v in baseline_counts)
    row_stride = int(header.get("row_stride", 0))
    n_channels  = int(header["channels"])
    payload_parts = []
    for row in rows_data:
        rb = bytearray(row_stride)
        spec = row.get("spectrum", [0] * n_channels)
        struct.pack_into(f"<{n_channels}H", rb, 0, *spec)
        for fd in header.get("row_fields", []):
            nm = fd["name"]
            if nm == "spectrum":
                continue
            off   = int(fd["offset"])
            dtype = fd.get("dtype", "uint16")
            val   = row.get(nm, 0)
            if dtype == "uint16":
                struct.pack_into("<H", rb, off, int(val))
            elif dtype == "uint32":
                struct.pack_into("<I", rb, off, int(val))
            elif dtype == "float32":
                struct.pack_into("<f", rb, off, float(val))
        payload_parts.append(bytes(rb))
    path = tmp_path / fname
    with open(path, "wb") as f:
        f.write(b"ASWF" + struct.pack("<I", header_len) + body + bl_bytes)
        for p in payload_parts:
            f.write(p)
    return str(path)


def _header_v3(**over):
    """Базовый v3-заголовок: 4 канала, поля spectrum+duration."""
    n = 4
    d = {
        "format": "atomspectra-waterfall", "version": 3, "channels": n,
        "interval_sec": 10, "calibration": [0.0, 1.0],
        "started_at": 1700000000, "saved_rows": 3,
        "row_stride": n * 2 + 2,
        "row_fields": [
            {"name": "spectrum", "dtype": "uint16", "offset": 0, "count": n},
            {"name": "duration", "dtype": "uint16", "offset": n * 2},
        ],
    }
    d.update(over)
    return d


def test_v3_uncompressed_basic(tmp_path):
    header = _header_v3()
    rows = [
        {"spectrum": [10, 20, 30, 40], "duration": 60},
        {"spectrum": [11, 21, 31, 41], "duration": 45},
        {"spectrum": [12, 22, 32, 42], "duration": 30},
    ]
    path = _build_aswf_v3(tmp_path, header, rows)
    sg = load_aswf(path)
    assert sg.n_slices == 3 and sg.n_channels == 4
    np.testing.assert_array_equal(sg.counts[0], [10, 20, 30, 40])
    np.testing.assert_array_equal(sg.counts[1], [11, 21, 31, 41])
    np.testing.assert_allclose(sg.real_time_s, [60.0, 45.0, 30.0])
    np.testing.assert_allclose(sg.time_offsets_s, [0.0, 60.0, 105.0])
    assert sg.baseline is None
    assert sg.dose_rate_usv_h is None
    assert sg.gps_track is None


def test_v3_baseline_section(tmp_path):
    header = _header_v3()
    header["baseline"] = {"count": 4}
    bl = [100, 200, 300, 400]
    rows = [{"spectrum": [1, 2, 3, 4], "duration": 10}] * 2
    header["saved_rows"] = 2
    path = _build_aswf_v3(tmp_path, header, rows, baseline_counts=bl)
    sg = load_aswf(path)
    assert sg.baseline is not None
    assert sg.baseline.dtype == np.int64
    np.testing.assert_array_equal(sg.baseline, [100, 200, 300, 400])


def test_v3_dose_and_gps(tmp_path):
    n = 4
    # spectrum(8) + duration(2) + dose_rate_usv_h(4) + latitude(4) + longitude(4) = 22
    stride = n * 2 + 2 + 4 + 4 + 4
    header = {
        "format": "atomspectra-waterfall", "version": 3, "channels": n,
        "interval_sec": 10, "calibration": [0.0, 1.0],
        "started_at": 1700000000, "saved_rows": 2,
        "row_stride": stride,
        "row_fields": [
            {"name": "spectrum",        "dtype": "uint16",  "offset": 0,       "count": n},
            {"name": "duration",        "dtype": "uint16",  "offset": n * 2},
            {"name": "dose_rate_usv_h", "dtype": "float32", "offset": n * 2 + 2},
            {"name": "latitude",        "dtype": "float32", "offset": n * 2 + 6},
            {"name": "longitude",       "dtype": "float32", "offset": n * 2 + 10},
        ],
    }
    rows = [
        {"spectrum": [1, 2, 3, 4], "duration": 60,
         "dose_rate_usv_h": 0.5, "latitude": 55.75, "longitude": 37.62},
        {"spectrum": [5, 6, 7, 8], "duration": 60,
         "dose_rate_usv_h": 0.7, "latitude": 55.76, "longitude": 37.63},
    ]
    path = _build_aswf_v3(tmp_path, header, rows)
    sg = load_aswf(path)
    assert sg.dose_rate_usv_h is not None
    np.testing.assert_allclose(sg.dose_rate_usv_h, [0.5, 0.7], rtol=1e-5)
    assert sg.gps_track is not None
    assert sg.gps_track.shape == (2, 2)
    np.testing.assert_allclose(sg.gps_track[0], [55.75, 37.62], rtol=1e-5)
    np.testing.assert_allclose(sg.gps_track[1], [55.76, 37.63], rtol=1e-5)


def test_v3_per_row_timestamp(tmp_path):
    n = 4; t0 = 1700000000
    # spectrum(8) + duration(2) + timestamp(4) = 14
    stride = n * 2 + 2 + 4
    header = {
        "format": "atomspectra-waterfall", "version": 3, "channels": n,
        "interval_sec": 10, "calibration": [0.0, 1.0],
        "started_at": t0, "saved_rows": 3,
        "row_stride": stride,
        "row_fields": [
            {"name": "spectrum",   "dtype": "uint16", "offset": 0,       "count": n},
            {"name": "duration",   "dtype": "uint16", "offset": n * 2},
            {"name": "timestamp",  "dtype": "uint32",  "offset": n * 2 + 2},
        ],
    }
    rows = [
        {"spectrum": [1, 0, 0, 0], "duration": 60, "timestamp": t0},
        {"spectrum": [2, 0, 0, 0], "duration": 60, "timestamp": t0 + 100},
        {"spectrum": [3, 0, 0, 0], "duration": 60, "timestamp": t0 + 250},
    ]
    path = _build_aswf_v3(tmp_path, header, rows)
    sg = load_aswf(path)
    np.testing.assert_allclose(sg.time_offsets_s, [0.0, 100.0, 250.0])


def test_v3_compressed_rle(tmp_path):
    n = 4
    header = {
        "format": "atomspectra-waterfall", "version": 3, "channels": n,
        "interval_sec": 10, "calibration": [0.0, 1.0],
        "started_at": 1700000000, "saved_rows": 0,
        "compressed": True,
        "row_fields": [
            {"name": "spectrum", "dtype": "uint16", "offset": 0, "count": n},
            {"name": "duration", "dtype": "uint16", "offset": 0},
        ],
    }
    # Row 0: spectrum [5,0,0,7] = literal 5, zero-run 2, literal 7; duration=30
    # Row 1: spectrum [0,0,0,0] = zero-run 4; duration=45
    payload = b""
    payload += struct.pack("<H", 5)           # literal 5
    payload += struct.pack("<H", 0x8002)      # 2 zeros
    payload += struct.pack("<H", 7)           # literal 7
    payload += struct.pack("<H", 30)          # duration row 0
    payload += struct.pack("<H", 0x8004)      # 4 zeros
    payload += struct.pack("<H", 45)          # duration row 1

    body = json.dumps(header).encode("utf-8")
    body += b"\x00" * (4096 - len(body))
    path = tmp_path / "v3_rle.aswf"
    with open(path, "wb") as f:
        f.write(b"ASWF" + struct.pack("<I", 4096) + body + payload)
    sg = load_aswf(str(path))
    assert sg.n_slices == 2 and sg.n_channels == 4
    np.testing.assert_array_equal(sg.counts[0], [5, 0, 0, 7])
    np.testing.assert_array_equal(sg.counts[1], [0, 0, 0, 0])
    np.testing.assert_allclose(sg.real_time_s, [30.0, 45.0])


def test_v3_rle_direct():
    from awf.io.aswf_loader import _rle_decode_row
    # [5, 0, 0, 7] encoded as: literal 5, zero-run 2, literal 7
    buf = struct.pack("<HHH", 5, 0x8002, 7)
    spec, pos = _rle_decode_row(buf, 0, 4)
    assert list(spec) == [5, 0, 0, 7]
    assert pos == 6


# ===== Задача #DATA-1: v4 — пер-строчная целостность CRC32 =====

def _row_crc4(n, spectrum, duration, covers):
    """CRC32 первых covers байт строки (spectrum..duration), zlib-совместим."""
    import zlib
    rb = bytearray(covers)
    struct.pack_into(f"<{n}H", rb, 0, *spectrum)
    struct.pack_into("<H", rb, n * 2, int(duration))
    return zlib.crc32(bytes(rb)) & 0xFFFFFFFF


def _header_v4(n=4):
    return {
        "format": "atomspectra-waterfall", "version": 4, "channels": n,
        "interval_sec": 10, "calibration": [0.0, 1.0],
        "started_at": 1700000000, "saved_rows": 3,
        "row_stride": n * 2 + 2 + 4,
        "row_fields": [
            {"name": "spectrum", "dtype": "uint16", "offset": 0, "count": n},
            {"name": "duration", "dtype": "uint16", "offset": n * 2},
            {"name": "crc32", "dtype": "uint32", "offset": n * 2 + 2, "covers": n * 2 + 2},
        ],
    }


def _v4_rows(n, cov):
    specs = [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]]
    return [{"spectrum": s, "duration": 60, "crc32": _row_crc4(n, s, 60, cov)} for s in specs]


def test_v4_crc32_all_ok(tmp_path):
    n = 4; cov = n * 2 + 2
    path = _build_aswf_v3(tmp_path, _header_v4(n), _v4_rows(n, cov), fname="v4_ok.aswf")
    rep = load_aswf(path).integrity_report
    assert rep is not None and rep["status"] == "ok"
    assert rep["checked"] == 3 and rep["bad"] == 0 and rep["version"] == 4


def test_v4_crc32_detects_corruption(tmp_path):
    n = 4; cov = n * 2 + 2
    rows = _v4_rows(n, cov)
    rows[1]["crc32"] = (rows[1]["crc32"] ^ 0xFFFF) & 0xFFFFFFFF   # испортить CRC строки 1
    path = _build_aswf_v3(tmp_path, _header_v4(n), rows, fname="v4_bad.aswf")
    rep = load_aswf(path).integrity_report
    assert rep["status"] == "corrupt" and rep["bad"] == 1 and rep["bad_rows"] == [1]