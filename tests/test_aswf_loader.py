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
