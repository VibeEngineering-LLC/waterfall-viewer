"""
Тесты Android-варианта .rcspg RadiaCode — табулированный ТЕКСТ (Задача #103).
iOS/JSON-вариант проверяется отдельно в test_rcspg_loader.py; здесь синтетический
текстовый файл: заголовок 'Spectrogram:' + строка 'Spectrum: <hex>' (калибровка) +
строки-интервалы '<FILETIME>\t<живых_сек>\t<отсчёты по каналам>'.
"""
from __future__ import annotations
import struct
import numpy as np
import pytest
from awf.io.rcspg_loader import load_rcspg, _filetime_to_iso

_BASE_FT = 133694269909800000   # FILETIME заголовка == 2024-08-29 18:43:10 UTC
_TICK = 10_000_000              # тиков (100 нс) в секунде


def _hex_base(coeffs=(1.0, 2.0, 0.0), base_counts=(0, 0, 0, 0, 0, 0, 0, 0)) -> str:
    raw = struct.pack("<I", 4918615)             # 4 служебных байта
    raw += struct.pack("<fff", *coeffs)          # 3*float32 калибровка
    raw += struct.pack("<%dI" % len(base_counts), *base_counts)  # N*uint32 базовый спектр
    return " ".join("%02X" % b for b in raw)


def _default_rows():
    ts0 = _BASE_FT + 2 * _TICK
    ts1 = ts0 + 3 * _TICK
    ts2 = ts1 + 1 * _TICK
    return [
        (ts0, 2, [0, 1, 2, 3]),                  # хвостовые нули опущены
        (ts1, 3, [5, 0, 0, 0, 0, 0, 0, 4]),
        (ts2, 1, [1, 1]),
    ]


def _write_rcspg(tmp_path, *, channels=8, include_channels=True, include_base=True, rows=None) -> str:
    if rows is None:
        rows = _default_rows()
    hdr = ["Spectrogram: test", "Time: 2024-08-29 18:43:10", f"Timestamp: {_BASE_FT}",
           "Accumulation time: 6"]
    if include_channels:
        hdr.append(f"Channels: {channels}")
    hdr += ["Device serial: TEST", "Flags: 1", "Comment: "]
    out = ["\t".join(hdr)]
    if include_base:
        out.append("Spectrum: " + _hex_base())
    for ts, dur, chans in rows:
        out.append("\t".join([str(ts), str(dur)] + [str(c) for c in chans]))
    path = tmp_path / "android.rcspg"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return str(path)


def test_text_basic_shape_and_dtype(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path))
    assert sg.n_slices == 3
    assert sg.n_channels == 8
    assert sg.counts.dtype == np.uint16


def test_text_counts_padding(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path))
    np.testing.assert_array_equal(sg.counts[0], [0, 1, 2, 3, 0, 0, 0, 0])
    np.testing.assert_array_equal(sg.counts[1], [5, 0, 0, 0, 0, 0, 0, 4])
    np.testing.assert_array_equal(sg.counts[2], [1, 1, 0, 0, 0, 0, 0, 0])


def test_text_calibration_from_base_spectrum(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path))
    np.testing.assert_allclose(sg.calibration.coeffs, [1.0, 2.0, 0.0], atol=1e-6)
    en = sg.energies()
    assert en[0] == pytest.approx(1.0, abs=1e-5)
    assert en[1] == pytest.approx(3.0, abs=1e-5)


def test_text_time_axes(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path))
    np.testing.assert_allclose(sg.live_time_s, [2.0, 3.0, 1.0])
    np.testing.assert_allclose(sg.real_time_s, [2.0, 3.0, 1.0])
    np.testing.assert_allclose(sg.time_offsets_s, [0.0, 2.0, 5.0])


def test_text_t0_iso(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path))
    assert sg.t0_iso == "2024-08-29T17:43:10Z"  # FILETIME = UTC; заголовочный Time 18:43:10 — локальное (UTC+1)
    assert _filetime_to_iso(None) is None
    assert _filetime_to_iso(_BASE_FT) == "2024-08-29T17:43:10Z"  # FILETIME = UTC; заголовочный Time 18:43:10 — локальное (UTC+1)


def test_text_max_slices(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path), max_slices=2)
    assert sg.n_slices == 2
    assert sg.n_channels == 8


def test_text_empty_rows_raises(tmp_path):
    with pytest.raises(ValueError):
        load_rcspg(_write_rcspg(tmp_path, rows=[]))


def test_text_channels_inferred(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path, include_channels=False))
    assert sg.n_channels == 8


def test_text_default_calibration_without_base(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path, include_base=False))
    np.testing.assert_allclose(sg.calibration.coeffs, [0.0, 1.0])