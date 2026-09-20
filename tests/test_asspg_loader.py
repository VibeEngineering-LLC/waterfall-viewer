"""
Задача #DATA-7

Тесты для загрузчика ASSPG формата.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from awf.io.asspg_loader import load_asspg, looks_like_asspg


def _slice_block(ts_ms: int, lat: float, lon: float, dur: float, counts: list[int]) -> list[str]:
    return [
        f"{ts_ms}",
        f"{lat:.6f}",
        f"{lon:.6f}",
        f"{dur:.6f}",
        "\t".join(map(str, counts)) + "\t"
    ]


_BASELINE = [0, 1, 2, 3, 0, 0, 3, 3]          # преднакопленный спектр эталонного файла, сумма 12
_COEFFS = ["2.50000000000", "1.75000000000", "3.20000000000e-06", "1.10000000000e-10"]


def _write_asspg(tmp_path, *, lines=None, n_channels=8, degree=2, slices=None) -> str:
    if lines is not None:
        path = tmp_path / "Spectrogram-TEST-DET-2026-03-14_09-41-07.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    if slices is None:
        slices = [
            {
                "ts": 1773470472000,
                "lat": 55.7501,
                "lon": 37.6167,
                "dur": 5.0,
                "counts": [0, 2, 0, 1, 0, 0, 4, 0]
            },
            {
                "ts": 1773470477000,
                "lat": 55.7502,
                "lon": 37.6168,
                "dur": 5.0,
                "counts": [1, 0, 3, 0, 0, 2, 0, 0]
            }
        ]

    preamble = [
        "FORMAT: 3",
        "2026.03.14 09:41:07 +0300 Counts: 12, ~cps: 2.000, Time: 6.00 s, Coord: 55°45'00.000\" N 37°37'00.000\" E at 2026.03.14 09:41:05 +0300, Captured from USB: 00000000",
        "1773470467000",
        "1773470465000",
        "55.750000000000000",
        "37.616666666666667",
        "TEST-DET",
        "USB: 00000000",
        "6.000000",
        f"{n_channels}",
        f"{degree}",
    ]
    # Коэффициентов ровно degree+1; базовый спектр — по ОДНОМУ значению в строке (как в приборе).
    preamble += _COEFFS[:degree + 1]
    preamble += [str(v) for v in _BASELINE[:n_channels]]

    content = preamble[:]
    for s in slices:
        content.extend(_slice_block(s["ts"], s["lat"], s["lon"], s["dur"], s["counts"]))

    path = tmp_path / "Spectrogram-TEST-DET-2026-03-14_09-41-07.txt"
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return str(path)


def test_counts_shape_and_values(tmp_path):
    path = _write_asspg(tmp_path)
    sg = load_asspg(path)
    assert sg.counts.shape == (2, 8)
    assert sg.counts[0].tolist() == [0, 2, 0, 1, 0, 0, 4, 0]
    assert int(sg.counts.sum()) == 13


def test_baseline_is_preaccumulated_spectrum(tmp_path):
    path = _write_asspg(tmp_path)
    sg = load_asspg(path)
    assert sg.baseline is not None
    assert sg.baseline.tolist() == [0, 1, 2, 3, 0, 0, 3, 3]
    assert int(sg.baseline.sum()) == 12


def test_calibration_reads_degree_plus_one_coefficients(tmp_path):
    path = _write_asspg(tmp_path)
    sg = load_asspg(path)
    assert sg.calibration.coeffs.size == 3
    assert np.allclose(sg.calibration.coeffs, [2.5, 1.75, 3.2e-6])


def test_baseline_offset_follows_coefficient_count(tmp_path):
    path = _write_asspg(tmp_path, degree=3, slices=[
        {
            "ts": 1773470472000,
            "lat": 55.7501,
            "lon": 37.6167,
            "dur": 5.0,
            "counts": [0, 2, 0, 1, 0, 0, 4, 0]
        },
        {
            "ts": 1773470477000,
            "lat": 55.7502,
            "lon": 37.6168,
            "dur": 5.0,
            "counts": [1, 0, 3, 0, 0, 2, 0, 0]
        }
    ])
    sg = load_asspg(path)
    assert sg.baseline.tolist() == [0, 1, 2, 3, 0, 0, 3, 3]


def test_time_offsets_start_at_zero(tmp_path):
    path = _write_asspg(tmp_path)
    sg = load_asspg(path)
    assert sg.time_offsets_s[0] == 0.0
    assert sg.time_offsets_s[1] == pytest.approx(5.0)


def test_real_and_live_time_equal_duration(tmp_path):
    path = _write_asspg(tmp_path)
    sg = load_asspg(path)
    assert sg.real_time_s.tolist() == [5.0, 5.0]
    assert sg.live_time_s.tolist() == [5.0, 5.0]


def test_t0_iso_keeps_file_timezone(tmp_path):
    path = _write_asspg(tmp_path)
    sg = load_asspg(path)
    assert sg.t0_iso == "2026-03-14T09:41:07+03:00"


def test_t0_iso_falls_back_to_utc_without_offset(tmp_path):
    lines = [
        "FORMAT: 3",
        "2026.03.14 09:41:07 no timezone here",
        "1773470467000",
        "1773470465000",
        "55.750000000000000",
        "37.616666666666667",
        "TEST-DET",
        "USB: 00000000",
        "6.000000",
        "8",
        "2",
        "2.50000000000",
        "1.75000000000",
        "3.20000000000e-06",
        *[str(v) for v in _BASELINE],
        "1773470472000",
        "55.750100000000000",
        "37.616700000000000",
        "5.000000",
        "0\t2\t0\t1\t0\t0\t4\t0\t"
    ]
    path = _write_asspg(tmp_path, lines=lines)
    sg = load_asspg(path)
    assert sg.t0_iso.endswith("Z")


def test_gps_track_per_slice(tmp_path):
    path = _write_asspg(tmp_path)
    sg = load_asspg(path)
    assert sg.gps_track.shape == (2, 2)
    assert sg.gps_track[0] == pytest.approx([55.7501, 37.6167])


def test_gps_track_none_when_no_fix(tmp_path):
    lines = [
        "FORMAT: 3",
        "2026.03.14 09:41:07 +0300 Counts: 12, ~cps: 2.000, Time: 6.00 s, Coord: 55°45'00.000\" N 37°37'00.000\" E at 2026.03.14 09:41:05 +0300, Captured from USB: 00000000",
        "1773470467000",
        "1773470465000",
        "55.750000000000000",
        "37.616666666666667",
        "TEST-DET",
        "USB: 00000000",
        "6.000000",
        "8",
        "2",
        "2.50000000000",
        "1.75000000000",
        "3.20000000000e-06",
        *[str(v) for v in _BASELINE],
        "1773470472000",
        "0.000000000000000",
        "0.000000000000000",
        "5.000000",
        "0\t2\t0\t1\t0\t0\t4\t0\t",
        "1773470477000",
        "0.000000000000000",
        "0.000000000000000",
        "5.000000",
        "1\t0\t3\t0\t0\t2\t0\t0\t"
    ]
    path = _write_asspg(tmp_path, lines=lines)
    sg = load_asspg(path)
    assert sg.gps_track is None


def test_max_slices_limits_rows(tmp_path):
    path = _write_asspg(tmp_path)
    sg = load_asspg(path, max_slices=1)
    assert sg.counts.shape == (1, 8)


def test_looks_like_asspg_accepts_and_rejects(tmp_path):
    path = _write_asspg(tmp_path)
    assert looks_like_asspg(path) is True

    lines = [
        "Spectrogram: something",
        "1\t2\t3"
    ]
    path = _write_asspg(tmp_path, lines=lines)
    assert looks_like_asspg(path) is False

    assert looks_like_asspg(tmp_path / "nonexistent.txt") is False


def test_rejects_unsupported_version(tmp_path):
    lines = [
        "FORMAT: 4",
        "2026.03.14 09:41:07 +0300 Counts: 12, ~cps: 2.000, Time: 6.00 s, Coord: 55°45'00.000\" N 37°37'00.000\" E at 2026.03.14 09:41:05 +0300, Captured from USB: 00000000",
        "1773470467000",
        "1773470465000",
        "55.750000000000000",
        "37.616666666666667",
        "TEST-DET",
        "USB: 00000000",
        "6.000000",
        "8",
        "2",
        "2.50000000000",
        "1.75000000000",
        "3.20000000000e-06",
        "0\t1\t2\t3\t0\t0\t3\t3"
    ]
    path = _write_asspg(tmp_path, lines=lines)
    with pytest.raises(ValueError):
        load_asspg(path)


def test_rejects_truncated_file(tmp_path):
    lines = [
        "FORMAT: 3",
        "2026.03.14 09:41:07 +0300 Counts: 12, ~cps: 2.000, Time: 6.00 s, Coord: 55°45'00.000\" N 37°37'00.000\" E at 2026.03.14 09:41:05 +0300, Captured from USB: 00000000",
        "1773470467000",
        "1773470465000",
        "55.750000000000000",
        "37.616666666666667",
        "TEST-DET",
        "USB: 00000000",
        "6.000000",
        "8",
        "2",
        "2.50000000000",
        "1.75000000000",
        "3.20000000000e-06",
        "0\t1\t2"
    ]
    path = _write_asspg(tmp_path, lines=lines)
    with pytest.raises(ValueError):
        load_asspg(path)


def test_rejects_file_without_slices(tmp_path):
    lines = [
        "FORMAT: 3",
        "2026.03.14 09:41:07 +0300 Counts: 12, ~cps: 2.000, Time: 6.00 s, Coord: 55°45'00.000\" N 37°37'00.000\" E at 2026.03.14 09:41:05 +0300, Captured from USB: 00000000",
        "1773470467000",
        "1773470465000",
        "55.750000000000000",
        "37.616666666666667",
        "TEST-DET",
        "USB: 00000000",
        "6.000000",
        "8",
        "2",
        "2.50000000000",
        "1.75000000000",
        "3.20000000000e-06",
        "0\t1\t2\t3\t0\t0\t3\t3"
    ]
    path = _write_asspg(tmp_path, lines=lines)
    with pytest.raises(ValueError):
        load_asspg(path)


def test_trailing_tab_in_counts_row_is_ignored(tmp_path):
    lines = [
        "FORMAT: 3",
        "2026.03.14 09:41:07 +0300 Counts: 12, ~cps: 2.000, Time: 6.00 s, Coord: 55°45'00.000\" N 37°37'00.000\" E at 2026.03.14 09:41:05 +0300, Captured from USB: 00000000",
        "1773470467000",
        "1773470465000",
        "55.750000000000000",
        "37.616666666666667",
        "TEST-DET",
        "USB: 00000000",
        "6.000000",
        "8",
        "2",
        "2.50000000000",
        "1.75000000000",
        "3.20000000000e-06",
        *[str(v) for v in _BASELINE],
        "1773470472000",
        "55.750100000000000",
        "37.616700000000000",
        "5.000000",
        "0\t2\t0\t1\t0\t0\t4\t0\t"
    ]
    path = _write_asspg(tmp_path, lines=lines)
    sg = load_asspg(path)
    assert sg.counts[0].tolist() == [0, 2, 0, 1, 0, 0, 4, 0]


def test_dispatcher_routes_txt_by_signature(tmp_path):
    """Задача #DATA-7: .txt само по себе формат не задаёт — диспетчер смотрит в содержимое."""
    from awf.ui.main_window import load_spectrogram
    sg = load_spectrogram(_write_asspg(tmp_path))
    assert sg.counts.shape == (2, 8)

    # .txt без сигнатуры FORMAT: уходит прежним маршрутом (N42/XML) и падает там, а не у нас.
    alien = tmp_path / "notes.txt"
    alien.write_text("просто текстовый файл\n", encoding="utf-8")
    with pytest.raises(Exception) as exc:
        load_spectrogram(str(alien))
    assert "AtomSpectra" not in str(exc.value)
