from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from awf.io.csv_loader import load_csv


def _write_csv(tmp_path, lines: list[str], name: str = "sample.csv") -> str:
    """lines — уже готовые строки CSV (без trailing newline). Возвращает str(path)."""
    path = Path(tmp_path) / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_load_csv_channel_counts_comma(tmp_path):
    """Тест загрузки CSV с двумя колонками channel,counts."""
    path = _write_csv(tmp_path, ["0,10", "1,20", "2,30", "3,40"])
    spg = load_csv(path)
    assert spg.n_slices == 1
    assert spg.n_channels == 4
    np.testing.assert_array_equal(spg.counts[0], np.array([10, 20, 30, 40], dtype=np.uint16))
    np.testing.assert_allclose(spg.calibration.coeffs, [0.0, 1.0])
    assert np.isnan(spg.real_time_s[0])
    assert np.isnan(spg.live_time_s[0])
    assert spg.t0_iso is None
    assert spg.source_path == path


def test_load_csv_energy_counts_linear_fit(tmp_path):
    """Тест загрузки CSV с energy,counts и автоматической калибровке."""
    path = _write_csv(tmp_path, ["10.0,5", "15.0,7", "20.0,9", "25.0,11"])
    spg = load_csv(path)
    assert spg.n_channels == 4
    np.testing.assert_allclose(spg.calibration.coeffs, [10.0, 5.0])
    np.testing.assert_array_equal(spg.counts[0], np.array([5, 7, 9, 11], dtype=np.uint16))


def test_load_csv_semicolon_separator(tmp_path):
    """Тест загрузки CSV с разделителем ;."""
    path = _write_csv(tmp_path, ["0;100", "1;200", "2;300"])
    spg = load_csv(path)
    np.testing.assert_array_equal(spg.counts[0], np.array([100, 200, 300], dtype=np.uint16))
    np.testing.assert_allclose(spg.calibration.coeffs, [0.0, 1.0])


def test_load_csv_tab_separator(tmp_path):
    """Тест загрузки CSV с разделителем \\t."""
    path = _write_csv(tmp_path, ["0\t1", "1\t2", "2\t3", "3\t4"])
    spg = load_csv(path)
    np.testing.assert_array_equal(spg.counts[0], np.array([1, 2, 3, 4], dtype=np.uint16))


def test_load_csv_header_skipped(tmp_path):
    """Тест пропуска заголовков."""
    path = _write_csv(tmp_path, ["Channel,Counts", "0,11", "1,22", "2,33"])
    spg = load_csv(path)
    assert spg.n_channels == 3
    np.testing.assert_array_equal(spg.counts[0], np.array([11, 22, 33], dtype=np.uint16))


def test_load_csv_comments_and_blanks_skipped(tmp_path):
    """Тест пропуска комментариев и пустых строк."""
    path = _write_csv(tmp_path, [
        "# spectrum export",
        "",
        "0,1",
        "# skip me",
        "1,2",
        "",
        "2,3",
    ])
    spg = load_csv(path)
    assert spg.n_channels == 3
    np.testing.assert_array_equal(spg.counts[0], np.array([1, 2, 3], dtype=np.uint16))


def test_load_csv_empty_raises(tmp_path):
    """Тест исключения при пустом файле."""
    path = _write_csv(tmp_path, ["# comment only", "  ", "# another"])
    with pytest.raises(ValueError):
        load_csv(path)


def test_load_csv_int32_when_gt_uint16(tmp_path):
    """Тест использования int32 при значениях > 65535."""
    path = _write_csv(tmp_path, ["0,1", "1,70000", "2,3"])
    spg = load_csv(path)
    assert spg.counts.dtype == np.int32
    assert int(spg.counts[0, 1]) == 70000


def test_load_csv_negative_clipped_to_zero(tmp_path):
    """Тест клиппирования отрицательных значений."""
    path = _write_csv(tmp_path, ["0,5", "1,-3", "2,7"])
    spg = load_csv(path)
    np.testing.assert_array_equal(spg.counts[0], np.array([5, 0, 7], dtype=np.uint16))


def test_load_csv_energy_single_channel_fallback_a1(tmp_path):
    """Тест fallback калибровки при одном канале и энергии."""
    path = _write_csv(tmp_path, ["100.5,42"])
    spg = load_csv(path)
    assert spg.n_channels == 1
    np.testing.assert_allclose(spg.calibration.coeffs, [100.5, 1.0])
    np.testing.assert_array_equal(spg.counts[0], np.array([42], dtype=np.uint16))
