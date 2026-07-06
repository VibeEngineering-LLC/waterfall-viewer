from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from awf.io.spe_loader import load_spe


def _write_spe(tmp_path, sections: list[tuple[str, list[str]]], name: str = "sample.spe") -> str:
    """sections — упорядоченный список (KEY, body_lines). Возвращает str(path).
    KEY без ведущего $ и хвостового :, body_lines — уже как строки текста."""
    lines: list[str] = []
    for key, body in sections:
        lines.append(f"${key}:")
        lines.extend(body)
    path = Path(tmp_path) / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_load_spe_minimal_data_only(tmp_path):
    """Только секция $DATA: — калибровка identity, real/live NaN, t0_iso None."""
    path = _write_spe(tmp_path, [("DATA", ["0 3", "10", "20", "30", "40"])])
    spg = load_spe(path)
    assert spg.n_slices == 1
    assert spg.n_channels == 4
    np.testing.assert_array_equal(spg.counts[0], np.array([10, 20, 30, 40], dtype=np.uint16))
    np.testing.assert_allclose(spg.calibration.coeffs, [0.0, 1.0])
    assert np.isnan(spg.real_time_s[0])
    assert np.isnan(spg.live_time_s[0])
    assert spg.time_offsets_s[0] == 0.0
    assert spg.t0_iso is None
    assert spg.source_path == path


def test_load_spe_ener_fit_only(tmp_path):
    """$ENER_FIT: — линейная калибровка."""
    path = _write_spe(tmp_path, [
        ("DATA", ["0 2", "1", "2", "3"]),
        ("ENER_FIT", ["1.5 0.25"]),
    ])
    spg = load_spe(path)
    np.testing.assert_allclose(spg.calibration.coeffs, [1.5, 0.25])


def test_load_spe_mca_cal_priority(tmp_path):
    """$MCA_CAL: приоритетнее $ENER_FIT: (даже если обе есть)."""
    path = _write_spe(tmp_path, [
        ("DATA", ["0 2", "1", "2", "3"]),
        ("ENER_FIT", ["9.9 9.9"]),
        ("MCA_CAL", ["3", "1.0 0.5 0.001 keV"]),
    ])
    spg = load_spe(path)
    np.testing.assert_allclose(spg.calibration.coeffs, [1.0, 0.5, 0.001])


def test_load_spe_mca_cal_without_units(tmp_path):
    """Тело $MCA_CAL: без хвостовой единицы (keV)."""
    path = _write_spe(tmp_path, [
        ("DATA", ["0 1", "5", "6"]),
        ("MCA_CAL", ["2", "0.3 0.7"]),
    ])
    spg = load_spe(path)
    np.testing.assert_allclose(spg.calibration.coeffs, [0.3, 0.7])


def test_load_spe_meas_tim_live_real(tmp_path):
    """$MEAS_TIM: <live> <real> — порядок Ortec (первый = live)."""
    path = _write_spe(tmp_path, [
        ("DATA", ["0 1", "1", "2"]),
        ("MEAS_TIM", ["1798.5 1800.0"]),
    ])
    spg = load_spe(path)
    assert spg.live_time_s[0] == pytest.approx(1798.5)
    assert spg.real_time_s[0] == pytest.approx(1800.0)


def test_load_spe_date_ortec(tmp_path):
    """Ortec-формат даты — конвертируется в ISO-8601 без tz."""
    path = _write_spe(tmp_path, [
        ("DATE_MEA", ["09/24/2018 14:05:37"]),
        ("DATA", ["0 1", "1", "2"]),
    ])
    spg = load_spe(path)
    assert spg.t0_iso == "2018-09-24T14:05:37"


def test_load_spe_date_iso(tmp_path):
    """ISO-8601 дата — сохраняется."""
    path = _write_spe(tmp_path, [
        ("DATE_MEA", ["2024-01-15T12:00:00"]),
        ("DATA", ["0 1", "1", "2"]),
    ])
    spg = load_spe(path)
    assert spg.t0_iso == "2024-01-15T12:00:00"


def test_load_spe_date_invalid_none(tmp_path):
    """Нераспарсимая дата — t0_iso is None."""
    path = _write_spe(tmp_path, [
        ("DATE_MEA", ["garbage-not-a-date"]),
        ("DATA", ["0 1", "1", "2"]),
    ])
    spg = load_spe(path)
    assert spg.t0_iso is None


def test_load_spe_start_nonzero_pads_left(tmp_path):
    """start > 0 → каналы 0..start-1 дополняются нулями; n_channels = end + 1."""
    path = _write_spe(tmp_path, [
        ("DATA", ["3 5", "7", "8", "9"]),
    ])
    spg = load_spe(path)
    assert spg.n_channels == 6
    np.testing.assert_array_equal(spg.counts[0], np.array([0, 0, 0, 7, 8, 9], dtype=np.uint16))


def test_load_spe_missing_data_raises(tmp_path):
    """Нет $DATA: → ValueError."""
    path = _write_spe(tmp_path, [
        ("SPEC_ID", ["only-id"]),
        ("MEAS_TIM", ["10 10"]),
    ])
    with pytest.raises(ValueError):
        load_spe(path)


def test_load_spe_no_calibration_fallback(tmp_path):
    """Нет ни $MCA_CAL:, ни $ENER_FIT: → identity [0.0, 1.0]."""
    path = _write_spe(tmp_path, [
        ("DATA", ["0 3", "0", "0", "0", "0"]),
    ])
    spg = load_spe(path)
    np.testing.assert_allclose(spg.calibration.coeffs, [0.0, 1.0])


def test_load_spe_int32_when_gt_uint16(tmp_path):
    """Есть значение > 65535 → dtype int32."""
    path = _write_spe(tmp_path, [
        ("DATA", ["0 2", "1", "70000", "2"]),
    ])
    spg = load_spe(path)
    assert spg.counts.dtype == np.int32
    assert int(spg.counts[0, 1]) == 70000
