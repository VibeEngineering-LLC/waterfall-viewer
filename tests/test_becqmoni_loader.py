from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from awf.io.becqmoni_loader import load_becqmoni


def _write_xml(tmp_path, body: str, name: str = "sample.xml") -> str:
    """body — content внутри <ResultDataFile>...</ResultDataFile>. Возвращает str(path)."""
    xml = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<ResultDataFile>\n"
        + body
        + "\n</ResultDataFile>\n"
    )
    path = Path(tmp_path) / name
    path.write_text(xml, encoding="utf-8")
    return str(path)


def test_load_becqmoni_datapoints_full(tmp_path):
    """Полный кейс: StartTime, MeasurementTime, EnergyCalibration, DataPoint-отсчёты."""
    body = (
        "<ResultDataList><ResultData>"
        "<StartTime>2020-01-01T12:00:00</StartTime>"
        "<EnergySpectrum>"
        "<NumberOfChannels>4</NumberOfChannels>"
        "<EnergyCalibration>"
        "<Coefficients>"
        "<Coefficient>0.5</Coefficient>"
        "<Coefficient>1.5</Coefficient>"
        "</Coefficients>"
        "</EnergyCalibration>"
        "<MeasurementTime>3600</MeasurementTime>"
        "<Spectrum>"
        "<DataPoint>10</DataPoint><DataPoint>20</DataPoint>"
        "<DataPoint>30</DataPoint><DataPoint>40</DataPoint>"
        "</Spectrum>"
        "</EnergySpectrum>"
        "</ResultData></ResultDataList>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    assert spg.n_slices == 1
    assert spg.n_channels == 4
    np.testing.assert_array_equal(spg.counts[0], np.array([10, 20, 30, 40], dtype=np.uint16))
    np.testing.assert_allclose(spg.calibration.coeffs, [0.5, 1.5])
    assert spg.real_time_s[0] == pytest.approx(3600.0)
    assert spg.live_time_s[0] == pytest.approx(3600.0)
    assert spg.t0_iso == "2020-01-01T12:00:00"
    assert spg.source_path == path


def test_load_becqmoni_channel_elements_with_n_attr(tmp_path):
    """Отсчёты как <Channel n="i">, порядок атрибута."""
    body = (
        "<EnergySpectrum><Channels>"
        "<Channel n=\"2\">30</Channel>"
        "<Channel n=\"0\">10</Channel>"
        "<Channel n=\"1\">20</Channel>"
        "</Channels></EnergySpectrum>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    np.testing.assert_array_equal(spg.counts[0], np.array([10, 20, 30], dtype=np.uint16))


def test_load_becqmoni_spectrum_text_whitespace(tmp_path):
    """Отсчёты в text-ноде <Spectrum>, whitespace-разделённые."""
    body = (
        "<EnergySpectrum>"
        "<Spectrum>1 2 3 4 5</Spectrum>"
        "</EnergySpectrum>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    np.testing.assert_array_equal(spg.counts[0], np.array([1, 2, 3, 4, 5], dtype=np.uint16))


def test_load_becqmoni_no_calibration_fallback(tmp_path):
    """Нет <EnergyCalibration> → identity [0.0, 1.0]."""
    body = (
        "<EnergySpectrum>"
        "<Spectrum><DataPoint>1</DataPoint><DataPoint>2</DataPoint></Spectrum>"
        "</EnergySpectrum>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    np.testing.assert_allclose(spg.calibration.coeffs, [0.0, 1.0])


def test_load_becqmoni_live_and_real_time(tmp_path):
    """Есть и <LiveTime> и <RealTime> — оба заполнены различно."""
    body = (
        "<EnergySpectrum>"
        "<LiveTime>1798.5</LiveTime>"
        "<RealTime>1800.0</RealTime>"
        "<Spectrum><DataPoint>1</DataPoint></Spectrum>"
        "</EnergySpectrum>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    assert spg.live_time_s[0] == pytest.approx(1798.5)
    assert spg.real_time_s[0] == pytest.approx(1800.0)


def test_load_becqmoni_missing_timings_nan(tmp_path):
    """Нет никаких тайминг-секций → real/live = NaN."""
    body = (
        "<EnergySpectrum>"
        "<Spectrum><DataPoint>1</DataPoint><DataPoint>2</DataPoint></Spectrum>"
        "</EnergySpectrum>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    assert np.isnan(spg.real_time_s[0])
    assert np.isnan(spg.live_time_s[0])


def test_load_becqmoni_start_time_invalid_none(tmp_path):
    """Битая <StartTime> → t0_iso is None."""
    body = (
        "<StartTime>garbage-not-a-date</StartTime>"
        "<EnergySpectrum>"
        "<Spectrum><DataPoint>1</DataPoint></Spectrum>"
        "</EnergySpectrum>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    assert spg.t0_iso is None


def test_load_becqmoni_number_of_channels_pads(tmp_path):
    """NumberOfChannels больше фактического количества → padding нулями справа."""
    body = (
        "<EnergySpectrum>"
        "<NumberOfChannels>5</NumberOfChannels>"
        "<Spectrum><DataPoint>1</DataPoint><DataPoint>2</DataPoint><DataPoint>3</DataPoint></Spectrum>"
        "</EnergySpectrum>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    assert spg.n_channels == 5
    np.testing.assert_array_equal(spg.counts[0], np.array([1, 2, 3, 0, 0], dtype=np.uint16))


def test_load_becqmoni_not_xml_raises(tmp_path):
    """Не-XML файл → ValueError."""
    path = Path(tmp_path) / "bad.xml"
    path.write_text("this is not xml at all", encoding="utf-8")
    with pytest.raises(ValueError):
        load_becqmoni(str(path))


def test_load_becqmoni_int32_when_gt_uint16(tmp_path):
    """Значение > 65535 → dtype int32."""
    body = (
        "<EnergySpectrum>"
        "<Spectrum><DataPoint>1</DataPoint><DataPoint>70000</DataPoint><DataPoint>3</DataPoint></Spectrum>"
        "</EnergySpectrum>"
    )
    path = _write_xml(tmp_path, body)
    spg = load_becqmoni(path)
    assert spg.counts.dtype == np.int32
    assert int(spg.counts[0, 1]) == 70000
