"""Задача #226 — тесты экспорта одномерного спектра.

Форматы (переделка #217): BecqMoni XML (.xml), LSRM бинарный SPE (.spe),
InterSpec/ANSI N42 (.n42). Round-trip через loader'ы там где есть.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from awf.io.becqmoni_writer import write_becqmoni_xml
from awf.io.becqmoni_loader import load_becqmoni
from awf.io.spe_writer import write_spe
from awf.io.n42_writer import write_n42


def _make_spectrum(n: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 500, size=n, dtype=np.int64)


def test_becqmoni_xml_round_trip(tmp_path: Path) -> None:
    spec = _make_spectrum(8)
    out = tmp_path / "out.xml"
    write_becqmoni_xml(out, spec, live_time_s=120.0, real_time_s=125.0,
                       calibration=[0.0, 0.3, 1e-5], sample_name="unit-test")
    sg = load_becqmoni(out)
    assert sg.counts.shape == (1, 8)
    assert list(sg.counts[0]) == list(spec.astype(int))
    assert abs(float(sg.live_time_s[0]) - 120.0) < 1e-4
    assert abs(float(sg.real_time_s[0]) - 125.0) < 1e-3
    coeffs = list(sg.calibration.coeffs)
    assert len(coeffs) == 3
    assert abs(coeffs[1] - 0.3) < 1e-9


def test_becqmoni_xml_structure(tmp_path: Path) -> None:
    spec = _make_spectrum(4)
    out = tmp_path / "s.xml"
    write_becqmoni_xml(out, spec, live_time_s=1.0, real_time_s=2.0, calibration=[0.0, 1.0])
    root = ET.parse(out).getroot()
    assert root.tag == "ResultDataFile"
    assert root.find("FormatVersion").text == "120920"
    es = root.find(".//EnergySpectrum")
    assert es is not None
    po = es.find("EnergyCalibration/PolynomialOrder")
    assert po is not None and po.text == "1"
    coefs = es.findall("EnergyCalibration/Coefficients/Coefficient")
    assert [float(c.text) for c in coefs] == [0.0, 1.0]
    dps = es.findall("Spectrum/DataPoint")
    assert len(dps) == 4


def test_becqmoni_empty_calibration_identity(tmp_path: Path) -> None:
    spec = _make_spectrum(4)
    out = tmp_path / "e.xml"
    write_becqmoni_xml(out, spec, live_time_s=1.0, real_time_s=1.0, calibration=None)
    root = ET.parse(out).getroot()
    coefs = root.findall(".//EnergyCalibration/Coefficients/Coefficient")
    assert [float(c.text) for c in coefs] == [0.0, 1.0]


def test_spe_binary_lsrm(tmp_path: Path) -> None:
    spec = _make_spectrum(16)
    out = tmp_path / "out.spe"
    write_spe(out, spec, live_time_s=60.0, real_time_s=65.0,
              calibration=[1.5, 0.3, 1e-5], spec_id="unit-test")
    raw = out.read_bytes()
    marker = b"SPECTR="
    idx = raw.find(marker)
    assert idx > 0
    header = raw[:idx].decode("cp1251")
    assert "SHIFR=unit-test\r\n" in header
    assert "TLIVE=60.00\r\n" in header
    assert "TREAL=65.00\r\n" in header
    assert "SPECTRSIZE=16\r\n" in header
    assert "ENERGY=2,1.5,0.3," in header
    binary = raw[idx + len(marker):]
    assert len(binary) == 16 * 4
    vals = np.frombuffer(binary, dtype="<u4")
    assert list(vals) == list(spec.astype(int))


def test_n42_valid_xml_and_channeldata(tmp_path: Path) -> None:
    spec = _make_spectrum(32)
    out = tmp_path / "out.n42"
    write_n42(out, spec, live_time_s=10.0, real_time_s=11.0, calibration=[0.0, 0.3, 1e-5])
    ns = "{http://physics.nist.gov/N42/2011/N42}"
    root = ET.fromstring(out.read_text(encoding="utf-8"))
    assert root.tag == ns + "RadInstrumentData"
    cal = root.find(f".//{ns}EnergyCalibration/{ns}CoefficientValues")
    assert cal is not None and cal.text and "0.3" in cal.text
    ch = root.find(f".//{ns}Spectrum/{ns}ChannelData")
    assert ch is not None and ch.text
    vals = [int(x) for x in ch.text.split()]
    assert vals == list(spec.astype(int))


def test_writers_accept_float_counts(tmp_path: Path) -> None:
    spec = np.array([1.4, 2.6, 3.5, 0.0], dtype=np.float64)
    write_becqmoni_xml(tmp_path / "f.xml", spec, live_time_s=1.0, real_time_s=1.0,
                       calibration=[0.0, 1.0])
    write_spe(tmp_path / "f.spe", spec, live_time_s=1.0, real_time_s=1.0, calibration=[0.0, 1.0])
    write_n42(tmp_path / "f.n42", spec, live_time_s=1.0, real_time_s=1.0, calibration=[0.0, 1.0])
    sg = load_becqmoni(tmp_path / "f.xml")
    assert list(sg.counts[0]) == [1, 3, 4, 0]


def test_reject_2d(tmp_path: Path) -> None:
    arr = np.zeros((2, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="1D"):
        write_becqmoni_xml(tmp_path / "x.xml", arr, live_time_s=1.0, real_time_s=1.0)
    with pytest.raises(ValueError, match="1D"):
        write_spe(tmp_path / "x.spe", arr, live_time_s=1.0, real_time_s=1.0)
    with pytest.raises(ValueError, match="1D"):
        write_n42(tmp_path / "x.n42", arr, live_time_s=1.0, real_time_s=1.0)
