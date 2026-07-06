"""Задача #217 — smoke-тесты экспорта одномерного спектра в .tka/.spe/.n42."""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from awf.io.tka_writer import write_tka
from awf.io.spe_writer import write_spe
from awf.io.n42_writer import write_n42


def _make_spectrum(n: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 500, size=n, dtype=np.int64)


def test_tka_header_and_counts(tmp_path: Path) -> None:
    spec = _make_spectrum(8)
    out = tmp_path / "out.tka"
    write_tka(out, spec, live_time_s=120.0, real_time_s=125.0)
    lines = out.read_text(encoding="ascii").strip().split("\n")
    assert lines[0] == "120"
    assert lines[1] == "125"
    assert [int(x) for x in lines[2:]] == list(spec.astype(int))


def test_spe_sections(tmp_path: Path) -> None:
    spec = _make_spectrum(16)
    out = tmp_path / "out.spe"
    write_spe(out, spec, live_time_s=60.0, real_time_s=65.0,
              calibration=[1.5, 0.3, 1e-5], spec_id="unit-test")
    text = out.read_text(encoding="ascii")
    assert "$SPEC_ID:" in text and "unit-test" in text
    assert "$DATE_MEA:" in text
    assert "$MEAS_TIM:" in text and "60 65" in text
    assert "$DATA:" in text and "0 15" in text
    assert "$ENER_FIT:" in text and "1.500000 0.300000" in text
    assert "$MCA_CAL:" in text


def test_n42_valid_xml_and_channeldata(tmp_path: Path) -> None:
    spec = _make_spectrum(32)
    out = tmp_path / "out.n42"
    write_n42(out, spec, live_time_s=10.0, real_time_s=11.0,
              calibration=[0.0, 0.3, 1e-5])
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
    write_tka(tmp_path / "f.tka", spec, live_time_s=1.0, real_time_s=1.0)
    write_spe(tmp_path / "f.spe", spec, live_time_s=1.0, real_time_s=1.0,
              calibration=[0.0, 1.0])
    write_n42(tmp_path / "f.n42", spec, live_time_s=1.0, real_time_s=1.0,
              calibration=[0.0, 1.0])
    # округление до int
    tka_lines = (tmp_path / "f.tka").read_text().strip().split("\n")[2:]
    assert [int(x) for x in tka_lines] == [1, 3, 4, 0]


def test_reject_2d(tmp_path: Path) -> None:
    arr = np.zeros((2, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="1D"):
        write_tka(tmp_path / "x.tka", arr, live_time_s=1.0, real_time_s=1.0)
    with pytest.raises(ValueError, match="1D"):
        write_spe(tmp_path / "x.spe", arr, live_time_s=1.0, real_time_s=1.0)
    with pytest.raises(ValueError, match="1D"):
        write_n42(tmp_path / "x.n42", arr, live_time_s=1.0, real_time_s=1.0)
