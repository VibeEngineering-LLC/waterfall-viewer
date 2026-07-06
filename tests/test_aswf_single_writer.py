"""Задача #216 — round-trip тесты single-file ASWF v3 writer."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from awf.io.aswf_single_writer import write_aswf, WF_CHANNELS
from awf.io.aswf_loader import load_aswf
from awf.model.spectrogram import Spectrogram, Calibration


def _make_sg(n_rows: int = 5, *, with_baseline: bool = False) -> Spectrogram:
    rng = np.random.default_rng(42)
    counts = rng.integers(0, 100, size=(n_rows, WF_CHANNELS), dtype=np.uint16)
    real_time = np.full(n_rows, 2.0, dtype=np.float64)
    live_time = np.full(n_rows, 1.9, dtype=np.float64)
    t_off = np.cumsum(real_time) - real_time[0]
    calib = Calibration(coeffs=np.array([1.5, 0.3, 1e-5], dtype=np.float64))
    baseline = None
    if with_baseline:
        baseline = rng.integers(0, 1000, size=WF_CHANNELS, dtype=np.uint32)
    return Spectrogram(
        counts=counts, calibration=calib,
        time_offsets_s=t_off, real_time_s=real_time, live_time_s=live_time,
        baseline=baseline,
    )


def test_roundtrip_basic(tmp_path: Path) -> None:
    sg = _make_sg(n_rows=7)
    out = tmp_path / "out.aswf"
    write_aswf(out, sg)
    sg2 = load_aswf(out)
    assert sg2.counts.shape == sg.counts.shape
    assert np.array_equal(sg2.counts, sg.counts.astype(sg2.counts.dtype))
    np.testing.assert_allclose(sg2.calibration.coeffs[:3], sg.calibration.coeffs[:3], rtol=0, atol=1e-12)


def test_roundtrip_with_baseline(tmp_path: Path) -> None:
    sg = _make_sg(n_rows=3, with_baseline=True)
    out = tmp_path / "b.aswf"
    write_aswf(out, sg)
    sg2 = load_aswf(out)
    assert sg2.baseline is not None
    assert np.array_equal(np.asarray(sg2.baseline, dtype=np.uint32),
                          np.asarray(sg.baseline, dtype=np.uint32))


def test_magic_header(tmp_path: Path) -> None:
    sg = _make_sg(n_rows=1)
    out = tmp_path / "m.aswf"
    write_aswf(out, sg, serial="XYZ", note="test note")
    with open(out, "rb") as f:
        assert f.read(4) == b"ASWF"
        (hdr_len,) = struct.unpack("<I", f.read(4))
        assert hdr_len == 4096


def test_bad_channels_raises() -> None:
    counts = np.zeros((2, 100), dtype=np.uint16)
    calib = Calibration(coeffs=np.array([0.0, 0.3], dtype=np.float64))
    sg = Spectrogram(counts=counts, calibration=calib,
                     time_offsets_s=np.zeros(2), real_time_s=np.ones(2),
                     live_time_s=np.ones(2))
    with pytest.raises(ValueError, match="только"):
        write_aswf("nope.aswf", sg)
