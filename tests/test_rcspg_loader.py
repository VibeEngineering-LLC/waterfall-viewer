from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from awf.io.rcspg_loader import load_rcspg, _epoch_ms_to_iso

def _write_rcspg(tmp_path, doc) -> str:
    path = tmp_path / "synthetic.rcspg"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return str(path)

def _doc():
    return {
      "channelCount": 8,
      "coefficients": [1.0, 2.0, 0.0],          # E(ch)=1+2*ch
      "startTimeTimestamp": 1700000000000,      # ms -> 2023-11-14T22:13:20Z UTC
      "deviceModel": "RadiaCode-110",
      "spectrums": [
        {"pulses": [0, 1, 2, 3],                 "collectTime": 2, "timestamp": 1700000000000},
        {"pulses": [5, 0, 0, 0, 0, 0, 0, 4],     "collectTime": 3, "timestamp": 1700000005000},
        {"pulses": [1, 1],                        "collectTime": 1, "timestamp": 1700000011000},
      ],
    }

def test_basic_shape_and_dtype(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path, _doc()))
    assert sg.n_slices == 3
    assert sg.n_channels == 8
    assert sg.counts.dtype == np.uint16

def test_counts_padding(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path, _doc()))
    np.testing.assert_array_equal(sg.counts[0], [0,1,2,3,0,0,0,0])
    np.testing.assert_array_equal(sg.counts[1], [5,0,0,0,0,0,0,4])
    np.testing.assert_array_equal(sg.counts[2], [1,1,0,0,0,0,0,0])

def test_calibration(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path, _doc()))
    np.testing.assert_allclose(sg.calibration.coeffs, [1.0, 2.0, 0.0])
    en = sg.energies()
    assert en[0] == pytest.approx(1.0)
    assert en[1] == pytest.approx(3.0)

def test_time_axes(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path, _doc()))
    np.testing.assert_allclose(sg.time_offsets_s, [0.0, 5.0, 11.0])
    np.testing.assert_allclose(sg.real_time_s, [2.0, 3.0, 1.0])
    np.testing.assert_allclose(sg.live_time_s, [2.0, 3.0, 1.0])

def test_t0_iso(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path, _doc()))
    assert sg.t0_iso == "2023-11-14T22:13:20Z"
    assert _epoch_ms_to_iso(None) is None
    assert _epoch_ms_to_iso(1700000000000) == "2023-11-14T22:13:20Z"

def test_max_slices(tmp_path):
    sg = load_rcspg(_write_rcspg(tmp_path, _doc()), max_slices=2)
    assert sg.n_slices == 2
    assert sg.n_channels == 8

def test_empty_spectrums_raises(tmp_path):
    doc = _doc()
    doc["spectrums"] = []
    with pytest.raises(ValueError):
        load_rcspg(_write_rcspg(tmp_path, doc))

def test_channelcount_inferred(tmp_path):
    doc = _doc()
    del doc["channelCount"]
    sg = load_rcspg(_write_rcspg(tmp_path, doc))
    assert sg.n_channels == 8
