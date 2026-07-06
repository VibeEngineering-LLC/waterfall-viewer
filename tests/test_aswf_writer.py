from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from awf.io.aswf_writer import (
    AswfSegmentedWriter,
    WaterfallRow,
    WF_CHANNELS,
    WF_ROW_STRIDE,
    _JSON_HEADER_LEN,
    _ASWF_MAGIC,
)
from awf.io.aswf_loader import load_aswf


def _row(counts_dict: dict[int, int], dur_sec: int) -> WaterfallRow:
    """WaterfallRow с counts=uint16(WF_CHANNELS,) — задать точечно ключами counts_dict."""
    counts = np.zeros(WF_CHANNELS, dtype=np.uint16)
    for ch, v in counts_dict.items():
        counts[ch] = v
    return WaterfallRow(counts=counts, dur_sec=dur_sec)


def _baseline_zero() -> np.ndarray:
    return np.zeros(WF_CHANNELS, dtype=np.uint32)


def _read_header(path: Path) -> dict:
    """Читает JSON-заголовок из .aswf файла (без loader-а)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == _ASWF_MAGIC
        (hlen,) = struct.unpack("<I", f.read(4))
        raw = f.read(hlen)
    return json.loads(raw.split(b"\x00")[0].decode("utf-8"))


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
    def __call__(self) -> float:
        return self.t
    def advance(self, dt: float) -> None:
        self.t += dt


def test_single_row_roundtrip_via_loader(tmp_path):
    """Один сегмент, 3 строки, close() → loader вернул те же counts и duration."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=5.0,
                            baseline=_baseline_zero(), calibration=[0.5, 1.5, 0.0])
    w.write_row(_row({100: 10}, dur_sec=5))
    w.write_row(_row({100: 20, 200: 3}, dur_sec=5))
    w.write_row(_row({500: 42}, dur_sec=7))
    path = w.close()
    
    assert path is not None
    assert path.exists()
    assert path.name == "seg_00000.aswf"
    
    spec = load_aswf(path)
    assert spec.counts.shape == (3, WF_CHANNELS)
    assert spec.counts[0, 100] == 10
    assert spec.counts[1, 100] == 20 and spec.counts[1, 200] == 3
    assert spec.counts[2, 500] == 42
    assert spec.real_time_s.tolist() == [5.0, 5.0, 7.0]
    assert list(spec.calibration.coeffs[:3]) == [0.5, 1.5, 0.0]


def test_rollover_by_max_rows(tmp_path):
    """max_rows=3, пишем 5 строк → 2 сегмента (первый 3, второй 2)."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0,
                            baseline=_baseline_zero(), max_rows=3, max_age_sec=10_000)
    for i in range(5):
        w.write_row(_row({100: i + 1}, dur_sec=1))
    final = w.close()
    
    assert (tmp_path / "seg_00000.aswf").exists()
    assert (tmp_path / "seg_00001.aswf").exists()
    
    s0 = load_aswf(tmp_path / "seg_00000.aswf")
    assert s0.counts.shape == (3, WF_CHANNELS)
    assert s0.counts[:, 100].tolist() == [1, 2, 3]
    
    s1 = load_aswf(tmp_path / "seg_00001.aswf")
    assert s1.counts.shape == (2, WF_CHANNELS)
    assert s1.counts[:, 100].tolist() == [4, 5]
    
    assert final == tmp_path / "seg_00001.aswf"


def test_rollover_by_age(tmp_path):
    """FakeClock. max_age_sec=10, пишем 2 строки при t=1000, продвинуть t на +11, tick() → сегмент закрыт."""
    clk = _FakeClock(1000.0)
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0,
                            baseline=_baseline_zero(), max_rows=1_000, max_age_sec=10, clock=clk)
    w.write_row(_row({0: 1}, dur_sec=1))
    w.write_row(_row({0: 2}, dur_sec=1))
    clk.advance(11.0)
    result = w.tick()
    
    assert result == tmp_path / "seg_00000.aswf"
    assert result.exists()
    
    s = load_aswf(result)
    assert s.counts.shape == (2, WF_CHANNELS)
    assert s.counts[:, 0].tolist() == [1, 2]


def test_baseline_roundtrip(tmp_path):
    """Baseline с уникальными значениями сохранён и прочитан."""
    bl = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bl[10] = 111
    bl[500] = 2_000_000
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0, baseline=bl)
    w.write_row(_row({0: 1}, dur_sec=1))
    path = w.close()
    
    s = load_aswf(path)
    assert s.baseline is not None
    assert int(s.baseline[10]) == 111
    assert int(s.baseline[500]) == 2_000_000
    assert int(s.baseline[0]) == 0


def test_close_on_empty_writer_returns_none(tmp_path):
    """Открытый writer без единого write_row → close() вернул None, никаких файлов не осталось."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0, baseline=_baseline_zero())
    assert w.close() is None
    assert list(tmp_path.iterdir()) == []


def test_close_is_idempotent(tmp_path):
    """Повторный close() → None; write_row после close → RuntimeError."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0, baseline=_baseline_zero())
    w.write_row(_row({0: 1}, dur_sec=1))
    first = w.close()
    assert first is not None
    assert w.close() is None
    with pytest.raises(RuntimeError):
        w.write_row(_row({0: 2}, dur_sec=1))


def test_write_row_wrong_shape_raises(tmp_path):
    """counts.shape != (WF_CHANNELS,) → ValueError."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0, baseline=_baseline_zero())
    bad = WaterfallRow(counts=np.zeros(100, dtype=np.uint16), dur_sec=1)
    with pytest.raises(ValueError):
        w.write_row(bad)


def test_write_row_dur_out_of_range_raises(tmp_path):
    """dur_sec < 0 или > 65535 → ValueError. Два вызова в одном тесте — оба должны бросить."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0, baseline=_baseline_zero())
    with pytest.raises(ValueError):
        w.write_row(WaterfallRow(counts=np.zeros(WF_CHANNELS, dtype=np.uint16), dur_sec=-1))
    with pytest.raises(ValueError):
        w.write_row(WaterfallRow(counts=np.zeros(WF_CHANNELS, dtype=np.uint16), dur_sec=65_536))


def test_baseline_wrong_shape_raises_in_init(tmp_path):
    """Конструктор с baseline неправильной длины → ValueError."""
    bad_bl = np.zeros(100, dtype=np.uint32)
    with pytest.raises(ValueError):
        AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0, baseline=bad_bl)


def test_header_fields_present(tmp_path):
    """После записи одного сегмента прочитать header напрямую (через _read_header) и проверить обязательные и опциональные ключи."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_042.5, interval_sec=3.0,
                            baseline=_baseline_zero(), calibration=[1.0, 2.0, 3.0],
                            serial="AS-42", note="unit-test")
    w.write_row(_row({0: 1}, dur_sec=3))
    path = w.close()
    hdr = _read_header(path)
    
    assert hdr["version"] == 3
    assert hdr["channels"] == WF_CHANNELS
    assert hdr["interval_sec"] == 3.0
    assert hdr["started_at"] == 1_700_000_042.5
    assert hdr["saved_rows"] == 0
    assert hdr["calibration"] == [1.0, 2.0, 3.0]
    assert hdr["row_stride"] == WF_ROW_STRIDE
    assert hdr["seg_index"] == 0
    assert hdr["serial"] == "AS-42"
    assert hdr["note"] == "unit-test"
    assert "baseline" in hdr and hdr["baseline"]["count"] == WF_CHANNELS
    assert hdr["baseline"]["offset"] == 8 + _JSON_HEADER_LEN
    
    fnames = {fd["name"] for fd in hdr["row_fields"]}
    assert fnames == {"spectrum", "duration"}


def test_seg_index_increments_across_rollovers(tmp_path):
    """max_rows=2, пишем 5 строк → сегменты 0, 1, 2 (2, 2, 1). Свойство seg_index в разные моменты."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0,
                            baseline=_baseline_zero(), max_rows=2, max_age_sec=10_000)
    assert w.seg_index == 0
    w.write_row(_row({0: 1}, dur_sec=1))
    w.write_row(_row({0: 2}, dur_sec=1))  # rollover → seg_index=1
    assert w.seg_index == 1
    w.write_row(_row({0: 3}, dur_sec=1))
    w.write_row(_row({0: 4}, dur_sec=1))  # rollover → seg_index=2
    assert w.seg_index == 2
    w.write_row(_row({0: 5}, dur_sec=1))
    last = w.close()
    assert w.seg_index == 3
    assert last == tmp_path / "seg_00002.aswf"
    
    files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".aswf")
    assert files == ["seg_00000.aswf", "seg_00001.aswf", "seg_00002.aswf"]


def test_serial_and_note_absent_when_none(tmp_path):
    """Если serial=None и note=None (по умолчанию) — ключей нет в header."""
    w = AswfSegmentedWriter(tmp_path, started_at=1_700_000_000.0, interval_sec=1.0, baseline=_baseline_zero())
    w.write_row(_row({0: 1}, dur_sec=1))
    path = w.close()
    hdr = _read_header(path)
    assert "serial" not in hdr
    assert "note" not in hdr
