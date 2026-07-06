from __future__ import annotations

import numpy as np
import pytest

from awf.usb.collector import (
    DeltaCollector,
    SpectrumSnapshot,
    WaterfallRow,
    WF_CHANNELS,
)


def _snap(bins: np.ndarray, total_time_sec: int) -> SpectrumSnapshot:
    """SpectrumSnapshot из bins с автоматическим total_counts=sum(bins)."""
    return SpectrumSnapshot(bins=bins, total_counts=int(bins.sum()), total_time_sec=total_time_sec)


def test_first_feed_yields_full_bins_as_delta():
    """Первый feed — reset (has_prev=False), delta должен равняться самим bins."""
    c = DeltaCollector()
    bins = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins[100] = 5
    bins[200] = 10
    snap = _snap(bins, total_time_sec=3)
    row = c.feed(snap)
    assert row.counts[100] == 5
    assert row.counts[200] == 10
    assert row.counts.dtype == np.uint16
    assert row.counts.shape == (WF_CHANNELS,)
    assert row.dur_sec == 3


def test_second_feed_yields_delta():
    """Второй feed даёт дельту по сравнению с предыдущим."""
    c = DeltaCollector()
    bins1 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins1[100] = 5
    snap1 = _snap(bins1, total_time_sec=3)
    c.feed(snap1)
    
    bins2 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins2[100] = 7
    snap2 = _snap(bins2, total_time_sec=8)
    row = c.feed(snap2)
    assert row.counts[100] == 2
    assert row.counts[200] == 0
    assert row.dur_sec == 5


def test_reset_when_total_counts_drop():
    """Сброс при уменьшении total_counts."""
    c = DeltaCollector()
    bins1 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins1[100] = 100
    snap1 = _snap(bins1, total_time_sec=10)
    c.feed(snap1)
    
    bins2 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins2[100] = 50
    snap2 = _snap(bins2, total_time_sec=15)
    row2 = c.feed(snap2)
    assert row2.counts[100] == 50
    assert row2.dur_sec == 15


def test_delta_clip_ceiling_at_65535():
    """Дельта клипуется вверх до 65535."""
    c = DeltaCollector()
    bins = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins[500] = 100_000
    snap = _snap(bins, total_time_sec=3)
    row = c.feed(snap)
    assert row.counts[500] == 65535
    assert row.counts.dtype == np.uint16


def test_delta_clip_floor_at_zero_no_reset():
    """Дельта клипуется вниз до 0, но reset не срабатывает."""
    c = DeltaCollector()
    bins1 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins1[100] = 100
    bins1[200] = 0
    snap1 = _snap(bins1, total_time_sec=5)
    c.feed(snap1)
    
    bins2 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins2[100] = 50
    bins2[200] = 200
    snap2 = _snap(bins2, total_time_sec=10)
    row = c.feed(snap2)
    assert row.counts[100] == 0
    assert row.counts[200] == 200
    assert row.dur_sec == 5


def test_time_delta_regress_yields_zero():
    """Отрицательная разница времени даёт dt=0."""
    c = DeltaCollector()
    bins1 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins1[100] = 10
    snap1 = _snap(bins1, total_time_sec=100)
    c.feed(snap1)
    
    bins2 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins2[100] = 15
    snap2 = _snap(bins2, total_time_sec=50)
    row = c.feed(snap2)
    assert row.dur_sec == 0
    assert row.counts[100] == 5


def test_dur_sec_clip_at_65535():
    """Длительность клипуется вверх до 65535."""
    c = DeltaCollector()
    bins1 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    snap1 = _snap(bins1, total_time_sec=0)
    c.feed(snap1)
    
    bins2 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins2[0] = 2
    snap2 = _snap(bins2, total_time_sec=100_000)
    row = c.feed(snap2)
    assert row.dur_sec == 65535


def test_reset_flag_first_dur_equals_cur_time():
    """Первый feed с total_time_sec=42 даёт dur_sec==42."""
    c = DeltaCollector()
    bins = np.zeros(WF_CHANNELS, dtype=np.uint32)
    snap = _snap(bins, total_time_sec=42)
    row = c.feed(snap)
    assert row.dur_sec == 42


def test_reset_method_resets_state():
    """Метод reset сбрасывает состояние."""
    c = DeltaCollector()
    bins1 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins1[0] = 100
    snap1 = _snap(bins1, total_time_sec=10)
    c.feed(snap1)
    
    c.reset()
    assert c.has_prev is False
    assert c.prev_total == 0
    assert c.prev_time == 0
    
    bins2 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins2[0] = 200
    snap2 = _snap(bins2, total_time_sec=15)
    row = c.feed(snap2)
    assert row.counts[0] == 200
    assert row.dur_sec == 15


def test_input_bins_not_mutated():
    """Входной массив bins не мутирует collector-ом."""
    c = DeltaCollector()
    bins = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins[100] = 5
    snap = _snap(bins, total_time_sec=1)
    original = bins.copy()
    c.feed(snap)
    
    bins[100] = 999
    bins2 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins2[100] = 10
    snap2 = _snap(bins2, total_time_sec=2)
    row = c.feed(snap2)
    assert row.counts[100] == 5


def test_has_prev_transitions():
    """has_prev меняется при feed и reset."""
    c = DeltaCollector()
    assert c.has_prev is False
    bins = np.zeros(WF_CHANNELS, dtype=np.uint32)
    snap = _snap(bins, total_time_sec=1)
    c.feed(snap)
    assert c.has_prev is True
    c.reset()
    assert c.has_prev is False


def test_prev_total_and_prev_time_updated():
    """prev_total и prev_time обновляются после feed."""
    c = DeltaCollector()
    bins1 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins1[0] = 42
    snap1 = _snap(bins1, total_time_sec=17)
    c.feed(snap1)
    assert c.prev_total == 42
    assert c.prev_time == 17
    
    bins2 = np.zeros(WF_CHANNELS, dtype=np.uint32)
    bins2[0] = 50
    snap2 = _snap(bins2, total_time_sec=25)
    c.feed(snap2)
    assert c.prev_total == 50
    assert c.prev_time == 25
    
    c.reset()
    assert c.prev_total == 0
    assert c.prev_time == 0
