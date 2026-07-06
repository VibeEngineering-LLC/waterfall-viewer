import sys
sys.stdout.reconfigure(encoding="utf-8")

import struct
import pytest
import numpy as np
from dataclasses import FrozenInstanceError

from awf.usb.assembler import (
    HistogramAssembler, SweepResult,
    CHANNELS, CMD_HISTOGRAM, CMD_STAT,
)

_CHUNK_BINS = 64
_N_CHUNKS = CHANNELS // _CHUNK_BINS

def _hist_chunk(offset: int, bins: list[int]) -> tuple[int, bytes]:
    payload = struct.pack("<H", offset) + struct.pack(f"<{len(bins)}I", *bins)
    return (CMD_HISTOGRAM, payload)

def _stat_packet(total_time_sec: int = 10, cpu_load: int = 5, cps: int = 42) -> tuple[int, bytes]:
    data = struct.pack("<I", total_time_sec) + struct.pack("<H", cpu_load) + struct.pack("<I", cps)
    return (CMD_STAT, data)

def _feed_full_sweep(assembler: HistogramAssembler, value: int = 1) -> SweepResult | None:
    result = None
    for i in range(_N_CHUNKS):
        offset = i * _CHUNK_BINS
        cmd, data = _hist_chunk(offset, [value] * _CHUNK_BINS)
        result = assembler.feed(cmd, data)
    return result

def test_no_result_before_full_sweep():
    assembler = HistogramAssembler()
    for i in range(_N_CHUNKS - 1):
        offset = i * _CHUNK_BINS
        cmd, data = _hist_chunk(offset, [1] * _CHUNK_BINS)
        assert assembler.feed(cmd, data) is None

def test_full_sweep_returns_sweep_result():
    assembler = HistogramAssembler()
    result = _feed_full_sweep(assembler)
    assert isinstance(result, SweepResult)
    assert result.bins.shape == (CHANNELS,)
    assert result.bins.sum() == CHANNELS
    assert result.total_counts == CHANNELS

def test_sweep_result_is_frozen_dataclass():
    assembler = HistogramAssembler()
    result = _feed_full_sweep(assembler)
    assert result is not None
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.cps = 999

def test_reset_clears_state():
    assembler = HistogramAssembler()
    for i in range(50):
        offset = i * _CHUNK_BINS
        cmd, data = _hist_chunk(offset, [1] * _CHUNK_BINS)
        assembler.feed(cmd, data)
    assembler.reset()
    result = _feed_full_sweep(assembler)
    assert result is not None
    assert result.bins.sum() == CHANNELS

def test_stat_updates_next_sweep_result():
    assembler = HistogramAssembler()
    result1 = _feed_full_sweep(assembler)
    assert result1 is not None
    assert result1.total_time_sec == 0
    assert result1.cps == 0
    cmd, data = _stat_packet(total_time_sec=10, cps=42)
    assert assembler.feed(cmd, data) is None
    result2 = _feed_full_sweep(assembler)
    assert result2.total_time_sec == 10
    assert result2.cps == 42

def test_gap_in_offsets_causes_drop():
    assembler = HistogramAssembler()
    cmd, data = _hist_chunk(0, [1] * _CHUNK_BINS)
    assert assembler.feed(cmd, data) is None
    cmd, data = _hist_chunk(128, [1] * _CHUNK_BINS)
    assert assembler.feed(cmd, data) is None
    assert assembler.drops == 1

def test_commits_counter():
    assembler = HistogramAssembler()
    _feed_full_sweep(assembler)
    _feed_full_sweep(assembler)
    assert assembler.commits == 2

def test_drops_counter_initial_zero():
    assembler = HistogramAssembler()
    assert assembler.drops == 0
    assert assembler.commits == 0

def test_bins_dtype_uint32_compatible():
    assembler = HistogramAssembler()
    result = _feed_full_sweep(assembler, value=65535)
    assert int(result.bins.max()) == 65535

def test_non_histogram_non_stat_cmd_ignored():
    assembler = HistogramAssembler()
    assert assembler.feed(0xFF, b"arbitrary data") is None
