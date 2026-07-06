"""Cumulative→delta преобразователь снапшотов спектра в строки водопада.

Порт `wf_task()` из `firmware/atomspectra-waterfall/main/spectrogram.c`.
Замечание #210. Приёмник (#212) кормит нас snapshot-ами; row+dur уходят в ASWF-writer (#211).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

WF_CHANNELS = 8192  # число каналов спектрограммы (тот же, что в firmware)


@dataclass(frozen=True)
class SpectrumSnapshot:
    """Кумулятивный снимок от прибора (мгновенный state)."""
    bins: np.ndarray          # dtype=uint32, shape=(WF_CHANNELS,) — счёты по каналам с начала записи
    total_counts: int         # сумма bins (uint32/uint64 в firmware — тут просто int)
    total_time_sec: int       # живое время прибора с начала записи, целые секунды


@dataclass(frozen=True)
class WaterfallRow:
    """Дельта-строка: приращение счётов за интервал + длительность интервала."""
    counts: np.ndarray  # dtype=uint16, shape=(WF_CHANNELS,) — клип [0, 65535]
    dur_sec: int        # длительность интервала (0..65535)


class DeltaCollector:
    def __init__(self, channels: int = WF_CHANNELS) -> None:
        self._channels = channels
        self._prev_bins: np.ndarray = np.zeros(channels, dtype=np.uint32)
        self._prev_total: int = 0
        self._prev_time: int = 0
        self._has_prev: bool = False

    def feed(self, snap: SpectrumSnapshot) -> WaterfallRow:
        """Преобразует снапшот в дельта-строку водопада."""
        reset = (not self._has_prev) or (snap.total_counts < self._prev_total)

        if reset:
            delta = snap.bins.astype(np.int64)
        else:
            delta = snap.bins.astype(np.int64) - self._prev_bins.astype(np.int64)
        np.clip(delta, 0, 65535, out=delta)
        counts = delta.astype(np.uint16)

        cur_time = snap.total_time_sec
        if reset:
            dt = cur_time
        elif cur_time >= self._prev_time:
            dt = cur_time - self._prev_time
        else:
            dt = 0
        dur_sec = 65535 if dt > 65535 else dt

        self._prev_bins = snap.bins.copy()
        self._prev_total = snap.total_counts
        self._prev_time = cur_time
        self._has_prev = True

        return WaterfallRow(counts=counts, dur_sec=dur_sec)

    def reset(self) -> None:
        """Сброс состояния коллекции."""
        self._prev_bins.fill(0)
        self._prev_total = 0
        self._prev_time = 0
        self._has_prev = False

    @property
    def prev_total(self) -> int:
        """Предыдущее значение total_counts."""
        return self._prev_total

    @property
    def prev_time(self) -> int:
        """Предыдущее значение total_time_sec."""
        return self._prev_time

    @property
    def has_prev(self) -> bool:
        """True, если был хотя бы один feed."""
        return self._has_prev
