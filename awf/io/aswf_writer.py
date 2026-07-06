"""Сегментный писатель ASWF v3: сериализация live-USB водопада в каталог сегментов.

Замечание #211. Читатель — `awf/io/aswf_loader.py` (uncompressed pathway).
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

# Порт из firmware/atomspectra-waterfall/main/spectrogram.h (константы сегментации).
WF_SEG_MAX_ROWS: int = 64             # закрыть сегмент по 64 строкам
WF_SEG_MAX_AGE_SEC: int = 600         # закрыть сегмент по возрасту 10 мин
WF_FSYNC_BATCH: int = 4               # fsync каждые N строк

WF_CHANNELS: int = 8192               # ширина спектра
WF_BASELINE_BYTES: int = WF_CHANNELS * 4   # 32768 — snapshot накопительного спектра

# Row layout (fixed stride): spectrum uint16 × N_CH + duration uint16.
# ВАЖНО: совпадает с firmware, чтобы был совместим с существующим loader-ом.
WF_ROW_STRIDE: int = WF_CHANNELS * 2 + 2   # 16386 байт

_ASWF_MAGIC: bytes = b"ASWF"
_JSON_HEADER_LEN: int = 4096          # фикс. длина заголовка (кратно 4096)

if TYPE_CHECKING:
    from awf.usb.collector import WaterfallRow  # только для типов


@dataclass(frozen=True)
class WaterfallRow:
    """Дельта-строка для writer-а.

    Совместимо с `awf.usb.collector.WaterfallRow`, но writer НЕ импортирует collector
    (обратная зависимость запрещена — writer чистая I/O-логика).
    """
    counts: np.ndarray    # dtype=uint16, shape=(WF_CHANNELS,)
    dur_sec: int


class AswfSegmentedWriter:
    def __init__(
        self,
        out_dir: Path | str,
        *,
        started_at: float,
        interval_sec: float,
        baseline: np.ndarray,
        calibration: Optional[list[float]] = None,
        serial: Optional[str] = None,
        note: Optional[str] = None,
        max_rows: int = WF_SEG_MAX_ROWS,
        max_age_sec: int = WF_SEG_MAX_AGE_SEC,
        fsync_batch: int = WF_FSYNC_BATCH,
        clock: "Callable[[], float]" = time.time,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._started_at = float(started_at)
        self._interval_sec = float(interval_sec)
        self._baseline = np.ascontiguousarray(baseline, dtype=np.uint32)
        if self._baseline.shape != (WF_CHANNELS,):
            raise ValueError(f"baseline must have shape ({WF_CHANNELS},), got {self._baseline.shape}")
        self._calibration = list(calibration) if calibration is not None else [0.0, 1.0, 0.0]
        self._serial = serial
        self._note = note
        self._max_rows = int(max_rows)
        self._max_age_sec = int(max_age_sec)
        self._fsync_batch = int(fsync_batch)
        self._clock = clock

        self._seg_index: int = 0          # номер следующего сегмента (0-based)
        self._file = None                 # BufferedWriter текущего сегмента
        self._rows_in_seg: int = 0        # строк в текущем сегменте
        self._seg_opened_at: float = 0.0  # когда открыт (clock())
        self._rows_since_fsync: int = 0
        self._closed: bool = False
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def write_row(self, row: "WaterfallRow") -> Path:
        if self._closed:
            raise RuntimeError("writer already closed")
        if self._file is None:
            self._open_new_segment()
        counts = np.ascontiguousarray(row.counts, dtype=np.uint16)
        if counts.shape != (WF_CHANNELS,):
            raise ValueError(f"row.counts must have shape ({WF_CHANNELS},)")
        dur = int(row.dur_sec)
        if dur < 0 or dur > 0xFFFF:
            raise ValueError(f"dur_sec out of range [0, 65535]: {dur}")
        payload = counts.tobytes() + struct.pack("<H", dur)
        assert len(payload) == WF_ROW_STRIDE
        self._file.write(payload)
        self._rows_in_seg += 1
        self._rows_since_fsync += 1
        if self._rows_since_fsync >= self._fsync_batch:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._rows_since_fsync = 0
        if (self._rows_in_seg >= self._max_rows or
                self._clock() - self._seg_opened_at >= self._max_age_sec):
            return self._finalize_current()
        return self._out_dir / f"seg_{self._seg_index:05d}.aswf"

    def tick(self, now: Optional[float] = None) -> Optional[Path]:
        if self._file is None:
            return None
        t = now if now is not None else self._clock()
        if t - self._seg_opened_at >= self._max_age_sec:
            return self._finalize_current()
        return None

    def close(self) -> Optional[Path]:
        if self._closed:
            return None
        if self._file is not None and self._rows_in_seg > 0:
            result = self._finalize_current()
            self._closed = True
            return result
        elif self._file is not None and self._rows_in_seg == 0:
            path = self._out_dir / f"seg_{self._seg_index:05d}.aswf"
            self._file.close()
            path.unlink(missing_ok=True)
            self._file = None
            self._closed = True
            return None
        else:
            self._closed = True
            return None

    @property
    def out_dir(self) -> Path:
        return self._out_dir

    @property
    def seg_index(self) -> int:
        """Индекс СЛЕДУЮЩЕГО сегмента, который будет открыт."""
        return self._seg_index

    @property
    def rows_in_seg(self) -> int:
        return self._rows_in_seg

    @property
    def is_open(self) -> bool:
        return self._file is not None and not self._closed

    def _open_new_segment(self) -> None:
        path = self._out_dir / f"seg_{self._seg_index:05d}.aswf"
        self._file = open(path, "wb")
        hdr_bytes = self._build_header_bytes()
        self._file.write(_ASWF_MAGIC)
        self._file.write(struct.pack("<I", _JSON_HEADER_LEN))
        self._file.write(hdr_bytes)
        self._file.write(self._baseline.tobytes())
        self._rows_in_seg = 0
        self._seg_opened_at = self._clock()
        self._rows_since_fsync = 0

    def _build_header_bytes(self) -> bytes:
        hdr = {
            "version": 3,
            "channels": WF_CHANNELS,
            "interval_sec": self._interval_sec,
            "started_at": self._started_at,
            "saved_rows": 0,
            "saved_at": 0,
            "calibration": self._calibration,
            "row_stride": WF_ROW_STRIDE,
            "row_fields": [
                {"name": "spectrum", "offset": 0,               "dtype": "uint16", "count": WF_CHANNELS},
                {"name": "duration", "offset": WF_CHANNELS * 2, "dtype": "uint16"},
            ],
            "baseline": {"count": WF_CHANNELS, "dtype": "uint32", "offset": 8 + _JSON_HEADER_LEN},
            "seg_index": self._seg_index,
        }
        if self._serial is not None:
            hdr["serial"] = self._serial
        if self._note is not None:
            hdr["note"] = self._note
        raw = json.dumps(hdr, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(raw) > _JSON_HEADER_LEN:
            raise ValueError(f"aswf: header too big: {len(raw)} > {_JSON_HEADER_LEN}")
        return raw + b"\x00" * (_JSON_HEADER_LEN - len(raw))

    def _finalize_current(self) -> Path:
        assert self._file is not None
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        path = self._out_dir / f"seg_{self._seg_index:05d}.aswf"
        self._seg_index += 1
        self._file = None
        self._rows_in_seg = 0
        self._rows_since_fsync = 0
        self._seg_opened_at = 0.0
        return path
