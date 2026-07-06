"""Задача #216 — писатель ASWF v3 в один файл (не сегментированный).

Совместим с `awf/io/aswf_loader.py` (uncompressed pathway v3): magic 'ASWF' +
uint32 header_len (4096) + JSON header + optional baseline uint32×N + N строк
(spectrum uint16×N_CH + duration uint16 = row_stride 16386 байт).

Отличие от `awf/io/aswf_writer.py`: там сегментированный writer для live-USB
записи в директорию, здесь монолитный dump уже накопленной Spectrogram в один
файл через QFileDialog. Используется пунктом «Файл → Сохранить как…».
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

WF_CHANNELS: int = 8192
WF_ROW_STRIDE: int = WF_CHANNELS * 2 + 2   # 16386 байт (spectrum + duration)
_ASWF_MAGIC: bytes = b"ASWF"
_JSON_HEADER_LEN: int = 4096


def _build_header_bytes(*, channels: int, interval_sec: float, started_at: float,
                         saved_rows: int, saved_at: float, calibration: list,
                         baseline_present: bool,
                         serial: Optional[str], note: Optional[str]) -> bytes:
    hdr = {
        "version": 3,
        "channels": int(channels),
        "interval_sec": float(interval_sec),
        "started_at": float(started_at),
        "saved_rows": int(saved_rows),
        "saved_at": float(saved_at),
        "calibration": list(calibration),
        "row_stride": WF_ROW_STRIDE,
        "row_fields": [
            {"name": "spectrum", "offset": 0, "dtype": "uint16", "count": int(channels)},
            {"name": "duration", "offset": int(channels) * 2, "dtype": "uint16"},
        ],
    }
    if baseline_present:
        hdr["baseline"] = {"count": int(channels), "dtype": "uint32",
                            "offset": 8 + _JSON_HEADER_LEN}
    if serial is not None:
        hdr["serial"] = str(serial)
    if note is not None:
        hdr["note"] = str(note)
    raw = json.dumps(hdr, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > _JSON_HEADER_LEN:
        raise ValueError(f"aswf: header too big: {len(raw)} > {_JSON_HEADER_LEN}")
    return raw + b"\x00" * (_JSON_HEADER_LEN - len(raw))


def write_aswf(path, spectrogram, *,
                serial: Optional[str] = None,
                note: Optional[str] = None,
                interval_sec: Optional[float] = None,
                started_at: Optional[float] = None) -> Path:
    """Сохранить Spectrogram в один файл ASWF v3 (uncompressed).

    - path: файл назначения (создаётся/перезаписывается).
    - spectrogram: awf.model.spectrogram.Spectrogram (calibration.coeffs -> hdr).
    - interval_sec: если не задано, берётся медиана real_time_s.
    - started_at: unix-секунды; если None -> now().
    """
    path = Path(path)
    sg = spectrogram
    counts = np.ascontiguousarray(sg.counts, dtype=np.uint16)
    if counts.ndim != 2:
        raise ValueError("Spectrogram.counts must be 2D")
    n_rows, n_ch = counts.shape
    if n_ch != WF_CHANNELS:
        raise ValueError(f"aswf_single_writer: только {WF_CHANNELS} каналов, а не {n_ch}")

    durations = np.asarray(sg.real_time_s, dtype=np.float64)
    if durations.shape != (n_rows,):
        raise ValueError(f"real_time_s.shape={durations.shape} != ({n_rows},)")
    # duration -> uint16 (клип для безопасности)
    dur_u16 = np.clip(np.rint(durations), 0, 0xFFFF).astype(np.uint16)

    if interval_sec is None:
        pos = durations[durations > 0]
        interval_sec = float(np.median(pos)) if pos.size else 1.0

    if started_at is None:
        started_at = float(time.time())

    calibration = list(np.asarray(sg.calibration.coeffs, dtype=np.float64).ravel())

    baseline_arr = None
    if getattr(sg, "baseline", None) is not None:
        b = np.asarray(sg.baseline)
        if b.size == n_ch:
            baseline_arr = np.ascontiguousarray(np.clip(b, 0, 0xFFFFFFFF), dtype=np.uint32)

    hdr_bytes = _build_header_bytes(
        channels=n_ch, interval_sec=interval_sec, started_at=started_at,
        saved_rows=n_rows, saved_at=float(time.time()), calibration=calibration,
        baseline_present=(baseline_arr is not None),
        serial=serial, note=note,
    )

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(_ASWF_MAGIC)
        f.write(struct.pack("<I", _JSON_HEADER_LEN))
        f.write(hdr_bytes)
        if baseline_arr is not None:
            f.write(baseline_arr.tobytes())
        # rows: spectrum uint16×N + duration uint16
        row_dur_bytes = dur_u16.astype("<u2").tobytes()
        row_spec_bytes = counts.astype("<u2").tobytes()
        # интерливинг батчем: для каждой строки — 16384 байт спектра + 2 байта dur
        for i in range(n_rows):
            f.write(row_spec_bytes[i * n_ch * 2 : (i + 1) * n_ch * 2])
            f.write(row_dur_bytes[i * 2 : (i + 1) * 2])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path
