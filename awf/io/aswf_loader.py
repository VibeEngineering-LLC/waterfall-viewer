from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import struct
import json
import numpy as np
from awf.model.spectrogram import Calibration, Spectrogram

def _epoch_s_to_iso(sec) -> str | None:
    """Unix-время в СЕКУНДАХ -> ISO-8601 UTC (или None при ошибке/None)."""
    if sec is None:
        return None
    try:
        return datetime.fromtimestamp(float(sec), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return None

def load_aswf(path, *, max_slices: int | None = None) -> Spectrogram:
    path = Path(path)
    with open(path, "rb") as f:
        # Чтение сигнатуры и длины заголовка
        head = f.read(8)
        magic = head[:4]
        if magic != b"ASWF":
            raise ValueError(f"ASWF: неверная сигнатура файла: {path}")
        header_len = struct.unpack("<I", head[4:8])[0]

        # Чтение и парсинг заголовка
        raw_header = f.read(header_len)
        header_raw = raw_header.split(b"\x00")[0].strip()
        hdr = json.loads(header_raw.decode("utf-8"))

        # Проверка числа каналов
        n_channels = int(hdr.get("channels") or 0)
        if n_channels <= 0:
            raise ValueError(f"ASWF: неверное число каналов: {n_channels}")

        # Определение размера данных
        data_off = 8 + header_len
        f.seek(0, 2)  # EOF
        file_size = f.tell()
        data_bytes = file_size - data_off
        row_bytes = n_channels * 2
        n_rows_file = data_bytes // row_bytes

        # Определение числа строк
        saved_rows = hdr.get("saved_rows")
        if saved_rows is not None:
            n_rows = min(int(saved_rows), n_rows_file)
        else:
            n_rows = n_rows_file

        if max_slices is not None:
            n_rows = min(n_rows, max_slices)

        if n_rows < 1:
            raise ValueError(f"ASWF: нет строк данных: {path}")

        # Чтение отсчётов
        f.seek(data_off)
        buf = np.frombuffer(f.read(n_rows * row_bytes), dtype="<u2")
        # np.array(copy) -> writable C-непрерывный массив, не держит файловый буфер
        counts = np.array(buf.reshape(n_rows, n_channels), dtype=np.uint16)

        # Временные оси
        interval = float(hdr.get("interval_sec") or 0.0)
        time_offsets_s = np.arange(n_rows, dtype=np.float64) * interval
        real_time_s = np.full(n_rows, interval if interval > 0 else np.nan, dtype=np.float64)
        live_time_s = real_time_s.copy()

        # Калибровка
        cal = hdr.get("calibration")
        if cal:
            calibration = Calibration(coeffs=np.asarray(cal, dtype=np.float64))
        else:
            calibration = Calibration(coeffs=np.array([0.0, 1.0], dtype=np.float64))

        # Время начала записи
        t0_iso = _epoch_s_to_iso(hdr.get("started_at"))

    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=time_offsets_s,
        real_time_s=real_time_s,
        live_time_s=live_time_s,
        t0_iso=t0_iso,
        source_path=str(path)
    )
