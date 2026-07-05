from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import struct
import json
import warnings
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
        # #191/P1-aswf-1: файл короче 8 байт → внятная ошибка вместо struct.error.
        if len(head) < 8:
            raise ValueError(f"ASWF: файл обрезан (< 8 байт): {path}")
        magic = head[:4]
        if magic != b"ASWF":
            raise ValueError(f"ASWF: неверная сигнатура файла: {path}")
        header_len = struct.unpack("<I", head[4:8])[0]

        # Чтение и парсинг заголовка
        raw_header = f.read(header_len)
        # #191/P1-aswf-2: файл обрезан посреди заголовка → UnicodeDecodeError/JSONDecodeError
        # превращаем в диагностичный ValueError вместо сырого traceback.
        try:
            header_raw = raw_header.split(b"\x00")[0].strip()
            hdr = json.loads(header_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"ASWF: повреждённый/обрезанный заголовок в {path}: {exc}") from exc

        # Проверка числа каналов
        n_channels = int(hdr.get("channels") or 0)
        if n_channels <= 0:
            raise ValueError(f"ASWF: неверное число каналов: {n_channels}")

        # Задача #180: v2-раскладка — row_stride из header (может включать per-row row_time поле).
        # Для v1 header row_stride отсутствует → fallback на n_channels*2 (совместимо).
        counts_bytes = n_channels * 2
        row_stride = int(hdr.get("row_stride") or counts_bytes)
        if row_stride < counts_bytes:
            raise ValueError(f"ASWF: row_stride={row_stride} < counts_bytes={counts_bytes}")

        # Определение размера данных
        data_off = 8 + header_len
        f.seek(0, 2)  # EOF
        file_size = f.tell()
        n_rows_file = (file_size - data_off) // row_stride
        # #191/P1-aswf-3: неполная последняя строка (краш прибора) — логируем потерю.
        _tail = (file_size - data_off) % row_stride
        if _tail:
            warnings.warn(
                f"ASWF: {path}: {_tail} байт неполного последнего интервала отброшены "
                f"(файл усечён крашем прибора?)"
            )

        # Задача #180: saved_rows==0 (прошивка v2 не обновила метку при выгрузке) → берём n_rows_file.
        saved_rows = hdr.get("saved_rows")
        if saved_rows is None or int(saved_rows) <= 0:
            n_rows = n_rows_file
        else:
            n_rows = min(int(saved_rows), n_rows_file)

        if max_slices is not None:
            n_rows = min(n_rows, max_slices)

        if n_rows < 1:
            raise ValueError(f"ASWF: нет строк данных: {path}")

        # Задача #180: читаем row_stride байт на строку; counts — первые counts_bytes байт строки.
        f.seek(data_off)
        raw_rows = np.frombuffer(f.read(n_rows * row_stride), dtype=np.uint8).reshape(n_rows, row_stride)
        counts = np.ascontiguousarray(raw_rows[:, :counts_bytes]).view("<u2").reshape(n_rows, n_channels).astype(np.uint16, copy=True)

        # Задача #180: real_time_s per-row — из row_time поля v2 (offset+dtype в pardefines),
        # для v1 fallback на interval_sec. time_offsets_s = кумулятивная сумма длительностей.
        interval = float(hdr.get("interval_sec") or 0.0)
        rt_meta = hdr.get("row_time") or None
        if rt_meta and row_stride > counts_bytes:
            off = int(rt_meta.get("offset") or counts_bytes)
            npdt = "<u4" if str(rt_meta.get("dtype") or "uint16") == "uint32" else "<u2"
            sz = 4 if npdt == "<u4" else 2
            real_time_s = np.ascontiguousarray(raw_rows[:, off:off+sz]).view(npdt).ravel().astype(np.float64)
        else:
            real_time_s = np.full(n_rows, interval if interval > 0 else np.nan, dtype=np.float64)
        time_offsets_s = np.zeros(n_rows, dtype=np.float64)
        if n_rows > 1:
            time_offsets_s[1:] = np.cumsum(real_time_s[:-1])
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
