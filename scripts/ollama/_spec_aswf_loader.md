# Спецификация: awf/io/aswf_loader.py

Загрузчик нативного waterfall-формата прибора AtomSpectra (расширение `.aswf`).
Формат подтверждён реверс-инжинирингом реального файла прибора.

## Бинарная раскладка файла `.aswf`
- Байты [0,4): магия — ASCII `ASWF` (b"ASWF"). Если не совпала — `ValueError`.
- Байты [4,8): `uint32` little-endian = длина JSON-заголовка в байтах (`header_len`, обычно 4096).
- Байты [8, 8+header_len): JSON-заголовок (UTF-8), дополнен padding-ом (нулевые байты и/или пробелы)
  до `header_len`. Парсить так: взять эти байты, `raw.split(b"\x00")[0].strip()`, затем `json.loads`.
- Байты [8+header_len, EOF): матрица отсчётов — подряд строки (временные срезы), КАЖДАЯ строка =
  `channels` значений типа `uint16` **little-endian** (`'<u2'`). Без per-row заголовков/таймстампов.

## Поля JSON-заголовка (могут отсутствовать — использовать .get с дефолтами)
- `channels` (int) — число каналов в строке (напр. 8192). Обязательное; если ≤0 — `ValueError`.
- `interval_sec` (число) — длительность одного среза в секундах (напр. 10). Может отсутствовать.
- `calibration` (list[float]) — коэффициенты ПОЛИНОМА энергии по возрастанию степеней:
  `E(ch) = c0 + c1*ch + c2*ch^2 + ...`. Передавать как есть в `Calibration(coeffs=...)`.
- `started_at` (int) — Unix-время старта записи в СЕКУНДАХ (не миллисекундах).
- `saved_rows` (int) — сколько строк записано (может отсутствовать; тогда вывести из размера файла).
- остальные ключи (`format`,`version`,`serial`,`source`,`saved_at`) — игнорировать.

## Импорты
```python
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import struct
import json
import numpy as np
from awf.model.spectrogram import Calibration, Spectrogram
```

## Вспомогательная функция
```python
def _epoch_s_to_iso(sec) -> str | None:
    """Unix-время в СЕКУНДАХ -> ISO-8601 UTC (или None при ошибке/None)."""
    if sec is None:
        return None
    try:
        return datetime.fromtimestamp(float(sec), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return None
```

## Основная функция
```python
def load_aswf(path, *, max_slices: int | None = None) -> Spectrogram:
```
Алгоритм:
1. `path = Path(path)`. Открыть файл в бинарном режиме (`"rb"`).
2. Прочитать первые 8 байт. `magic = head[:4]`; если `magic != b"ASWF"` — поднять
   `ValueError(f"ASWF: неверная сигнатура файла: {path}")`.
   `header_len = struct.unpack("<I", head[4:8])[0]`.
3. Прочитать `header_len` байт заголовка; распарсить JSON как описано выше
   (`hdr = json.loads(raw.split(b"\x00")[0].strip().decode("utf-8"))`).
4. `n_channels = int(hdr.get("channels") or 0)`; если `<= 0` — `ValueError`.
5. `data_off = 8 + header_len`. Определить размер данных:
   `import os; data_bytes = os.path.getsize(path) - data_off` (или через `f.seek(0,2)`).
   `row_bytes = n_channels * 2`. `n_rows_file = data_bytes // row_bytes` (целочисленно — хвостовые
   неполные байти игнорируются, файл может быть оборван).
6. Определить число строк: если `saved_rows` задан — `n_rows = min(int(saved_rows), n_rows_file)`,
   иначе `n_rows = n_rows_file`. Затем, если `max_slices` не None — `n_rows = min(n_rows, max_slices)`.
   Если `n_rows < 1` — `ValueError(f"ASWF: нет строк данных: {path}")`.
7. Прочитать отсчёты: `f.seek(data_off)`; `buf = np.frombuffer(f.read(n_rows * row_bytes), dtype="<u2")`.
   `counts = np.ascontiguousarray(buf.reshape(n_rows, n_channels))` (копия — чтобы массив был
   writable и не держал файловый буфер; dtype остаётся uint16).
8. Временные оси: `interval = float(hdr.get("interval_sec") or 0.0)`.
   `time_offsets_s = np.arange(n_rows, dtype=np.float64) * interval`.
   `real_time_s = np.full(n_rows, interval if interval > 0 else np.nan, dtype=np.float64)`.
   `live_time_s = real_time_s.copy()` (прибор не даёт отдельный live-time — берём интервал).
9. Калибровка: `cal = hdr.get("calibration")`; если непустой список —
   `calibration = Calibration(coeffs=np.asarray(cal, dtype=np.float64))`, иначе
   `Calibration(coeffs=np.array([0.0, 1.0], dtype=np.float64))`.
10. `t0_iso = _epoch_s_to_iso(hdr.get("started_at"))`.
11. Вернуть `Spectrogram(counts=counts, calibration=calibration, time_offsets_s=time_offsets_s,
    real_time_s=real_time_s, live_time_s=live_time_s, t0_iso=t0_iso, source_path=str(path))`.

## Требования
- Чистый Python 3.12+, только перечисленные импорты.
- Комментарии кратко по-русски.
- Файл закрывать (использовать `with open(...) as f:` для всех чтений, либо открыть один раз
  и читать последовательно: head -> header -> seek(data_off) -> data).
- Никаких сторонних зависимостей кроме numpy и стандартной библиотеки.
- Не печатать ничего (не использовать print). Только определения функций.
- Вернуть ТОЛЬКО код модуля, без markdown-ограждений.
