# OLLAMA EXECUTION TASK SPEC — rcspg_loader.py

Сгенерировать **полный** Python-модуль `awf/io/rcspg_loader.py` — загрузчик waterfall-спектрограмм
RadiaCode (`.rcspg`). Вернуть ТОЛЬКО код модуля, без объяснений, без markdown-ограждений.

## Контекст: целевые типы (НЕ переопределять, только импортировать)

Модуль `awf.model.spectrogram` уже содержит:

```python
@dataclass(frozen=True)
class Calibration:
    coeffs: np.ndarray            # полином E(ch) в ВОЗРАСТАЮЩЕМ порядке [c0, c1, c2, ...]
    # конструируется как Calibration(coeffs=np.asarray([...], dtype=np.float64))

class Spectrogram:
    def __init__(self, *, counts, calibration, time_offsets_s,
                 real_time_s, live_time_s, t0_iso=None, source_path=None): ...
    # counts: 2D ndarray [n_slices, n_channels]
    # time_offsets_s/real_time_s/live_time_s: 1D ndarray длины n_slices
```

Импорт: `from awf.model.spectrogram import Calibration, Spectrogram`

## Формат .rcspg (JSON, несмотря на бинарное расширение) — проверено на реальном файле RadiaCode-110

Верхний уровень — объект:
- `channelCount` : int — число каналов (напр. 1024)
- `coefficients` : `[c0, c1, c2]` — полином энергии E(ch)=c0+c1·ch+c2·ch², возрастающий порядок, float с точкой
- `startTimeTimestamp` : int — эпоха в **миллисекундах**, начало записи
- `spectrums` : массив объектов `{ "pulses": [int...], "collectTime": int, "timestamp": int }`
  - `pulses` : **дифференциальный счёт по каналам** этого среза. `pulses[j]` = число отсчётов в канале `j`
    за интервал. Длина массива ≤ `channelCount` (хвостовые нулевые каналы опущены). Класть в
    `counts[i, 0:len(pulses)] = pulses`, остальное оставить нулём.
  - `collectTime` : int — длительность набора среза, **секунды**
  - `timestamp` : int — эпоха в **миллисекундах**, момент среза (монотонно растёт)
- прочие ключи (`deviceModel`, `deviceId`, `title`, `id`) — игнорировать.

rcspg **не разделяет** real/live time → `real_time_s == live_time_s == collectTime`.

## Требуемый публичный интерфейс

```python
def load_rcspg(path, *, max_slices: int | None = None) -> Spectrogram: ...
```

Поведение `load_rcspg`:
1. `path = Path(path)`; открыть `open(path, "r", encoding="utf-8")`; `doc = json.load(f)`.
2. `spectrums = doc.get("spectrums") or []`; если `max_slices` не None — `spectrums = spectrums[:max_slices]`.
3. Если `spectrums` пуст → `raise ValueError(f"RCSPG: в файле нет спектров (spectrums): {path}")`.
4. `n_channels = int(doc.get("channelCount") or 0)`; если ≤0 — вывести из макс. длины `pulses` по всем срезам;
   если всё ещё ≤0 → `raise ValueError(f"RCSPG: не удалось определить число каналов: {path}")`.
5. `n_slices = len(spectrums)`.
6. Глобальный максимум значения pulses `gmax`; `dtype = np.uint16 if gmax <= 65535 else np.int32`.
7. `counts = np.zeros((n_slices, n_channels), dtype=dtype)`.
   `real_arr`, `live_arr` = `np.full(n_slices, np.nan, np.float64)`; `offsets = np.zeros(n_slices, np.float64)`.
8. База времени `base_ms` = `timestamp` первого среза, иначе `startTimeTimestamp`.
9. Цикл по срезам `i, s`:
   - `pulses = s.get("pulses") or ()`; `m = min(len(pulses), n_channels)`; если m>0 — `counts[i, :m] = np.asarray(pulses[:m], dtype=dtype)`.
   - `ct = s.get("collectTime")`; если не None — `real_arr[i] = live_arr[i] = float(ct)`.
   - `ts = s.get("timestamp")`; если ts и base_ms не None — `offsets[i] = (float(ts) - float(base_ms)) / 1000.0`.
10. `coeffs = doc.get("coefficients")`; если есть — `Calibration(coeffs=np.asarray(coeffs, dtype=np.float64))`,
    иначе `Calibration(coeffs=np.array([0.0, 1.0], dtype=np.float64))`.
11. `t0_iso` через хелпер `_epoch_ms_to_iso(startTimeTimestamp or base_ms)`.
12. Вернуть `Spectrogram(counts=..., calibration=..., time_offsets_s=offsets, real_time_s=real_arr,
    live_time_s=live_arr, t0_iso=t0_iso, source_path=str(path))`.

Хелпер:
```python
def _epoch_ms_to_iso(ms) -> str | None:
    # эпоха (мс) -> ISO-8601 UTC "2026-05-05T07:36:09Z"; None/ошибка -> None
```
Реализация: `datetime.fromtimestamp(float(ms)/1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")`,
в `try/except (OSError, OverflowError, ValueError)` → None; если `ms is None` → None.

## Требования к коду
- `from __future__ import annotations` первой строкой.
- Импорты: `from datetime import datetime, timezone`, `from pathlib import Path`, `import json`, `import numpy as np`, целевые типы.
- Комментарии на русском, лаконичные. Без сторонних зависимостей (только stdlib + numpy).
- Никакого кода вне функций (модуль-уровень — только импорты и комментарии-шапка).
- Не печатать ничего (no print), не читать argv.
