# Спецификация: tests/test_aswf_loader.py

Pytest-тесты для загрузчика `awf/io/aswf_loader.py` (формат `.aswf` прибора AtomSpectra).
Тесты строят СИНТЕТИЧЕСКИЙ `.aswf`-файл в памяти и пишут во временную папку pytest (`tmp_path`).
Стиль и структура — как в существующем `tests/test_rcspg_loader.py` (helper-сборщик + отдельные
функции-тесты, `np.testing.assert_*`, `pytest.approx`, `pytest.raises`).

## Импорты
```python
from __future__ import annotations
import struct
import json
from pathlib import Path
import numpy as np
import pytest
from awf.io.aswf_loader import load_aswf, _epoch_s_to_iso
```

## Helper: сборка бинарного `.aswf`
Функция `_build_aswf(tmp_path, header: dict, rows: list[list[int]], header_len: int = 4096) -> str`:
1. `body = json.dumps(header).encode("utf-8")`.
2. Если `len(body) < header_len` — дополнить нулевыми байтами `b"\x00"` до `header_len`
   (`body = body + b"\x00" * (header_len - len(body))`). (Считать, что тест не передаёт заголовок
   длиннее `header_len`.)
3. `magic = b"ASWF"`; `len_field = struct.pack("<I", header_len)`.
4. Данные строк: для каждой строки `row` (список int) — `np.array(row, dtype="<u2").tobytes()`.
   Склеить все строки подряд.
5. Записать файл `tmp_path / "synthetic.aswf"` в бинарном режиме: `magic + len_field + body + data`.
6. Вернуть `str(path)`.

## Helper: эталонный заголовок
Функция `_header(**over)` возвращает dict со значениями по умолчанию, перекрываемыми `over`:
```python
{
    "format": "aswf",
    "version": 1,
    "channels": 4,
    "interval_sec": 10,
    "calibration": [1.0, 2.0, 0.0],     # E(ch) = 1 + 2*ch
    "started_at": 1700000000,           # СЕКУНДЫ -> 2023-11-14T22:13:20Z UTC
    "saved_rows": 3,
    "serial": "TEST",
}
```
и эталонные строки `_rows()`:
```python
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
```

## Тесты (каждый — отдельная функция)
1. `test_basic_shape_and_dtype(tmp_path)`: загрузить эталон; `sg.n_slices == 3`,
   `sg.n_channels == 4`, `sg.counts.dtype == np.uint16`.
2. `test_counts_values(tmp_path)`: `np.testing.assert_array_equal(sg.counts[0], [0,1,2,3])`,
   строка `sg.counts[2]` == `[8,9,10,11]`.
3. `test_counts_writable(tmp_path)`: `assert sg.counts.flags.writeable is True`
   (массив должен быть копией, не read-only view на файловый буфер).
4. `test_calibration(tmp_path)`: `np.testing.assert_allclose(sg.calibration.coeffs, [1.0,2.0,0.0])`;
   `en = sg.energies()`; `en[0] == pytest.approx(1.0)`, `en[1] == pytest.approx(3.0)`.
5. `test_time_axes(tmp_path)`: `np.testing.assert_allclose(sg.time_offsets_s, [0.0,10.0,20.0])`;
   `np.testing.assert_allclose(sg.real_time_s, [10.0,10.0,10.0])`;
   `np.testing.assert_allclose(sg.live_time_s, [10.0,10.0,10.0])`.
6. `test_t0_iso(tmp_path)`: `assert sg.t0_iso == "2023-11-14T22:13:20Z"`;
   `assert _epoch_s_to_iso(None) is None`;
   `assert _epoch_s_to_iso(1700000000) == "2023-11-14T22:13:20Z"`.
7. `test_max_slices(tmp_path)`: загрузить с `max_slices=2`; `sg.n_slices == 2`, `sg.n_channels == 4`.
8. `test_bad_magic_raises(tmp_path)`: записать файл, где первые 4 байта НЕ `ASWF`
   (например `b"XXXX" + struct.pack("<I", 4096) + b"\x00"*4096`); `pytest.raises(ValueError)` при
   `load_aswf(path)`.
9. `test_saved_rows_caps_file(tmp_path)`: заголовок с `saved_rows=2`, но в файле 3 строки данных;
   результат `sg.n_slices == 2` (берётся минимум saved_rows и фактического числа строк в файле).
10. `test_saved_rows_absent_infers_from_size(tmp_path)`: убрать ключ `saved_rows` из заголовка
    (`header.pop("saved_rows", None)`); при 3 строках данных в файле `sg.n_slices == 3`
    (число строк выводится из размера файла).
11. `test_default_calibration_when_missing(tmp_path)`: убрать ключ `calibration`; загрузка не падает,
    `sg.energies()[0] == pytest.approx(0.0)`, `sg.energies()[1] == pytest.approx(1.0)`
    (дефолт `[0,1]` => E(ch)=ch).

## Требования
- Только перечисленные импорты + helper-функции, описанные выше.
- Комментарии кратко по-русски, где нужно.
- Не использовать print; не обращаться к сети/реальным файлам прибора — только `tmp_path`.
- Вернуть ТОЛЬКО код модуля, без markdown-ограждений.
