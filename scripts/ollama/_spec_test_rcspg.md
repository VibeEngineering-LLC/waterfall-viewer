# OLLAMA EXECUTION TASK SPEC — tests/test_rcspg_loader.py

Сгенерировать **полный** pytest-модуль `tests/test_rcspg_loader.py` для загрузчика `load_rcspg`
из `awf.io.rcspg_loader`. Вернуть ТОЛЬКО код Python, без пояснений, без markdown-ограждений.

Тесты НЕ зависят от внешних файлов: синтетический .rcspg строится в коде и пишется в `tmp_path`
(встроенная фикстура pytest). Формат .rcspg = JSON (`json.dump`).

## Импорты
```python
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from awf.io.rcspg_loader import load_rcspg, _epoch_ms_to_iso
```

## Хелпер построения синтетического файла
Функция `_write_rcspg(tmp_path, doc) -> str`: пишет `doc` (dict) как JSON в
`tmp_path / "synthetic.rcspg"` (encoding utf-8, `json.dump`), возвращает str-путь.

Базовый документ `_doc()` возвращает dict:
```python
{
  "channelCount": 8,
  "coefficients": [1.0, 2.0, 0.0],          # E(ch)=1+2*ch
  "startTimeTimestamp": 1700000000000,      # ms -> 2023-11-14T22:13:20Z UTC
  "deviceModel": "RadiaCode-110",
  "spectrums": [
    {"pulses": [0, 1, 2, 3],                 "collectTime": 2, "timestamp": 1700000000000},
    {"pulses": [5, 0, 0, 0, 0, 0, 0, 4],     "collectTime": 3, "timestamp": 1700000005000},
    {"pulses": [1, 1],                        "collectTime": 1, "timestamp": 1700000011000},
  ],
}
```

## Тесты (каждый — отдельная функция, принимает `tmp_path` где нужно)

1. `test_basic_shape_and_dtype(tmp_path)`:
   - sg = load_rcspg(_write_rcspg(tmp_path, _doc()))
   - assert sg.n_slices == 3; sg.n_channels == 8; sg.counts.dtype == np.uint16

2. `test_counts_padding(tmp_path)` — хвостовые нулевые каналы дополняются нулём:
   - np.testing.assert_array_equal(sg.counts[0], [0,1,2,3,0,0,0,0])
   - np.testing.assert_array_equal(sg.counts[1], [5,0,0,0,0,0,0,4])
   - np.testing.assert_array_equal(sg.counts[2], [1,1,0,0,0,0,0,0])

3. `test_calibration(tmp_path)`:
   - np.testing.assert_allclose(sg.calibration.coeffs, [1.0, 2.0, 0.0])
   - en = sg.energies(); assert en[0] == pytest.approx(1.0); en[1] == pytest.approx(3.0)

4. `test_time_axes(tmp_path)`:
   - np.testing.assert_allclose(sg.time_offsets_s, [0.0, 5.0, 11.0])
   - np.testing.assert_allclose(sg.real_time_s, [2.0, 3.0, 1.0])
   - np.testing.assert_allclose(sg.live_time_s, [2.0, 3.0, 1.0])   # real==live (rcspg не разделяет)

5. `test_t0_iso(tmp_path)`:
   - assert sg.t0_iso == "2023-11-14T22:13:20Z"
   - и отдельно: assert _epoch_ms_to_iso(None) is None
   - assert _epoch_ms_to_iso(1700000000000) == "2023-11-14T22:13:20Z"

6. `test_max_slices(tmp_path)`:
   - sg = load_rcspg(_write_rcspg(tmp_path, _doc()), max_slices=2)
   - assert sg.n_slices == 2; sg.n_channels == 8

7. `test_empty_spectrums_raises(tmp_path)`:
   - doc = _doc(); doc["spectrums"] = []
   - with pytest.raises(ValueError): load_rcspg(_write_rcspg(tmp_path, doc))

8. `test_channelcount_inferred(tmp_path)` — channelCount отсутствует → выводится из макс. длины pulses:
   - doc = _doc(); del doc["channelCount"]
   - sg = load_rcspg(_write_rcspg(tmp_path, doc))
   - assert sg.n_channels == 8   # макс. длина pulses в _doc() == 8

## Требования
- `from __future__ import annotations` первой строкой.
- Каждый тест сам строит sg (вызывает _write_rcspg + load_rcspg) — не полагаться на общий fixture-объект.
- Комментарии краткие, по-русски. Только stdlib + numpy + pytest.
- Никакого кода на уровне модуля кроме импортов и определения хелперов `_write_rcspg`/`_doc`.
