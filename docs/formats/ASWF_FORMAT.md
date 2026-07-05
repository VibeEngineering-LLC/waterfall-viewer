# Формат файла ASWF — спецификация

[🇬🇧 English](ASWF_FORMAT.en.md)

**ASWF** (AtomSpectra Waterfall Format) — бинарный формат хранения спектрограммы
(водопада): последовательность спектров, снятых через фиксированные интервалы.
Каждый файл `.aswf` самодостаточен и содержит полный заголовок и данные, пригодные
для автономного парсинга без внешней схемы.

---

## Структура файла

```
┌────────────────────────────────────────────────────────┐
│  Offset 0 │ 4 байта │ Магическое число "ASWF" (ASCII)  │
│  Offset 4 │ 4 байта │ uint32 LE: длина JSON-области     │
│  Offset 8 │ N байт  │ JSON-заголовок (UTF-8 + пробелы)  │
│  Offset 8+N         │ Строки данных (payload)            │
└────────────────────────────────────────────────────────┘
```

| Поле          | Смещение | Размер   | Тип       | Значение                        |
|---------------|----------|----------|-----------|---------------------------------|
| Магия         | 0        | 4        | ASCII     | `41 53 57 46` = `"ASWF"`       |
| `hlen`        | 4        | 4        | uint32 LE | Зарезервированный размер JSON   |
| JSON-шапка    | 8        | `hlen`   | UTF-8     | Метаданные (доби́та пробелами)  |
| Payload       | 8+`hlen` | N×stride | binary    | Строки спектра, старейшая первой |

**Важно:** `hlen` — это **зарезервированный** (выровненный) размер JSON-области,
а не длина самого JSON-текста. Текущее значение всегда `4096`. Парсер обязан
использовать именно `hlen` для вычисления смещения payload, а не сканировать `}`.

---

## JSON-заголовок

Заголовок содержит полное описание формата; все поля, нужные для декодирования,
указаны внутри файла — внешняя схема не нужна.

### Поля заголовка

| Ключ           | Тип              | Обязательный | Описание |
|----------------|------------------|:---:|------|
| `format`       | string           | да  | Всегда `"atomspectra-waterfall"` |
| `version`      | int              | да  | Версия формата: `1` или `2` |
| `channels`     | int              | да  | Каналов в строке; текущее значение `8192` |
| `dtype`        | string           | да  | Тип элемента: `"uint16"` |
| `byte_order`   | string           | да  | Порядок байт: `"little"` |
| `row_stride`   | int              | v2  | Размер строки в байтах `16386` (v2); отсутствует в v1 |
| `row_time`     | object           | v2  | Описание поля длительности (см. ниже) |
| `interval_sec` | int              | да  | Номинальный интервал записи, секунды |
| `started_at`   | int (unix ts)    | да  | Время первой строки (UTC, секунды от эпохи) |
| `saved_rows`   | int              | да  | Число строк в файле; `0` у незакрытого сегмента |
| `saved_at`     | int (unix ts)    | да  | Время финализации файла; `0` у незакрытого сегмента |
| `serial`       | string           | нет | Серийный номер устройства |
| `calibration`  | array of floats  | нет | Коэффициенты энергетической калибровки |

#### Объект `row_time` (только v2)

```json
{
  "dtype":  "uint16",
  "unit":   "sec",
  "offset": 16384
}
```

Поле `offset` — смещение поля длительности **внутри строки** (в байтах).

### Пример заголовка v2

```json
{
  "saved_rows": 660,
  "saved_at": 1783198621,
  "format": "atomspectra-waterfall",
  "version": 2,
  "channels": 8192,
  "dtype": "uint16",
  "byte_order": "little",
  "row_stride": 16386,
  "row_time": {"dtype": "uint16", "unit": "sec", "offset": 16384},
  "interval_sec": 60,
  "started_at": 1783157403,
  "serial": "AS-001",
  "calibration": [0.0, 0.298, 0.0]
}
```

---

## Форматы строк

### v1 — строка 16384 байта

```
Байты [0 .. 16383]:  8192 × uint16 LE — счёт за интервал, канал за каналом
```

Реальная длительность строки = `interval_sec` (номинал).

### v2 — строка 16386 байт (текущая)

```
Байты [0     .. 16383]:  8192 × uint16 LE — счёт за интервал
Байты [16384 .. 16385]:  uint16 LE — реальная длительность, секунды
```

Поле длительности (`duration`) содержит **фактическое живое время прибора**
за этот интервал в секундах. Если значение равно `0` — использовать `interval_sec`.

### Определение версии

```python
is_v2 = "row_stride" in header
stride = header.get("row_stride", header["channels"] * 2)
```

### Число строк

```python
n_rows = (file_size - (8 + hlen)) // stride
```

Если `saved_rows` в заголовке равен `0` (незакрытый сегмент), вычислить `n_rows`
по формуле выше.

---

## Данные спектра

Каждый канал строки — **дельта** накопительного спектра: сколько импульсов
зафиксировано за данный интервал. Тип `uint16` — значения от `0` до `65535`.

### Абсолютный спектр (сумма за всё время)

```python
absolute = [0] * channels
for row in rows:
    for ch in range(channels):
        absolute[ch] += row[ch]
```

### Скорость счёта (отсчётов/с) для строки i

```python
dur_i = duration[i] if duration[i] > 0 else interval_sec
rate  = [row[i][ch] / dur_i for ch in range(channels)]
```

---

## Временны́е метки строк

```
t[0] = started_at
t[i] = started_at + sum(duration[0..i-1])
```

Для строки с `duration[j] == 0` вместо реального значения подставить `interval_sec`.

В v1 формуле `duration[j] == interval_sec` для всех строк.

---

## Калибровка: канал → энергия (кэВ)

Если поле `calibration` присутствует, энергия вычисляется как полином:

```
E(ch) = a[0] + a[1]·ch + a[2]·ch² + …
```

где `a = calibration` (массив коэффициентов, индекс = степень полинома).

Пример из заголовка выше: `E(ch) = 0.0 + 0.298·ch + 0.0·ch²` — линейная шкала.

Если поле отсутствует, энергетическая шкала неизвестна; ось X — номер канала.

---

## Python: минимальный парсер

```python
import json
import struct
from pathlib import Path

def read_aswf(path):
    """Читает .aswf, возвращает (header, rows, durations).

    rows       — list из N кортежей по channels элементов uint16.
    durations  — list из N значений uint16 (0 = использовать interval_sec).
                 Для v1 все значения равны header["interval_sec"].
    """
    buf   = Path(path).read_bytes()
    magic = buf[:4]
    if magic != b"ASWF":
        raise ValueError(f"Не ASWF-файл: magic={magic!r}")

    hlen   = struct.unpack_from("<I", buf, 4)[0]
    header = json.loads(buf[8:8 + hlen].decode("utf-8"))

    ch     = header["channels"]
    stride = header.get("row_stride", ch * 2)
    is_v2  = "row_stride" in header

    payload = buf[8 + hlen:]
    n_rows  = len(payload) // stride

    rows      = []
    durations = []
    for i in range(n_rows):
        off  = i * stride
        row  = struct.unpack_from(f"<{ch}H", payload, off)
        rows.append(row)
        if is_v2:
            dur = struct.unpack_from("<H", payload, off + ch * 2)[0]
        else:
            dur = header["interval_sec"]
        durations.append(dur)

    return header, rows, durations


def row_timestamp(header, durations, index):
    """Unix-timestamp начала строки index."""
    ts = header["started_at"]
    iv = header["interval_sec"]
    for j in range(index):
        ts += durations[j] if durations[j] > 0 else iv
    return ts


def channel_to_kev(header, ch):
    """Энергия канала ch в кэВ (None если калибровки нет)."""
    cal = header.get("calibration")
    if not cal:
        return None
    return sum(a * ch**i for i, a in enumerate(cal))
```

---

## HTTP API устройства

Сегменты хранятся на Flash и доступны по HTTP **без авторизации** (список) и
**с авторизацией** (содержимое файла).

### Список сегментов

```
GET /api/waterfall/segments
```

Ответ `200 application/json`:

```json
{
  "segments": [
    { "name": "seg_00000.aswf", "size": 1056870, "finalized": true  },
    { "name": "seg_00001.aswf", "size": 524904,  "finalized": false }
  ],
  "ring_capacity": 64,
  "seg_count": 2
}
```

| Поле        | Описание |
|-------------|----------|
| `name`      | Имя файла, используется в запросе скачивания |
| `size`      | Размер файла в байтах |
| `finalized` | `false` — сегмент ещё открыт (записывается); заголовок имеет `saved_rows=0` |

Незакрытый сегмент (`finalized: false`) можно читать, но число строк
в заголовке равно `0` — вычислить по размеру файла.

### Скачивание сегмента

```
GET /api/waterfall/segment?name=seg_00000.aswf
Authorization: Basic <base64(login:password)>
```

Ответ `200 application/octet-stream` — бинарный `.aswf` файл.

### Удаление сегмента (подтверждение приёма)

```
POST /api/waterfall/segment/delete?name=seg_00000.aswf
Authorization: Basic <base64(login:password)>
```

Ответ `200` — файл удалён с Flash.

---

## Кольцевой буфер сегментов

Устройство хранит ограниченное количество финализированных сегментов.
При превышении лимита старейший сегмент удаляется автоматически.
Незакрытый сегмент (`finalized: false`) в лимит не входит.

Каждый сегмент содержит до `64` строк (~1 МБ payload) и не превышает
10 минут записи при любом интервале.

---

## Склейка сегментов

Для получения непрерывного файла за длительный период:

1. Забрать все сегменты (`GET /api/waterfall/segment?name=…`).
2. Отсортировать по `started_at` из заголовка.
3. Объединить payload в хронологическом порядке.
4. Взять метаданные (calibration, serial, interval_sec) из первого сегмента.
5. Пересчитать `saved_rows` = сумма строк всех сегментов.

Пример:

```python
import json, struct
from pathlib import Path

def merge_aswf(paths_sorted, out_path):
    HDR_RESERVE = 4096
    files  = [Path(p).read_bytes() for p in paths_sorted]
    first  = files[0]
    hlen   = struct.unpack_from("<I", first, 4)[0]
    hdr    = json.loads(first[8:8 + hlen].decode("utf-8"))
    stride = hdr.get("row_stride", hdr["channels"] * 2)

    # собрать payload
    payload = b""
    for buf in files:
        h2 = struct.unpack_from("<I", buf, 4)[0]
        payload += buf[8 + h2:]

    total_rows = len(payload) // stride
    hdr["saved_rows"] = total_rows
    hdr["saved_at"]   = 0  # неизвестно при ручной склейке

    hdr_bytes = json.dumps(hdr, ensure_ascii=False).encode("utf-8")
    hdr_bytes = hdr_bytes.ljust(HDR_RESERVE)  # добить пробелами

    with open(out_path, "wb") as f:
        f.write(b"ASWF")
        f.write(struct.pack("<I", HDR_RESERVE))
        f.write(hdr_bytes)
        f.write(payload)
```

---

## Ограничения и граничные случаи

| Ситуация | Поведение |
|----------|-----------|
| `saved_rows == 0` | Сегмент не финализирован. Вычислить строки по размеру файла. |
| `saved_at == 0`   | Время финализации неизвестно (открытый или ручная склейка). |
| `duration == 0`   | Реальная длительность неизвестна. Подставить `interval_sec`. |
| Неизвестные ключи JSON | Игнорировать (возможны расширения в будущих версиях). |
| `hlen` отличается от 4096 | Зарезервировано для будущего. Использовать фактический `hlen`. |
| Усечённый последний ряд | `len(payload) % stride != 0` → отбросить неполный хвост. |

---

## v3 — Самоописывающиеся строки (расширенный формат)

v3 заменяет фиксированную компоновку строки на полностью самоописывающийся массив
`row_fields` в заголовке. Каждая строка может нести произвольные типизированные поля:
спектр, длительность, метку времени, координаты GPS, мощность дозы и любые будущие
расширения — без нарушения обратной совместимости с парсерами, игнорирующими незнакомые поля.

### Структура файла (v3)

```
ASWF(4) + hlen(4) + JSON(hlen) + baseline_section(B) + payload(N строк)
```

`B = baseline.count × 4` (нулевое при отсутствии ключа `baseline` в заголовке).  
`payload_offset = 8 + hlen + B`.

### Поля заголовка v3

| Ключ           | Тип              | v3   | Описание |
|----------------|------------------|:----:|----------|
| `version`      | int              | да   | `3` |
| `row_fields`   | array of objects | да   | Дескрипторы полей — см. ниже |
| `row_stride`   | int              | нет* | Размер строки в байтах; **отсутствует** при `compressed=true` |
| `compressed`   | bool             | нет  | `true` → RLE-кодирование строк (по умолчанию `false`) |
| `baseline`     | object           | нет  | `{"count": N}` — N значений uint32 LE после JSON-шапки |

(*) Обязателен для несжатого v3; отсутствует для сжатого.

#### Дескриптор `row_fields`

```json
{
  "name":   "spectrum",
  "dtype":  "uint16",
  "offset": 0,
  "count":  8192
}
```

| Поле     | Описание |
|----------|----------|
| `name`   | `"spectrum"`, `"duration"`, `"timestamp"`, `"latitude"`, `"longitude"`, `"dose_rate_usv_h"` или любое будущее имя |
| `dtype`  | `"uint16"`, `"uint32"`, `"float32"` |
| `offset` | Смещение поля в байтах **внутри строки** (несжатый режим); при сжатом — поля следуют после RLE-спектра в порядке возрастания `offset` |
| `count`  | Число элементов; присутствует у `spectrum` (= `channels`); скалярные поля опускают или указывают `1` |

### Секция baseline

Если заголовок содержит `"baseline": {"count": N}`, после JSON-области вставляется
бинарный блок из `N × 4` байт uint32 LE. Это накопительный спектр, зафиксированный до
начала текущей сессии (используется для вычитания фона).

```
payload_offset = 8 + hlen + (baseline.count × 4)
```

### Несжатый v3

`row_stride` обязателен. Каждая строка — `row_stride` байт; поля расположены по своим `offset`.
Спектр занимает байты `[0 .. channels×2 − 1]`.

### RLE-сжатый v3 (`compressed: true`)

`row_stride` отсутствует. Каждая строка в payload кодируется:
1. **RLE-спектр** (переменной длины) — см. кодирование ниже.
2. **Не-спектральные поля** в порядке возрастания `offset`, каждое своей шириной dtype.

#### RLE-кодирование

Каждый элемент — uint16 LE:

| Значение             | Смысл |
|----------------------|-------|
| `v < 0x8000`         | Литеральное значение канала `v` |
| `0x8000 ≤ v < 0xFFFF` | `v & 0x7FFF` последовательных нулевых каналов |
| `0xFFFF`             | Зарезервировано — парсер обязан выбросить ошибку |

### Поддерживаемые имена полей `name` (v3)

| Имя                | dtype     | Описание |
|--------------------|-----------|----------|
| `spectrum`         | uint16    | Счёты за интервал, `channels` элементов |
| `duration`         | uint16    | Реальное живое время, секунды (0 → использовать `interval_sec`) |
| `timestamp`        | uint32    | Unix-метка этой строки (UTC, секунды от эпохи); заменяет кумсумму при > 0 |
| `latitude`         | float32   | GPS-широта (градусы); NaN при отсутствии сигнала |
| `longitude`        | float32   | GPS-долгота (градусы); NaN при отсутствии сигнала |
| `dose_rate_usv_h`  | float32   | Мощность дозы, мкЗв/ч; NaN при отсутствии данных |

### Пример заголовка v3 (несжатый)

```json
{
  "format":       "atomspectra-waterfall",
  "version":      3,
  "channels":     8192,
  "dtype":        "uint16",
  "byte_order":   "little",
  "interval_sec": 60,
  "started_at":   1783157403,
  "saved_rows":   128,
  "saved_at":     1783165103,
  "serial":       "AS-003",
  "calibration":  [0.0, 0.298, 0.0],
  "row_stride":   16406,
  "baseline":     {"count": 8192},
  "row_fields": [
    {"name": "spectrum",       "dtype": "uint16",  "offset": 0,     "count": 8192},
    {"name": "duration",       "dtype": "uint16",  "offset": 16384},
    {"name": "dose_rate_usv_h","dtype": "float32", "offset": 16386},
    {"name": "latitude",       "dtype": "float32", "offset": 16390},
    {"name": "longitude",      "dtype": "float32", "offset": 16394},
    {"name": "timestamp",      "dtype": "uint32",  "offset": 16398},
    {"name": "dose_rate_msvh", "dtype": "float32", "offset": 16402}
  ]
}
```

---

## История версий формата

| Версия | Изменения |
|--------|-----------|
| v1     | 16384 байт/строка. Нет поля длительности. `row_stride` отсутствует в заголовке. |
| v2     | +2 байта uint16 LE длительности в конце каждой строки. `row_stride=16386`, `row_time` в заголовке. |
| v3     | Самоописывающийся `row_fields`; опциональная секция `baseline`; опциональное RLE-сжатие; поля `timestamp`, GPS и `dose_rate_usv_h` на строку. |
