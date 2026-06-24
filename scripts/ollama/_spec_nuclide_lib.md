# Спецификация: awf/io/nuclide_lib.py

Модуль работы с библиотекой гамма-нуклидов. Парсит формат LSRM SpectraLine `.lib`
(windows-1251 XML, десятичный разделитель — ЗАПЯТАЯ) и загружает наш собственный JSON.
Чистый Python + stdlib только (`xml.etree.ElementTree`, `json`, `dataclasses`, `pathlib`,
`typing`). НИКАКИХ внешних зависимостей (без numpy, без lxml).

## Формат входного .lib (факты, проверены на реальном файле)

```xml
<?xml version="1.0" encoding="windows-1251"?>
<!--SpectraLine Nuclear Data Information file ... -->
<Library library_type="gamma" library_version="2.0" database_version="26">
  <Comment></Comment>
  <Nuclide name="Am-241" half_life_value="432,6" half_life_unit="year" gamma_constant="0,118348" atomic_mass="241">
    <Line energy="59,5409" d_energy="0,0001" intensity="35,9" d_intensity="0,4"/>
  </Nuclide>
  <Nuclide name="Bi-207" half_life_value="31,55" half_life_unit="year" gamma_constant="15,122747" atomic_mass="207">
    <Line energy="74,969" d_energy="0,0009" intensity="36,8" d_intensity="1,1" line_type="X" used="false"/>
    <Line energy="569,698" d_energy="0,002" intensity="97,75" d_intensity="0,03"/>
  </Nuclide>
</Library>
```

Ключевые факты:
- Все числа — десятичная ЗАПЯТАЯ: `"432,6"` означает 432.6, `"59,5409"` означает 59.5409.
  Некоторые без дробной части и без запятой: `half_life_value="1248000000"` (K-40), `atomic_mass="241"`.
- Атрибуты `<Nuclide>`: `name` (str, обяз.), `half_life_value` (число), `half_life_unit`
  (str: "year"/"day"), `gamma_constant` (число, может отсутствовать), `atomic_mass` (целое, может отсутствовать).
- Атрибуты `<Line>`: `energy` (число, кэВ, обяз.), `intensity` (число, %, обяз.),
  `d_energy` (число, может отсутствовать), `d_intensity` (число, может отсутствовать),
  `line_type` (str, опц., напр. "X"), `used` (str "false" -> линия НЕ используется по умолчанию; отсутствует -> используется).
- Возможен пустой `<Comment></Comment>` — игнорировать.

## Требуемый API

### `_cfloat(s)` -> float | None
Распарсить строку-число с запятой-разделителем. `None` или пустая строка -> `None`.
Реализация: если `s is None`, вернуть None; заменить запятую на точку; `float(...)`.

### `_cint(s)` -> int | None
Целое или None. `None`/пусто -> None; иначе `int(float(s.replace(",", ".")))`.

### `@dataclass(frozen=True) GammaLine`
Поля:
- `energy: float`        — энергия линии, кэВ
- `intensity: float`     — интенсивность (выход), %
- `d_energy: float | None = None`
- `d_intensity: float | None = None`
- `line_type: str | None = None`
- `used: bool = True`

### `@dataclass(frozen=True) Nuclide`
Поля:
- `name: str`
- `half_life_value: float | None = None`
- `half_life_unit: str | None = None`
- `gamma_constant: float | None = None`
- `atomic_mass: int | None = None`
- `lines: tuple[GammaLine, ...] = ()`

Методы:
- `half_life_seconds(self) -> float | None`
  Перевести период полураспада в секунды. `unit`=="year" -> *365.25*86400; "day" -> *86400;
  "hour" -> *3600; "minute" -> *60; "second" -> *1. Если value None или unit неизвестен -> None.
- `major_lines(self, min_intensity: float = 0.0, only_used: bool = True) -> list[GammaLine]`
  Вернуть линии с `intensity >= min_intensity`, при `only_used=True` — только те, у кого `used` True;
  отсортированные по `intensity` по УБЫВАНИЮ. Не мутирует `self.lines`.

### `parse_lsrm_lib(path) -> list[Nuclide]`
Прочитать файл как windows-1251, распарсить через ElementTree, вернуть список `Nuclide`
в порядке появления. Для каждого `<Nuclide>`:
- name из атрибута;
- half_life_value через `_cfloat`, half_life_unit как есть;
- gamma_constant через `_cfloat`, atomic_mass через `_cint`;
- lines: по каждому дочернему `<Line>` -> `GammaLine(energy=_cfloat(...), intensity=_cfloat(...),
  d_energy=_cfloat(...), d_intensity=_cfloat(...), line_type=attr.get("line_type"),
  used=(attr.get("used","").lower() != "false"))`. Пропускать `<Line>` без energy или без intensity
  (если `_cfloat` дал None) — НЕ добавлять такую линию.
Чтение файла: `Path(path).read_bytes().decode("windows-1251")`, затем `ET.fromstring(text)`.

### `to_json_obj(nuclides, provenance: dict | None = None) -> dict`
Сериализовать список `Nuclide` в dict для нашего nuclides.json:
```python
{"_provenance": provenance or {}, "nuclides": [ {...}, ... ]}
```
каждый нуклид:
```python
{"name": ..., "half_life_value": ..., "half_life_unit": ..., "gamma_constant": ...,
 "atomic_mass": ..., "lines": [
    {"energy": ..., "intensity": ..., "d_energy": ..., "d_intensity": ...,
     "line_type": ..., "used": ...}, ... ]}
```
Значения None сохранять как None (-> JSON null). НЕ округлять числа.

### `load_nuclides_json(path) -> list[Nuclide]`
Загрузить наш JSON (utf-8) формата `to_json_obj`. Вернуть список `Nuclide`.
Принимать как объект с ключом "nuclides", так и (на всякий случай) голый список нуклидов.
Для каждой линии собирать `GammaLine` из ключей (отсутствующие -> дефолты dataclass).
`used` брать как есть (bool), при отсутствии -> True.

### `default_library() -> list[Nuclide]`
Загрузить встроенный `awf/data/nuclides.json` относительно модуля:
`Path(__file__).resolve().parent.parent / "data" / "nuclides.json"`.
Если файла нет -> вернуть пустой список `[]` (НЕ кидать исключение).

## Стиль
- `from __future__ import annotations` первой строкой.
- Комментарии и docstring — кратко по-русски.
- Без сторонних импортов. Без побочных эффектов на уровне модуля.
- Код должен импортироваться и работать на Python 3.12+.
