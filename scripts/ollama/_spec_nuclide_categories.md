# SPEC: awf/io/nuclide_categories.py

Сгенерировать ТОЛЬКО Python-код модуля (без markdown-ограждений, без пояснений).
Чистый Python 3.14, без Qt, без сети. Зависимости: stdlib `json`, `pathlib`,
`dataclasses` (функция `replace`), и `from awf.io.nuclide_lib import Nuclide`.

Назначение: категоризация нуклидов по происхождению (origin) и времени жизни (lifetime),
плюс хелперы выборки. Карта категорий лежит в JSON-файле рядом с `nuclides.json`.

## Шапка модуля
- `from __future__ import annotations`
- docstring модуля (1-2 строки на русском): «Категоризация нуклидов (Задача 10): origin из
  awf/data/nuclide_categories.json + lifetime по порогу T½. Qt-free.»
- импорты: `import json`; `from dataclasses import replace`; `from pathlib import Path`;
  `from typing import List, Optional`; `from awf.io.nuclide_lib import Nuclide`

## Константы
- `CATEGORIES = ("natural", "technogenic", "medical", "fission")`
- `LIFETIMES = ("short", "long")`
- `_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nuclide_categories.json"`
- `_DEFAULT_THRESHOLD_DAYS = 60.0`

## Кэш карты категорий
- Модульная переменная `_MAP_CACHE = None` (dict | None).
- Функция `_load_map() -> dict`:
  - если `_MAP_CACHE` не None — вернуть его;
  - иначе прочитать `_DATA_PATH` (utf-8) как JSON; взять `data.get("categories", {})` (dict
    имя->category) и `data.get("long_lifetime_threshold_days", _DEFAULT_THRESHOLD_DAYS)`;
  - сохранить в модульные переменные `_MAP_CACHE` (dict категорий) и
    `_THRESHOLD_DAYS_CACHE`; вернуть `_MAP_CACHE`;
  - при ЛЮБОМ исключении чтения/парсинга — вернуть пустой dict `{}` и threshold по умолчанию
    (fail-soft: отсутствие файла не должно ронять загрузку библиотеки). Использовать
    глобальные переменные через `global`.
- Функция `threshold_days() -> float`: гарантировать вызов `_load_map()`, вернуть
  `_THRESHOLD_DAYS_CACHE` (или `_DEFAULT_THRESHOLD_DAYS`).
- Функция `threshold_seconds() -> float`: `threshold_days() * 86400.0`.

## category_of
`def category_of(name: str) -> Optional[str]:`
- вернуть `_load_map().get(name)` (точное совпадение по имени, напр. "Cs-137");
- если имя не в карте — вернуть None (НЕ выдумывать категорию).

## classify_lifetime_seconds
`def classify_lifetime_seconds(half_life_s: Optional[float]) -> Optional[str]:`
- если `half_life_s` is None — вернуть None;
- `"long"` если `half_life_s >= threshold_seconds()`, иначе `"short"`.

## classify_lifetime
`def classify_lifetime(nuclide: Nuclide) -> Optional[str]:`
- вернуть `classify_lifetime_seconds(nuclide.half_life_seconds())`.

## enrich_nuclide
`def enrich_nuclide(nuclide: Nuclide) -> Nuclide:`
- вычислить `cat = category_of(nuclide.name)` и `lt = classify_lifetime(nuclide)`;
- вернуть `replace(nuclide, category=cat, lifetime=lt)` (frozen dataclass — только через replace).

## enrich_all
`def enrich_all(nuclides) -> List[Nuclide]:`
- вернуть `[enrich_nuclide(n) for n in nuclides]`.

## nuclides_by_category
`def nuclides_by_category(nuclides, category: str) -> List[Nuclide]:`
- вернуть список тех `n`, у кого `n.category == category`.

## nuclides_by_lifetime
`def nuclides_by_lifetime(nuclides, lifetime: str) -> List[Nuclide]:`
- вернуть список тех `n`, у кого `n.lifetime == lifetime`.

## __all__
Перечислить все публичные имена: CATEGORIES, LIFETIMES, threshold_days, threshold_seconds,
category_of, classify_lifetime_seconds, classify_lifetime, enrich_nuclide, enrich_all,
nuclides_by_category, nuclides_by_lifetime.

Код должен компилироваться (py_compile) и быть идемпотентным при повторном импорте.
