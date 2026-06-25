# SPEC: tests/test_nuclide_categories.py

Сгенерировать ТОЛЬКО Python-код pytest-файла (без markdown-ограждений, без пояснений).
Стиль — как tests/test_nuclide_lib.py (pytest-функции, без классов). Без сети, без Qt.

## Импорты
- `import pytest`
- `from awf.io.nuclide_lib import Nuclide, GammaLine, default_library`
- `from awf.io.nuclide_categories import (category_of, classify_lifetime_seconds,
  classify_lifetime, enrich_nuclide, enrich_all, nuclides_by_category,
  nuclides_by_lifetime, threshold_seconds, threshold_days, CATEGORIES, LIFETIMES)`

## Хелпер
`def _nuc(name, hv=None, hu=None):` — вернуть `Nuclide(name=name, half_life_value=hv,
half_life_unit=hu, lines=(GammaLine(100.0, 50.0),))`.

## Тесты (каждый — отдельная функция):

1. `test_categories_constant`: `CATEGORIES == ("natural", "technogenic", "medical", "fission")`;
   `LIFETIMES == ("short", "long")`.

2. `test_category_of_known`: `category_of("Cs-137") == "fission"`;
   `category_of("K-40") == "natural"`; `category_of("Co-60") == "technogenic"`;
   `category_of("Tc-99m") == "medical"`.

3. `test_category_of_unknown`: `category_of("Zz-999") is None` (неизвестный -> None).

4. `test_threshold`: `threshold_days() == 60` (или `== pytest.approx(60.0)`);
   `threshold_seconds() == pytest.approx(60 * 86400.0)`.

5. `test_classify_lifetime_seconds`: ниже порога -> "short", на пороге и выше -> "long",
   None -> None. Использовать `thr = threshold_seconds()`:
   `classify_lifetime_seconds(thr - 1.0) == "short"`;
   `classify_lifetime_seconds(thr) == "long"`;
   `classify_lifetime_seconds(thr + 1.0) == "long"`;
   `classify_lifetime_seconds(None) is None`.

6. `test_classify_lifetime_nuclide`: `classify_lifetime(_nuc("Cs-137", 30.0, "year")) == "long"`;
   `classify_lifetime(_nuc("I-131", 8.0, "day")) == "short"` (8 сут < 60 сут);
   `classify_lifetime(_nuc("X")) is None` (нет T½).

7. `test_enrich_nuclide`: `e = enrich_nuclide(_nuc("Cs-137", 30.0, "year"))`;
   `e.category == "fission"`; `e.lifetime == "long"`;
   исходный nuclide не мутирован (frozen) — проверить, что `e` это другой объект с
   проставленными полями (например `e.name == "Cs-137"` и `e.lines` сохранены).

8. `test_enrich_all_and_filters`: построить список
   `nucs = enrich_all([_nuc("Cs-137",30.0,"year"), _nuc("K-40",1.25e9,"year"),
   _nuc("I-131",8.0,"day"), _nuc("Zz-999",1.0,"day")])`;
   - `nuclides_by_category(nucs, "fission")` содержит Cs-137 и I-131 (по name), не содержит K-40;
   - `nuclides_by_category(nucs, "natural")` содержит K-40;
   - `nuclides_by_lifetime(nucs, "short")` содержит I-131 (8 сут) и Zz-999 (1 сут);
   - `nuclides_by_lifetime(nucs, "long")` содержит Cs-137 и K-40;
   - Zz-999 имеет `category is None` (неизвестный).
   Проверки делать по множеству имён, например:
   `names = lambda lst: {n.name for n in lst}` и сравнивать `>=`/`==` с ожидаемыми множествами.

9. `test_default_library_enrichable`: `lib = enrich_all(default_library())`;
   если библиотека непуста — у Cs-137 (найти по name) `category == "fission"`.
   Если в библиотеке нет Cs-137 — тест просто проверяет, что enrich_all не падает и
   возвращает список той же длины, что и default_library().

Использовать только публичный API. Код должен проходить pytest.
