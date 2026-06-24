# Спецификация: tests/test_nuclide_lib.py

pytest-тесты для `awf/io/nuclide_lib.py`. Только stdlib + pytest. Без numpy.

Импорты:
```python
from awf.io.nuclide_lib import (
    _cfloat, _cint, GammaLine, Nuclide,
    parse_lsrm_lib, to_json_obj, load_nuclides_json, default_library,
)
```

## Синтетический .lib (фикстура)

Хелпер `_write_lib(tmp_path) -> str`: пишет ASCII-XML в файл `lib.lib` кодировкой
windows-1251 (`path.write_bytes(text.encode("windows-1251"))`) и возвращает str(path).
Содержимое (запятая = десятичный разделитель, как в реальном LSRM):

```xml
<?xml version="1.0" encoding="windows-1251"?>
<Library library_type="gamma" library_version="2.0" database_version="26">
  <Comment></Comment>
  <Nuclide name="Am-241" half_life_value="432,6" half_life_unit="year" gamma_constant="0,118348" atomic_mass="241">
    <Line energy="59,5409" d_energy="0,0001" intensity="35,9" d_intensity="0,4"/>
  </Nuclide>
  <Nuclide name="Test-2" half_life_value="10" half_life_unit="day">
    <Line energy="100,0" intensity="50,0" line_type="X" used="false"/>
    <Line energy="200,0" intensity="90,0"/>
    <Line energy="300,0" intensity="5,0"/>
    <Line d_energy="0,1" intensity="1,0"/>
  </Nuclide>
</Library>
```
Заметь: у `Test-2` нет gamma_constant/atomic_mass; одна линия имеет used="false" и line_type="X";
последняя линия БЕЗ energy (должна быть пропущена парсером).

## Тесты

1. `test_cfloat`: `_cfloat("432,6")==432.6`; `_cfloat("1248000000")==1248000000.0`;
   `_cfloat(None) is None`; `_cfloat("") is None`.
2. `test_cint`: `_cint("241")==241`; `_cint(None) is None`; `_cint("") is None`.
3. `test_parse_counts(tmp_path)`: `parse_lsrm_lib(_write_lib(tmp_path))` -> 2 нуклида;
   у Am-241 1 линия; у Test-2 — 3 линии (4-я без energy пропущена).
4. `test_parse_values(tmp_path)`: Am-241: half_life_value==432.6, half_life_unit=="year",
   gamma_constant==0.118348, atomic_mass==241; первая линия energy==59.5409, intensity==35.9,
   d_energy==0.0001, d_intensity==0.4.
5. `test_used_flag(tmp_path)`: у Test-2 линия energy==100.0 имеет `used is False`;
   линия energy==200.0 имеет `used is True`.
6. `test_line_type(tmp_path)`: линия energy==100.0 -> line_type=="X"; линия energy==200.0 -> line_type is None.
7. `test_missing_optional(tmp_path)`: у Test-2 gamma_constant is None и atomic_mass is None.
8. `test_half_life_seconds`: `Nuclide("X", 1.0, "year").half_life_seconds() == 365.25*86400`;
   `Nuclide("X", 2.0, "day").half_life_seconds() == 2*86400`;
   `Nuclide("X", None, "year").half_life_seconds() is None`;
   `Nuclide("X", 1.0, None).half_life_seconds() is None`.
9. `test_major_lines`:
   ```python
   n = Nuclide("X", lines=(
       GammaLine(100.0, 50.0, used=False),
       GammaLine(200.0, 90.0),
       GammaLine(300.0, 5.0),
   ))
   ml = n.major_lines(only_used=True)
   assert [l.energy for l in ml] == [200.0, 300.0]   # used=False отброшена, сорт по интенсивности убыв.
   ml2 = n.major_lines(min_intensity=10.0, only_used=False)
   assert [l.energy for l in ml2] == [200.0, 100.0]  # >=10%, сорт убыв.
   ```
10. `test_json_roundtrip(tmp_path)`: `src = parse_lsrm_lib(_write_lib(tmp_path))`;
    `obj = to_json_obj(src, provenance={"k":"v"})`; записать obj в файл json (utf-8) и
    `back = load_nuclides_json(jsonpath)`; сверить: имена, число линий, и для каждой линии
    energy/intensity/used/line_type совпадают; `obj["_provenance"]=={"k":"v"}`;
    в obj есть ключ "nuclides" длиной 2.
11. `test_default_library_present`: `lib = default_library()`; `len(lib) >= 1`; среди имён есть
    "Cs-137"; у Cs-137 первая линия energy примерно 661.657 (`abs(e-661.657) < 0.01`).
    (Это регресс-гард на встроенный awf/data/nuclides.json.)

Все тесты — обычные функции `def test_...():`, ASCII-имена, комментарии кратко по-русски.
Используй `tmp_path` фикстуру pytest где нужен файл.
