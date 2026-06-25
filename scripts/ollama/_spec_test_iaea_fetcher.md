# SPEC: tests/test_iaea_fetcher.py

Сгенерировать ТОЛЬКО Python-код pytest-файла (без markdown-ограждений, без пояснений).
Стиль — pytest-функции (без классов). БЕЗ РЕАЛЬНОЙ СЕТИ. Сеть мокается через monkeypatch.

## Импорты
```
import pytest
from awf.io.iaea_fetcher import (
    IaeaGammaLine, fetch_iaea_gamma_lines, load_iaea_gamma_lines_from_cache,
    merge_iaea_into_internal,
    _normalize_nuclide_name, _denormalize_nuclide_name, _cache_path, _parse_iaea_csv,
)
from pathlib import Path
```

## Синтетический CSV-хелпер
```
def _csv():
    return ("energy,unc_e,intensity,unc_i,decay,p_energy,parent\n"
            "63.29,0.02,4.8,0.5,B-,0,234TH\n"
            "92.38,0.01,2.81,0.2,B-,0,234TH\n"
            "1001.03,0.03,0.837,0.05,B-,0,234TH\n")
```

## Фейковый HTTP-ответ (для monkeypatch urlopen)
```
class _FakeResp:
    def __init__(self, data): self._data = data
    def read(self): return self._data
    def __enter__(self): return self
    def __exit__(self, *a): return False
```

## Тесты (каждый — отдельная функция):

1. `test_parse_csv`: `lines = _parse_iaea_csv(_csv())`; `len(lines) == 3`;
   `lines[0].energy_keV == 63.29`; `lines[0].intensity_pct == 4.8`;
   `lines[0].energy_uncertainty_keV == 0.02`; `lines[0].parent_nuclide == "234TH"`;
   `lines[2].energy_keV == 1001.03`.

2. `test_parse_skips_empty_energy`: CSV с пустой энергией в строке пропускается. Дать
   `"energy,intensity,parent\n,5.0,234TH\n100.0,3.0,234TH\n"` -> `len == 1`,
   единственная линия `energy_keV == 100.0`.

3. `test_normalize_ok`: `_normalize_nuclide_name` для `"Th-234"`, `"234Th"`, `"234TH"`,
   `"th234"` -> все `== "234th"`.

4. `test_normalize_fail_loud`: параметризовать
   `@pytest.mark.parametrize("bad", ["?evil=1", "../etc/passwd", "x"*33, "", "Th--/"])`;
   тело: `with pytest.raises(ValueError): _normalize_nuclide_name(bad)`.

5. `test_denormalize`: `_denormalize_nuclide_name("234TH") == "Th-234"`;
   `_denormalize_nuclide_name("234th") == "Th-234"`.

6. `test_cache_path`: `p = _cache_path(Path("/tmp/c"), "Th-234", "g")`;
   `p.name == "234th_g.csv"`.

7. `test_cache_roundtrip` (tmp_path): записать `_csv()` в файл
   `tmp_path / "234th_g.csv"` (utf-8); `lines = load_iaea_gamma_lines_from_cache("Th-234",
   cache_dir=tmp_path)`; `lines is not None`; `len(lines) == 3`.
   Затем `load_iaea_gamma_lines_from_cache("Cs-137", cache_dir=tmp_path) is None` (нет файла).

8. `test_fetch_uses_cache_no_network` (tmp_path, monkeypatch): записать `_csv()` в
   `tmp_path / "234th_g.csv"`; замокать `urllib.request.urlopen` так, чтобы при вызове падал
   (`def _boom(*a, **k): raise AssertionError("network called")`;
   `monkeypatch.setattr("urllib.request.urlopen", _boom)`);
   `lines = fetch_iaea_gamma_lines("Th-234", cache_dir=tmp_path)` (force_refresh=False);
   `len(lines) == 3` (взято из кэша, сеть не вызвана).

9. `test_fetch_writes_cache_atomically` (tmp_path, monkeypatch): замокать urlopen на
   `_FakeResp(b"energy,intensity,parent\n100.0,5.0,234TH\n")`:
   `monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None:
   _FakeResp(b"energy,intensity,parent\\n100.0,5.0,234TH\\n"))`;
   `lines = fetch_iaea_gamma_lines("Th-234", cache_dir=tmp_path, force_refresh=True)`;
   `len(lines) == 1` и `lines[0].energy_keV == 100.0`;
   файл `tmp_path / "234th_g.csv"` существует и его текст начинается с `"energy"`;
   временного файла `.tmp` не осталось (`not (tmp_path / "234th_g.csv.tmp").exists()`).

10. `test_merge_basic`: `lines = _parse_iaea_csv(_csv())`;
    `d = merge_iaea_into_internal(lines, "Th-234")`; `d` — dict с ключом `"lines"`;
    `len(d["lines"]) == 3`; отсортировано по энергии: `d["lines"][0][0] == 63.29`,
    `d["lines"][2][0] == 1001.03`; каждый элемент — `[E, I, dI]` длиной 3.

11. `test_merge_min_intensity`: `d = merge_iaea_into_internal(_parse_iaea_csv(_csv()),
    "Th-234", min_intensity_pct=1.0)`; линия 0.837% отфильтрована -> `len(d["lines"]) == 2`;
    минимальная оставшаяся интенсивность >= 1.0.

12. `test_merge_excited_parent_excluded`: построить вручную список
    `[IaeaGammaLine(50.0, None, 10.0, None, "234TH", "B-", 0.0),
      IaeaGammaLine(60.0, None, 20.0, None, "234TH", "B-", 74.0)]`;
    `d = merge_iaea_into_internal(lines, "Th-234", only_ground_state_parent=True)`;
    линия с parent_energy_keV=74.0 исключена -> `len(d["lines"]) == 1`,
    единственная `d["lines"][0][0] == 50.0`.

Использовать только указанный API. Код должен проходить pytest без сети.
