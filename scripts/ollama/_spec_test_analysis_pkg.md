# Спецификация: tests/test_analysis_pkg.py

Pytest-тесты для каркаса пакета `awf/analysis/` (Задача 1).
Проверяют: импорт публичного API, конструирование frozen-типов, и хелпер
`spectrum_from_selection`, который должен поканально совпадать с `Spectrogram.sum_spectrum`.
Стиль — как в `tests/test_data_layer.py` (отдельные функции-тесты, numpy, pytest, фикс. seed).

## Импорты (ровно эти)
```python
from __future__ import annotations
import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from awf.model.spectrogram import Calibration, Spectrogram
from awf.analysis import (
    FoundPeak, PeakArea, MdaResult, LineMatch, IdentResult,
    spectrum_from_selection,
)
```

## Helper: синтетическая спектрограмма
Функция `_make_sg(n_slices: int = 4, n_channels: int = 5) -> Spectrogram`:
1. `counts = np.arange(n_slices * n_channels, dtype=np.int64).reshape(n_slices, n_channels)`.
2. `cal = Calibration(coeffs=[0.0, 1.0])`  (E(ch)=ch).
3. `t = np.arange(n_slices, dtype=np.float64) * 10.0`.
4. `rt = np.full(n_slices, 10.0)`; `lt = np.full(n_slices, 10.0)`.
5. Вернуть `Spectrogram(counts=counts, calibration=cal, time_offsets_s=t, real_time_s=rt, live_time_s=lt)`.

## Тесты (каждый — отдельная функция)

1. `test_public_api_importable()`: проверить, что `FoundPeak, PeakArea, MdaResult, LineMatch,
   IdentResult` — это типы (`isinstance(X, type)`), а `spectrum_from_selection` — вызываемый
   (`callable(spectrum_from_selection)`).

2. `test_found_peak_fields()`: создать
   `p = FoundPeak(channel=10.0, energy=661.7, height=500.0, fwhm_channels=3.0, significance=8.0, area_estimate=1250.0)`;
   проверить `p.channel == 10.0`, `p.energy == pytest.approx(661.7)`, `p.significance == 8.0`.

3. `test_peak_area_fields()`: создать
   `a = PeakArea(net=900.0, d_net=40.0, gross=1100.0, baseline=200.0, roi_lo=8, roi_hi=14)`;
   проверить `a.net == 900.0`, `a.gross - a.baseline == pytest.approx(a.net)`, `a.roi_hi > a.roi_lo`.

4. `test_mda_result_fields()`: создать `m = MdaResult(l_c=12.0, l_d=27.0, a_mda=3.5)`;
   проверить `m.l_c == 12.0 and m.l_d == 27.0 and m.a_mda == pytest.approx(3.5)`.

5. `test_ident_result_defaults()`: создать `r = IdentResult(nuclide="K-40", confidence=0.9)`;
   проверить `r.matches == ()` (дефолт — пустой tuple) и `r.category is None`;
   затем `lm = LineMatch(nuclide="K-40", line_energy=1460.8, peak_energy=1461.0, delta_keV=0.2,
   intensity_pct=10.66)`; создать `r2 = IdentResult(nuclide="K-40", confidence=0.9, matches=(lm,),
   category="natural")`; проверить `r2.matches[0].nuclide == "K-40"` и `r2.category == "natural"`.

6. `test_types_are_frozen()`: создать `m = MdaResult(l_c=1.0, l_d=2.0, a_mda=3.0)`;
   проверить, что мутация поля бросает `FrozenInstanceError`:
   `with pytest.raises(FrozenInstanceError): m.l_c = 5.0`.

7. `test_spectrum_from_selection_matches_sum()`: `sg = _make_sg()`; для окон
   `windows = [(None, None), (0, 2), (1, 4), (2, 3)]`:
   - `got = spectrum_from_selection(sg, lo, hi)`;
   - `assert got.dtype == np.float64`;
   - `assert np.array_equal(got, sg.sum_spectrum(lo, hi).astype(np.float64))`.

8. `test_spectrum_from_selection_full_equals_total()`: `sg = _make_sg()`;
   `assert np.array_equal(spectrum_from_selection(sg), sg.total_spectrum().astype(np.float64))`.

## Требования
- Только перечисленные импорты + helper `_make_sg`.
- Не использовать print; не обращаться к сети/файлам — только синтетика в памяти.
- Комментарии кратко по-русски, где нужно.
- Вернуть ТОЛЬКО код модуля, без markdown-ограждений.
