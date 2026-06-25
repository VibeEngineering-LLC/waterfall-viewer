# Спецификация: awf/analysis/types.py

Модуль общих типов результатов численного анализа waterfall-спектрограмм.
Это КАРКАС (Задача 1): только структуры данных, БЕЗ алгоритмов и методов.
Все структуры — `@dataclass(frozen=True)`. Поля и порядок — строго как ниже, ничего не добавлять.

Единицы: энергия — кэВ; канал — дробный (float, субканальная точность); счёт/площадь — float; активность — Бк.

## Импорты (ровно эти, ничего лишнего)
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
```

## Классы (точные поля, в этом порядке, с краткими русскими комментариями)

### FoundPeak — найденный фотопик (Задача 2)
- `channel: float`         — центр пика, дробный канал
- `energy: float`          — энергия центра, кэВ
- `height: float`          — высота над фоном, отсчёты
- `fwhm_channels: float`   — ширина на полувысоте, каналы
- `significance: float`    — значимость Currie, в сигмах
- `area_estimate: float`   — предварительная площадь ~2.507*sigma*height

### PeakArea — интегральная площадь пика и нетто (Задача 6)
- `net: float`        — нетто-площадь (gross - baseline)
- `d_net: float`      — погрешность нетто, ~sqrt(gross + baseline)
- `gross: float`      — полный счёт в ROI
- `baseline: float`   — интеграл континуума под пиком
- `roi_lo: int`       — левая граница ROI (канал, включительно)
- `roi_hi: int`       — правая граница ROI (канал, исключительно)

### MdaResult — пределы по ISO 11929 (Задача 7)
- `l_c: float`     — порог принятия решения, counts
- `l_d: float`     — предел обнаружения, counts
- `a_mda: float`   — минимально детектируемая активность, Бк

### LineMatch — сопоставление пика с библиотечной линией (Задача 11)
- `nuclide: str`          — имя нуклида, напр. "Cs-137"
- `line_energy: float`    — табличная энергия линии, кэВ
- `peak_energy: float`    — энергия найденного пика, кэВ
- `delta_keV: float`      — peak_energy - line_energy
- `intensity_pct: float`  — табличная интенсивность линии, %

### IdentResult — кандидат-нуклид с набором совпавших линий (Задача 11)
- `nuclide: str`
- `confidence: float`                                  — индекс уверенности [0..1]
- `matches: Tuple[LineMatch, ...] = field(default_factory=tuple)`
- `category: Optional[str] = None`                     — natural/technogenic/medical/fission (Задача 10)

## Требования
- Модульный docstring одной строкой по-русски, описывающий назначение файла.
- Каждый класс — `@dataclass(frozen=True)` с docstring-строкой по-русски.
- НЕ добавлять методов, валидации, `__post_init__`, свойств — только поля.
- Имена полей/классов — ASCII, ровно как указано. Комментарии — по-русски.
- Поля без значения по умолчанию идут перед полями со значением по умолчанию (требование dataclass).
- Вернуть ТОЛЬКО код модуля, без markdown-ограждений.
