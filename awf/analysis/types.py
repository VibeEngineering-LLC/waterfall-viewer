"""Модуль общих типов результатов численного анализа waterfall-спектрограмм."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class FoundPeak:
    """Найденный фотопик (Задача 2)"""
    channel: float         # центр пика, дробный канал
    energy: float          # энергия центра, кэВ
    height: float          # высота над фоном, отсчёты
    fwhm_channels: float   # ширина на полувысоте, каналы
    significance: float    # значимость Currie, в сигмах
    area_estimate: float   # предварительная площадь ~2.507*sigma*height

@dataclass(frozen=True)
class PeakArea:
    """Интегральная площадь пика и нетто (Задача 6)"""
    net: float        # нетто-площадь (gross - baseline)
    d_net: float      # погрешность нетто, ~sqrt(gross + baseline)
    gross: float      # полный счёт в ROI
    baseline: float   # интеграл континуума под пиком
    roi_lo: int       # левая граница ROI (канал, включительно)
    roi_hi: int       # правая граница ROI (канал, исключительно)

@dataclass(frozen=True)
class MdaResult:
    """Пределы по ISO 11929 (Задача 7)"""
    l_c: float     # порог принятия решения, counts
    l_d: float     # предел обнаружения, counts
    a_mda: float   # минимально детектируемая активность, Бк

@dataclass(frozen=True)
class LineMatch:
    """Сопоставление пика с библиотечной линией (Задача 11)"""
    nuclide: str          # имя нуклида, напр. "Cs-137"
    line_energy: float    # табличная энергия линии, кэВ
    peak_energy: float    # энергия найденного пика, кэВ
    delta_keV: float      # peak_energy - line_energy
    intensity_pct: float  # табличная интенсивность линии, %

@dataclass(frozen=True)
class IdentResult:
    """Кандидат-нуклид с набором совпавших линий (Задача 11)"""
    nuclide: str
    confidence: float                                  # индекс уверенности [0..1]
    matches: Tuple[LineMatch, ...] = field(default_factory=tuple)
    category: Optional[str] = None                     # natural/technogenic/medical/fission (Задача 10)
