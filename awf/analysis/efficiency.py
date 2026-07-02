"""
Задача #156: нормализация водопада по эффективности регистрации ε(E).

Эффективность фотопика падает с энергией на порядок и более, поэтому линии высоких
энергий (K-40 1461, Tl-208 2615 кэВ) на сыром водопаде выглядят слабее реальной
эмиссии. Нормализация умножает отсчёты канала на ε_ref/ε(E): в максимуме кривой 1.0,
дальше растёт — рельеф показывает интенсивность излучения, а не отклик детектора.

Кривая по умолчанию — измеренная кривая Гамма-1С (Gamma-1S-NB1, NaI(Tl) 63×63 мм,
точечная геометрия 15 см) из проекта SpectraVibe: LSRM .efr, источники Eu-152 /
Na-22 / Ba-133 / Cs-137 / Am-241, 16 точек 59.5–1408 кэВ. Интерполяция — линейная
в log-log (как relative_efficiency, #130), за краями — плоская экстраполяция.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from awf.model.spectrogram import Spectrogram

# Измеренные точки Гамма-1С (E_кэВ, ε_абс): Gamma-1S-NB1, Point-15cm (LSRM .efr).
_GAMMA1S_POINTS = (
    (59.537, 1.579658e-03),    # Am-241
    (79.614, 1.507212e-03),    # Ba-133
    (80.997, 1.528032e-03),    # Ba-133
    (121.782, 1.8015e-03),     # Eu-152
    (244.697, 1.323961e-03),   # Eu-152
    (276.402, 1.275997e-03),   # Ba-133
    (302.852, 1.297139e-03),   # Ba-133
    (344.279, 1.049628e-03),   # Eu-152
    (356.014, 1.022029e-03),   # Ba-133
    (383.852, 8.316944e-04),   # Ba-133
    (511.0, 7.501414e-04),     # Na-22
    (661.66, 4.8809e-04),      # Cs-137
    (778.904, 3.172642e-04),   # Eu-152
    (1112.069, 2.203497e-04),  # Eu-152
    (1274.543, 2.017937e-04),  # Na-22
    (1408.006, 1.674492e-04),  # Eu-152
)


@dataclass(frozen=True)
class EfficiencyCurve:
    """Кривая эффективности регистрации: точки (E_кэВ, ε), сортированные по энергии."""
    name: str
    energies: tuple
    values: tuple

    def efficiency(self, E_keV):
        """ε(E): лог-лог интерполяция по точкам кривой; за краями диапазона — плоская
        экстраполяция краевыми значениями (поведение np.interp). Скаляр или массив."""
        e = np.asarray(E_keV, dtype=np.float64)
        xs = np.log(np.asarray(self.energies, dtype=np.float64))
        ys = np.log(np.asarray(self.values, dtype=np.float64))
        out = np.exp(np.interp(np.log(np.maximum(e, 1e-12)), xs, ys))
        return float(out) if out.ndim == 0 else out

    def factors(self, energies) -> np.ndarray:
        """Множители нормализации ε_ref/ε(E), ε_ref = максимум кривой: в максимуме
        кривой 1.0, дальше растут. Отсчёты канала умножаются на этот множитель."""
        eps = self.efficiency(np.asarray(energies, dtype=np.float64))
        return float(max(self.values)) / np.maximum(eps, 1e-30)


def default_gamma1s() -> EfficiencyCurve:
    """Кривая Гамма-1С по умолчанию (см. докстринг модуля — провенанс точек)."""
    return EfficiencyCurve(name="Гамма-1С (NaI 63×63, 15 см)",
                           energies=tuple(p[0] for p in _GAMMA1S_POINTS),
                           values=tuple(p[1] for p in _GAMMA1S_POINTS))


# Строка энергии LSRM .efr: «E=eps,d_eps_pct,нуклид,S,dS,I_pct» (ключ — число).
_EFR_ENERGY_LINE = re.compile(r"^\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*=\s*(.+)$")


def _parse_lsrm(text: str) -> list:
    """Точки (E, ε) из LSRM .efr/.efa: метаданные (Detector=…) не совпадают с
    числовым ключом регэкспа и пропускаются; первое поле значения — ε."""
    pts = []
    for line in text.splitlines():
        m = _EFR_ENERGY_LINE.match(line)
        if not m:
            continue
        parts = [p.strip() for p in m.group(2).split(",")]
        try:
            e, eps = float(m.group(1)), float(parts[0])
        except ValueError:
            continue
        if e > 0.0 and eps > 0.0:
            pts.append((e, eps))
    return pts


def _parse_columns(text: str) -> list:
    """Точки (E, ε) из двухколоночного текста: разделители запятая/;/пробел/таб,
    строки-комментарии (#, //) и нечисловые строки пропускаются."""
    pts = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        cols = [c for c in re.split(r"[,;\s]+", s) if c]
        if len(cols) < 2:
            continue
        try:
            e, eps = float(cols[0]), float(cols[1])
        except ValueError:
            continue
        if e > 0.0 and eps > 0.0:
            pts.append((e, eps))
    return pts


def _parse_json(text: str) -> list:
    """Точки (E, ε) из JSON: [[E, eps], …] либо {"points": [[E, eps], …]}."""
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("points", [])
    pts = []
    for row in data:
        e, eps = float(row[0]), float(row[1])
        if e > 0.0 and eps > 0.0:
            pts.append((e, eps))
    return pts


def load_efficiency_curve(path) -> EfficiencyCurve:
    """Кривая из файла: .efr/.efa (LSRM, cp1251), .json, иначе — две колонки E ε.
    Дубли энергий усредняются; <2 валидных точек → ValueError."""
    p = Path(path)
    raw = p.read_bytes()
    suffix = p.suffix.lower()
    if suffix in (".efr", ".efa"):
        try:
            text = raw.decode("cp1251")
        except UnicodeDecodeError:
            text = raw.decode("latin1")
        pts = _parse_lsrm(text)
    elif suffix == ".json":
        pts = _parse_json(raw.decode("utf-8"))
    else:
        pts = _parse_columns(raw.decode("utf-8", errors="replace"))
    if len(pts) < 2:
        raise ValueError(f"в файле кривой эффективности меньше 2 точек: {p.name}")
    grouped: dict = {}
    for e, eps in pts:
        grouped.setdefault(e, []).append(eps)
    es = sorted(grouped)
    vs = [sum(grouped[e]) / len(grouped[e]) for e in es]
    return EfficiencyCurve(name=p.stem, energies=tuple(es), values=tuple(vs))


def apply_efficiency(sg: Spectrogram, curve: EfficiencyCurve) -> Spectrogram:
    """Новая спектрограмма: отсчёты каждого канала × ε_ref/ε(E_канала).
    Калибровка/времена/метаданные исходной сохраняются."""
    f = curve.factors(np.asarray(sg.energies(), dtype=np.float64))
    scaled = np.asarray(sg.counts, dtype=np.float64) * f[None, :]
    return Spectrogram(counts=scaled, calibration=sg.calibration,
                       time_offsets_s=sg.time_offsets_s, real_time_s=sg.real_time_s,
                       live_time_s=sg.live_time_s, t0_iso=sg.t0_iso,
                       source_path=sg.source_path)
