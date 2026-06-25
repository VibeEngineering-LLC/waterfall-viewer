from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple


def _cfloat(s: str | None) -> float | None:
    """Распарсить строку-число с запятой-разделителем."""
    if s is None or s == "":
        return None
    return float(s.replace(",", "."))


def _cint(s: str | None) -> int | None:
    """Целое или None."""
    if s is None or s == "":
        return None
    return int(float(s.replace(",", ".")))


@dataclass(frozen=True)
class GammaLine:
    """Гамма-линия нуклида."""
    energy: float
    intensity: float
    d_energy: float | None = None
    d_intensity: float | None = None
    line_type: str | None = None
    used: bool = True


@dataclass(frozen=True)
class Nuclide:
    """Нуклид с гамма-линиями."""
    name: str
    half_life_value: float | None = None
    half_life_unit: str | None = None
    gamma_constant: float | None = None
    atomic_mass: int | None = None
    lines: Tuple[GammaLine, ...] = ()
    category: str | None = None     # natural/technogenic/medical/fission (Задача 10)
    lifetime: str | None = None     # short/long по T½-порогу (Задача 10)

    def half_life_seconds(self) -> float | None:
        """Перевести период полураспада в секунды."""
        if self.half_life_value is None or self.half_life_unit is None:
            return None
        unit_to_sec = {
            "year": 365.25 * 86400,
            "day": 86400,
            "hour": 3600,
            "minute": 60,
            "second": 1,
        }
        return self.half_life_value * unit_to_sec.get(self.half_life_unit, 1.0)

    def major_lines(self, min_intensity: float = 0.0, only_used: bool = True) -> List[GammaLine]:
        """Вернуть линии с заданной интенсивностью и фильтром по использованию."""
        lines = [line for line in self.lines if line.intensity >= min_intensity]
        if only_used:
            lines = [line for line in lines if line.used]
        return sorted(lines, key=lambda x: x.intensity, reverse=True)


def parse_lsrm_lib(path) -> List[Nuclide]:
    """Распарсить файл LSRM SpectraLine .lib."""
    text = Path(path).read_bytes().decode("windows-1251")
    root = ET.fromstring(text)
    nuclides = []
    for nuclide_elem in root.findall("Nuclide"):
        attrs = nuclide_elem.attrib
        name = attrs.get("name", "")
        half_life_value = _cfloat(attrs.get("half_life_value"))
        half_life_unit = attrs.get("half_life_unit")
        gamma_constant = _cfloat(attrs.get("gamma_constant"))
        atomic_mass = _cint(attrs.get("atomic_mass"))
        lines = []
        for line_elem in nuclide_elem.findall("Line"):
            line_attrs = line_elem.attrib
            energy = _cfloat(line_attrs.get("energy"))
            intensity = _cfloat(line_attrs.get("intensity"))
            if energy is None or intensity is None:
                continue  # Пропускаем линии без energy или intensity
            d_energy = _cfloat(line_attrs.get("d_energy"))
            d_intensity = _cfloat(line_attrs.get("d_intensity"))
            line_type = line_attrs.get("line_type")
            used = (line_attrs.get("used", "").lower() != "false")
            lines.append(GammaLine(
                energy=energy,
                intensity=intensity,
                d_energy=d_energy,
                d_intensity=d_intensity,
                line_type=line_type,
                used=used
            ))
        nuclides.append(Nuclide(
            name=name,
            half_life_value=half_life_value,
            half_life_unit=half_life_unit,
            gamma_constant=gamma_constant,
            atomic_mass=atomic_mass,
            lines=tuple(lines)
        ))
    return nuclides


def to_json_obj(nuclides: List[Nuclide], provenance: dict | None = None) -> dict:
    """Сериализовать список нуклидов в JSON-объект."""
    def line_to_dict(line: GammaLine) -> dict:
        return {
            "energy": line.energy,
            "intensity": line.intensity,
            "d_energy": line.d_energy,
            "d_intensity": line.d_intensity,
            "line_type": line.line_type,
            "used": line.used
        }

    def nuclide_to_dict(nuclide: Nuclide) -> dict:
        return {
            "name": nuclide.name,
            "half_life_value": nuclide.half_life_value,
            "half_life_unit": nuclide.half_life_unit,
            "gamma_constant": nuclide.gamma_constant,
            "atomic_mass": nuclide.atomic_mass,
            "category": nuclide.category,
            "lifetime": nuclide.lifetime,
            "lines": [line_to_dict(line) for line in nuclide.lines]
        }

    return {
        "_provenance": provenance or {},
        "nuclides": [nuclide_to_dict(nuclide) for nuclide in nuclides]
    }


def load_nuclides_json(path) -> List[Nuclide]:
    """Загрузить список нуклидов из JSON-файла."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nuclides_data = data.get("nuclides", [])
    if not isinstance(nuclides_data, list):
        nuclides_data = [nuclides_data]

    def line_from_dict(line_data: dict) -> GammaLine:
        return GammaLine(
            energy=line_data["energy"],
            intensity=line_data["intensity"],
            d_energy=line_data.get("d_energy"),
            d_intensity=line_data.get("d_intensity"),
            line_type=line_data.get("line_type"),
            used=line_data.get("used", True)
        )

    def nuclide_from_dict(nuclide_data: dict) -> Nuclide:
        lines = [line_from_dict(line) for line in nuclide_data.get("lines", [])]
        return Nuclide(
            name=nuclide_data["name"],
            half_life_value=nuclide_data.get("half_life_value"),
            half_life_unit=nuclide_data.get("half_life_unit"),
            gamma_constant=nuclide_data.get("gamma_constant"),
            atomic_mass=nuclide_data.get("atomic_mass"),
            category=nuclide_data.get("category"),
            lifetime=nuclide_data.get("lifetime"),
            lines=tuple(lines)
        )

    return [nuclide_from_dict(nuclide) for nuclide in nuclides_data]


def default_library() -> List[Nuclide]:
    """Загрузить встроенный файл библиотеки нуклидов."""
    try:
        path = Path(__file__).resolve().parent.parent / "data" / "nuclides.json"
        return load_nuclides_json(path)
    except Exception:
        return []
