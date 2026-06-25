"""Категоризация нуклидов (Задача 10): origin из awf/data/nuclide_categories.json + lifetime по порогу T½. Qt-free."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from awf.io.nuclide_lib import Nuclide

CATEGORIES = ("natural", "technogenic", "medical", "fission")
LIFETIMES = ("short", "long")

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nuclide_categories.json"
_DEFAULT_THRESHOLD_DAYS = 60.0

_MAP_CACHE = None
_THRESHOLD_DAYS_CACHE = _DEFAULT_THRESHOLD_DAYS


def _load_map() -> dict:
    global _MAP_CACHE, _THRESHOLD_DAYS_CACHE
    if _MAP_CACHE is not None:
        return _MAP_CACHE

    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        categories = data.get("categories", {})
        threshold = data.get("long_lifetime_threshold_days", _DEFAULT_THRESHOLD_DAYS)
    except Exception:
        categories = {}
        threshold = _DEFAULT_THRESHOLD_DAYS

    _MAP_CACHE = categories
    _THRESHOLD_DAYS_CACHE = threshold
    return _MAP_CACHE


def threshold_days() -> float:
    _load_map()
    return _THRESHOLD_DAYS_CACHE


def threshold_seconds() -> float:
    return threshold_days() * 86400.0


def category_of(name: str) -> Optional[str]:
    return _load_map().get(name)


def classify_lifetime_seconds(half_life_s: Optional[float]) -> Optional[str]:
    if half_life_s is None:
        return None
    return "long" if half_life_s >= threshold_seconds() else "short"


def classify_lifetime(nuclide: Nuclide) -> Optional[str]:
    return classify_lifetime_seconds(nuclide.half_life_seconds())


def enrich_nuclide(nuclide: Nuclide) -> Nuclide:
    cat = category_of(nuclide.name)
    lt = classify_lifetime(nuclide)
    return replace(nuclide, category=cat, lifetime=lt)


def enrich_all(nuclides) -> List[Nuclide]:
    return [enrich_nuclide(n) for n in nuclides]


def nuclides_by_category(nuclides, category: str) -> List[Nuclide]:
    return [n for n in nuclides if n.category == category]


def nuclides_by_lifetime(nuclides, lifetime: str) -> List[Nuclide]:
    return [n for n in nuclides if n.lifetime == lifetime]


__all__ = [
    "CATEGORIES",
    "LIFETIMES",
    "threshold_days",
    "threshold_seconds",
    "category_of",
    "classify_lifetime_seconds",
    "classify_lifetime",
    "enrich_nuclide",
    "enrich_all",
    "nuclides_by_category",
    "nuclides_by_lifetime",
]
