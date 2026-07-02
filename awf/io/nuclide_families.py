"""Свёртка распадных семейств в родителей (Задача #154).

Оператор: «Убрать из списка изотопы из семейств. Оставить только родительские
Th-232 и Ra-226, а линии дочерних включить в состав родителей». Родительские
записи awf/data/nuclides.json УЖЕ несут равновесные линии дочек — Th-232:
238.6 (Pb-212), 583.2/2614.5 (Tl-208 с ветвлением Bi-212→Tl-208 0.36:
99.75×0.36=35.9), 911.2/338.3/969.0 (Ac-228), 727.3 (Bi-212); Ra-226:
351.9/295.2 (Pb-214), 609.3/1120.3/1764.5 (Bi-214). Поэтому свёртка — скрыть
дочек из списка; линии не синтезируются и не дублируются. Дочка скрывается
ТОЛЬКО если её родитель присутствует в списке (библиотека без родителя не
теряет нуклид). Qt-free."""
from __future__ import annotations

FAMILY_PARENTS = {
    "Th-232": ("Ra-228", "Ac-228", "Th-228", "Pb-212", "Bi-212", "Tl-208"),
    "Ra-226": ("Pb-214", "Bi-214", "Pb-210"),
}

# Обратная карта: дочка -> родитель семейства.
FAMILY_DAUGHTERS = {d: p for p, ds in FAMILY_PARENTS.items() for d in ds}


def family_parent_of(name: str):
    """Родитель семейства для дочки, иначе None."""
    return FAMILY_DAUGHTERS.get(name)


def collapse_families(nuclides):
    """Убрать из списка дочек семейств, чей родитель присутствует (идемпотентно)."""
    present = {n.name for n in nuclides}
    return [n for n in nuclides
            if FAMILY_DAUGHTERS.get(n.name) not in present]


__all__ = ["FAMILY_PARENTS", "FAMILY_DAUGHTERS",
           "family_parent_of", "collapse_families"]
