"""Задача #154: свёртка распадных семейств в родителей (Th-232, Ra-226).

Дочки скрываются из списка панели; их равновесные линии УЖЕ в родительских
записях nuclides.json (закреплено инвариантами данных ниже). Дочка без
родителя в списке НЕ скрывается."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from awf.io.nuclide_families import (
    FAMILY_PARENTS, FAMILY_DAUGHTERS, family_parent_of, collapse_families,
)
from awf.io.nuclide_lib import default_library, Nuclide, GammaLine

ALL_DAUGHTERS = ("Ra-228", "Ac-228", "Th-228", "Pb-212", "Bi-212", "Tl-208",
                 "Pb-214", "Bi-214", "Pb-210")


def test_collapse_removes_daughters_when_parent_present():
    lib = default_library()
    names = {n.name for n in lib}
    assert set(ALL_DAUGHTERS) <= names       # в сырой библиотеке дочки есть
    out = {n.name for n in collapse_families(lib)}
    assert "Th-232" in out and "Ra-226" in out
    for d in ALL_DAUGHTERS:
        assert d not in out, f"{d} должен быть свёрнут в родителя"


def test_daughter_kept_when_parent_absent():
    tl = Nuclide(name="Tl-208", lines=(GammaLine(energy=2614.5, intensity=99.75),))
    cs = Nuclide(name="Cs-137", lines=(GammaLine(energy=661.66, intensity=85.1),))
    out = collapse_families([tl, cs])
    assert {n.name for n in out} == {"Tl-208", "Cs-137"}


def test_collapse_idempotent_and_preserves_order():
    once = collapse_families(default_library())
    twice = collapse_families(once)
    assert [n.name for n in once] == [n.name for n in twice]


def test_family_parent_of():
    assert family_parent_of("Tl-208") == "Th-232"
    assert family_parent_of("Bi-214") == "Ra-226"
    assert family_parent_of("Cs-137") is None
    # обратная карта согласована с прямой
    for parent, daughters in FAMILY_PARENTS.items():
        for d in daughters:
            assert FAMILY_DAUGHTERS[d] == parent


def _lines_of(name):
    nuc = next(n for n in default_library() if n.name == name)
    return {round(ln.energy, 1): ln.intensity for ln in nuc.lines}


def test_th232_entry_carries_daughters_equilibrium_lines():
    # инвариант данных: линии дочек уже в родительской записи Th-232
    th = _lines_of("Th-232")
    for e in (238.6, 583.2, 911.2, 338.3, 727.3, 2614.5):
        assert e in th, f"линия {e} кэВ отсутствует в записи Th-232"


def test_ra226_entry_carries_daughters_equilibrium_lines():
    # инвариант данных: линии дочек уже в родительской записи Ra-226
    ra = _lines_of("Ra-226")
    for e in (351.9, 295.2, 609.3, 1120.3, 1764.5, 186.2):
        assert e in ra, f"линия {e} кэВ отсутствует в записи Ra-226"


def test_th232_tl208_branching_applied_in_parent():
    # в родителе интенсивности Tl-208 умножены на ветвление Bi-212->Tl-208 ~0.36
    th, tl = _lines_of("Th-232"), _lines_of("Tl-208")
    for e in (2614.5, 583.2):
        assert abs(th[e] / tl[e] - 0.36) < 0.01, f"ветвление не применено @ {e}"


# ---------- интеграция с панелью нуклидов ----------

@pytest.fixture(scope="module")
def app():
    from PyQt5 import QtWidgets
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _tree_names(tree):
    from PyQt5 import QtWidgets, QtCore
    names, it = set(), QtWidgets.QTreeWidgetItemIterator(tree)
    while it.value():
        nm = it.value().data(0, QtCore.Qt.UserRole)
        if nm is not None:
            names.add(nm)
        it += 1
    return names


def test_panel_tree_has_parents_not_daughters(app):
    from awf.ui.nuclide_panel import NuclidePanel
    p = NuclidePanel(default_library())
    names = _tree_names(p._tree)
    assert "Th-232" in names and "Ra-226" in names
    for d in ALL_DAUGHTERS:
        assert d not in names, f"{d} не должен показываться в дереве"


def _th_peak(e):
    # площадь ∝ интенсивность×отн.эффективность — реалистичный торий-спектр
    from awf.analysis.types import FoundPeak
    from awf.analysis.identify import relative_efficiency
    th = _lines_of("Th-232")
    area = 1e4 * th[e] * relative_efficiency(e)
    return FoundPeak(channel=e, energy=e, height=area / 10.0,
                     fwhm_channels=10.0, significance=20.0, area_estimate=area)


def test_identification_reports_parent_not_daughters(app):
    # ториевый спектр -> кандидат Th-232, не дочки. Задача #162: гейт «все значимые
    # линии» требует в наборе пиков и 338.3/968.9 (>=20% характеристической 238.6) —
    # без них, при той же статистике, реальный спектр их тоже бы показал.
    from awf.ui.nuclide_panel import NuclidePanel
    p = NuclidePanel(default_library())
    peaks = [_th_peak(e) for e in (238.6, 338.3, 583.2, 911.2, 969.0, 2614.5)]
    p.show_candidates(peaks, min_confidence=0.30)
    names = [p._cand.topLevelItem(i).text(0)
             for i in range(p._cand.topLevelItemCount())]
    assert "Th-232" in names, f"Th-232 не идентифицирован: {names}"
    for d in ALL_DAUGHTERS:
        assert d not in names, f"дочка {d} не должна быть кандидатом"
