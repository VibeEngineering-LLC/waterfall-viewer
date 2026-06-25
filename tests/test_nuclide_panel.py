import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets, QtCore

from awf.ui.nuclide_panel import NuclidePanel
from awf.io.nuclide_lib import default_library, Nuclide, GammaLine
from awf.analysis.types import FoundPeak


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _leaf_names(tree):
    names = []
    it = QtWidgets.QTreeWidgetItemIterator(tree)
    while it.value():
        item = it.value()
        nm = item.data(0, QtCore.Qt.UserRole)
        if nm is not None:
            names.append(nm)
        it += 1
    return names


def _find_leaf(tree, name):
    it = QtWidgets.QTreeWidgetItemIterator(tree)
    while it.value():
        item = it.value()
        if item.data(0, QtCore.Qt.UserRole) == name:
            return item
        it += 1
    return None


def _fp(e, area=1000.0, sig=30.0):
    # синтетический найденный пик: только energy/area влияют на идентификацию
    return FoundPeak(channel=e, energy=e, height=area / (sig * 2.5),
                     fwhm_channels=sig, significance=10.0, area_estimate=area)


def test_tree_builds_all_nuclides(app):
    lib = default_library()
    p = NuclidePanel(lib)
    names = _leaf_names(p._tree)
    assert len(names) == len(lib)
    assert "Cs-137" in names and "K-40" in names
    # верхний уровень — категории (в встроенной библиотеке: natural/technogenic/fission)
    assert p._tree.topLevelItemCount() == 3


def test_category_filter_hides_branch(app):
    lib = default_library()
    p = NuclidePanel(lib)
    full = len(_leaf_names(p._tree))
    natural = sum(1 for n in lib if n.category == "natural")
    assert natural > 0
    p._cat_checks["natural"].setChecked(False)
    assert len(_leaf_names(p._tree)) == full - natural
    p._cat_checks["natural"].setChecked(True)
    assert len(_leaf_names(p._tree)) == full


def test_lifetime_filter_empties_when_all_long(app):
    lib = default_library()  # все 21 нуклида — long
    p = NuclidePanel(lib)
    full = len(_leaf_names(p._tree))
    assert full > 0
    p._lt_checks["long"].setChecked(False)
    assert len(_leaf_names(p._tree)) == 0
    p._lt_checks["long"].setChecked(True)
    assert len(_leaf_names(p._tree)) == full


def test_check_emits_lines_and_persists_across_rebuild(app):
    p = NuclidePanel(default_library())
    emitted = []
    p.linesChanged.connect(lambda lines: emitted.append(lines))
    leaf = _find_leaf(p._tree, "Cs-137")
    leaf.setCheckState(0, QtCore.Qt.Checked)
    assert "Cs-137" in p._checked
    assert len(p.selected_lines()) >= 1
    assert emitted  # сигнал linesChanged сработал
    # переключение фильтра перестраивает дерево, но отметка сохраняется
    p._cat_checks["natural"].setChecked(False)
    p._cat_checks["natural"].setChecked(True)
    assert "Cs-137" in p._checked
    releaf = _find_leaf(p._tree, "Cs-137")
    assert releaf.checkState(0) == QtCore.Qt.Checked


def test_clear_selection(app):
    p = NuclidePanel(default_library())
    _find_leaf(p._tree, "Cs-137").setCheckState(0, QtCore.Qt.Checked)
    assert p._checked
    p.clear_selection()
    assert not p._checked
    assert p.selected_lines() == []


def test_candidates_correct_first(app):
    p = NuclidePanel(default_library())
    peaks = [_fp(661.66), _fp(1460.8), _fp(1173.2, 800.0), _fp(1332.5, 720.0)]
    p.show_candidates(peaks, min_confidence=0.5)
    names = [p._cand.topLevelItem(i).text(0)
             for i in range(p._cand.topLevelItemCount())]
    assert names, "кандидаты не построены"
    for expected in ("Cs-137", "K-40", "Co-60"):
        assert expected in names
    top_conf = float(p._cand.topLevelItem(0).text(1))
    assert top_conf >= 0.9
    # у Co-60 две совпавшие линии -> два дочерних элемента
    co_row = next(p._cand.topLevelItem(i)
                  for i in range(p._cand.topLevelItemCount())
                  if p._cand.topLevelItem(i).text(0) == "Co-60")
    assert co_row.childCount() == 2


def test_candidates_cleared(app):
    p = NuclidePanel(default_library())
    p.show_candidates([_fp(661.66)], min_confidence=0.0)
    assert p._cand.topLevelItemCount() >= 1
    p.clear_candidates()
    assert p._cand.topLevelItemCount() == 0


def test_add_nuclide_replaces_not_duplicates(app):
    p = NuclidePanel(default_library())
    before = len(_leaf_names(p._tree))
    g = (GammaLine(energy=1001.0, intensity=0.84, d_intensity=0.05),)
    p.add_nuclide(Nuclide(name="Th-234", lines=g))
    after = _leaf_names(p._tree)
    assert "Th-234" in after
    assert len(after) == before + 1
    # категория проставлена обогащением (Th-234 -> natural в карте категорий)
    th = next(n for n in p.library() if n.name == "Th-234")
    assert th.category is not None
    # повторное добавление заменяет, не дублирует
    p.add_nuclide(Nuclide(name="Th-234", lines=g))
    assert _leaf_names(p._tree).count("Th-234") == 1