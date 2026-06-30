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
    # верхний уровень — категории (после #86 в библиотеке все 4:
    # natural/technogenic/medical/fission)
    assert p._tree.topLevelItemCount() == 4


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


def test_lifetime_filter_partitions_long_short(app):
    # после #86 в библиотеке есть и long, и short — фильтр должен их разделять
    lib = default_library()
    p = NuclidePanel(lib)
    full = len(_leaf_names(p._tree))
    n_long = sum(1 for n in lib if n.lifetime == "long")
    n_short = sum(1 for n in lib if n.lifetime == "short")
    assert n_long > 0 and n_short > 0
    # снять «long» -> остаются только short
    p._lt_checks["long"].setChecked(False)
    assert len(_leaf_names(p._tree)) == n_short
    # снять оба -> дерево пусто
    p._lt_checks["short"].setChecked(False)
    assert len(_leaf_names(p._tree)) == 0
    # вернуть оба -> полный список
    p._lt_checks["long"].setChecked(True)
    p._lt_checks["short"].setChecked(True)
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
    # Задача #130: приоритеты демотируют техногенные/осколочные нуклиды (~0.5);
    # порог снижен 0.5→0.30 (продакшн-дефолт), природный K-40 (~0.999) доминирует
    p.show_candidates(peaks, min_confidence=0.30)
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
    # Po-210 — чистый альфа-излучатель, нет в библиотеке (после #86),
    # но есть в карте категорий -> подходит для проверки добавления
    p = NuclidePanel(default_library())
    before = len(_leaf_names(p._tree))
    g = (GammaLine(energy=1001.0, intensity=0.84, d_intensity=0.05),)
    p.add_nuclide(Nuclide(name="Po-210", lines=g))
    after = _leaf_names(p._tree)
    assert "Po-210" in after
    assert len(after) == before + 1
    # категория проставлена обогащением (Po-210 -> natural в карте категорий)
    po = next(n for n in p.library() if n.name == "Po-210")
    assert po.category is not None
    # повторное добавление заменяет, не дублирует
    p.add_nuclide(Nuclide(name="Po-210", lines=g))
    assert _leaf_names(p._tree).count("Po-210") == 1

# ---------- Задача #127: идентификация по найденным пикам ----------

def _strong_peaks():
    # сильные пики: Cs-137 (662), K-40 (1460), Co-60 (1173+1332)
    return [_fp(661.66), _fp(1460.8), _fp(1173.2, 800.0), _fp(1332.5, 720.0)]


def test_show_candidates_uses_spinbox_default_threshold(app):
    # #127: без явного min_confidence порог берётся из спинбокса (дефолт 0.30)
    p = NuclidePanel(default_library())
    assert abs(p._ident_min_conf.value() - 0.30) < 1e-9
    p.show_candidates(_strong_peaks())
    assert abs(p._ident_min_conf.value() - 0.30) < 1e-9  # спинбокс не сдвинут
    names = [p._cand.topLevelItem(i).text(0)
             for i in range(p._cand.topLevelItemCount())]
    for expected in ("Cs-137", "K-40", "Co-60"):
        assert expected in names


def test_ident_status_reports_counts(app):
    # #127: статус «идентифицировано N нуклид(ов) по M пик(ам)»
    p = NuclidePanel(default_library())
    peaks = _strong_peaks()
    p.show_candidates(peaks, min_confidence=0.5)
    txt = p._ident_status.text()
    assert "Идентификация:" in txt
    assert f"по {len(peaks)} пик" in txt
    n_top = p._cand.topLevelItemCount()
    assert f"{n_top} нуклид" in txt


def test_clear_candidates_resets_status(app):
    # #127: очистка возвращает статус к прочерку
    p = NuclidePanel(default_library())
    p.show_candidates(_strong_peaks(), min_confidence=0.5)
    assert "нуклид" in p._ident_status.text()
    p.clear_candidates()
    assert p._ident_status.text() == "Идентификация: —"


def test_empty_peaks_status_dash(app):
    # #127: пустой список пиков -> прочерк, дерево пусто
    p = NuclidePanel(default_library())
    p.show_candidates([])
    assert p._cand.topLevelItemCount() == 0
    assert p._ident_status.text() == "Идентификация: —"


def test_click_candidate_checks_nuclide_and_highlights(app):
    # #127: клик по кандидату -> нуклид отмечен в библиотеке, linesChanged эмитнут
    p = NuclidePanel(default_library())
    emitted = []
    p.linesChanged.connect(lambda lines: emitted.append(lines))
    # Задача #130: порог 0.5→0.30 — Cs-137 с приоритетом ~0.4999 теперь ниже 0.5
    p.show_candidates(_strong_peaks(), min_confidence=0.30)
    cs = next(p._cand.topLevelItem(i)
              for i in range(p._cand.topLevelItemCount())
              if p._cand.topLevelItem(i).text(0) == "Cs-137")
    p._on_candidate_clicked(cs)
    assert "Cs-137" in p._checked
    assert emitted and len(emitted[-1]) >= 1
    releaf = _find_leaf(p._tree, "Cs-137")
    assert releaf.checkState(0) == QtCore.Qt.Checked


def test_click_child_line_uses_parent_nuclide(app):
    # #127: клик по дочерней строке (линии) отмечает родительский нуклид
    p = NuclidePanel(default_library())
    # Задача #130: порог 0.5→0.30 — Co-60 с приоритетом ~0.4997 теперь ниже 0.5
    p.show_candidates(_strong_peaks(), min_confidence=0.30)
    co = next(p._cand.topLevelItem(i)
              for i in range(p._cand.topLevelItemCount())
              if p._cand.topLevelItem(i).text(0) == "Co-60")
    assert co.childCount() >= 1
    p._on_candidate_clicked(co.child(0))
    assert "Co-60" in p._checked


def test_threshold_change_rerenders_and_filters(app):
    # #127: повышение порога не увеличивает число кандидатов (valueChanged -> re-render)
    p = NuclidePanel(default_library())
    p.show_candidates(_strong_peaks())
    p._ident_min_conf.setValue(0.0)
    count_low = p._cand.topLevelItemCount()
    p._ident_min_conf.setValue(0.95)
    count_high = p._cand.topLevelItemCount()
    assert count_low >= 1
    assert count_high <= count_low