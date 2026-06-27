from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets

from awf.io.nuclide_categories import (
    CATEGORIES, LIFETIMES, enrich_nuclide,
)
from awf.io.nuclide_lib import Nuclide, GammaLine
from awf.analysis.identify import identify_peaks

COLORS = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324",
          "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9")

# человекочитаемые подписи категорий / времени жизни для веток дерева и фильтров
CATEGORY_LABELS = {
    "natural": "Природные",
    "technogenic": "Техногенные",
    "medical": "Медицинские",
    "fission": "Осколочные (деления)",
    None: "Без категории",
}
LIFETIME_LABELS = {
    "short": "Короткоживущие",
    "long": "Долгоживущие",
    None: "T½ неизвестно",
}


class IaeaFetchThread(QtCore.QThread):
    """Фоновая загрузка γ-линий нуклида из IAEA LiveChart (или офлайн-кэша) — не блокирует UI.
    Импорты iaea_fetcher выполняются внутри run(), чтобы конструирование панели не тянуло сеть."""
    fetched = QtCore.Signal(object)   # Nuclide
    failed = QtCore.Signal(str)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._name = name

    def run(self) -> None:
        try:
            from awf.io.iaea_fetcher import (
                fetch_iaea_gamma_lines, merge_iaea_into_internal,
            )
            lines = fetch_iaea_gamma_lines(self._name)
            merged = merge_iaea_into_internal(lines, self._name)
            gamma = tuple(
                GammaLine(energy=float(e), intensity=float(i), d_intensity=float(di))
                for e, i, di in merged["lines"]
            )
            if not gamma:
                self.failed.emit(f"{self._name}: гамма-линий не найдено")
                return
            self.fetched.emit(enrich_nuclide(Nuclide(name=self._name, lines=gamma)))
        except Exception as exc:  # любую ошибку отдать в UI-поток, не падать
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class NuclidePanel(QtWidgets.QWidget):
    """Расширенная панель нуклидов (Задача 12).

    - дерево: Категория → Время жизни → нуклид (чекбокс + цвет линий);
    - фильтры: мин. интенсивность, только основные линии, набор категорий, набор времён жизни;
    - панель «Кандидаты» под выбранным пиком (Задача 11, identify_peaks);
    - кнопка «Добавить из IAEA…» (Задача 9, фоновый QThread).

    Публичный контракт прежней панели сохранён: сигнал linesChanged(list),
    методы set_library(), selected_lines(), clear_selection()."""

    linesChanged = QtCore.Signal(object)  # list[(energy_keV: float, color: str, label: str)]

    def __init__(self, nuclides=None, parent=None):
        super().__init__(parent)
        self._nuclides: list = []
        self._color_by_name: dict = {}
        self._checked: set = set()
        self._found_peaks: list = []
        self._fetch_thread = None

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel("Библиотека нуклидов"))

        # --- фильтр по интенсивности / основным линиям ---
        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(QtWidgets.QLabel("Мин. интенс., %:"))
        self._min_int = QtWidgets.QDoubleSpinBox()
        self._min_int.setRange(0, 100)
        self._min_int.setSingleStep(1.0)
        self._min_int.setValue(5.0)
        self._min_int.setDecimals(1)
        frow.addWidget(self._min_int)
        self._only_used = QtWidgets.QCheckBox("только основные")
        self._only_used.setChecked(True)
        frow.addWidget(self._only_used)
        frow.addStretch(1)
        root.addLayout(frow)

        # --- фильтр по категориям (Задача 12.2) ---
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(QtWidgets.QLabel("Категории:"))
        self._cat_checks: dict = {}
        for cat in CATEGORIES:
            cb = QtWidgets.QCheckBox(CATEGORY_LABELS[cat])
            cb.setChecked(True)
            cb.stateChanged.connect(self._rebuild_tree)
            self._cat_checks[cat] = cb
            crow.addWidget(cb)
        crow.addStretch(1)
        root.addLayout(crow)

        # --- фильтр по времени жизни (Задача 12.2) ---
        lrow = QtWidgets.QHBoxLayout()
        lrow.addWidget(QtWidgets.QLabel("Время жизни:"))
        self._lt_checks: dict = {}
        for lt in LIFETIMES:
            cb = QtWidgets.QCheckBox(LIFETIME_LABELS[lt])
            cb.setChecked(True)
            cb.stateChanged.connect(self._rebuild_tree)
            self._lt_checks[lt] = cb
            lrow.addWidget(cb)
        lrow.addStretch(1)
        root.addLayout(lrow)

        # --- дерево библиотеки (Задача 12.1) ---
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        root.addWidget(self._tree, stretch=3)

        # --- кнопки ---
        brow = QtWidgets.QHBoxLayout()
        self._btn_iaea = QtWidgets.QPushButton("Добавить из IAEA…")
        self._btn_none = QtWidgets.QPushButton("Снять все")
        brow.addWidget(self._btn_iaea)
        brow.addWidget(self._btn_none)
        root.addLayout(brow)

        # --- панель кандидатов (Задача 12.3 / 11) ---
        root.addWidget(QtWidgets.QLabel("Кандидаты по выбранному пику"))
        self._cand = QtWidgets.QTreeWidget()
        self._cand.setHeaderLabels(["Нуклид", "Уверен.", "Категория", "Линий"])
        self._cand.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        root.addWidget(self._cand, stretch=2)

        self._status = QtWidgets.QLabel("")
        root.addWidget(self._status)

        # --- сигналы ---
        self._min_int.valueChanged.connect(self._recompute)
        self._only_used.stateChanged.connect(self._recompute)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._btn_none.clicked.connect(self.clear_selection)
        self._btn_iaea.clicked.connect(self._on_add_iaea)

        if nuclides is not None:
            self.set_library(nuclides)

    # ---------- библиотека / дерево ----------
    def set_library(self, nuclides) -> None:
        self._nuclides = [self._ensure_enriched(n) for n in nuclides]
        self._color_by_name = {}
        for i, name in enumerate(sorted({n.name for n in self._nuclides})):
            self._color_by_name[name] = COLORS[i % len(COLORS)]
        self._rebuild_tree()

    @staticmethod
    def _ensure_enriched(n):
        if n.category is None or n.lifetime is None:
            return enrich_nuclide(n)
        return n

    def _color(self, name: str) -> str:
        return self._color_by_name.get(name, COLORS[0])

    def _rebuild_tree(self, *args) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        cats_on = {c for c in CATEGORIES if self._cat_checks[c].isChecked()}
        lt_on = {l for l in LIFETIMES if self._lt_checks[l].isChecked()}
        cat_order = list(CATEGORIES) + [None]   # «без категории» — последней веткой
        lt_order = list(LIFETIMES) + [None]      # «T½ неизвестно» показываем всегда
        for cat in cat_order:
            if cat is not None and cat not in cats_on:
                continue
            cat_nucs = [n for n in self._nuclides if n.category == cat]
            if not cat_nucs:
                continue
            cat_item = None
            for lt in lt_order:
                if lt is not None and lt not in lt_on:
                    continue
                grp = [n for n in cat_nucs if n.lifetime == lt]
                if not grp:
                    continue
                if cat_item is None:
                    cat_item = QtWidgets.QTreeWidgetItem(self._tree, [CATEGORY_LABELS[cat]])
                    cat_item.setFlags(QtCore.Qt.ItemIsEnabled)
                    cat_item.setExpanded(True)
                lt_item = QtWidgets.QTreeWidgetItem(cat_item, [LIFETIME_LABELS[lt]])
                lt_item.setFlags(QtCore.Qt.ItemIsEnabled)
                lt_item.setExpanded(True)
                for n in sorted(grp, key=lambda x: x.name):
                    leaf = QtWidgets.QTreeWidgetItem(lt_item, [n.name])
                    leaf.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    state = QtCore.Qt.Checked if n.name in self._checked else QtCore.Qt.Unchecked
                    leaf.setCheckState(0, state)
                    leaf.setForeground(0, QtGui.QColor(self._color(n.name)))
                    leaf.setData(0, QtCore.Qt.UserRole, n.name)
        self._tree.blockSignals(False)
        self._recompute()

    def _on_item_changed(self, item, col: int = 0) -> None:
        name = item.data(0, QtCore.Qt.UserRole)
        if name is None:
            return
        if item.checkState(0) == QtCore.Qt.Checked:
            self._checked.add(name)
        else:
            self._checked.discard(name)
        self._recompute()

    def clear_selection(self) -> None:
        self._checked.clear()
        self._rebuild_tree()

    # ---------- линии нуклидов -> спектр ----------
    def _collect_lines(self):
        min_int = self._min_int.value()
        only_used = self._only_used.isChecked()
        by_name = {n.name: n for n in self._nuclides}
        lines = []
        for name in self._checked:
            nuc = by_name.get(name)
            if nuc is None:
                continue
            color = self._color(name)
            for ln in nuc.major_lines(min_intensity=min_int, only_used=only_used):
                # Задача #69: 4-й элемент — интенсивность линии (вероятность испускания,
                # доля), нужен для маркеров нуклидов на плоскостях высотой ∝ интенсивности.
                lines.append((float(ln.energy), color, name, float(ln.intensity)))
        return lines

    def _recompute(self, *args) -> None:
        lines = self._collect_lines()
        self._status.setText(f"выбрано нуклидов: {len(self._checked)}, линий: {len(lines)}")
        self.linesChanged.emit(lines)

    def selected_lines(self) -> list:
        return self._collect_lines()

    # ---------- кандидаты (Задача 12.3 / 11) ----------
    def show_candidates(self, found_peaks, *, min_confidence: float = 0.0) -> None:
        """Прогнать identify_peaks по текущей библиотеке для переданных найденных пиков и
        показать кандидатов: имя, уверенность, категория, число совпавших линий + Δ/I по линиям."""
        self._found_peaks = list(found_peaks)
        self._cand.clear()
        if not self._nuclides or not self._found_peaks:
            return
        results = identify_peaks(
            self._found_peaks, self._nuclides, min_confidence=min_confidence)
        for r in results:
            top = QtWidgets.QTreeWidgetItem(
                self._cand,
                [r.nuclide, f"{r.confidence:.2f}", r.category or "—", str(len(r.matches))])
            top.setForeground(0, QtGui.QColor(self._color(r.nuclide)))
            for m in r.matches:
                QtWidgets.QTreeWidgetItem(
                    top,
                    [f"{m.line_energy:.1f} кэВ", f"Δ={m.delta_keV:+.2f}",
                     f"I={m.intensity_pct:.1f}%", ""])
            top.setExpanded(True)
        for i in range(self._cand.columnCount()):
            self._cand.resizeColumnToContents(i)

    def clear_candidates(self) -> None:
        self._found_peaks = []
        self._cand.clear()

    # ---------- IAEA (Задача 12.4) ----------
    def _on_add_iaea(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Добавить нуклид из IAEA",
            "Имя нуклида (например, Th-234, Cs-137):")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._btn_iaea.setEnabled(False)
        self._status.setText(f"IAEA: загрузка {name} …")
        self._fetch_thread = IaeaFetchThread(name, parent=self)
        self._fetch_thread.fetched.connect(self._on_iaea_fetched)
        self._fetch_thread.failed.connect(self._on_iaea_failed)
        self._fetch_thread.start()

    @QtCore.Slot(object)
    def _on_iaea_fetched(self, nuclide) -> None:
        self.add_nuclide(nuclide)
        self._btn_iaea.setEnabled(True)
        self._status.setText(
            f"IAEA: добавлен {nuclide.name} ({len(nuclide.lines)} линий)")

    @QtCore.Slot(str)
    def _on_iaea_failed(self, message: str) -> None:
        self._btn_iaea.setEnabled(True)
        self._status.setText(f"IAEA: ошибка — {message}")
        QtWidgets.QMessageBox.warning(
            self, "IAEA", f"Не удалось загрузить нуклид:\n{message}")

    def add_nuclide(self, nuclide) -> None:
        """Добавить/заменить нуклид в библиотеке и перестроить дерево (тестируемо без сети)."""
        nuclide = self._ensure_enriched(nuclide)
        self._nuclides = [n for n in self._nuclides if n.name != nuclide.name]
        self._nuclides.append(nuclide)
        if nuclide.name not in self._color_by_name:
            idx = len(self._color_by_name)
            self._color_by_name[nuclide.name] = COLORS[idx % len(COLORS)]
        self._rebuild_tree()

    def library(self) -> list:
        return list(self._nuclides)