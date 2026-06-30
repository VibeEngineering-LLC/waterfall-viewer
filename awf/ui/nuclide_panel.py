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
        # Задача #130: модель FWHM(E) детектора для окна матчинга идентификации;
        # main_window прокидывает авто-калиброванную (#120), иначе грубый дефолт.
        self._ident_fwhm_model = None
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

        # --- Задача #127: модуль идентификации нуклидов по НАЙДЕННЫМ пикам ---
        # Раньше панель называлась «Кандидаты по выбранному пику» и НИКОГДА не получала
        # пиков (метод show_candidates не вызывался). Теперь main_window кормит её
        # _found_peaks() из подсистемы поиска (#110/#111) — identify_peaks по библиотеке.
        irow = QtWidgets.QHBoxLayout()
        self._ident_title = QtWidgets.QLabel("Идентификация по найденным пикам")
        irow.addWidget(self._ident_title)
        irow.addStretch(1)
        irow.addWidget(QtWidgets.QLabel("мин. увер.:"))
        self._ident_min_conf = QtWidgets.QDoubleSpinBox()
        self._ident_min_conf.setRange(0.0, 1.0)
        self._ident_min_conf.setSingleStep(0.05)
        self._ident_min_conf.setValue(0.30)
        self._ident_min_conf.setDecimals(2)
        self._ident_min_conf.setToolTip(
            "Порог уверенности: кандидаты ниже порога не показываются (меньше ложных)")
        irow.addWidget(self._ident_min_conf)
        root.addLayout(irow)
        self._cand = QtWidgets.QTreeWidget()
        self._cand.setHeaderLabels(["Нуклид", "Уверен.", "Категория", "Линий"])
        self._cand.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        root.addWidget(self._cand, stretch=2)
        # Задача #127: «идентифицировано N нуклид(ов) по M пик(ам)».
        self._ident_status = QtWidgets.QLabel("Идентификация: —")
        self._ident_status.setWordWrap(True)
        root.addWidget(self._ident_status)

        self._status = QtWidgets.QLabel("")
        root.addWidget(self._status)

        # --- сигналы ---
        self._min_int.valueChanged.connect(self._recompute)
        self._only_used.stateChanged.connect(self._recompute)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._btn_none.clicked.connect(self.clear_selection)
        self._btn_iaea.clicked.connect(self._on_add_iaea)
        # Задача #127: порог уверенности → перерисовать кандидатов; клик по кандидату →
        # отметить нуклид в библиотеке (его линии подсветятся на 3D/2D/срезе через linesChanged).
        self._ident_min_conf.valueChanged.connect(self._render_candidates)
        self._cand.itemClicked.connect(self._on_candidate_clicked)

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

    # ---------- идентификация по найденным пикам (Задача #127 / 12.3 / 11) ----------
    def show_candidates(self, found_peaks, *, min_confidence: float = None,
                        fwhm_model=None) -> None:
        """Задача #127: принять НАЙДЕННЫЕ пики (из подсистемы поиска #110/#111) и показать
        идентифицированные нуклиды. Порог уверенности берётся из спинбокса панели; если
        min_confidence передан явно (старый контракт/тесты) — выставляет спинбокс под него.

        Задача #130: fwhm_model(E)->FWHM(E) — модель ширины детектора для окна матчинга
        (main_window прокидывает авто-калиброванную #120); None → грубый дефолт."""
        self._found_peaks = list(found_peaks)
        if fwhm_model is not None:
            self._ident_fwhm_model = fwhm_model
        if min_confidence is not None:
            self._ident_min_conf.blockSignals(True)
            self._ident_min_conf.setValue(float(min_confidence))
            self._ident_min_conf.blockSignals(False)
        self._render_candidates()

    def _render_candidates(self, *args) -> None:
        """Задача #127: прогнать identify_peaks по библиотеке для сохранённых найденных пиков
        и заполнить дерево; порог — из спинбокса; статус — «идентифицировано N по M пикам»."""
        self._cand.clear()
        n_peaks = len(self._found_peaks)
        if not self._nuclides or not self._found_peaks:
            self._ident_status.setText("Идентификация: —")
            return
        # Задача #130: apply_priors=True — давить ложные кандидаты (осколочные/медицинские/
        # космогенные) априорами правдоподобия; fwhm_model — реальная ширина детектора (#120).
        results = identify_peaks(
            self._found_peaks, self._nuclides,
            fwhm_model=self._ident_fwhm_model,
            min_confidence=float(self._ident_min_conf.value()),
            apply_priors=True)
        for r in results:
            self._add_candidate_row(r)
        for i in range(self._cand.columnCount()):
            self._cand.resizeColumnToContents(i)
        self._update_ident_status(len(results), n_peaks)

    def _add_candidate_row(self, r) -> None:
        """Задача #127: одна строка-кандидат в дереве + дочерние строки линий. UserRole
        хранит имя нуклида — ключ для клика (отметить в библиотеке → подсветка линий)."""
        top = QtWidgets.QTreeWidgetItem(
            self._cand,
            [r.nuclide, f"{r.confidence:.2f}", r.category or "—", str(len(r.matches))])
        top.setForeground(0, QtGui.QColor(self._color(r.nuclide)))
        top.setData(0, QtCore.Qt.UserRole, r.nuclide)
        for m in r.matches:
            QtWidgets.QTreeWidgetItem(
                top,
                [f"{m.line_energy:.1f} кэВ", f"Δ={m.delta_keV:+.2f}",
                 f"I={m.intensity_pct:.1f}%", ""])
        top.setExpanded(True)

    def _update_ident_status(self, n_results: int, n_peaks: int) -> None:
        """Задача #127: метка «идентифицировано N нуклид(ов) по M пик(ам)»."""
        self._ident_status.setText(
            f"Идентификация: {n_results} нуклид(ов) по {n_peaks} пик(ам)")

    def _on_candidate_clicked(self, item, col: int = 0) -> None:
        """Задача #127: клик по кандидату → отметить нуклид в библиотеке (линии подсветятся
        на 3D/2D/срезе через linesChanged). Клик по дочерней линии берёт родителя."""
        name = item.data(0, QtCore.Qt.UserRole)
        if name is None and item.parent() is not None:
            name = item.parent().data(0, QtCore.Qt.UserRole)
        if name:
            self._check_nuclide(str(name))

    def _check_nuclide(self, name: str) -> None:
        """Задача #127: отметить нуклид по имени (если не отмечен) и перестроить дерево —
        линии уходят в linesChanged → подсветка на спектрограммах."""
        if name in self._checked:
            return
        self._checked.add(name)
        self._rebuild_tree()

    def clear_candidates(self) -> None:
        self._found_peaks = []
        self._cand.clear()
        self._ident_status.setText("Идентификация: —")

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