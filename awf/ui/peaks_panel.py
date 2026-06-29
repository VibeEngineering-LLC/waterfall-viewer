"""Задача #111: панель «Найденные пики» — список пиков + регулятор чувствительности.

PeaksPanel(QWidget) отображает результаты find_peaks из Waterfall3DView:
- QDoubleSpinBox для порога значимости σ (диапазон 1.5–6.0, шаг 0.5, дефолт 3.0);
- метка «Найдено: N»;
- QTableWidget с 6 колонками (Энергия/Канал/Значимость/Высота/FWHM/Площадь);
  числовая сортировка через setData(Qt.EditRole, float).
Тёмная тема наследуется из APP_QSS (QDoubleSpinBox стрелки уже стилизованы в #97).
"""
from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets


class PeaksPanel(QtWidgets.QWidget):
    """Задача #111: виджет панели найденных пиков.

    Сигналы:
        sigmaChanged(float) — пользователь изменил порог значимости.

    Методы:
        set_peaks(peaks: list) — заполнить таблицу из объектов FoundPeak.
    """

    sigmaChanged = QtCore.Signal(float)
    # Задача #124: клик по строке пика → подсветка гребня на 3D (энергия, кэВ);
    # переключение чекбокса → показать/скрыть гребень пика (энергия, видим?).
    peakSelected = QtCore.Signal(float)
    peakVisibilityChanged = QtCore.Signal(float, bool)

    # Русские заголовки колонок (ключи i18n). Задача #123: у Высоты/Площади указаны
    # единицы — отсчёты (нетто-счёт над континуумом в суммарном по времени спектре).
    # Задача #124: колонка 0 — чекбокс «Показать» (видимость гребня пика на 3D).
    _COL_CHECK = 0
    _COL_ENERGY = 1
    _COL_HEADERS_RU = [
        "Показать",
        "Энергия, кэВ",
        "Канал",
        "Значимость",
        "Высота, отсч.",
        "FWHM",
        "Площадь, отсч.",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        # Задача #124: флаг подавляет itemChanged при программном заполнении таблицы
        # (иначе каждая вставка чекбокса слала бы ложный peakVisibilityChanged).
        self._populating = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- регулятор чувствительности ---
        sigma_row = QtWidgets.QHBoxLayout()
        sigma_label = QtWidgets.QLabel("Порог значимости, σ")
        sigma_label.setObjectName("knobTitle")
        self._sigma_box = QtWidgets.QDoubleSpinBox()
        self._sigma_box.setObjectName("peaksSigmaBox")
        self._sigma_box.setRange(1.5, 6.0)
        self._sigma_box.setSingleStep(0.5)
        self._sigma_box.setValue(3.0)
        self._sigma_box.setDecimals(1)
        self._sigma_box.setToolTip(
            "Порог значимости Currie L_C (σ): выше → меньше пиков, чище результат"
        )
        sigma_row.addWidget(sigma_label)
        sigma_row.addWidget(self._sigma_box)
        sigma_row.addStretch(1)
        layout.addLayout(sigma_row)

        # --- метка «Найдено: N» ---
        self._count_label = QtWidgets.QLabel("Найдено: 0")
        self._count_label.setObjectName("knobTitle")
        layout.addWidget(self._count_label)

        # --- Задача #123: метка временно́го окна поиска ---
        # Поиск идёт по суммарному (интегральному) по времени спектру всего файла,
        # значит «окно» = весь водопад. Высота/площадь в таблице — отсчёты этого
        # интеграла. Дополнительно отмечаются транзиентные пики (Задача #113).
        self._win_slices = None
        self._win_seconds = None
        self._window_label = QtWidgets.QLabel("Окно поиска: —")
        self._window_label.setObjectName("knobTitle")
        self._window_label.setWordWrap(True)
        self._window_label.setToolTip(
            "Пики ищутся в суммарном по времени (интегральном) спектре всего файла; "
            "дополнительно отмечаются транзиентные пики, значимые в отдельных срезах "
            "(Задача #113). «Высота» и «Площадь» — в отсчётах (сумма по всему файлу)."
        )
        layout.addWidget(self._window_label)

        # --- таблица пиков ---
        self._table = QtWidgets.QTableWidget(0, len(self._COL_HEADERS_RU))
        self._table.setHorizontalHeaderLabels(self._COL_HEADERS_RU)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        # По умолчанию — сортировка по значимости (колонка 3 после #124) по убыванию
        self._table.sortItems(3, QtCore.Qt.DescendingOrder)
        _hdr = self._table.horizontalHeader()
        _hdr.setStretchLastSection(True)
        _hdr.setSectionsClickable(True)
        # Задача #124: узкая колонка чекбоксов — по содержимому
        _hdr.setSectionResizeMode(self._COL_CHECK,
                                  QtWidgets.QHeaderView.ResizeToContents)
        # Задача #116: QSS красит ячейки/заголовок, но ПУСТУЮ область viewport ниже строк
        # Qt заливает из палитры (роль Base / фон viewport) — оставалась белой поверх тёмной
        # темы. Тёмная Base/AlternateBase + autoFill фона viewport кладут пустое поле под тему.
        _pal = self._table.palette()
        _pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#26282b"))
        _pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#2d2f33"))
        self._table.setPalette(_pal)
        _vp = self._table.viewport()
        _vp_pal = _vp.palette()
        _vp_pal.setColor(_vp.backgroundRole(), QtGui.QColor("#26282b"))
        _vp.setPalette(_vp_pal)
        _vp.setAutoFillBackground(True)
        layout.addWidget(self._table)

        # --- сигналы ---
        self._sigma_box.valueChanged.connect(self._on_sigma_changed)
        # Задача #124: чекбокс отображения → видимость гребня; клик по строке → подсветка.
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.cellClicked.connect(self._on_cell_clicked)

    def _on_sigma_changed(self, value: float) -> None:
        self.sigmaChanged.emit(float(value))

    def _on_item_changed(self, item) -> None:
        """Задача #124: переключение чекбокса колонки 0 → peakVisibilityChanged(E, видим?).
        Подавлено при программном заполнении (_populating)."""
        if self._populating or item.column() != self._COL_CHECK:
            return
        energy = item.data(QtCore.Qt.UserRole)
        if energy is None:
            return
        visible = item.checkState() == QtCore.Qt.Checked
        self.peakVisibilityChanged.emit(float(energy), bool(visible))

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Задача #124: клик по любой данные-ячейке строки → peakSelected(E) (подсветка
        гребня на 3D). Клик по самому чекбоксу (col 0) только переключает видимость."""
        if col == self._COL_CHECK:
            return
        eitem = self._table.item(row, self._COL_ENERGY)
        if eitem is None:
            return
        energy = eitem.data(QtCore.Qt.EditRole)
        if energy is not None:
            self.peakSelected.emit(float(energy))

    def _snapshot_check_states(self) -> dict:
        """Задача #124: снять {энергия-ключ: checked?} с текущих строк — сохранить выбор
        видимости при перезаполнении тем же набором пиков (round(E,3) как в view3d)."""
        snap = {}
        for r in range(self._table.rowCount()):
            it = self._table.item(r, self._COL_CHECK)
            if it is None:
                continue
            e = it.data(QtCore.Qt.UserRole)
            if e is not None:
                snap[round(float(e), 3)] = (it.checkState() == QtCore.Qt.Checked)
        return snap

    def _make_check_item(self, energy: float, prev: dict):
        """Задача #124: чекбокс видимости гребня пика (колонка 0). UserRole хранит энергию;
        состояние из prev по энергии-ключу, по умолчанию — отмечен (гребень виден)."""
        it = QtWidgets.QTableWidgetItem()
        it.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsSelectable)
        checked = prev.get(round(float(energy), 3), True)
        it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        it.setData(QtCore.Qt.UserRole, float(energy))
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        return it

    def set_peaks(self, peaks: list) -> None:
        """Заполнить таблицу из списка FoundPeak. Корректная числовая сортировка
        через setData(Qt.EditRole, float_val) — НЕ строковая. Задача #124: колонка 0 —
        чекбокс видимости гребня; состояние галочек сохраняется между перезаполнениями
        (по энергии-ключу), чтобы попутные ре-рендеры 3D не сбрасывали выбор."""
        header = self._table.horizontalHeader()
        sort_col = header.sortIndicatorSection()
        sort_order = header.sortIndicatorOrder()
        prev_checked = self._snapshot_check_states()   # Задача #124: сохранить галочки
        self._populating = True                        # подавить itemChanged при заливке
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        for pk in peaks:
            row = self._table.rowCount()
            self._table.insertRow(row)
            energy = float(pk.energy)
            self._table.setItem(row, self._COL_CHECK,
                                self._make_check_item(energy, prev_checked))
            values = [
                energy,
                float(pk.channel),
                float(pk.significance),
                float(pk.height),
                float(pk.fwhm_channels),
                float(pk.area_estimate),
            ]
            formats = ["{:.1f}", "{:.1f}", "{:.1f}", "{:.0f}", "{:.1f}", "{:.0f}"]
            for k, (val, fmt) in enumerate(zip(values, formats)):
                item = QtWidgets.QTableWidgetItem(fmt.format(val))
                item.setData(QtCore.Qt.EditRole, val)   # числовая сортировка по float
                item.setTextAlignment(
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                )
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self._table.setItem(row, k + 1, item)   # сдвиг на колонку чекбокса

        n = len(peaks)
        self._count_label.setText(f"Найдено: {n}")

        # Восстанавливаем сортировку
        self._table.setSortingEnabled(True)
        self._table.sortItems(sort_col, sort_order)
        self._populating = False   # Задача #124: снять подавление itemChanged

    def _format_duration(self, sec) -> str:
        """Задача #123: длительность с/мин/ч (локализованные единицы)."""
        from awf.ui.i18n import tr
        s = float(sec or 0.0)
        if s < 60.0:
            return f"{s:.0f} {tr('с')}"
        if s < 3600.0:
            return f"{s / 60.0:.1f} {tr('мин')}"
        return f"{s / 3600.0:.2f} {tr('ч')}"

    def set_window_info(self, n_slices, total_seconds) -> None:
        """Задача #123: задать временно́е окно поиска пиков (для метки панели).
        n_slices=None → «—» (файл не загружен)."""
        self._win_slices = n_slices
        self._win_seconds = total_seconds
        self._render_window_label()

    def _render_window_label(self) -> None:
        """Задача #123: собрать локализованную метку окна поиска из сохранённых чисел."""
        from awf.ui.i18n import tr
        if self._win_slices is None:
            self._window_label.setText(f"{tr('Окно поиска')}: —")
            return
        dur = self._format_duration(self._win_seconds)
        self._window_label.setText(
            f"{tr('Окно поиска')}: {tr('весь файл')} "
            f"({self._win_slices} {tr('срезов')}, {dur})"
        )

    def retranslate(self) -> None:
        """Задача #111/#106: применить переводчик tr к заголовкам колонок и метке порога.
        Вызывается из MainWindow при смене языка."""
        from awf.ui.i18n import tr
        translated = [tr(h) for h in self._COL_HEADERS_RU]
        self._table.setHorizontalHeaderLabels(translated)
        # Метка «Найдено: N» — сохраняем N, переводим только префикс
        current = self._count_label.text()
        try:
            n = int(current.split()[-1])
        except (ValueError, IndexError):
            n = 0
        self._count_label.setText(f"{tr('Найдено: ')}{n}")
        # Задача #123: метка временно́го окна тоже зависит от языка
        self._render_window_label()
