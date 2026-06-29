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

    # Русские заголовки колонок (ключи i18n)
    _COL_HEADERS_RU = [
        "Энергия, кэВ",
        "Канал",
        "Значимость",
        "Высота",
        "FWHM",
        "Площадь",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
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

        # --- таблица пиков ---
        self._table = QtWidgets.QTableWidget(0, len(self._COL_HEADERS_RU))
        self._table.setHorizontalHeaderLabels(self._COL_HEADERS_RU)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        # По умолчанию — сортировка по значимости (колонка 2) по убыванию
        self._table.sortItems(2, QtCore.Qt.DescendingOrder)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionsClickable(True)
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

    def _on_sigma_changed(self, value: float) -> None:
        self.sigmaChanged.emit(float(value))

    def set_peaks(self, peaks: list) -> None:
        """Заполнить таблицу из списка FoundPeak. Корректная числовая сортировка
        через setData(Qt.EditRole, float_val) — НЕ строковая."""
        header = self._table.horizontalHeader()
        sort_col = header.sortIndicatorSection()
        sort_order = header.sortIndicatorOrder()

        # Отключаем сортировку при заполнении (иначе каждая вставка — пересортировка)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        for pk in peaks:
            row = self._table.rowCount()
            self._table.insertRow(row)
            # Значения в порядке колонок
            values = [
                float(pk.energy),
                float(pk.channel),
                float(pk.significance),
                float(pk.height),
                float(pk.fwhm_channels),
                float(pk.area_estimate),
            ]
            # Форматы отображения
            formats = ["{:.1f}", "{:.1f}", "{:.1f}", "{:.0f}", "{:.1f}", "{:.0f}"]
            for col, (val, fmt) in enumerate(zip(values, formats)):
                item = QtWidgets.QTableWidgetItem(fmt.format(val))
                # Числовая сортировка: setData(Qt.EditRole, float)
                item.setData(QtCore.Qt.EditRole, val)
                # Правое выравнивание числовых ячеек
                item.setTextAlignment(
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                )
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self._table.setItem(row, col, item)

        n = len(peaks)
        self._count_label.setText(f"Найдено: {n}")

        # Восстанавливаем сортировку
        self._table.setSortingEnabled(True)
        self._table.sortItems(sort_col, sort_order)

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
