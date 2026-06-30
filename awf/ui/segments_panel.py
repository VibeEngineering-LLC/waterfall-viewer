"""Задача #131: панель «Сегментация по времени» — авто-сегменты записи + посегментная
идентификация нуклидов.

Запись часто временно-структурирована (источник A → источник B → фон); в интегральном по
времени спектре яркий/долгий сегмент забивает слабые. Панель показывает дерево:
сегмент (временной интервал) → идентифицированные в нём нуклиды → совпавшие линии.
Сам расчёт (awf.analysis.segment) выполняет MainWindow и передаёт результат в set_segments().

Сигналы:
    recomputeRequested(float) — запрошен пересчёт сегментации с заданным pen_factor (штраф BIC).
    nuclideSelected(str)      — клик по строке нуклида → подсветить его в библиотеке/на 3D.

Тёмная тема дерева (Base/AlternateBase) — как в PeaksPanel (#116), чтобы пустая область
viewport не белела поверх тёмной темы.
"""
from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from awf.ui.i18n import tr


class SegmentsPanel(QtWidgets.QWidget):
    recomputeRequested = QtCore.Signal(float)
    nuclideSelected = QtCore.Signal(str)

    _COL_HEADERS_RU = [
        "Сегмент / Нуклид",
        "Время / линия",
        "Увер. / счёт",
        "Категория / Δ"
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Строка управления
        ctrl_layout = QtWidgets.QHBoxLayout()

        label = QtWidgets.QLabel("Чувствительность (штраф BIC)")
        label.setObjectName("knobTitle")
        self._pen_box = QtWidgets.QDoubleSpinBox()
        self._pen_box.setObjectName("segPenBox")
        self._pen_box.setRange(0.5, 8.0)
        self._pen_box.setSingleStep(0.5)
        self._pen_box.setValue(2.0)
        self._pen_box.setDecimals(1)
        self._pen_box.setToolTip("Штраф BIC за новый сегмент: больше → меньше (крупнее) сегментов")

        self._btn = QtWidgets.QPushButton("Сегментировать")
        self._btn.setToolTip("Разбить запись по времени и идентифицировать нуклиды в каждом сегменте")

        ctrl_layout.addWidget(label)
        ctrl_layout.addWidget(self._pen_box)
        ctrl_layout.addWidget(self._btn)
        ctrl_layout.addStretch(1)

        layout.addLayout(ctrl_layout)

        # Метка статуса
        self._status = QtWidgets.QLabel("Сегментов: —")
        self._status.setObjectName("knobTitle")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Дерево
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(len(self._COL_HEADERS_RU))
        self._tree.setHeaderLabels(self._COL_HEADERS_RU)
        self._tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._tree.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Тёмная тема
        _pal = self._tree.palette()
        _pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#26282b"))
        _pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#2d2f33"))
        self._tree.setPalette(_pal)

        _vp = self._tree.viewport()
        _vp_pal = _vp.palette()
        _vp_pal.setColor(_vp.backgroundRole(), QtGui.QColor("#26282b"))
        _vp.setPalette(_vp_pal)
        _vp.setAutoFillBackground(True)

        layout.addWidget(self._tree, 1)

        self.setLayout(layout)

        # Сигналы
        self._btn.clicked.connect(self._on_recompute)
        self._pen_box.valueChanged.connect(lambda _v: self._on_recompute())
        self._tree.itemClicked.connect(self._on_item_clicked)

    def _on_recompute(self, *args):
        self.recomputeRequested.emit(float(self._pen_box.value()))

    def _on_item_clicked(self, item, col=0):
        name = item.data(0, QtCore.Qt.UserRole)
        if name is None and item.parent():
            name = item.parent().data(0, QtCore.Qt.UserRole)
        if name:
            self.nuclideSelected.emit(str(name))

    def _fmt_time(self, sec) -> str:
        s = float(sec or 0.0)
        if s < 60:
            return f"{s:.0f} {tr('с')}"
        elif s < 3600:
            return f"{s/60.0:.1f} {tr('мин')}"
        else:
            return f"{s/3600.0:.2f} {tr('ч')}"

    def _fmt_counts(self, n) -> str:
        return f"{int(n):,}".replace(",", " ")

    def set_segments(self, segment_idents):
        self._tree.clear()

        for k, si in enumerate(segment_idents):
            seg = si.segment
            top = QtWidgets.QTreeWidgetItem(
                self._tree,
                [
                    f"{tr('Сегмент')} {k+1}",
                    f"{self._fmt_time(seg.t_start_s)}–{self._fmt_time(seg.t_end_s)}",
                    self._fmt_counts(seg.total_counts),
                    f"{seg.n_slices} {tr('срезов')}"
                ]
            )

            # Жирный шрифт для сегмента
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)

            if si.idents:
                for r in si.idents:
                    child = QtWidgets.QTreeWidgetItem(
                        top,
                        [r.nuclide, "", f"{r.confidence:.2f}", r.category or "—"]
                    )
                    child.setData(0, QtCore.Qt.UserRole, r.nuclide)
                    for m in r.matches:
                        QtWidgets.QTreeWidgetItem(
                            child,
                            [
                                f"{m.line_energy:.1f} {tr('кэВ')}",
                                "",
                                f"I={m.intensity_pct:.1f}%",
                                f"Δ={m.delta_keV:+.2f}"
                            ]
                        )
                    child.setExpanded(True)
            else:
                QtWidgets.QTreeWidgetItem(
                    top,
                    [tr("нуклиды не идентифицированы"), "", "", ""]
                )

            top.setExpanded(True)

        # Подгон ширины колонок
        for i in range(self._tree.columnCount()):
            self._tree.resizeColumnToContents(i)

        n = len(segment_idents)
        self._status.setText(f"{tr('Сегментов')}: {n}")

    def clear_segments(self):
        self._tree.clear()
        self._status.setText(f"{tr('Сегментов')}: —")

    def retranslate(self):
        self._tree.setHeaderLabels([tr(h) for h in self._COL_HEADERS_RU])
        self._btn.setText(tr("Сегментировать"))
        cur = self._status.text()
        suffix = cur.split(":", 1)[1].strip() if ":" in cur else "—"
        self._status.setText(f"{tr('Сегментов')}: {suffix}")
