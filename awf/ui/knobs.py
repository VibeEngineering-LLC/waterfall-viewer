"""Горизонтальные движки-фейдеры панели регулировок (Задача #58; раньше — вертикальные
фейдеры #57, ещё раньше поворотные рукоятки #55, имя классов сохранено ради внешнего API).

`Knob` — горизонтальный фейдер: тёмный паз, зелёный «уровень», риски-шкала сверху/снизу и
светлая металлическая каретка; управление горизонтальным drag мыши. API совместим с
QSlider (value/setValue/setRange/valueChanged), поэтому обработчики main_window читают
значение тем же способом (имя Knob и _*_slider-алиасы целы — внешний код не трогаем).
`KnobRow` — горизонтальный ряд: подпись + движок + значение + индивидуальные вкл/выкл и сброс.
`AdjustPanel` собирает ряды одной колонкой (Задача #58) и кнопку «Сброс всех». Bypass —
по каждому ряду отдельно (per-row вкл/выкл); общий выключатель убран (Задача #91).
"""
from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets


class Knob(QtWidgets.QWidget):
    """Горизонтальный движок-фейдер (Задача #58; имя Knob сохранено ради внешнего API).
    Работает в целых «тиках» (как QSlider), чтобы обработчики main_window читали value()
    и делили на 100 без изменений."""
    valueChanged = QtCore.Signal(int)

    def __init__(self, minimum=0, maximum=100, value=0, parent=None):
        super().__init__(parent)
        self._min = int(minimum)
        self._max = int(maximum)
        self._val = max(self._min, min(self._max, int(value)))
        self._drag_x = None
        self._drag_v0 = 0
        self.setMinimumWidth(80)                   # Задача #58: горизонтальный фейдер —
        self.setFixedHeight(24)                    #   низкий, тянется по ширине ряда
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setCursor(QtCore.Qt.SizeHorCursor)

    def sizeHint(self):
        return QtCore.QSize(120, 24)

    # --- QSlider-совместимый API (минимум, чтобы интегрироваться без правок логики) ---
    def minimum(self) -> int:
        return self._min

    def maximum(self) -> int:
        return self._max

    def value(self) -> int:
        return self._val

    def setRange(self, lo, hi) -> None:
        self._min, self._max = int(lo), int(hi)
        self.setValue(self._val)

    def setValue(self, v) -> None:
        v = max(self._min, min(self._max, int(round(v))))
        changed = (v != self._val)
        self._val = v
        if changed:
            self.valueChanged.emit(v)
        self.update()

    def _frac(self) -> float:
        """Доля заполнения [0..1] по текущему значению (для дуги и указателя)."""
        rng = self._max - self._min
        return 0.0 if rng <= 0 else (self._val - self._min) / float(rng)

    # --- управление мышью: горизонтальный drag (Задача #58; вправо = +, влево = −) ---
    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._drag_x = e.position().x()
            self._drag_v0 = self._val
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_x is None:
            return
        dx = e.position().x() - self._drag_x      # вправо — положительно
        span = max(1, self._max - self._min)
        # полный ход движка ≈ 150 px; Shift — точная подстройка ×0.25
        sens = span / 150.0 * (0.25 if e.modifiers() & QtCore.Qt.ShiftModifier else 1.0)
        self.setValue(self._drag_v0 + dx * sens)
        e.accept()

    def mouseReleaseEvent(self, e):
        self._drag_x = None
        e.accept()

    def wheelEvent(self, e):
        step = 1 if e.angleDelta().y() > 0 else -1
        self.setValue(self._val + step)
        e.accept()

    # --- отрисовка: горизонтальный движок-фейдер (Задача #58) — паз, зелёный «уровень»,
    #     риски-шкала сверху/снизу и светлая металлическая каретка-указатель ---
    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        cy = h / 2.0
        x0, x1 = 7.0, w - 7.0                  # лево (мин.) и право (макс.) хода каретки
        track_w = x1 - x0
        frac = self._frac()
        hx = x0 + frac * track_w               # центр каретки по текущему значению
        gy = cy - 3.5                          # жёлоб: горизонтальный паз
        gg = QtGui.QLinearGradient(0.0, gy, 0.0, gy + 7.0)
        gg.setColorAt(0.0, QtGui.QColor("#1c1e21"))
        gg.setColorAt(1.0, QtGui.QColor("#34373c"))
        p.setPen(QtGui.QPen(QtGui.QColor("#15161a"), 1.0))
        p.setBrush(QtGui.QBrush(gg))
        p.drawRoundedRect(QtCore.QRectF(x0, gy, track_w, 7.0), 3.0, 3.0)

        if frac > 0.0:                         # зелёный «уровень» слева до каретки
            fill = QtGui.QColor("#4a7d4a") if self.isEnabled() else QtGui.QColor("#55585e")
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QBrush(fill))
            p.drawRoundedRect(QtCore.QRectF(x0 + 1.0, gy + 1.0, hx - x0, 5.0), 2.0, 2.0)
        p.setPen(QtGui.QPen(QtGui.QColor("#55585e"), 1.0))   # риски-шкала сверху/снизу паза
        for k in range(5):
            tx = x0 + track_w * k / 4.0
            p.drawLine(QtCore.QPointF(tx, cy + 6.0), QtCore.QPointF(tx, cy + 9.0))
            p.drawLine(QtCore.QPointF(tx, cy - 9.0), QtCore.QPointF(tx, cy - 6.0))
        self._draw_cap(p, cy, hx, h)
        p.end()

    def _draw_cap(self, p, cy, hx, h):
        """Каретка-указатель фейдера (Задача #58): светлая металлическая планка с прорезью."""
        hh = h / 2.0 - 3.0
        cg = QtGui.QLinearGradient(hx - 5.0, 0.0, hx + 5.0, 0.0)
        if self.isEnabled():
            cg.setColorAt(0.0, QtGui.QColor("#d8dbe0"))
            cg.setColorAt(1.0, QtGui.QColor("#8a8d92"))
        else:
            cg.setColorAt(0.0, QtGui.QColor("#6a6d72"))
            cg.setColorAt(1.0, QtGui.QColor("#4a4d52"))
        p.setPen(QtGui.QPen(QtGui.QColor("#15161a"), 1.0))
        p.setBrush(QtGui.QBrush(cg))
        p.drawRoundedRect(QtCore.QRectF(hx - 5.0, cy - hh, 10.0, 2 * hh), 3.0, 3.0)
        p.setPen(QtGui.QPen(QtGui.QColor("#3a3d42"), 1.4))
        p.drawLine(QtCore.QPointF(hx, cy - hh + 3.0), QtCore.QPointF(hx, cy + hh - 3.0))


class KnobRow(QtWidgets.QWidget):
    """Ячейка панели: подпись + рукоятка + значение + индивидуальные вкл/выкл и сброс.
    `changed` испускается при любом изменении (значение ручки ИЛИ переключение вкл/выкл).
    `effective_value()` — значение для применения: само значение, если ряд включён, иначе
    дефолт (bypass). Позиция ручки при выключении сохраняется."""
    changed = QtCore.Signal()

    def __init__(self, key, label, lo, hi, default, fmt=None, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)   # #35: красить фон ряда
        self._key = str(key)
        self._default = int(default)
        self._fmt = fmt or (lambda v: str(v))
        self.knob = Knob(lo, hi, default, self)
        self.knob.valueChanged.connect(self._on_knob)

        self._title = QtWidgets.QLabel(label, self)
        self._title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._title.setObjectName("knobTitle")
        self._title.setFixedWidth(96)                    # Задача #58: ряд в одну колонку
        self._readout = QtWidgets.QLabel(self._fmt(default), self)
        self._readout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._readout.setObjectName("knobValue")
        self._readout.setFixedWidth(56)

        self._chk = QtWidgets.QToolButton(self)         # индивидуальный вкл/выкл
        self._chk.setCheckable(True)
        self._chk.setChecked(False)                     # Задача #60: по умолчанию выкл
        self._chk.setText("выкл")
        self._chk.setObjectName("knobToggle")
        self._chk.setToolTip("Включить/выключить эту регулировку")
        self._chk.toggled.connect(self._on_toggle)
        self.knob.setEnabled(False)                     # #60: ручка/значение погашены под «выкл»
        self._readout.setEnabled(False)

        self._reset = QtWidgets.QToolButton(self)       # сброс к дефолту
        self._reset.setText("⟲")
        self._reset.setObjectName("knobReset")
        self._reset.setToolTip("Сбросить к значению по умолчанию")
        self._reset.clicked.connect(self.reset)

        # Задача #58: ряд горизонтальный (подпись | движок | значение | вкл | сброс);
        # вся панель — одна колонка из таких рядов.
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(6)
        lay.addWidget(self._title)
        lay.addWidget(self.knob, 1)             # горизонтальный движок тянется по ширине
        lay.addWidget(self._readout)
        lay.addWidget(self._chk)
        lay.addWidget(self._reset)

    # --- реакции ---
    def _on_knob(self, v):
        self._readout.setText(self._fmt(int(v)))
        self.changed.emit()

    def _on_toggle(self, on):
        self._chk.setText("вкл" if on else "выкл")
        self.knob.setEnabled(bool(on))
        self._readout.setEnabled(bool(on))
        self.changed.emit()

    def reset(self):
        """Сбросить значение ручки к дефолту (вкл/выкл не трогаем)."""
        self.knob.setValue(self._default)
        self.changed.emit()

    # --- API ---
    def key(self):
        return self._key

    def value(self) -> int:
        return self.knob.value()

    def default(self) -> int:
        return self._default

    def setValue(self, v):
        self.knob.setValue(v)

    def is_on(self) -> bool:
        return self._chk.isChecked()

    def set_on(self, on: bool):
        self._chk.setChecked(bool(on))

    def effective_value(self) -> int:
        """Значение для применения: ручка, если ряд включён, иначе дефолт (bypass)."""
        return self.knob.value() if self._chk.isChecked() else self._default


# Спецификации регулировок отображения (ключ, подпись, lo, hi, дефолт, форматтер).
# Дефолты = «нейтральные» значения, при которых apply_z_scale/сглаживание/свет не меняют картину
# (gain 1.0, gamma 1.0, отсечка 100%, сглаживание 0, свет 0) — это и есть bypass-состояние (#55).
# Задача #56: «Окно t» — относит. ширина выборки по времени (v/100); дефолт 100 (×1.00) нейтрален
# (стандартный max_time=400), главное окно переводит её в число временны́х бинов 3D-водопада.
_SPECS = (
    ("gain",   "Усиление",    20, 500, 100, lambda v: f"{v / 100:.2f}×"),
    ("gamma",  "Гамма",       20, 300, 100, lambda v: f"{v / 100:.2f}"),
    ("clip",   "Отсечка",     80, 100, 100, lambda v: f"{v}%"),
    ("smooth", "Сглаживание",  0,  15,   0, lambda v: f"{v}"),
    ("light",  "Освещение",    0, 100,   0, lambda v: f"{v}%"),
    ("tbin",   "Окно t",      25, 400, 100, lambda v: f"{v / 100:.2f}×"),
)


class AdjustPanel(QtWidgets.QWidget):
    """Панель регулировок отображения в стиле knob-плагина (Задача #55). Собирает ячейки
    `KnobRow` и общий выключатель. `changed` — при любом изменении; `values()` отдаёт
    эффективные значения с учётом per-row и глобального bypass (выкл → дефолты)."""
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)   # #35
        self.setObjectName("adjustPanel")
        self.rows = {}
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(4)
        for i, (key, label, lo, hi, dflt, fmt) in enumerate(_SPECS):
            row = KnobRow(key, label, lo, hi, dflt, fmt, self)
            row.changed.connect(self.changed)
            self.rows[key] = row
            grid.addWidget(row, i, 0)           # Задача #58: одна колонка (ряды друг под другом)

        # Задача #91: общий выключатель «Регулировки: ВКЛ/ВЫКЛ» убран — он был нефункционален
        # (per-row тумблеры уже дают bypass на каждую ручку, а мастер-гейт по дефолту ВЫКЛ
        # глушил все ряды, из-за чего включение отдельной регулировки не давало эффекта).
        self._reset_all = QtWidgets.QToolButton(self)
        self._reset_all.setText("Сброс всех")
        self._reset_all.setObjectName("knobResetAll")
        self._reset_all.clicked.connect(self.reset_all)

        top = QtWidgets.QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self._reset_all)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addLayout(top)
        outer.addLayout(grid)
        outer.addStretch(1)

    # --- реакции / API ---
    def values(self) -> dict:
        """Эффективные значения регулировок: по каждому ряду — значение, если ряд включён,
        иначе его дефолт (per-row bypass). Задача #91: общий выключатель убран."""
        return {k: r.effective_value() for k, r in self.rows.items()}

    def reset_all(self):
        """Сброс всей панели к умолчанию (#60): значения = дефолты, все ряды ВЫКЛ
        (стартовое состояние). Сигналы рядов глушим — пересчёт отображения один раз в конце.
        Гашение ручек делает _on_toggle при set_on(False) (сигнал _chk не заглушён)."""
        for r in self.rows.values():
            r.blockSignals(True)
            r.set_on(False)
            r.reset()
            r.blockSignals(False)
        self.changed.emit()
