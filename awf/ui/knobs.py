"""Поворотные рукоятки (knob) в стиле iZotope/ircam VERB (Задача #55).

`Knob` — круглый регулятор с зелёной дугой-индикатором и светлым указателем,
управление вертикальным drag мыши. API совместим с QSlider (value/setValue/setRange/
valueChanged), поэтому обработчики main_window читают значение тем же способом.
`KnobRow` добавляет подпись, поле значения, индивидуальные вкл/выкл и сброс.
`AdjustPanel` собирает ряды и общий выключатель (bypass к дефолтам).
"""
from __future__ import annotations
import math
from PySide6 import QtCore, QtGui, QtWidgets


class Knob(QtWidgets.QWidget):
    """Круглый поворотный регулятор. Работает в целых «тиках» (как QSlider), чтобы
    обработчики main_window читали value() и делили на 100 без изменений."""
    valueChanged = QtCore.Signal(int)

    # Дуга индикатора: 240° раскрыта вниз (как на скрине), от -210° до +30° по экрану.
    _A0 = 210.0
    _SPAN = 240.0

    def __init__(self, minimum=0, maximum=100, value=0, parent=None):
        super().__init__(parent)
        self._min = int(minimum)
        self._max = int(maximum)
        self._val = max(self._min, min(self._max, int(value)))
        self._drag_y = None
        self._drag_v0 = 0
        self.setFixedSize(50, 50)
        self.setCursor(QtCore.Qt.SizeVerCursor)

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

    # --- управление мышью: вертикальный drag (вверх = +, вниз = −), как у аудио-knob ---
    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._drag_y = e.position().y()
            self._drag_v0 = self._val
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_y is None:
            return
        dy = self._drag_y - e.position().y()      # вверх — положительно
        span = max(1, self._max - self._min)
        # полный ход рукоятки ≈ 150 px; Shift — точная подстройка ×0.25
        sens = span / 150.0 * (0.25 if e.modifiers() & QtCore.Qt.ShiftModifier else 1.0)
        self.setValue(self._drag_v0 + dy * sens)
        e.accept()

    def mouseReleaseEvent(self, e):
        self._drag_y = None
        e.accept()

    def wheelEvent(self, e):
        step = 1 if e.angleDelta().y() > 0 else -1
        self.setValue(self._val + step)
        e.accept()

    # --- отрисовка: тёмное тело + металлический обод + зелёная дуга + светлый указатель ---
    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        rim = min(w, h) / 2.0 - 3.0           # радиус обода
        body = rim - 4.0                       # радиус тёмного тела
        # тело: радиальный градиент (центр светлее края) — объём «шайбы»
        grad = QtGui.QRadialGradient(cx, cy - body * 0.3, body * 1.6)
        grad.setColorAt(0.0, QtGui.QColor("#3a3d42"))
        grad.setColorAt(1.0, QtGui.QColor("#1f2124"))
        p.setPen(QtGui.QPen(QtGui.QColor("#15161a"), 1.0))
        p.setBrush(QtGui.QBrush(grad))
        p.drawEllipse(QtCore.QPointF(cx, cy), body, body)

        # дуга-индикатор: трек (тёмный) + заполнение (зелёный акцент) на радиусе обода
        arc_rect = QtCore.QRectF(cx - rim, cy - rim, 2 * rim, 2 * rim)
        frac = self._frac()
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor("#15171a"), 3.0, QtCore.Qt.SolidLine,
                            QtCore.Qt.FlatCap))
        p.drawArc(arc_rect, int(self._A0 * 16), int(-self._SPAN * 16))
        if frac > 0.0:
            fill = QtGui.QColor("#4a7d4a") if self.isEnabled() else QtGui.QColor("#55585e")
            p.setPen(QtGui.QPen(fill, 3.0, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
            p.drawArc(arc_rect, int(self._A0 * 16), int(-self._SPAN * frac * 16))

        # указатель: радиальная риска от ~0.45R до края тела на угле текущего значения
        th = math.radians(self._A0 - self._SPAN * frac)
        ux, uy = math.cos(th), -math.sin(th)          # экранный y вниз → знак минус
        ptr = QtGui.QColor("#e3e5e9") if self.isEnabled() else QtGui.QColor("#6a6d72")
        p.setPen(QtGui.QPen(ptr, 2.4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawLine(QtCore.QPointF(cx + ux * body * 0.42, cy + uy * body * 0.42),
                   QtCore.QPointF(cx + ux * (body - 1.5), cy + uy * (body - 1.5)))
        p.end()


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
        self._title.setAlignment(QtCore.Qt.AlignCenter)
        self._title.setObjectName("knobTitle")
        self._readout = QtWidgets.QLabel(self._fmt(default), self)
        self._readout.setAlignment(QtCore.Qt.AlignCenter)
        self._readout.setObjectName("knobValue")

        self._chk = QtWidgets.QToolButton(self)         # индивидуальный вкл/выкл
        self._chk.setCheckable(True)
        self._chk.setChecked(True)
        self._chk.setText("вкл")
        self._chk.setObjectName("knobToggle")
        self._chk.setToolTip("Включить/выключить эту регулировку")
        self._chk.toggled.connect(self._on_toggle)

        self._reset = QtWidgets.QToolButton(self)       # сброс к дефолту
        self._reset.setText("⟲")
        self._reset.setObjectName("knobReset")
        self._reset.setToolTip("Сбросить к значению по умолчанию")
        self._reset.clicked.connect(self.reset)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        lay.addWidget(self._title)
        lay.addWidget(self.knob, 0, QtCore.Qt.AlignHCenter)
        lay.addWidget(self._readout)
        btns = QtWidgets.QHBoxLayout()
        btns.setSpacing(3)
        btns.addStretch(1)
        btns.addWidget(self._chk)
        btns.addWidget(self._reset)
        btns.addStretch(1)
        lay.addLayout(btns)

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

    def set_global_enabled(self, on: bool):
        """Глобальный выключатель: визуально гасит ручку (значение/состояние сохраняются)."""
        vis = bool(on) and self._chk.isChecked()
        self.knob.setEnabled(vis)
        self._readout.setEnabled(vis)


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
            grid.addWidget(row, i // 3, i % 3)

        self._global = QtWidgets.QToolButton(self)      # общий выключатель всех регулировок
        self._global.setCheckable(True)
        self._global.setChecked(True)
        self._global.setText("Регулировки: ВКЛ")
        self._global.setObjectName("knobGlobal")
        self._global.setToolTip("Выключить все регулировки — отображение как без них (bypass)")
        self._global.toggled.connect(self._on_global)
        self._reset_all = QtWidgets.QToolButton(self)
        self._reset_all.setText("Сброс всех")
        self._reset_all.setObjectName("knobResetAll")
        self._reset_all.clicked.connect(self.reset_all)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self._global)
        top.addStretch(1)
        top.addWidget(self._reset_all)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addLayout(top)
        outer.addLayout(grid)
        outer.addStretch(1)

    # --- реакции / API ---
    def _on_global(self, on):
        self._global.setText("Регулировки: ВКЛ" if on else "Регулировки: ВЫКЛ")
        for r in self.rows.values():
            r.set_global_enabled(bool(on))
        self.changed.emit()

    def is_global_on(self) -> bool:
        return self._global.isChecked()

    def values(self) -> dict:
        """Эффективные значения регулировок. Глобальный выкл → все дефолты (полный bypass);
        иначе по каждому ряду: значение, если ряд включён, иначе его дефолт."""
        if not self._global.isChecked():
            return {k: r.default() for k, r in self.rows.items()}
        return {k: r.effective_value() for k, r in self.rows.items()}

    def reset_all(self):
        """Сброс всей панели к умолчанию: значения = дефолты, все ряды и общий — ВКЛ.
        Сигналы рядов глушим, чтобы пересчёт отображения сработал один раз в конце."""
        for r in self.rows.values():
            r.blockSignals(True)
            r.set_on(True)
            r.reset()
            r.blockSignals(False)
        self._global.blockSignals(True)
        self._global.setChecked(True)
        self._global.setText("Регулировки: ВКЛ")
        self._global.blockSignals(False)
        for r in self.rows.values():
            r.set_global_enabled(True)
        self.changed.emit()
