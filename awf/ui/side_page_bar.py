from __future__ import annotations
from PySide6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QStackedWidget,
                                QStackedLayout, QSizePolicy)
from PySide6.QtGui import QPainter, QColor, QFontMetrics
from PySide6.QtCore import Qt, QRect, QPropertyAnimation, QEasingCurve


class SidePageTab(QPushButton):
    def __init__(self, title: str, side: str, parent=None):
        super().__init__(parent)
        self.setFixedWidth(24)
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(title)
        self._title = title
        self._side = side
        self._active = False

    def set_active(self, active: bool):
        self._active = active
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._active:
            bg = QColor("#3d6b3d")
        elif self.underMouse():
            bg = QColor("#3d4044")
        else:
            bg = QColor("#2b2d31")
        painter.fillRect(self.rect(), bg)
        accent = QColor("#4a7d4a") if self._active else QColor("#6a8faf")
        if self._side == 'left':
            painter.fillRect(self.width() - 2, 0, 2, self.height(), accent)
        else:
            painter.fillRect(0, 0, 2, self.height(), accent)
        painter.setPen(QColor("#d8dade"))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        angle = -90 if self._side == 'left' else 90
        painter.rotate(angle)
        fm = QFontMetrics(painter.font())
        text = fm.elidedText(self._title, Qt.ElideRight, self.height() - 8)
        rect = QRect(-self.height() // 2, -self.width() // 2, self.height(), self.width())
        painter.drawText(rect, Qt.AlignCenter, text)
        painter.restore()


class SidePageContent(QWidget):
    def __init__(self, side: str, parent=None):
        super().__init__(parent)
        self._side = side
        self._anim = QPropertyAnimation(self, b"maximumWidth")
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.InOutQuart)
        layout = QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)
        self.setMaximumWidth(0)
        self.setMinimumWidth(0)
        self._panels: dict[str, tuple[QWidget, int]] = {}
        self._current: str | None = None

    def add_panel(self, title: str, widget: QWidget, preferred_width: int = 350) -> None:
        self._panels[title] = (widget, preferred_width)
        self._stack.addWidget(widget)

    def show_panel(self, title: str) -> None:
        if title not in self._panels:
            return
        widget, width = self._panels[title]
        self._stack.setCurrentWidget(widget)
        self._current = title
        self._anim.stop()
        self._anim.setStartValue(self.maximumWidth())
        self._anim.setEndValue(width)
        self.setMinimumWidth(0)
        self._anim.start()

    def hide_panel(self) -> None:
        self._anim.stop()
        self._anim.setStartValue(self.maximumWidth())
        self._anim.setEndValue(0)
        try:
            self._anim.finished.connect(self._on_hide_done)
        except RuntimeError:
            pass
        self._anim.start()
        self._current = None

    def _on_hide_done(self):
        try:
            self._anim.finished.disconnect(self._on_hide_done)
        except RuntimeError:
            pass

    def is_visible_panel(self) -> bool:
        return self.maximumWidth() > 0

    def current_title(self) -> str | None:
        return self._current


class SidePageBar(QWidget):
    def __init__(self, side: str, content: SidePageContent, parent=None):
        super().__init__(parent)
        self._side = side
        self._content = content
        self._tabs: dict[str, SidePageTab] = {}
        self.setFixedWidth(24)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(1)
        layout.addStretch()

    def add_panel(self, title: str, widget: QWidget, preferred_width: int = 350) -> None:
        self._content.add_panel(title, widget, preferred_width)
        tab = SidePageTab(title, self._side, self)
        self._tabs[title] = tab
        self.layout().insertWidget(self.layout().count() - 1, tab)
        tab.clicked.connect(lambda checked=False, t=title: self._on_tab_clicked(t))

    def _on_tab_clicked(self, title: str) -> None:
        if self._content.current_title() == title and self._content.is_visible_panel():
            self._content.hide_panel()
            self._tabs[title].set_active(False)
        else:
            prev = self._content.current_title()
            if prev and prev in self._tabs:
                self._tabs[prev].set_active(False)
            self._content.show_panel(title)
            self._tabs[title].set_active(True)

    def toggle_panel(self, title: str) -> None:
        self._on_tab_clicked(title)

    def is_panel_open(self, title: str) -> bool:
        return (self._content.current_title() == title
                and self._content.is_visible_panel())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#222426"))
        if self._side == 'left':
            painter.fillRect(self.width() - 1, 0, 1, self.height(), QColor("#45484d"))
        else:
            painter.fillRect(0, 0, 1, self.height(), QColor("#45484d"))