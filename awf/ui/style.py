"""Серая градиентная схема оформления окна (Замечание IV-R1).

QSS в стиле «металлический хром» iZotope Insight: тёмно-серые градиенты для тулбара/меню/
доков/вкладок, светлый текст, зелёный акцент выделения. Стиль НАМЕРЕННО ограничен конкретными
классами Qt-виджетов (QToolBar/QMenuBar/QDockWidget/QTabBar/QLabel/QCheckBox/QComboBox/QSlider/
QPushButton/QSpinBox/QTreeView/QHeaderView/QStatusBar/QMainWindow), чтобы НЕ перекрашивать
внутренние тёмные сцены pyqtgraph (GraphicsLayoutWidget/PlotWidget) и GLViewWidget — у них
собственная отрисовка. Цвета нуклидов в дереве заданы per-item (Qt.ForegroundRole) и поэтому
сохраняются поверх дефолтного цвета текста из этого QSS.
"""

APP_QSS = """
QMainWindow { background-color: #2b2d31; }

QMenuBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4a4d52, stop:1 #303338);
    color: #d8dade;
}
QMenuBar::item { background: transparent; padding: 4px 10px; }
QMenuBar::item:selected { background: #5a5e64; }
QMenu { background: #303338; color: #d8dade; border: 1px solid #1c1e21; }
QMenu::item:selected { background: #4a7d4a; }

QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #484b50, stop:0.5 #34373c, stop:1 #2a2c30);
    border: 0px;
    spacing: 3px;
    padding: 3px;
}

QStatusBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3a3d42, stop:1 #2a2c30);
    color: #c0c2c6;
}

QDockWidget { color: #d8dade; titlebar-close-icon: none; }
QDockWidget::title {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #45484d, stop:1 #303338);
    padding: 5px;
    color: #d8dade;
}

QLabel { color: #cfd2d6; background: transparent; }
QCheckBox { color: #cfd2d6; spacing: 5px; }

QComboBox {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #44474c, stop:1 #34373c);
    color: #e0e2e6;
    border: 1px solid #1c1e21;
    border-radius: 3px;
    padding: 2px 6px;
}
QComboBox:hover { background: #50545a; }
QComboBox QAbstractItemView {
    background: #303338; color: #d8dade;
    selection-background-color: #4a7d4a; selection-color: #ffffff;
    border: 1px solid #1c1e21;
}

QSlider::groove:horizontal {
    height: 6px; border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1f2124, stop:1 #3a3d42);
}
QSlider::handle:horizontal {
    width: 12px; margin: -5px 0; border-radius: 3px;
    border: 1px solid #1c1e21;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #c8cbd0, stop:1 #8a8d92);
}
QSlider::handle:horizontal:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #d8dbe0, stop:1 #9aa0a6);
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #45484d, stop:1 #2f3236);
    color: #d8dade;
    border: 1px solid #1c1e21;
    border-radius: 3px;
    padding: 3px 10px;
}
QPushButton:hover { background: #50545a; }
QPushButton:pressed { background: #4a7d4a; }

QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #34373c; color: #e0e2e6;
    border: 1px solid #1c1e21; border-radius: 3px; padding: 1px 4px;
}

QTabWidget::pane { border: 1px solid #1c1e21; background: #2b2d31; }
QTabBar::tab {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #45484d, stop:1 #303338);
    color: #cfd2d6;
    padding: 5px 12px;
    border: 1px solid #1c1e21;
    border-bottom: 0px;
    border-top-left-radius: 3px; border-top-right-radius: 3px;
}
QTabBar::tab:selected { background: #4a7d4a; color: #ffffff; }
QTabBar::tab:hover:!selected { background: #50545a; }

QTreeView, QTreeWidget, QListView {
    background: #26282b; color: #d0d2d6;
    border: 1px solid #1c1e21;
    alternate-background-color: #2d2f33;
}
QTreeView::item:selected, QTreeWidget::item:selected { background: #3d6b3d; }
QHeaderView::section {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #45484d, stop:1 #303338);
    color: #d0d2d6; padding: 3px; border: 1px solid #1c1e21;
}

QScrollBar:vertical { background: #2a2c30; width: 12px; margin: 0px; }
QScrollBar::handle:vertical {
    background: #50545a; border-radius: 4px; min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background: #2a2c30; height: 12px; margin: 0px; }
QScrollBar::handle:horizontal {
    background: #50545a; border-radius: 4px; min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""