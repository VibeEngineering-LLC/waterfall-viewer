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
/* Задача #97: диалоги (напр. «Выбор фона») — тот же тёмный фон, что и главное окно.
   QSS каскадируется на дочерний диалог (parent=MainWindow), но фон top-level окна
   полезно задать явно. */
QDialog { background-color: #2b2d31; }

QMenuBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4a4d52, stop:1 #303338);
    color: #d8dade;
    font-size: 14px;
}
QMenuBar::item { background: transparent; padding: 5px 12px; }
QMenuBar::item:selected { background: #5a5e64; }
QMenu { background: #303338; color: #d8dade; border: 1px solid #1c1e21;
        font-size: 14px; }
QMenu::item { padding: 5px 24px; }
QMenu::item:selected { background: #4a7d4a; }

QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #484b50, stop:0.5 #34373c, stop:1 #2a2c30);
    border: 0px;
    spacing: 8px;            /* Задача #62: было 3px — тулбар «Вид» скучен */
    padding: 7px 8px;        /* Задача #62: было 3px — выше панель, больше воздух */
    font-size: 14px;         /* Задача #93: 15px→14px — как главное меню (QMenuBar) */
}
/* Задача #62: контролы тулбара «Вид» крупнее и выше. Целюсь по потомкам QToolBar —
   доки-панели (рукоятки/сечения) не вложены в тулбар и не затрагиваются.
   Задача #93: шрифт приведён к 14px — как у главного меню (QMenuBar/QMenu, строки выше). */
QToolBar QLabel { font-size: 14px; }
QToolBar QCheckBox { font-size: 14px; spacing: 7px; }
QToolBar QComboBox { font-size: 14px; padding: 5px 10px; min-height: 28px; }
QToolBar QPushButton { font-size: 14px; padding: 6px 16px; min-height: 28px; }
/* Задача #109: кнопка палитры — QToolButton, у неё не было toolbar-правила, поэтому
   «Viridis» рендерилась дефолтным (мельче и тусклее соседних QLabel/QPushButton).
   Приводим цвет (#d8dade) и размер шрифта (14px) к остальным контролам тулбара. */
QToolBar QToolButton {
    font-size: 14px; color: #d8dade;
    background: #303338; border: 1px solid #1c1e21; border-radius: 3px;
    padding: 6px 14px; min-height: 28px;
}
QToolBar QToolButton:hover { background: #3a3d42; }
QToolBar QToolButton:pressed { background: #4a7d4a; color: #ffffff; }

QStatusBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3a3d42, stop:1 #2a2c30);
    color: #c0c2c6;
    font-size: 14px;         /* Задача #62: строка статуса была мелкой и скученной */
    min-height: 26px;        /* Задача #62: выше строка — путь файла не лепится к тулбару */
    padding: 4px 10px;
}
QStatusBar QLabel { font-size: 14px; }
QStatusBar::item { border: 0px; }   /* без рамок-сепараторов между секциями */

QDockWidget { color: #d8dade; titlebar-close-icon: none; }
QDockWidget::title {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #45484d, stop:1 #303338);
    padding: 5px;
    color: #d8dade;
}
/* Тело открепляемого дока (Задача #35): в плавающем состоянии док становится
   отдельным top-level окном и без явного правила тело виджета-содержимого красится
   системным светлым фоном. Прямой потомок QDockWidget ('>') = виджет содержимого
   (NuclidePanel / SectionControls / SlicePanel); вложенные PlotWidget/GraphicsLayoutWidget
   pyqtgraph — потомки панели, а не дока, поэтому комбинатор '>' их НЕ затрагивает и сцены
   сохраняют собственную тёмную отрисовку. */
QDockWidget > QWidget { background-color: #2b2d31; }

QLabel { color: #cfd2d6; background: transparent; }
QCheckBox { color: #cfd2d6; spacing: 5px; }

/* Задача #97: радиокнопки (диалог «Выбор фона») — тёмный индикатор с зелёным акцентом.
   Без правила QRadioButton рисовался дефолтным синим индикатором Windows и выбивался
   из тёмной темы. Зелёный #4a7d4a — тот же акцент, что у выделения меню/вкладок. */
QRadioButton { color: #cfd2d6; spacing: 6px; }
QRadioButton::indicator {
    width: 14px; height: 14px; border-radius: 8px;
    border: 1px solid #1c1e21; background: #34373c;
}
QRadioButton::indicator:checked { background: #4a7d4a; border: 1px solid #2f5a2f; }
QRadioButton::indicator:hover { border: 1px solid #50545a; }

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
/* Задача #97: кнопки-стрелки спинбоксов — тёмные под тему. Поле было стилизовано,
   а дефолтные ::up/down-button рисовались светлыми/системными и выбивались из
   тёмного диалога «Выбор фона». Стрелки — CSS-треугольники светлого цвета. */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #45484d; border: 1px solid #1c1e21; width: 16px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #50545a; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0; height: 0; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-bottom: 5px solid #cfd2d6;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0; height: 0; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid #cfd2d6;
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

/* Задача #116: таблицы (панель «Найденные пики» #111 — QTableWidget). В теме были
   правила QTreeView/QTreeWidget/QListView, но НЕ QTableView/QTableWidget — поэтому
   viewport таблицы пиков рисовался дефолтным светлым фоном Qt. Цвета — как у дерева
   нуклидов; gridline под тёмный фон, угловая кнопка — как секция заголовка. */
QTableView, QTableWidget {
    background: #26282b; color: #d0d2d6;
    border: 1px solid #1c1e21;
    gridline-color: #1c1e21;
    alternate-background-color: #2d2f33;
    selection-background-color: #3d6b3d; selection-color: #ffffff;
}
QTableView::item:selected, QTableWidget::item:selected {
    background: #3d6b3d; color: #ffffff;
}
QTableCornerButton::section {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #45484d, stop:1 #303338);
    border: 1px solid #1c1e21;
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

/* Задача #55 — панель регулировок «рукоятки» (knob-стиль). */
#adjustPanel { background-color: #2b2d31; }
QLabel#knobTitle { color: #b6b9be; font-size: 11px; }
QLabel#knobValue { color: #9fd29f; font-size: 12px; font-weight: bold; }
QToolButton#knobToggle, QToolButton#knobReset {
    background: #34373c; color: #cfd2d6;
    border: 1px solid #1c1e21; border-radius: 3px; padding: 1px 5px;
}
QToolButton#knobToggle:checked { background: #3d6b3d; color: #ffffff; }
QToolButton#knobToggle:!checked { background: #4a3030; color: #c8b0b0; }
QToolButton#knobReset:hover, QToolButton#knobToggle:hover { background: #50545a; }
QToolButton#knobResetAll {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #45484d, stop:1 #2f3236);
    color: #d8dade; border: 1px solid #1c1e21;
    border-radius: 3px; padding: 3px 10px;
}
QToolButton#knobResetAll:hover { background: #50545a; }
"""