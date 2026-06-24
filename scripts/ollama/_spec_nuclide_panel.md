# Спецификация: awf/ui/nuclide_panel.py

Виджет-панель выбора нуклидов из библиотеки. Пользователь отмечает галочками нуклиды
(можно несколько = «семейство»), задаёт порог интенсивности и флаг «только основные линии».
Панель испускает сигнал со списком линий для подсветки на графике спектра.

## Импорты
```python
from __future__ import annotations
from PySide6 import QtCore, QtWidgets
```
(numpy/pyqtgraph НЕ нужны. Тип Nuclide НЕ импортировать — работаем утиной типизацией:
у объекта есть атрибут `.name` и метод `.major_lines(min_intensity, only_used)`.)

## Палитра цветов
Модульная константа — кортеж из 12 различимых hex-цветов (строки), напр.:
```python
COLORS = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324")
```
Цвет нуклида = `COLORS[index % len(COLORS)]`, где index — позиция нуклида в библиотеке
(стабилен независимо от того, какие отмечены).

## Класс `NuclidePanel(QtWidgets.QWidget)`

### Сигнал
```python
linesChanged = QtCore.Signal(object)  # несёт list[tuple[float, str, str]] = (energy_keV, color_hex, label)
```

### `__init__(self, nuclides=None, parent=None)`
Построить UI вертикальным layout:
1. Заголовок `QtWidgets.QLabel("Библиотека нуклидов")`.
2. Строка фильтра (горизонтальный layout):
   - `QtWidgets.QLabel("Мин. интенсивность, %:")`
   - `self._min_int = QtWidgets.QDoubleSpinBox()`; range 0..100, single step 1.0, value 5.0, decimals 1.
   - `self._only_used = QtWidgets.QCheckBox("только основные линии")`; setChecked(True).
3. `self._list = QtWidgets.QListWidget()` — список нуклидов с галочками. Растягивается (stretch=1).
4. Строка кнопок:
   - `self._btn_none = QtWidgets.QPushButton("Снять все")`.
5. `self._status = QtWidgets.QLabel("")` — статус «выбрано N нуклидов, M линий».

Сигналы:
- `self._min_int.valueChanged.connect(self._recompute)`
- `self._only_used.stateChanged.connect(self._recompute)`
- `self._list.itemChanged.connect(self._on_item_changed)`
- `self._btn_none.clicked.connect(self.clear_selection)`

Поля: `self._nuclides = []`. В конце, если `nuclides` не None — вызвать `self.set_library(nuclides)`.

### `set_library(self, nuclides) -> None`
Заполнить список. Чтобы не ловить ложные `itemChanged` при заполнении — на время блокировать
сигналы списка: `self._list.blockSignals(True)` … `self._list.blockSignals(False)`.
- `self._nuclides = list(nuclides)`.
- `self._list.clear()`.
- Для каждого нуклида с индексом i:
  - `item = QtWidgets.QListWidgetItem(nuclide.name)`.
  - `item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)`.
  - `item.setCheckState(QtCore.Qt.Unchecked)`.
  - цвет = `COLORS[i % len(COLORS)]`; пометить элемент этим цветом текста:
    `item.setForeground(QtGui.QColor(color))` — для этого добавь импорт `QtGui` в строку
    `from PySide6 import QtCore, QtGui, QtWidgets`.
  - сохранить в элементе данные: `item.setData(QtCore.Qt.UserRole, i)` (индекс нуклида).
  - `self._list.addItem(item)`.
- После заполнения вызвать `self._recompute()`.

### `_on_item_changed(self, item) -> None`
Просто вызвать `self._recompute()`.

### `clear_selection(self) -> None`
Снять все галочки (с блокировкой сигналов, затем один `_recompute`):
перебрать `self._list.item(row)` для всех строк, `setCheckState(Unchecked)`; затем `_recompute()`.

### `_recompute(self) -> None`
Собрать линии всех отмеченных нуклидов и испустить сигнал.
```
min_int = self._min_int.value()
only_used = self._only_used.isChecked()
lines = []
checked = 0
for row in range(self._list.count()):
    item = self._list.item(row)
    if item.checkState() != QtCore.Qt.Checked:
        continue
    checked += 1
    i = item.data(QtCore.Qt.UserRole)
    nuc = self._nuclides[i]
    color = COLORS[i % len(COLORS)]
    for ln in nuc.major_lines(min_intensity=min_int, only_used=only_used):
        label = nuc.name
        lines.append((float(ln.energy), color, label))
self._status.setText(f"выбрано нуклидов: {checked}, линий: {len(lines)}")
self.linesChanged.emit(lines)
```

### `selected_lines(self) -> list`
Публичный геттер: вернуть тот же список, что был бы испущен (для тестов/smoke).
Реализуй через ту же логику, что `_recompute` (можно вынести во внутренний хелпер
`self._collect_lines()` и звать из обоих), но без emit.

## Стиль
- Кратко по-русски комментарии. Без сторонних зависимостей кроме PySide6.
- Класс должен инстанцироваться без QApplication-краша при наличии QApplication.
