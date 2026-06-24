# AtomSpectra Waterfall Viewer

Просмотрщик и анализатор **waterfall-спектрограмм** гамма-спектрометра.
3D-просмотр с вращением и зумом, 2D-карта Время×Энергия, срезы / сечения / прямоугольные выборки.

**Поддерживаемые форматы:** ANSI N42.42-2011 (`.n42`/`.xml`) и RadiaCode (`.rcspg`, JSON — проверено
на модели RadiaCode-110). Калибровка энергии берётся из самого файла (полином из `coefficients`/`CoefficientValues`).

![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue) ![GUI](https://img.shields.io/badge/GUI-PySide6%20(Qt)-green) ![3D](https://img.shields.io/badge/3D-pyqtgraph%20%2B%20OpenGL-orange) ![Format](https://img.shields.io/badge/format-ANSI%20N42.42--2011-yellowgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

Источник данных-прибора: [atomspectra-waterfall-esp32](https://github.com/VibeEngineering-LLC/atomspectra-waterfall-esp32).
Архитектурные идеи — по мотивам [InterSpec (Sandia)](https://github.com/sandialabs/interspec).

## Возможности v1

- **3D-поверхность** Время × Энергия × Отсчёты (OpenGL, вращение ЛКМ, зум колесом, панорама СКМ).
- **2D-карта** Время×Энергия (лог-контраст) с прямоугольной выборкой (ROI).
- **Срезы (по времени)** — спектр выбранного временного среза.
- **Сечения (по каналу/полосе)** — временной ряд интенсивности в выбранной полосе энергий.
- **Выборки (ROI)** — спектр окна времени + временной ряд полосы + суммарные отсчёты в прямоугольнике.
- **Масштаб без верхнего предела** — суточные записи (десятки тысяч срезов) грузятся потоково
  (`lxml.iterparse`, ограниченная память), 3D и 2D рендерятся через LOD-прорежку (`Spectrogram.downsample`).

## Установка

> Подробная установка с нуля (без опыта Python) — в [`INSTALL.md`](INSTALL.md).
> Известные проблемы и совместимость — в [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

**Вариант A — venv (изолированно, рекомендуется):**

```bat
cd "путь\к\atomspectra-waterfall-viewer"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Вариант B — глобально (без venv):**

```bat
py -3.14 -m pip install -r requirements.txt
```

`run.bat` сам выберет: при наличии `.venv` запустит из него, иначе — глобальным Python.

## Запуск

```bat
run.bat                              REM пустое окно, файл открыть через меню
run.bat "C:\путь\waterfall (7).n42"  REM сразу открыть файл
```

или напрямую:

```bat
.venv\Scripts\python.exe -m awf "C:\путь\к\файлу.n42"
```

## Управление

| Действие | Как |
|---|---|
| Открыть файл | Меню **Файл → Открыть…** (Ctrl+O) |
| Вращать 3D | Левая кнопка мыши + перемещение |
| Зум 3D | Колесо мыши |
| Панорама 3D | Средняя кнопка мыши |
| Выборка (ROI) | Вкладка **2D Карта** → тянуть жёлтый прямоугольник; срезы справа обновляются |

## Архитектура

```
awf/
  model/spectrogram.py   — численное ядро: Calibration (полиномиальная), Spectrogram
                           (срезы, сечения, ROI-суммы, LOD-downsample). Без зависимости от Qt.
  io/n42_loader.py       — потоковый загрузчик N42 (CountedZeroes-декодер: векторный + скалярный
                           эталон; iterparse с освобождением памяти для суточных файлов).
  io/rcspg_loader.py     — загрузчик RadiaCode (.rcspg, JSON): pulses->counts, calib из coefficients.
  ui/view3d.py           — Waterfall3DView (GLViewWidget + GLSurfacePlotItem).
  ui/panels.py           — HeatmapPanel (2D + ROI), SlicePanel (спектр + временной ряд).
  ui/main_window.py      — MainWindow + фоновая загрузка (QThread).
  __main__.py            — точка входа (python -m awf).
```

Численное ядро отделено от UI: его можно тестировать и использовать без графики.

## Тесты

Слой данных покрыт автотестами (опорные значения — независимый оракул из байтов реального файла):

```bat
.venv\Scripts\python.exe -m pytest -q
```

(16 тестов: ISO-длительности, CountedZeroes hand-cases + fuzz vec≡scalar, полиномиальная калибровка,
обратное преобразование энергия→канал, загрузка реального образца, лимит срезов, аналитические примитивы;
+ загрузчик RadiaCode `.rcspg`: форма/dtype, дополнение каналов нулём, калибровка, временные оси,
t0, лимит срезов, пустой файл, авто-вывод числа каналов.)

Графические модули проверяются smoke-скриптами в `scripts/` (строят виджеты на реальном образце без GUI-взаимодействия).

## Стек

Python 3.12 / 3.14 · numpy · lxml · PySide6 (Qt) · pyqtgraph + PyOpenGL.

## Дорожная карта (v2)

Спектроскопия: поиск пиков, идентификация изотопов, калибровка по реперам, ROI-нетто/фон, экспорт.
Возможна упаковка в `.exe` (PyInstaller).

## Лицензия

[MIT](LICENSE) © 2026 Verter73.
