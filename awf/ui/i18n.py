"""Задача #106: лёгкая i18n-подсистема для UI.

Стратегия: ключ перевода = русская строка. Если перевода нет — возвращаем
ru-text (graceful fallback). Это позволяет переводить интерфейс инкрементально:
обернул `tr("…")` — уже работает на русском, добавил в TRANSLATIONS[en] —
заработало на английском. При смене языка эмитим Qt-сигнал `signals.changed`,
панели слушают и пересоздают подписи через свои `retranslate_ui()` методы.

API:
  tr(ru)                  -> str           — перевод (или ru если нет)
  set_language(code)      -> None          — меняет язык и эмитит сигнал
  current_language()      -> "ru" | "en"
  signals.changed         -> QtCore.Signal(str)  — глобальный сигнал
  TRANSLATIONS[en][ru]    -> en перевод
"""
from __future__ import annotations
from PySide6 import QtCore

LANG_RU = "ru"
LANG_EN = "en"
SUPPORTED = (LANG_RU, LANG_EN)
DEFAULT = LANG_RU


class _Signals(QtCore.QObject):
    """Глобальный эмиттер: signals.changed.emit('ru'|'en')."""
    changed = QtCore.Signal(str)


signals = _Signals()
_state = {"lang": DEFAULT}


# Словарь переводов RU -> EN. Ключ = ru-строка, значение = en-строка.
# Покрытие наращивается инкрементально; ru-строки без записи в TRANSLATIONS[en]
# просто возвращаются как есть (видны на русском).
TRANSLATIONS: dict[str, dict[str, str]] = {
    LANG_EN: {
        # Главное меню
        "Файл": "File",
        "Открыть…": "Open…",
        "Выход": "Quit",
        "Изотопы": "Isotopes",
        "Анализ": "Analysis",
        "Инструменты": "Tools",
        "Сервис": "Service",
        "Помощь": "Help",
        "О программе": "About",
        "— наполняется позже —": "— coming later —",
        "Окно изотопов (нуклиды)": "Isotopes (nuclides) window",
        # Задача #115: меню «Инструменты» — окна-доки
        "Срезы / Сечения / Выборки": "Slices / Sections / Samples",
        "Сечения (3D)": "Sections (3D)",
        "Регулировки отображения": "Display adjustments",
        # Меню «Анализ»
        "Выбор фона…": "Select background…",
        "Наложение фона": "Overlay background",
        # Задача #142: тумблеры видимости элементов наложения (тулбар «Вид»)
        "Простыня фона": "Background sheet",
        "Фон среза": "Slice background",
        # Задача #143: тумблер видимости основного 3D-рельефа (тулбар «Вид»)
        "Простыня образца": "Sample sheet",
        # Задача #145: подписи комбобоксов стиля простыней (тулбар «Вид»)
        "  Стиль обр.: ": "  Sample style: ",
        "  Стиль фона: ": "  Bg style: ",
        "Вычет фона": "Subtract background",
        "Мощность дозы (RadiaCode)": "Dose rate (RadiaCode)",
        "Калибровка дозы доступна только для RadiaCode (.rcspg)":
            "Dose calibration is only available for RadiaCode (.rcspg)",
        # Задача #110: поиск фотопиков на 3D-спектрограмме
        "Поиск пиков": "Peak search",
        "Отметить найденные фотопики на 3D-спектрограмме":
            "Mark detected photopeaks on the 3D spectrogram",
        # Задача #111: панель «Найденные пики» и её строки
        "Найденные пики": "Found peaks",
        "Порог значимости, σ": "Significance threshold, σ",
        "Найдено: ": "Found: ",
        # Задача #124: заголовок колонки чекбокса видимости гребня пика
        "Показать": "Show",
        "Энергия, кэВ": "Energy, keV",
        "Канал": "Channel",
        "Значимость": "Significance",
        "Высота": "Height",
        "FWHM": "FWHM",
        "Площадь": "Area",
        # Задача #123: единицы Высоты/Площади (отсчёты) + метка окна поиска
        "Высота, отсч.": "Height, cnt",
        "Площадь, отсч.": "Area, cnt",
        "Окно поиска": "Search window",
        "весь файл": "whole file",
        "срезов": "slices",
        "с": "s",
        "мин": "min",
        "ч": "h",
        # Задача #156: нормализация по эффективности регистрации ε(E)
        "Нормализация по эффективности": "Efficiency normalization",
        "Умножить отсчёты каналов на ε_ref/ε(E) — компенсация падения "
        "эффективности фотопика с энергией":
            "Multiply channel counts by ε_ref/ε(E) — compensate the drop of "
            "photopeak efficiency with energy",
        "Загрузить кривую эффективности…": "Load efficiency curve…",
        "Кривые эффективности (*.efr *.efa *.csv *.txt *.json);;Все файлы (*)":
            "Efficiency curves (*.efr *.efa *.csv *.txt *.json);;All files (*)",
        "Кривая эффективности": "Efficiency curve",
        "Не удалось загрузить кривую эффективности:":
            "Failed to load the efficiency curve:",
        "Кривая:": "Curve:",
        # Задача #131: панель «Сегментация по времени» и её строки
        "Сегментация по времени": "Time segmentation",
        "Сегментация по времени…": "Time segmentation…",
        "Разбить запись по времени и идентифицировать нуклиды в каждом сегменте":
            "Split the record over time and identify nuclides in each segment",
        "Сегментировать": "Segment",
        "Сегментов": "Segments",
        "Сегмент": "Segment",
        "кэВ": "keV",
        "нуклиды не идентифицированы": "no nuclides identified",
        "Сегмент / Нуклид": "Segment / Nuclide",
        "Время / линия": "Time / line",
        "Увер. / счёт": "Confid. / count",
        "Категория / Δ": "Category / Δ",
        # Меню «Сервис → Язык»
        "Язык": "Language",
        "Русский": "Russian",
        "English": "English",
        # Тулбар «Вид»
        "Вид": "View",
        " Z-шкала: ": " Z-scale: ",
        # Заголовок окна и статусные сообщения
        "AtomSpectra Waterfall Viewer": "AtomSpectra Waterfall Viewer",
        "Ошибка загрузки": "Load error",
        "Открыть файл": "Open file",
        # Подписи вкладок
        "3D Waterfall": "3D Waterfall",
        "2D Карта (Время×Энергия)": "2D Map (Time×Energy)",
        "Аналитика": "Analytics",
        # Сообщения статусбара и диалоги открытия
        "Готов. Файл → Открыть… (Ctrl+O)": "Ready. File → Open… (Ctrl+O)",
        "Открыть спектрограмму": "Open spectrogram",
        ("Спектрограммы (*.n42 *.xml *.rcspg *.aswf);;N42 / XML (*.n42 *.xml);;"
         "RadiaCode (*.rcspg);;AtomSpectra (*.aswf);;Все файлы (*)"):
            ("Spectrograms (*.n42 *.xml *.rcspg *.aswf);;N42 / XML (*.n42 *.xml);;"
             "RadiaCode (*.rcspg);;AtomSpectra (*.aswf);;All files (*)"),
        # Задача #169: тулбар «Вид» — палитра/единицы/тумблеры/стили/сброс
        "  Палитра: ": "  Palette: ",
        "Выбрать цветовую палитру": "Choose a color palette",
        "  Единицы: ": "  Units: ",
        "отсчёты": "counts",
        "отсч/с (cps)": "cps",
        "  Время: ": "  Time: ",
        "Оси": "Axes",
        "Подсветка": "Highlight",
        "Подложка": "Floor",
        "Палитра": "Palette",
        "Однотонный": "Solid",
        "Каркас": "Wireframe",
        "Стиль простыни образца": "Sample sheet style",
        "Стиль простыни фона": "Background sheet style",
        "Сброс": "Reset",
        "Показать/скрыть подложку (плоское дно рельефа)":
            "Show/hide the floor (flat base of the relief)",
        "Показать/скрыть простыню образца (основной 3D-рельеф)":
            "Show/hide the sample sheet (the main 3D relief)",
        "Показать/скрыть простыню фона на 3D-спектрограмме":
            "Show/hide the background sheet on the 3D spectrogram",
        "Показать/скрыть кривую фона в окне среза":
            "Show/hide the background curve in the slice plot",
        "Вернуть настройки отображения к значениям по умолчанию":
            "Reset display settings to their defaults",
        # Задача #169: подписи Z-шкалы (awf/ui/zscale.py Z_MODES)
        "Линейная": "Linear",
        "Корень √": "Square root √",
        "Логарифм log10": "Logarithm log10",
        # Задача #169: статусы главного окна, док «Нуклиды», регулировки (knobs)
        "Сначала откройте файл, затем выбирайте фон.":
            "Open a file first, then select the background.",
        "Фон выбран. Доступны «Наложение» и «Вычет».":
            'Background selected. "Overlay" and "Subtract" are available.',
        "Выбор фона": "Background selection",
        "Загрузка": "Loading",
        "каналов": "channels",
        "всего отсчётов": "total counts",
        "Нуклиды": "Nuclides",
        "Усиление": "Gain",
        "Гамма": "Gamma",
        "Отсечка": "Clip",
        "Сглаживание": "Smoothing",
        "Сглаживание E": "Smoothing E",
        "Сглаж. по t": "t-smooth",
        "по сегм.": "by segs",
        "Сглаживать по оси времени внутри каждого временного сегмента независимо": "Smooth time axis within each segment independently",
        "Освещение": "Lighting",
        "Окно t": "t window",
        "вкл": "on",
        "выкл": "off",
        "Включить/выключить эту регулировку": "Enable/disable this adjustment",
        "Сбросить к значению по умолчанию": "Reset to the default value",
        "Сброс всех": "Reset all",
        # Задача #169: док «Сечения (3D)» и панель срезов
        "Секущие плоскости (2 на ось)": "Cutting planes (2 per axis)",
        "Время (с)": "Time (s)",
        "Энергия (кэВ)": "Energy (keV)",
        "Отсчёты (выс.)": "Counts (height)",
        "Файл не загружен": "No file loaded",
        "Энергоокно:": "Energy window:",
        "лог Y": "log Y",
        "Сброс зума": "Reset zoom",
        "Вернуть полный вид графиков среза и времени":
            "Restore the full view of the slice and time plots",
        "Отсчёты": "Counts",
        "Отсчёты/с": "Counts/s",
        "Время, с": "Time, s",
        "Отсчёты в полосе": "Counts in band",
        "Скорость в полосе, отсч/с": "Band rate, cps",
        "полоса ROI": "ROI band",
        "энергоокно": "energy window",
        "доза": "dose",
        "Мощность дозы, ": "Dose rate, ",
        "мощность дозы, ": "dose rate, ",
        "мЗв/ч": "mSv/h",
        "мкЗв/ч": "µSv/h",
        "Срез времени": "Time slice",
        "Канал (энергия)": "Channel (energy)",
        "Время (срез)": "Time (slice)",
        "Загружено: срезов": "Loaded: slices",
        "Интегральный спектр и полная полоса.": "Integral spectrum, full band.",
        "Выборка: срезы": "Sample: slices",
        "каналы": "channels",
        "Сумма отсчётов": "Total counts",
        # Задача #169: панель «Нуклиды»
        "Библиотека нуклидов": "Nuclide library",
        "Мин. интенс., %:": "Min. intensity, %:",
        "только основные": "major lines only",
        "Категории:": "Categories:",
        "Время жизни:": "Lifetime:",
        "Природные": "Natural",
        "Техногенные": "Technogenic",
        "Медицинские": "Medical",
        "Осколочные (деления)": "Fission products",
        "Без категории": "Uncategorized",
        "Короткоживущие": "Short-lived",
        "Долгоживущие": "Long-lived",
        "T½ неизвестно": "T½ unknown",
        "Добавить из IAEA…": "Add from IAEA…",
        "Снять все": "Uncheck all",
        "выбрано нуклидов": "nuclides selected",
        "линий": "lines",
        "Идентификация по найденным пикам": "Identification from found peaks",
        "мин. увер.:": "min. conf.:",
        "Порог уверенности: кандидаты ниже порога не показываются (меньше ложных)":
            "Confidence threshold: candidates below it are hidden (fewer false hits)",
        "Нуклид": "Nuclide",
        "Уверен.": "Confid.",
        "Категория": "Category",
        "Линий": "Lines",
        "Идентификация": "Identification",
        "нуклид(ов) по": "nuclide(s) from",
        "пик(ам)": "peak(s)",
        "Добавить нуклид из IAEA": "Add nuclide from IAEA",
        "Имя нуклида (например, Th-234, Cs-137):":
            "Nuclide name (e.g. Th-234, Cs-137):",
        "IAEA: загрузка": "IAEA: loading",
        "IAEA: добавлен": "IAEA: added",
        "IAEA: ошибка —": "IAEA: error —",
        "Не удалось загрузить нуклид:": "Failed to load nuclide:",
        "гамма-линий не найдено": "no gamma lines found",
        # Задача #169: вкладка «Аналитика»
        "Проекция:": "Projection:",
        "Кластеры:": "Clusters:",
        "Нормировка": "Normalize",
        "Пересчитать": "Recompute",
        "Загрузите файл и нажмите «Пересчитать».": 'Load a file and press "Recompute".',
        "нажмите «Пересчитать» вручную.": 'press "Recompute" manually.',
        "Срезов": "Slices",
        "Метод недоступен (пакет не установлен):":
            "Method unavailable (package not installed):",
        "Ошибка": "Error",
        "Метод": "Method",
        "точек": "points",
        "кластеров": "clusters",
        "дисперсия PCA": "PCA variance",
        "шум": "noise",
        "Кластер": "Cluster",
        "Компонента 1": "Component 1",
        "Компонента 2": "Component 2",
        # Задача #169: диалог «Выбор фона»
        "Из текущего измерения (диапазон срезов)":
            "From the current measurement (slice range)",
        "срезы:": "slices:",
        "Из сечений": "From section planes",
        "Диапазон между секущими плоскостями Времени (3D)":
            "The range between the Time cutting planes (3D)",
        "Из файла (.aswf / .rcspg / .n42)": "From a file (.aswf / .rcspg / .n42)",
        "файл не выбран": "no file selected",
        "Обзор…": "Browse…",
        "Файл фона": "Background file",
        "Файл фона не выбран.": "No background file selected.",
        "Пустой диапазон срезов.": "Empty slice range.",
        "диапазон времени ≈": "time range ≈",
        "Спектрограммы (*.n42 *.xml *.rcspg *.aswf);;Все файлы (*)":
            "Spectrograms (*.n42 *.xml *.rcspg *.aswf);;All files (*)",
        # Задача #169: окно «Цветовая палитра» — заголовок и описания палитр
        "Цветовая палитра": "Color palette",
        "фирменная: синяя база → чёрный → оранжевый":
            "signature: blue base → black → orange",
        "тёмная, контрастная": "dark, high-contrast",
        "мягкая, фиолетово-розовая": "soft, purple-pink",
        "фиолет → жёлтый, без чёрного": "violet → yellow, no black",
        "перцептивная, друг-к-другу слепых": "perceptual, colorblind-friendly",
        "для дальтоников, синь→жёлт": "for colorblind users, blue→yellow",
        "яркая, плавный синий→красн": "vivid, smooth blue→red",
        "классика matlab": "matlab classic",
        "чёрный→красн→жёлт→белый": "black→red→yellow→white",
        "чёрный→синий→белый, спокойная": "black→blue→white, calm",
        "matlab по умолчанию, синь→жёлт": "matlab default, blue→yellow",
        "яркость растёт, ч/б-совместима": "increasing lightness, grayscale-safe",
        "диверг., синий→жёлт→красн": "diverging, blue→yellow→red",
        "голубой→пурпурный, яркая": "cyan→magenta, vivid",
        "моно, для печати": "monochrome, print-friendly",
        # Задача #169: тултипы панелей «Найденные пики» и «Сегментация по времени»
        "Порог значимости Currie L_C (σ): выше → меньше пиков, чище результат":
            "Currie L_C significance threshold (σ): higher → fewer peaks, "
            "cleaner result",
        "Пики ищутся в суммарном по времени (интегральном) спектре всего файла; "
        "дополнительно отмечаются транзиентные пики, значимые в отдельных срезах "
        "(Задача #113). «Высота» и «Площадь» — в отсчётах (сумма по всему файлу).":
            "Peaks are searched in the time-integrated spectrum of the whole file; "
            "transient peaks significant in individual slices are marked as well "
            '(Task #113). "Height" and "Area" are in counts (whole-file sum).',
        "Чувствительность (штраф BIC)": "Sensitivity (BIC penalty)",
        "Штраф BIC за новый сегмент: больше → меньше (крупнее) сегментов":
            "BIC penalty per segment: higher → fewer (coarser) segments",
    },
}


def current_language() -> str:
    """Код текущего языка ('ru' | 'en')."""
    return _state["lang"]


def tr(ru: str) -> str:
    """Перевод ru-строки на текущий язык. Fallback — сам ru-text."""
    lang = _state["lang"]
    if lang == LANG_RU:
        return ru
    return TRANSLATIONS.get(lang, {}).get(ru, ru)


def set_language(code: str) -> None:
    """Сменить язык. code ∈ SUPPORTED; неизвестный код игнорируется молча
    (UI-вызовы безопасны). Эмитит signals.changed(code) только при реальной смене."""
    if code not in SUPPORTED:
        return
    if _state["lang"] == code:
        return
    _state["lang"] = code
    signals.changed.emit(code)


def reset_for_tests() -> None:
    """Тест-хелпер: сбросить состояние к дефолту (RU). Не для прод-кода."""
    _state["lang"] = DEFAULT