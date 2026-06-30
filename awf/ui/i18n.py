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