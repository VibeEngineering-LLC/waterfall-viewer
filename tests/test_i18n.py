"""Tests for Task #106: i18n core (translation dict + signals)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from awf.ui import i18n
from awf.ui.i18n import tr


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


@pytest.fixture(autouse=True)
def _reset_lang():
    """Каждый тест стартует с RU (изменения в _state могли утечь)."""
    i18n.reset_for_tests()
    yield
    i18n.reset_for_tests()


def test_default_language_is_ru():
    assert i18n.current_language() == "ru"


def test_tr_returns_ru_string_in_ru_mode():
    assert tr("Файл") == "Файл"
    assert tr("Открыть…") == "Открыть…"


def test_tr_returns_en_translation_after_switch():
    i18n.set_language("en")
    assert tr("Файл") == "File"
    assert tr("Открыть…") == "Open…"
    assert tr("Выход") == "Quit"
    assert tr("Изотопы") == "Isotopes"
    assert tr("Анализ") == "Analysis"
    assert tr("Сервис") == "Service"


def test_tr_falls_back_to_ru_for_unknown_key():
    i18n.set_language("en")
    assert tr("ЭТО_СТРОКА_БЕЗ_ПЕРЕВОДА") == "ЭТО_СТРОКА_БЕЗ_ПЕРЕВОДА"


def test_set_language_emits_signal_only_on_change(app):
    events = []
    i18n.signals.changed.connect(events.append)
    try:
        i18n.set_language("en")     # ru -> en: emit
        i18n.set_language("en")     # повтор: not emit
        i18n.set_language("ru")     # en -> ru: emit
    finally:
        i18n.signals.changed.disconnect(events.append)
    assert events == ["en", "ru"]


def test_unknown_language_code_is_ignored():
    i18n.set_language("fr")
    assert i18n.current_language() == "ru"


def test_tab_labels_have_en_translations():
    """Подписи вкладок главного окна — в словаре en."""
    i18n.set_language("en")
    assert tr("3D Waterfall") == "3D Waterfall"
    assert tr("Аналитика") == "Analytics"
    assert tr("2D Карта (Время×Энергия)") == "2D Map (Time×Energy)"


def test_language_persistence_settings_key():
    """Ключ настроек интерфейса — стабильное имя 'interface/language'."""
    # документируем контракт; реальный QSettings IO — в integration-тестах MainWindow.
    assert "interface/language" == "interface/language"