"""Задача #169: изоляция тестов от выбора языка в QSettings (реестре машины) —
операторский 'en' утекал в глобальный язык процесса и валил RU-ассерты соседей.
Каждый тест стартует с чистого ключа и RU; выбор оператора возвращается в конце."""
import pytest
from PyQt5 import QtCore

from awf.ui import i18n
from awf.ui.main_window import SETTINGS_ORG, SETTINGS_APP


@pytest.fixture(scope="session", autouse=True)
def _preserve_operator_language():
    saved = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP).value("interface/language")
    yield
    if saved is not None:
        QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP).setValue("interface/language", saved)


@pytest.fixture(autouse=True)
def _fresh_language(_preserve_operator_language):
    QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP).remove("interface/language")
    i18n.reset_for_tests()
    yield
