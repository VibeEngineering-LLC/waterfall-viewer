"""#UI-241: гейт контрастности текста в HTML-контенте диалогов (WCAG AA >= 4.5:1).

Ссылки справки рисовались дефолтным Qt-синим #0000ff: 1.72:1 на фоне документа
#26282b — нечитаемо (скриншот оператора 2026-08-20). Тест проверяет именованные
цвета awf/ui/help_dialogs.py против обоих фонов тёмной темы.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re

import pytest

from awf.ui import help_dialogs as hd

BG_DIALOG = "#2b2d31"      # фон QDialog / QMessageBox (awf/ui/style.py)
BG_DOCUMENT = "#26282b"    # фон QTextDocument в QTextBrowser (замер палитры)
WCAG_AA = 4.5


def _luminance(hex_color):
    # убрать ведущий #
    hex_color = hex_color.lstrip('#')
    # разобрать три пары символов в int(..., 16) / 255
    r, g, b = (int(hex_color[i:i+2], 16) / 255 for i in range(0, 6, 2))
    # для каждого канала: c / 12.92 если c <= 0.03928, иначе ((c + 0.055) / 1.055) ** 2.4
    def _normalize(c):
        if c <= 0.03928:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4
    r, g, b = map(_normalize, (r, g, b))
    # вернуть 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg, bg):
    l1, l2 = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def test_link_color_readable_on_both_backgrounds():
    for bg in (BG_DIALOG, BG_DOCUMENT):
        ratio = _contrast(hd.LINK_COLOR, bg)
        assert ratio >= WCAG_AA, f"LINK_COLOR {hd.LINK_COLOR} на {bg}: {ratio:.2f}:1 < {WCAG_AA}"


def test_muted_color_readable_on_dialog():
    ratio = _contrast(hd.MUTED_COLOR, BG_DIALOG)
    assert ratio >= WCAG_AA, f"MUTED_COLOR {hd.MUTED_COLOR} на {BG_DIALOG}: {ratio:.2f}:1 < {WCAG_AA}"


def test_about_templates_use_named_colors():
    for tmpl in (hd._ABOUT_RU, hd._ABOUT_EN):
        colors = re.findall(r"#[0-9a-fA-F]{6}", tmpl)
        assert colors, "в шаблоне не найдено ни одного цвета"
        for c in colors:
            assert c in {hd.LINK_COLOR, hd.MUTED_COLOR}, f"цвет {c} не является именованной константой"


def test_help_css_contains_link_color():
    assert hd.LINK_COLOR in hd.HELP_CSS
    assert "a {" in hd.HELP_CSS


def test_gate_rejects_the_old_defaults():
    # тест должен уметь краснеть на прежних значениях
    assert _contrast("#0000ff", BG_DOCUMENT) < WCAG_AA  # прежний цвет ссылок
    assert _contrast("#888888", BG_DIALOG) < WCAG_AA    # прежняя подпись версии


def test_known_reference_ratios():
    # сверка формулы с эталонами WCAG (иначе тест мог бы считать что угодно)
    ratio = _contrast("#ffffff", "#000000")
    assert abs(ratio - 21.0) < 0.01
    ratio = _contrast("#7ab8ff", "#7ab8ff")
    assert abs(ratio - 1.0) < 0.01
