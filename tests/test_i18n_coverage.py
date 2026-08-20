"""Задача #I18N-1: гейт покрытия i18n.

Падает, если tr()-ключ не имеет EN-перевода: на EN такая строка осталась бы
русской. Класс дефекта пойман на скриншоте оператора (меню «Калибровка»,
вкладка Device, диалог калибровки) 2026-08-20.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tr_keys(path):
    """Собрать все ключи tr() вызовов из Python-файла."""
    code = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Проверяем, что это вызов tr()
            if (
                (isinstance(node.func, ast.Name) and node.func.id == "tr") or
                (isinstance(node.func, ast.Attribute) and node.func.attr == "tr")
            ):
                if node.args and isinstance(node.args[0], ast.Constant):
                    yield (node.args[0].value, node.lineno)


def _missing_en():
    """Найти все tr() ключи, отсутствующие в EN-переводах."""
    from awf.ui.i18n import TRANSLATIONS
    en = TRANSLATIONS["en"]
    missing = {}
    for py in sorted((ROOT / "awf").rglob("*.py")):
        for key, ln in _tr_keys(py):
            if key not in en:
                key_path = f"{py.relative_to(ROOT).as_posix()}:{ln}"
                if key not in missing:
                    missing[key] = []
                missing[key].append(key_path)
    return missing


def test_all_tr_keys_have_en_translation():
    """Тест: все tr() ключи должны быть переведены на английский."""
    missing = _missing_en()
    if missing:
        lines = [f"tr()-ключей без EN-перевода: {len(missing)}"]
        for k in sorted(missing):
            lines.append(f"  {k!r} <- {', '.join(missing[k])}")
        pytest.fail("\n".join(lines))


def test_gate_detects_a_missing_key(tmp_path):
    """Мутационный тест: гейт должен поймать отсутствующий ключ."""
    mutant = tmp_path / "mutant.py"
    mutant.write_text('x = tr("ЗАВЕДОМО_ОТСУТСТВУЮЩИЙ_КЛЮЧ_I18N_1")\n', encoding="utf-8")
    keys = [k for k, _ln in _tr_keys(mutant)]
    assert "ЗАВЕДОМО_ОТСУТСТВУЮЩИЙ_КЛЮЧ_I18N_1" in keys
    from awf.ui.i18n import TRANSLATIONS
    assert "ЗАВЕДОМО_ОТСУТСТВУЮЩИЙ_КЛЮЧ_I18N_1" not in TRANSLATIONS["en"]


def test_bom_files_are_parsed():
    """Регрессия: файлы с BOM должны корректно парситься."""
    bom_files = [
        ROOT / "awf" / "io" / "aswf_loader.py",
        ROOT / "awf" / "ui" / "help_dialogs.py"
    ]
    for p in bom_files:
        if not p.exists():
            pytest.skip(f"нет файла: {p}")
        list(_tr_keys(p))  # Просто проверяем, что не бросает исключение
