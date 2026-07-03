"""Задача #169: аудит покрытия i18n.

Находит tr("…")-вызовы в awf/, чей ru-ключ отсутствует в TRANSLATIONS[en]
(на EN такие строки останутся русскими). Запуск:
    py -3.14 scripts/i18n_audit.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from awf.ui.i18n import TRANSLATIONS  # noqa: E402

EN = TRANSLATIONS["en"]


def tr_keys(path):
    """Все литеральные аргументы tr(...) в файле: (строка, номер строки)."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        is_tr = (isinstance(node, ast.Call)
                 and ((isinstance(node.func, ast.Name) and node.func.id == "tr")
                      or (isinstance(node.func, ast.Attribute) and node.func.attr == "tr")))
        if (is_tr and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            yield node.args[0].value, node.lineno


missing: dict[str, list[str]] = {}
for py in sorted((ROOT / "awf").rglob("*.py")):
    for key, ln in tr_keys(py):
        if key not in EN:
            missing.setdefault(key, []).append(f"{py.relative_to(ROOT)}:{ln}")
print(f"tr()-ключей без EN-перевода: {len(missing)}")
for k in sorted(missing):
    print(f"  {k!r}  <-  {', '.join(missing[k])}")
