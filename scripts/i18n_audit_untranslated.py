"""Аудит #169+: найти русские строковые литералы в awf/, НЕ обёрнутые в tr().

Дополняет scripts/i18n_audit.py (тот проверяет tr()-обёрнутые ключи против EN).
Этот — обратное: где остались голые ru-строки без tr().
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CYR = re.compile(r"[А-Яа-яЁё]")


def is_tr_call(node):
    return (isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == "tr")
                 or (isinstance(node.func, ast.Attribute) and node.func.attr == "tr")))


def collect_tr_arg_ids(tree):
    tr_ids = set()
    for node in ast.walk(tree):
        if is_tr_call(node) and node.args and isinstance(node.args[0], ast.Constant):
            tr_ids.add(id(node.args[0]))
    return tr_ids


def docstring_ids(tree):
    ds = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ds.add(id(body[0].value))
    return ds


def cyr_literals(path):
    src = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    skip = collect_tr_arg_ids(tree) | docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and CYR.search(node.value):
            if id(node) in skip:
                continue
            yield node.value, node.lineno


SKIP_FILES = {"i18n.py"}
hits: dict[str, list[str]] = {}
for py in sorted((ROOT / "awf").rglob("*.py")):
    if py.name in SKIP_FILES:
        continue
    for text, ln in cyr_literals(py):
        rel = py.relative_to(ROOT).as_posix()
        hits.setdefault(text, []).append(f"{rel}:{ln}")

print(f"голых ru-литералов (без tr()): {len(hits)}")
for k in sorted(hits):
    where = ", ".join(hits[k][:3])
    more = f" (+{len(hits[k])-3})" if len(hits[k]) > 3 else ""
    prev = k.replace("\n", " ⏎ ")
    if len(prev) > 80:
        prev = prev[:77] + "..."
    print(f"  {prev!r}\n    <- {where}{more}")