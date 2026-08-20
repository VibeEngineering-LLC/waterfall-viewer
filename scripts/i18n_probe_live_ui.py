"""#I18N-1: живой зонд — собрать видимые подписи MainWindow в EN и найти кириллицу.

Статический аудит (scripts/i18n_audit.py) видит только tr()-ключи. Здесь строится
реальное окно, язык переключается на EN, и собираются подписи, которые остались
русскими: это и есть то, что видит оператор.

Запуск: py -3.14 scripts/i18n_probe_live_ui.py
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6 import QtGui, QtWidgets

from awf.ui import i18n
from awf.ui.main_window import MainWindow

CYR = re.compile(r"[А-Яа-яЁё]")


def collect(w):
    """Собрать видимые подписи с кириллицей из виджетов и действий."""
    hits = set()
    # pylint: disable=protected-access
    for obj in w.findChildren(QtWidgets.QWidget) + w.findChildren(QtGui.QAction):
        for meth in (
            "text",
            "title",
            "windowTitle",
            "placeholderText",
            "toolTip",
            "suffix",
        ):
            f = getattr(obj, meth, None)
            if not callable(f):
                continue
            try:
                v = f()
            except Exception:  # pylint: disable=broad-except
                continue
            if isinstance(v, str) and CYR.search(v):
                hits.add((type(obj).__name__, meth, v.replace("\n", " ")[:70]))
    return hits


def main():
    """Точка входа."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    i18n.set_language("en")
    hits = collect(w)
    print(f"видимых подписей с кириллицей в EN-режиме: {len(hits)}")
    for cls, meth, v in sorted(hits):
        print(f"  [{cls}.{meth}] {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
