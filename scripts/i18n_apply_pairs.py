"""#I18N-1: вставить пары ru->en из JSON в конец TRANSLATIONS[en] (awf/ui/i18n.py)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "awf/ui/i18n.py").read_text(encoding="utf-8")
pairs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
anchor = "    },\n}\n"
assert src.count(anchor) == 1, f"якорь не уникален: {src.count(anchor)}"

block = ["        # --- #I18N-1: вкладка «Прибор» и диалог калибровки (2026-08-20) ---\n"]
for ru, en in pairs.items():
    if f'\n        "{ru}":' in src:
        continue                      # уже есть — не дублируем
    # json.dumps, а не repr: корректно экранирует внутренние кавычки
    k = json.dumps(ru, ensure_ascii=False)
    v = json.dumps(en, ensure_ascii=False)
    block.append(f'        {k}: {v},\n')
src = src.replace(anchor, "".join(block) + anchor)
(ROOT / "awf/ui/i18n.py").write_text(src, encoding="utf-8", newline="\n")
print(f"добавлено пар: {len(block) - 1}")
