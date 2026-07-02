# -*- coding: utf-8 -*-
"""#159: включить used=True равновесным линиям Th-232 в awf/data/nuclides.json.

Идемпотентно; та же кураторская поправка, что USED_OVERRIDES в awf/io/nuclide_lib.py
(регенерация через scripts/gen_nuclides_json.py применяет её автоматически).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from awf.io.nuclide_lib import USED_OVERRIDES

path = Path(__file__).resolve().parent.parent / "awf" / "data" / "nuclides.json"
data = json.loads(path.read_text(encoding="utf-8"))
n = 0
for nuc in data["nuclides"]:
    es = USED_OVERRIDES.get(nuc["name"])
    if not es:
        continue
    for ln in nuc["lines"]:
        if any(abs(ln["energy"] - e) < 0.01 for e in es) and not ln["used"]:
            ln["used"] = True
            n += 1
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"used=True включено линиям: {n}")
