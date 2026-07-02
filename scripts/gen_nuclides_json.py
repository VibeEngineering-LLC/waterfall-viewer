"""Сгенерировать awf/data/nuclides.json из LSRM SpectraLine .lib.

Запуск:  PYTHONPATH=. python scripts/gen_nuclides_json.py "<путь к .lib>"
Путь к .lib передаётся аргументом (не зашит в код — он машинно-специфичен).
Гамма-линии (энергии/интенсивности) — физические константы; это извлечение, не копия БД.
"""
import json
import sys
from pathlib import Path
from awf.io.nuclide_lib import parse_lsrm_lib, to_json_obj, apply_used_overrides

src = sys.argv[1] if len(sys.argv) > 1 else None
if not src:
    raise SystemExit("usage: gen_nuclides_json.py <path-to-.lib>")
# Задача #159: поверх LSRM-флагов включаем равновесные линии (USED_OVERRIDES).
nucs = apply_used_overrides(parse_lsrm_lib(src))
prov = {
    "source": "LSRM SpectraLine gamma nuclide library (УДС-ГЦ 2024)",
    "library_version": "2.0",
    "database_version": "26",
    "note": ("Энергии и интенсивности "
             "гамма-линий — физические константы. "
             "Извлечено парсером awf/io/nuclide_lib.py."),
    "nuclide_count": len(nucs),
    "line_count": sum(len(n.lines) for n in nucs),
}
obj = to_json_obj(nucs, provenance=prov)
out = Path(__file__).resolve().parent.parent / "awf" / "data" / "nuclides.json"
out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"WROTE {out}  nuclides={prov['nuclide_count']} lines={prov['line_count']}")
