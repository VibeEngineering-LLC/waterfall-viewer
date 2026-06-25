"""Обогатить awf/data/nuclides.json полями category/lifetime (Задача 10). Идемпотентно.
Запуск:  PYTHONPATH=. python scripts/enrich_nuclide_categories.py
category — из awf/data/nuclide_categories.json (origin); lifetime — по порогу T½ (60 сут).
"""
import json
from pathlib import Path
from awf.io.nuclide_lib import load_nuclides_json, to_json_obj
from awf.io.nuclide_categories import enrich_all

root = Path(__file__).resolve().parent.parent
path = root / "awf" / "data" / "nuclides.json"
prov = json.loads(path.read_text(encoding="utf-8")).get("_provenance", {})
enriched = enrich_all(load_nuclides_json(path))
prov["categorized"] = True
path.write_text(json.dumps(to_json_obj(enriched, provenance=prov), ensure_ascii=False, indent=2),
                encoding="utf-8")
n_cat = sum(1 for n in enriched if n.category)
print(f"WROTE {path}  nuclides={len(enriched)} categorized={n_cat}")
