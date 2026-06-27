#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задача #86 — догрузка гамма-линий недостающих изотопов из IAEA LiveChart (ENSDF).

Считает разницу между категоризированным списком (awf/data/nuclide_categories.json)
и текущей библиотекой с линиями (awf/data/nuclides.json) и в порядке приоритета
medical -> natural -> technogenic -> fission тянет decay_rads (rad_types=g) для каждого
недостающего изотопа, строит запись в схеме nuclides.json и (с флагом --merge)
дописывает её в библиотеку.

ВСЕ значения (энергии/интенсивности/период полураспада) берутся из реального CSV IAEA,
НИКОГДА из памяти (анти-галлюцинация §23). Изотопы без пригодных гамма-линий (чистые
бета-излучатели и т. п.) честно помечаются FAIL, а не выдумываются.

Метастабильные (Tc-99m, Ba-137m, Ag-110m): запрос "<n>m" в decay_rads пуст; изомерный
переход живёт в запросе ОСНОВНОГО состояния со строкой p_energy = энергия изомерного
уровня. Уровни проверены прямым обращением к API.

Сеть (тянет всё, пишет кэш и кандидатов, БЕЗ записи в библиотеку):
    python scripts/fetch_missing_nuclides.py
Слияние в библиотеку (из кэша, сеть не нужна):
    python scripts/fetch_missing_nuclides.py --merge

Провенанс: ENSDF / IAEA LiveChart of Nuclides, https://www-nds.iaea.org/
"""
from __future__ import annotations
import csv
import io
import json
import re
import sys
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "awf" / "data"
CAT_JSON = DATA / "nuclide_categories.json"
LIB_JSON = DATA / "nuclides.json"
HERE = Path(__file__).resolve().parent
CAND_JSON = HERE / "_missing_candidates.json"
REPORT_TXT = HERE / "_missing_report.txt"

sys.path.insert(0, str(ROOT))
from awf.io.iaea_fetcher import fetch_iaea_gamma_lines, _cache_path, DEFAULT_CACHE_DIR

PRIORITY = ["medical", "natural", "technogenic", "fission"]
THRESHOLD_SEC = 60 * 86400          # порог "long" из nuclide_categories.json (60 дней)
MIN_INTENSITY_PCT = 0.1             # как в merge_iaea_into_internal
MAX_LINES = 30                      # верхняя граница числа линий на изотоп
YEAR_SEC = 365.25 * 86400
KNOWN_UNIT = {"s": "second", "m": "minute", "h": "hour", "d": "day", "y": "year"}
UNIT_SEC = {"second": 1.0, "minute": 60.0, "hour": 3600.0,
            "day": 86400.0, "year": YEAR_SEC}

# Метастабильные: (запрос основного состояния, энергия изомерного уровня, кэВ).
METASTABLE = {
    "Tc-99m": ("Tc-99", 142.6836),
    "Ba-137m": ("Ba-137", 661.659),
    "Ag-110m": ("Ag-110", 117.59),
}


def _mass_number(name):
    m = re.search(r"-(\d+)", name)
    return int(m.group(1)) if m else None


def _f(row, key):
    v = (row.get(key) or "").strip()
    try:
        return float(v) if v else None
    except ValueError:
        return None


def _value_unit(sec, raw_value, raw_unit):
    """(value, unit_word) для half_life: известную сырую единицу берём как есть,
    иначе выводим из секунд так, чтобы half_life_seconds() дала те же секунды."""
    u = (raw_unit or "").strip().lower()
    if raw_value and u in KNOWN_UNIT:
        try:
            return float(raw_value), KNOWN_UNIT[u]
        except ValueError:
            pass
    s = float(sec)
    if s >= YEAR_SEC:
        return round(s / YEAR_SEC, 6), "year"
    if s >= 86400:
        return round(s / 86400, 6), "day"
    if s >= 3600:
        return round(s / 3600, 6), "hour"
    if s >= 60:
        return round(s / 60, 6), "minute"
    return round(s, 6), "second"


def _read_csv_rows(name_query):
    path = _cache_path(Path(DEFAULT_CACHE_DIR), name_query, "g")
    txt = Path(path).read_text(encoding="utf-8")
    if not txt.strip() or txt.strip() == "0":
        return []
    return list(csv.DictReader(io.StringIO(txt)))


def _halflife_normal(rows):
    """T1/2 родителя из строки основного состояния (p_energy ~ 0)."""
    for r in rows:
        pe = _f(r, "p_energy")
        if pe is not None and abs(pe) >= 1e-3:
            continue
        hl_sec = _f(r, "half_life_sec")
        raw_v = (r.get("half_life") or "").strip()
        raw_u = (r.get("unit_hl") or "").strip()
        if hl_sec:
            return hl_sec, raw_v, raw_u
        u = raw_u.lower()
        if raw_v and u in KNOWN_UNIT:
            try:
                return float(raw_v) * UNIT_SEC[KNOWN_UNIT[u]], raw_v, raw_u
            except ValueError:
                pass
    return None, None, None


def _halflife_isomer(rows, level):
    """T1/2 изомера: start_level_hl строк с start_level_energy ~ level (секунды)."""
    for r in rows:
        sle = _f(r, "start_level_energy")
        slh = _f(r, "start_level_hl")
        if sle is not None and abs(sle - level) < 0.5 and slh:
            return slh, None, None
    return None, None, None


def _gamma_lines(objs, level):
    """Линии: level=None — основное состояние (p_E~0), иначе изомер (p_E~level)."""
    out = []
    for ln in objs:
        if ln.intensity_pct < MIN_INTENSITY_PCT:
            continue
        pe = ln.parent_energy_keV
        if level is None:
            if pe is not None and abs(pe) > 1e-3:
                continue
        elif pe is None or abs(pe - level) > 0.5:
            continue
        out.append({
            "energy": ln.energy_keV,
            "intensity": ln.intensity_pct,
            "d_energy": ln.energy_uncertainty_keV,
            "d_intensity": ln.intensity_uncertainty_pct,
            "line_type": None,
            "used": True,
        })
    out.sort(key=lambda x: -x["intensity"])
    capped = len(out) > MAX_LINES
    out = out[:MAX_LINES]
    out.sort(key=lambda x: x["energy"])
    return out, capped


def build_entry(name, category):
    meta = METASTABLE.get(name)
    query = meta[0] if meta else name
    level = meta[1] if meta else None
    objs = fetch_iaea_gamma_lines(query)            # сеть + кэш
    rows = _read_csv_rows(query)
    lines, capped = _gamma_lines(objs, level)
    if not lines:
        return None, "нет пригодных гамма-линий (>= %.2f%%)" % MIN_INTENSITY_PCT
    if level is None:
        hl_sec, raw_v, raw_u = _halflife_normal(rows)
    else:
        hl_sec, raw_v, raw_u = _halflife_isomer(rows, level)
    if hl_sec is None:
        return None, "не найден период полураспада в CSV"
    hv, hu = _value_unit(hl_sec, raw_v, raw_u)
    lifetime = "long" if hl_sec >= THRESHOLD_SEC else "short"
    entry = {
        "name": name,
        "half_life_value": hv,
        "half_life_unit": hu,
        "gamma_constant": None,
        "atomic_mass": _mass_number(name),
        "category": category,
        "lifetime": lifetime,
        "lines": lines,
        "_source": "IAEA LiveChart ENSDF (decay_rads, rad_types=g)",
    }
    status = "OK (%d линий%s, T=%.4g с, %s)" % (
        len(lines), ", capped" if capped else "", hl_sec, lifetime)
    return entry, status


def missing_in_priority():
    cats = json.loads(CAT_JSON.read_text(encoding="utf-8"))["categories"]
    have = {n["name"] for n in json.loads(LIB_JSON.read_text(encoding="utf-8"))["nuclides"]}
    bycat = {}
    for nm, c in cats.items():
        bycat.setdefault(c, []).append(nm)
    ordered = []
    for c in PRIORITY:
        for nm in sorted(bycat.get(c, [])):
            if nm not in have:
                ordered.append((nm, c))
    return ordered


def main():
    do_merge = "--merge" in sys.argv
    todo = missing_in_priority()
    cands, report = [], []
    report.append("Задача #86 — догрузка изотопов из IAEA (decay_rads, g)")
    report.append("Недостающих: %d (порядок: %s)\n" % (len(todo), " -> ".join(PRIORITY)))
    ok = fail = 0
    for nm, cat in todo:
        try:
            entry, status = build_entry(nm, cat)
        except Exception as e:
            entry, status = None, "ОШИБКА %s: %s" % (type(e).__name__, e)
        if entry:
            cands.append(entry)
            ok += 1
        else:
            fail += 1
        report.append("[%-11s] %-9s %s" % (cat, nm, status))
    report.append("\nИтого: OK=%d FAIL=%d из %d" % (ok, fail, len(todo)))
    CAND_JSON.write_text(json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_TXT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\nКандидаты: %s" % CAND_JSON)
    print("Отчёт:     %s" % REPORT_TXT)

    if do_merge:
        lib = json.loads(LIB_JSON.read_text(encoding="utf-8"))
        have = {n["name"] for n in lib["nuclides"]}
        added = [e for e in cands if e["name"] not in have]
        lib["nuclides"].extend(added)
        line_count = sum(len(n["lines"]) for n in lib["nuclides"])
        prov = lib.setdefault("_provenance", {})
        prov["source"] = ("LSRM SpectraLine (УДС-ГЦ 2024) + IAEA LiveChart ENSDF "
                          "(decay_rads, rad_types=g)")
        prov["iaea_added"] = len(added)
        prov["iaea_endpoint"] = ("https://www-nds.iaea.org/relnsd/v1/data?"
                                 "fields=decay_rads&rad_types=g")
        prov["iaea_fetch_date"] = datetime.date.today().isoformat()
        prov["nuclide_count"] = len(lib["nuclides"])
        prov["line_count"] = line_count
        LIB_JSON.write_text(json.dumps(lib, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nСлито в %s: добавлено %d, всего %d нуклидов / %d линий" % (
            LIB_JSON, len(added), len(lib["nuclides"]), line_count))


if __name__ == "__main__":
    main()