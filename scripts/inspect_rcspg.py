"""Инспектор структуры .rcspg RadiaCode (Android-текст). Не печатает Device serial.
Использование: python inspect_rcspg.py <path.rcspg>   (Задачи #103/#104)."""
import sys

def main(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]
    print("всего непустых строк:", len(lines))

    # заголовок (имена полей; значение Device serial маскируем)
    print("\n--- заголовок ---")
    for part in lines[0].split("\t"):
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip()
            v = "<masked>" if k.lower().startswith("device") else v.strip()
            print(f"  {k} = {v}")

    has_base = len(lines) > 1 and lines[1].startswith("Spectrum:")
    print("\nстрока Spectrum(base) присутствует:", has_base)
    data = lines[2:] if has_base else lines[1:]
    print("строк-интервалов:", len(data))

    counts = [len(r.split("\t")) for r in data]
    print("токенов в строке: min/max =", min(counts), "/", max(counts))

    print("\n--- первые 8 строк-интервалов: первые 5 токенов ---")
    for r in data[:8]:
        t = r.split("\t")
        print("  n=%4d | t0=%s | t1=%s | t2=%s | t3=%s" % (
            len(t), t[0], t[1] if len(t) > 1 else "-",
            t[2] if len(t) > 2 else "-", t[3] if len(t) > 3 else "-"))

    # где в строке встречаются нецелые токены (float с точкой)?
    float_cols = {}
    for r in data:
        for i, tok in enumerate(r.split("\t")):
            if "." in tok or "e" in tok.lower():
                float_cols[i] = float_cols.get(i, 0) + 1
    print("\nстолбцы с нецелыми (float) токенами {индекс: сколько строк}:", dict(sorted(float_cols.items())))

    # сводка по столбцу 1 и 2 (кандидаты в длительность/дозу)
    def col_summary(idx):
        vals = []
        for r in data:
            t = r.split("\t")
            if len(t) > idx:
                try:
                    vals.append(float(t[idx]))
                except ValueError:
                    pass
        if not vals:
            return "нет данных"
        return "n=%d min=%g max=%g sum=%g mean=%g uniq<=10:%s" % (
            len(vals), min(vals), max(vals), sum(vals), sum(vals)/len(vals),
            sorted(set(vals))[:10])
    print("\nстолбец[1]:", col_summary(1))
    print("столбец[2]:", col_summary(2))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")