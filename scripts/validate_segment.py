"""Валидация авто-сегментации (Задача #131) на реальном файле.

Использование:
    set PYTHONPATH=<корень проекта>
    py -3.14 scripts/validate_segment.py <path.aswf> [pen_factor]

Путь к файлу передаётся аргументом (не хардкодится — может содержать PII/IP).
"""
import sys
import numpy as np

from awf.io.aswf_loader import load_aswf
from awf.io.nuclide_lib import default_library
from awf.analysis.segment import segment_by_time, identify_segments


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: validate_segment.py <path.aswf> [pen_factor]")
        return 2
    path = sys.argv[1]
    pen_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    sg = load_aswf(path)
    print(f"Загружено: n_slices={sg.n_slices} n_channels={sg.n_channels} "
          f"live_total={float(np.asarray(sg.live_time_s).sum()):.0f} c")

    segs = segment_by_time(sg, pen_factor=pen_factor)
    print(f"\npen_factor={pen_factor}  ->  {len(segs)} сегмент(ов):")
    for i, s in enumerate(segs):
        print(f"  [{i}] срезы {s.t_lo:4d}..{s.t_hi:<4d} "
              f"t={s.t_start_s:7.0f}..{s.t_end_s:7.0f} c  "
              f"live={s.live_time_s:7.0f} c  counts={s.total_counts}")

    print("\nИдентификация по сегментам:")
    lib = default_library()
    sidents = identify_segments(sg, lib, segs)
    for i, si in enumerate(sidents):
        s = si.segment
        names = [(r.nuclide, round(float(r.confidence), 3)) for r in si.idents[:6]]
        print(f"  [{i}] t={s.t_start_s:7.0f}..{s.t_end_s:7.0f} c  "
              f"пиков={len(si.peaks):2d}  ->  {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())