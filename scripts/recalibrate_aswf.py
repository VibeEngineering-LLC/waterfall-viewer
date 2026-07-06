"""
Перезапись калибровки в .aswf файле.
Использование:
  py -3.14 scripts/recalibrate_aswf.py <input.aswf> <output.aswf> c0 c1 c2 c3 c4
"""
from __future__ import annotations
import sys, struct, json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq


def recalibrate(src: Path, dst: Path, new_cal: list[float]) -> None:
    data = src.read_bytes()
    if data[:4] != b"ASWF":
        raise ValueError(f"Не ASWF: {src}")
    header_len = struct.unpack("<I", data[4:8])[0]
    raw_hdr = data[8:8 + header_len]
    hdr = json.loads(raw_hdr.split(b"\x00")[0].strip().decode("utf-8"))
    old_cal = hdr.get("calibration", [])
    hdr["calibration"] = new_cal
    new_json = json.dumps(hdr, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(new_json) > header_len:
        raise ValueError(f"Новый заголовок {len(new_json)} > слот {header_len} байт")
    new_raw = new_json + b"\x00" * (header_len - len(new_json))
    out = data[:8] + new_raw + data[8 + header_len:]
    dst.write_bytes(out)
    print(f"  src:     {src}")
    print(f"  dst:     {dst}")
    print(f"  old cal: {old_cal}")
    print(f"  new cal: {new_cal}")
    print(f"  OK: {len(out)} байт записано")


def fit_calibration(current_cal: list[float],
                    peak_pairs: list[tuple[float, float]],
                    deg: int = 4) -> list[float]:
    """Фит нового полинома по парам (измеренная_E, истинная_E).
    Инвертирует текущую калибровку, находит ch, потом polyfit(ch -> true_E)."""
    def E_cur(ch):
        return np.polyval(current_cal[::-1], ch)
    chs = []
    true_Es = []
    for e_meas, e_true in peak_pairs:
        ch = brentq(lambda c: E_cur(c) - e_meas, 0, 8191)
        chs.append(ch)
        true_Es.append(e_true)
    co = np.polyfit(np.array(chs), np.array(true_Es), deg)[::-1]
    return co.tolist()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    new_cal = [float(x) for x in sys.argv[3:]]
    recalibrate(src, dst, new_cal)