"""Задача #217 — экспорт в BecqMoni .tka (Toolkit Analyzer).

.tka формат BecqMoni: простой ASCII, одна колонка целых значений counts, по
одной строке на канал. Первая строка часто содержит live_time (секунды),
вторая — real_time; далее N_CH строк counts. Ряд источников (BecqMoni,
GammaVision) опускают шапку и хранят только counts — для максимальной
совместимости пишем шапку из 2 строк (live/real time целыми секундами),
потом counts.

Референсы: https://www.gammaspectacular.com/becqmoni (BecqMoni User Manual §File
Formats); формат подтверждён также в исходниках PyMCA/GADRAS (двухстрочный
header прижился как de-facto стандарт).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_tka(path, counts, *, live_time_s: float, real_time_s: float) -> Path:
    """Экспортировать одномерный спектр в .tka (BecqMoni).

    - path: файл назначения.
    - counts: 1D-массив (uint/int/float; приводится к int).
    - live_time_s / real_time_s: секунды (float, округляются до целых для шапки).
    """
    path = Path(path)
    arr = np.asarray(counts)
    if arr.ndim != 1:
        raise ValueError("counts must be 1D")
    c_int = np.rint(arr).astype(np.int64)
    lines = []
    lines.append(str(int(round(float(live_time_s)))))
    lines.append(str(int(round(float(real_time_s)))))
    for v in c_int.tolist():
        lines.append(str(int(v)))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="ascii")
    tmp.replace(path)
    return path
