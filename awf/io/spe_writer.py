"""Задача #217 — экспорт в LSRM/IAEA .spe (ASCII IAEA SPE).

Формат IAEA .spe — ASCII, набор секций-«записей», каждая начинается с
`$KEY:` на своей строке, дальше строки данных до пустой строки или новой
секции. Основные секции: $SPEC_ID / $DATE_MEA / $MEAS_TIM / $DATA /
$ENER_FIT / $ROI. Используется LSRM (Россия), ORTEC MAESTRO, GADRAS.

Референс: IAEA-TECDOC-1275 «Specifications for Radionuclide Metrology
Instrumentation» app. B; ORTEC A65-B32 MAESTRO Software User's Manual §File
Formats (SPE); подтверждено в реальных экспортах LSRM SpectraLine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def write_spe(path, counts, *, live_time_s: float, real_time_s: float,
              calibration=None, mea_time: Optional[datetime] = None,
              spec_id: str = "waterfall-viewer export") -> Path:
    """Экспортировать одномерный спектр в IAEA .spe (LSRM-совместимый).

    calibration — итерируемое [c0..cN] (энергия = c0 + c1*ch + c2*ch^2 + ...);
    в $ENER_FIT пишем первые два коэффициента (c0, c1), общий формат IAEA.
    """
    path = Path(path)
    arr = np.asarray(counts)
    if arr.ndim != 1:
        raise ValueError("counts must be 1D")
    c = arr
    n = int(c.size)
    c_int = np.rint(c).astype(np.int64)

    if mea_time is None:
        mea_time = datetime.now(timezone.utc)
    date_str = mea_time.strftime("%m/%d/%Y")
    time_str = mea_time.strftime("%H:%M:%S")

    coeffs = list(calibration) if calibration is not None else [0.0, 1.0]
    c0 = float(coeffs[0]) if len(coeffs) >= 1 else 0.0
    c1 = float(coeffs[1]) if len(coeffs) >= 2 else 1.0

    lines = []
    lines.append("$SPEC_ID:")
    lines.append(str(spec_id))
    lines.append("$DATE_MEA:")
    lines.append(f"{date_str} {time_str}")
    lines.append("$MEAS_TIM:")
    lines.append(f"{int(round(live_time_s))} {int(round(real_time_s))}")
    lines.append("$DATA:")
    lines.append(f"0 {n - 1}")
    for v in c_int.tolist():
        lines.append(f"       {int(v)}")
    lines.append("$ENER_FIT:")
    lines.append(f"{c0:.6f} {c1:.6f}")
    lines.append("$MCA_CAL:")
    lines.append(str(min(3, len(coeffs))))
    lines.append(" ".join(f"{float(x):.6E}" for x in coeffs[:3]) + " keV")
    lines.append("$ROI:")
    lines.append("0")
    lines.append("$PRESETS:")
    lines.append("None")
    lines.append("0")
    lines.append("0")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(("\r\n".join(lines) + "\r\n").encode("ascii"))
    tmp.replace(path)
    return path
