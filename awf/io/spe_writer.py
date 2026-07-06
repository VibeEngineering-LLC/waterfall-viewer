"""
LSRM SpectraLine .spe writer.

Spec: awf/io/spe_writer.py — LSRM SpectraLine binary .spe writer
Ref: #226 (replace #217, "correctly Bkmon xml, LSRM spe (binary)")

Writes binary LSRM SpectraLine 2.0 format spectra files.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np


def write_spe(path, counts, *, live_time_s: float, real_time_s: float,
              calibration=None, mea_time: Optional[datetime] = None,
              spec_id: str = "waterfall-viewer export") -> Path:
    path = Path(path)
    arr = np.asarray(counts)
    if arr.ndim != 1:
        raise ValueError("counts must be 1D")
    
    c_u32 = np.clip(np.rint(arr).astype(np.int64), 0, 0xFFFFFFFF).astype("<u4")
    n = int(c_u32.size)

    if mea_time is None:
        mea_time = datetime.now(timezone.utc)
    measbegin = mea_time.strftime("%d-%m-%y %H:%M:%S") + f".{mea_time.microsecond // 10000:02d}"

    if calibration is None:
        calibration = [0.0, 1.0]
    coeffs = [float(x) for x in calibration]
    slots = coeffs + [0.0] * max(0, 7 - len(coeffs))
    slots = slots[:7]
    degree = max(0, min(len(coeffs) - 1, 6))
    energy_line = ",".join([str(degree)] + [f"{float(x):.10g}" for x in slots])

    header_pairs = [
        ("SHIFR", spec_id),
        ("NOMER", "0"),
        ("CONFIGNAME", "waterfall-viewer"),
        ("MEASBEGIN", measbegin),
        ("TLIVE", f"{float(live_time_s):.2f}"),
        ("TREAL", f"{float(real_time_s):.2f}"),
        ("ENERGY", energy_line),
        ("SPECTRSIZE", str(n)),
    ]
    
    header_text = "".join(f"{k}={v}\r\n" for k, v in header_pairs)
    header_bytes = header_text.encode("cp1251", errors="replace")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header_bytes)
        f.write(b"SPECTR=")
        f.write(c_u32.tobytes())
    tmp.replace(path)
    return path
