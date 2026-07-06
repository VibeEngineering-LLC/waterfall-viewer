from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
from awf.model.spectrogram import Calibration, Spectrogram

_SECTIONS_RE = re.compile(r"^\$[A-Z0-9_]+:$")

def _parse_sections(text: str) -> dict[str, list[str]]:
    sections = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _SECTIONS_RE.match(line.strip()):
            key = line.strip()[1:-1]  # убираем $ и :
            if key in sections:
                i += 1
                continue
            sections[key] = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if _SECTIONS_RE.match(next_line.strip()):
                    break
                sections[key].append(next_line)
                i += 1
        else:
            i += 1
    return sections

def _first_nonempty(body: list[str]) -> str | None:
    for line in body:
        if line.strip() != "":
            return line
    return None

def _parse_ener_fit(body: list[str]) -> np.ndarray | None:
    line = _first_nonempty(body)
    if line is None:
        return None
    tokens = line.split()
    if len(tokens) < 2:
        return None
    try:
        a0, a1 = float(tokens[0]), float(tokens[1])
    except ValueError:
        return None
    return np.array([a0, a1], dtype=np.float64)

def _parse_mca_cal(body: list[str]) -> np.ndarray | None:
    nonempty = [line for line in body if line.strip() != ""]
    if len(nonempty) < 2:
        return None
    try:
        N = int(nonempty[0].strip())
    except ValueError:
        return None
    if N <= 0:
        return None
    tokens = nonempty[1].split()
    floats = []
    for tok in tokens:
        try:
            f = float(tok)
            floats.append(f)
            if len(floats) == N:
                break
        except ValueError:
            break
    if len(floats) < N:
        return None
    return np.array(floats[:N], dtype=np.float64)

def _parse_spe_date(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    try:
        dt = datetime.strptime(text, "%m/%d/%Y %H:%M:%S")
        return dt.isoformat()
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.isoformat()
    except ValueError:
        pass
    return None

def load_spe(path) -> Spectrogram:
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    sections = _parse_sections(text)
    data_body = sections.get("DATA")
    if not data_body:
        raise ValueError(f"SPE: секция $DATA: не найдена: {path}")
    first_line = _first_nonempty(data_body)
    if not first_line:
        raise ValueError(f"SPE: некорректный заголовок $DATA:: {first_line}")
    try:
        start, end = map(int, first_line.split())
    except ValueError:
        raise ValueError(f"SPE: некорректный заголовок $DATA:: {first_line}")
    values_raw = [int(x) for x in data_body[1:] if x.strip()]
    expected = end - start + 1
    if len(values_raw) < expected:
        warnings.warn(f"SPE: $DATA: усечён — {len(values_raw)} из {expected} каналов: {path}")
        values_raw.extend([0] * (expected - len(values_raw)))
    elif len(values_raw) > expected:
        values_raw = values_raw[:expected]
    n_channels = end + 1
    counts = np.zeros(n_channels, dtype=np.uint16 if max(values_raw, default=0) <= 65535 else np.int32)
    counts[start:start + len(values_raw)] = values_raw
    counts = counts.reshape(1, -1)
    real, live = np.nan, np.nan
    meas_tim_body = sections.get("MEAS_TIM")
    if meas_tim_body:
        line = _first_nonempty(meas_tim_body)
        if line:
            try:
                tokens = line.split()
                if len(tokens) >= 2:
                    live, real = float(tokens[0]), float(tokens[1])
            except ValueError:
                warnings.warn(f"SPE: $MEAS_TIM: ошибка парсинга: {path}")
    # Калибровка
    calibration = None
    mca_cal_body = sections.get("MCA_CAL")
    if mca_cal_body:
        calibration = _parse_mca_cal(mca_cal_body)
    if calibration is None:
        ener_fit_body = sections.get("ENER_FIT")
        if ener_fit_body:
            calibration = _parse_ener_fit(ener_fit_body)
    if calibration is None:
        calibration = np.array([0.0, 1.0], dtype=np.float64)
    calibration = Calibration(coeffs=calibration)
    date_body = sections.get("DATE_MEA")
    t0_iso = _parse_spe_date(_first_nonempty(date_body) or "") if date_body else None
    time_offsets_s = np.array([0.0], dtype=np.float64)
    real_time_s = np.array([real], dtype=np.float64)
    live_time_s = np.array([live], dtype=np.float64)
    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=time_offsets_s,
        real_time_s=real_time_s,
        live_time_s=live_time_s,
        t0_iso=t0_iso,
        source_path=str(path)
    )
