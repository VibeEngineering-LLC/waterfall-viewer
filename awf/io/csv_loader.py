from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
from awf.model.spectrogram import Calibration, Spectrogram

_warned_short = False
_warned_bad = False
_warned_round = False
_warned_neg = False


def _detect_delimiter(lines: list[str]) -> str:
    candidates = [",", ";", "\t"]
    counts = [0] * len(candidates)
    n_lines = min(20, len(lines))
    for i in range(n_lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for j, sep in enumerate(candidates):
            if len(line.split(sep)) >= 2:
                counts[j] += 1
    max_count = max(counts)
    if max_count == 0:
        return ","
    best_idx = counts.index(max_count)
    return candidates[best_idx]


def _try_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_channel_index(xs: list[float]) -> bool:
    if len(xs) == 0:
        return False
    for i, v in enumerate(xs):
        if abs(v - round(v)) > 1e-9:
            return False
        if int(round(v)) != i:
            return False
    return True


def load_csv(path) -> Spectrogram:
    global _warned_short, _warned_bad, _warned_round, _warned_neg

    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        text = f.read()
    lines = text.splitlines()

    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        filtered_lines.append(line)

    if not filtered_lines:
        raise ValueError(f"CSV: файл пуст: {path}")

    sep = _detect_delimiter(filtered_lines)
    header_line = None
    header_index = -1

    for i, line in enumerate(filtered_lines):
        parts = [c.strip() for c in line.split(sep)]
        if len(parts) < 2:
            continue
        try:
            float(parts[0])
            float(parts[1])
        except ValueError:
            continue
        header_line = line
        header_index = i
        break

    if header_line is not None:
        parts = [c.strip() for c in header_line.split(sep)]
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                pass
            else:
                header_index = -1

    data_lines = []
    for i, line in enumerate(filtered_lines):
        if i == header_index:
            continue
        parts = [c.strip() for c in line.split(sep)]
        if len(parts) < 2:
            global _warned_short
            if not _warned_short:
                warnings.warn(f"CSV: строка {i} не содержит двух колонок")
                _warned_short = True
            continue
        try:
            a = float(parts[0])
            b = float(parts[1])
        except ValueError:
            global _warned_bad
            if not _warned_bad:
                warnings.warn(f"CSV: строка {i} содержит некорректные числа")
                _warned_bad = True
            continue
        data_lines.append((a, b))

    if not data_lines:
        raise ValueError(f"CSV: нет валидных числовых строк: {path}")

    xs = [x for x, y in data_lines]
    ys = [y for x, y in data_lines]

    n_channels = len(xs)

    counts_1d = np.zeros(n_channels, dtype=np.int32)
    for i, y in enumerate(ys):
        if abs(y - round(y)) > 1e-9:
            global _warned_round
            if not _warned_round:
                warnings.warn("CSV: округление значений отсчетов")
                _warned_round = True
        y_int = int(round(y))
        if y_int < 0:
            global _warned_neg
            if not _warned_neg:
                warnings.warn("CSV: отрицательные значения отсчетов клипнуты до 0")
                _warned_neg = True
            y_int = 0
        counts_1d[i] = y_int

    if counts_1d.max() <= 65535:
        counts_1d = counts_1d.astype(np.uint16)
    else:
        counts_1d = counts_1d.astype(np.int32)

    counts = counts_1d.reshape(1, n_channels)

    if _is_channel_index(xs):
        calibration = Calibration(coeffs=np.array([0.0, 1.0], dtype=np.float64))
    else:
        a0 = xs[0]
        if n_channels >= 2:
            a1 = (xs[-1] - xs[0]) / (n_channels - 1)
        else:
            a1 = 1.0
        calibration = Calibration(coeffs=np.array([a0, a1], dtype=np.float64))

    time_offsets_s = np.array([0.0], dtype=np.float64)
    real_time_s = np.array([np.nan], dtype=np.float64)
    live_time_s = np.array([np.nan], dtype=np.float64)

    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=time_offsets_s,
        real_time_s=real_time_s,
        live_time_s=live_time_s,
        t0_iso=None,
        source_path=str(path)
    )
