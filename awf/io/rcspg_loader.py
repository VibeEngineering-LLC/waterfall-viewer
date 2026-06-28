from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import struct
import numpy as np
from awf.model.spectrogram import Calibration, Spectrogram

# .rcspg существует в ДВУХ вариантах экспорта приложения RadiaCode (Задача #103):
#   * iOS     — JSON-документ (channelCount / coefficients / spectrums[].pulses / ...);
#   * Android — табулированный ТЕКСТ: строка-заголовок "Spectrogram: ...",
#               строка "Spectrum: <hex>" (служебное + калибровка + преднакопленный спектр)
#               и далее по строке на интервал "<FILETIME>\t<живых_сек>\t<отсчёты по каналам>".
# load_rcspg() определяет формат по первому непустому символу файла и выбирает парсер.


# --- iOS / JSON -------------------------------------------------------------

def _epoch_ms_to_iso(ms) -> str | None:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return None


def _load_rcspg_json(path: Path, *, max_slices: int | None = None) -> Spectrogram:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    spectrums = doc.get("spectrums") or []
    if max_slices is not None:
        spectrums = spectrums[:max_slices]
    if not spectrums:
        raise ValueError(f"RCSPG(JSON): в файле нет спектров (spectrums): {path}")

    n_channels = int(doc.get("channelCount") or 0)
    if n_channels <= 0:
        max_len = max(len(s.get("pulses", [])) for s in spectrums)
        if max_len <= 0:
            raise ValueError(f"RCSPG(JSON): не удалось определить число каналов: {path}")
        n_channels = max_len

    n_slices = len(spectrums)

    gmax = 0
    for s in spectrums:
        pulses = s.get("pulses") or []
        if pulses:
            gmax = max(gmax, max(pulses))
    dtype = np.uint16 if gmax <= 65535 else np.int32

    counts = np.zeros((n_slices, n_channels), dtype=dtype)
    real_arr = np.full(n_slices, np.nan, np.float64)
    live_arr = np.full(n_slices, np.nan, np.float64)
    offsets = np.zeros(n_slices, np.float64)

    base_ms = spectrums[0].get("timestamp")
    if base_ms is None:
        base_ms = doc.get("startTimeTimestamp")

    for i, s in enumerate(spectrums):
        pulses = s.get("pulses") or ()
        m = min(len(pulses), n_channels)
        if m > 0:
            counts[i, :m] = np.asarray(pulses[:m], dtype=dtype)
        ct = s.get("collectTime")
        if ct is not None:
            real_arr[i] = live_arr[i] = float(ct)
        ts = s.get("timestamp")
        if ts is not None and base_ms is not None:
            offsets[i] = (float(ts) - float(base_ms)) / 1000.0

    coeffs = doc.get("coefficients")
    if coeffs is not None:
        calibration = Calibration(coeffs=np.asarray(coeffs, dtype=np.float64))
    else:
        calibration = Calibration(coeffs=np.array([0.0, 1.0], dtype=np.float64))

    t0_iso = _epoch_ms_to_iso(doc.get("startTimeTimestamp") or base_ms)

    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=offsets,
        real_time_s=real_arr,
        live_time_s=live_arr,
        t0_iso=t0_iso,
        source_path=str(path),
    )


# --- Android / текст --------------------------------------------------------

# Windows FILETIME -> Unix: тики по 100 нс от 1601-01-01 UTC; 11644473600 с между 1601 и 1970.
_FILETIME_EPOCH_DIFF = 11644473600.0


def _filetime_to_iso(ft) -> str | None:
    if ft is None:
        return None
    try:
        unix_s = float(ft) / 1e7 - _FILETIME_EPOCH_DIFF
        return datetime.fromtimestamp(unix_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return None


def _parse_header(line: str) -> dict[str, str]:
    # "Spectrogram: <имя>\tTime: ...\tTimestamp: ...\tChannels: 1024\t..." -> словарь ключ->значение.
    out: dict[str, str] = {}
    for part in line.split("\t"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_base_calibration(line: str):
    # "Spectrum: <hex>" -> (a0, a1, a2). Структура hex: 4 служебных байта + 3*float32 LE калибровка
    # + N*uint32 LE преднакопленный спектр (как срез НЕ используем — берём только калибровку).
    body = line.split(":", 1)[1].strip() if ":" in line else ""
    if not body:
        return None
    try:
        raw = bytes(int(tok, 16) for tok in body.split())
    except ValueError:
        return None
    if len(raw) < 16:
        return None
    return struct.unpack("<fff", raw[4:16])


def _load_rcspg_text(path: Path, *, max_slices: int | None = None) -> Spectrogram:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]

    if not lines or not lines[0].startswith("Spectrogram:"):
        raise ValueError(f"RCSPG(текст): не похоже на RadiaCode-спектрограмму: {path}")

    header = _parse_header(lines[0])
    base_ft = header.get("Timestamp")
    base_ft = int(base_ft) if base_ft and base_ft.isdigit() else None

    cal_coeffs = None
    data_start = 1
    if len(lines) > 1 and lines[1].startswith("Spectrum:"):
        cal_coeffs = _parse_base_calibration(lines[1])
        data_start = 2

    rows = lines[data_start:]
    if max_slices is not None:
        rows = rows[:max_slices]
    if not rows:
        raise ValueError(f"RCSPG(текст): в файле нет временны́х срезов: {path}")

    ts_int: list[int] = []
    durations: list[float] = []
    spectra: list[list[int]] = []
    for ln in rows:
        parts = ln.split("\t")
        tok0 = parts[0].strip()
        ts_int.append(int(tok0) if tok0.lstrip("-").isdigit() else 0)
        durations.append(float(parts[1]) if len(parts) > 1 and parts[1].strip() else 1.0)
        spectra.append([int(x) for x in parts[2:] if x.strip()])

    declared = int(header.get("Channels") or 0)
    longest = max((len(s) for s in spectra), default=0)
    n_channels = max(declared, longest)
    if n_channels <= 0:
        raise ValueError(f"RCSPG(текст): не удалось определить число каналов: {path}")
    n_slices = len(spectra)

    gmax = max((max(s) if s else 0 for s in spectra), default=0)
    dtype = np.uint16 if gmax <= 65535 else np.int32
    counts = np.zeros((n_slices, n_channels), dtype=dtype)
    for i, s in enumerate(spectra):
        m = min(len(s), n_channels)
        if m > 0:
            counts[i, :m] = np.asarray(s[:m], dtype=dtype)

    # Живое время среза = поле «живых секунд» прибора (Σ == Accumulation time из заголовка).
    # Реальное время и смещение по оси — из FILETIME-меток (точные, суб-секундные); старт
    # интервала i = метка предыдущего интервала (для i=0 — Timestamp заголовка). Целочисленная
    # арифметика по тикам (метки 18-значные) — затем деление на 1e7, без потери точности.
    live_arr = np.asarray(durations, dtype=np.float64)
    base = base_ft if base_ft is not None else (ts_int[0] if ts_int else 0)
    prev = [base] + ts_int[:-1]
    real_arr = np.array([(ts_int[i] - prev[i]) / 1e7 for i in range(n_slices)], dtype=np.float64)
    real_arr = np.where(real_arr > 0, real_arr, live_arr)
    offsets = np.array([(prev[i] - base) / 1e7 for i in range(n_slices)], dtype=np.float64)
    offsets = np.maximum.accumulate(offsets)

    if cal_coeffs is not None:
        calibration = Calibration(coeffs=np.asarray(cal_coeffs, dtype=np.float64))
    else:
        calibration = Calibration(coeffs=np.array([0.0, 1.0], dtype=np.float64))

    t0_iso = _filetime_to_iso(base)

    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=offsets,
        real_time_s=real_arr,
        live_time_s=live_arr,
        t0_iso=t0_iso,
        source_path=str(path),
    )


# --- диспетчер по формату ---------------------------------------------------

def load_rcspg(path, *, max_slices: int | None = None) -> Spectrogram:
    """Загрузить waterfall-спектрограмму RadiaCode (.rcspg). Поддерживаются ОБА формата экспорта
    (Задача #103): iOS — JSON-документ; Android — табулированный текст с заголовком 'Spectrogram:'.
    Формат определяется по первому непустому символу файла. max_slices ограничивает число срезов."""
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(4096)
    stripped = head.lstrip("﻿ \t\r\n")
    if stripped[:1] == "{":
        return _load_rcspg_json(path, max_slices=max_slices)
    return _load_rcspg_text(path, max_slices=max_slices)