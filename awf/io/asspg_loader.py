"""Задача #DATA-7: загрузчик текстовых спектрограмм мобильного приложения AtomSpectra
(`Spectrogram-<детектор>-<дата>_<время>.txt`, заголовок `FORMAT: 3`). В отличие от `.rcspg`
RadiaCode, где срез несёт только метку и счётчики, здесь у каждого среза есть собственные
GPS-координаты, а преднакопленный до старта записи спектр лежит отдельным блоком."""
from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

from awf.model.spectrogram import Calibration, Spectrogram


def looks_like_asspg(path: str | Path) -> bool:
    """Проверяет, похож ли файл на спектрограмму AtomSpectra (FORMAT: 3)."""
    try:
        with open(path, "rb") as f:
            data = f.read(4096)
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return False
        first_line = text.split("\n")[0].strip()
        return first_line.startswith("FORMAT:")
    except Exception:
        return False


def load_asspg(path: str | Path, *, max_slices: int | None = None) -> Spectrogram:
    """
    Загружает спектрограмму из текстового файла AtomSpectra (FORMAT: 3).
    Задача #DATA-7
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\r\n") for line in f.readlines()]

    # Убираем хвостовые пустые строки
    while lines and not lines[-1]:
        lines.pop()

    if not lines or not lines[0].startswith("FORMAT:"):
        raise ValueError(f"не похоже на спектрограмму AtomSpectra: {path}")

    version = int(lines[0][7:].strip())
    if version != 3:
        raise ValueError(f"неподдерживаемая версия формата: {version} ({path})")

    # Преамбула
    try:
        t0_ms = int(lines[2])
        n_channels = int(lines[9])
        d = int(lines[10])
        if n_channels <= 0:
            raise ValueError(f"файл обрезан / не удалось определить число каналов ({path})")
    except (ValueError, IndexError):
        raise ValueError(f"файл обрезан / не удалось определить число каналов ({path})")

    # Калибровка: строка 10 — СТЕПЕНЬ полинома, коэффициентов на один больше (E = c0 + c1·ch + … + cD·ch^D).
    try:
        coeffs = [float(lines[i]) for i in range(11, 11 + d + 1)]
    except (ValueError, IndexError):
        raise ValueError(f"ошибка чтения калибровки ({path})")
    calibration = Calibration(np.array(coeffs, dtype=np.float64))

    # Базовый спектр
    baseline_start = 11 + d + 1
    if len(lines) < baseline_start + n_channels:
        raise ValueError(f"файл обрезан / не удалось определить число каналов ({path})")
    try:
        baseline = np.array(
            [int(lines[i]) for i in range(baseline_start, baseline_start + n_channels)],
            dtype=np.int64
        )
    except (ValueError, IndexError):
        raise ValueError(f"ошибка чтения базового спектра ({path})")

    # Срезы
    slices_start = baseline_start + n_channels
    if len(lines) <= slices_start:
        raise ValueError(f"в файле нет временных срезов ({path})")

    slice_lines = lines[slices_start:]
    if max_slices is not None:
        # Ограничиваем количество срезов, но читаем до конца блоков
        full_blocks = len(slice_lines) // 5
        actual_blocks = min(full_blocks, max_slices)
        slice_lines = slice_lines[:actual_blocks * 5]

    if len(slice_lines) % 5 != 0:
        # Неполный блок — отбрасываем
        slice_lines = slice_lines[: (len(slice_lines) // 5) * 5]

    n_slices = len(slice_lines) // 5

    if n_slices == 0:
        raise ValueError(f"в файле нет временных срезов ({path})")

    # Собираем данные по срезам
    ts_ms = np.zeros(n_slices, dtype=np.int64)
    latitudes = np.zeros(n_slices, dtype=np.float64)
    longitudes = np.zeros(n_slices, dtype=np.float64)
    durations_s = np.zeros(n_slices, dtype=np.float64)
    counts_data = []

    for i in range(n_slices):
        idx = i * 5
        try:
            ts_ms[i] = int(slice_lines[idx])
            latitudes[i] = float(slice_lines[idx + 1])
            longitudes[i] = float(slice_lines[idx + 2])
            durations_s[i] = float(slice_lines[idx + 3])
        except (ValueError, IndexError):
            raise ValueError(f"ошибка чтения среза {i} ({path})")

        # Счётчики
        counts_str = slice_lines[idx + 4]
        try:
            tokens = [t for t in counts_str.split("\t") if t]
            counts = np.array(tokens, dtype=np.int64)
            if len(counts) > n_channels:
                counts = counts[:n_channels]
            elif len(counts) < n_channels:
                padded = np.zeros(n_channels, dtype=np.int64)
                padded[:len(counts)] = counts
                counts = padded
            counts_data.append(counts)
        except (ValueError, IndexError):
            raise ValueError(f"ошибка чтения счётчиков среза {i} ({path})")

    # Формируем массив counts
    counts = np.array(counts_data, dtype=np.int64)
    if counts.max() <= 65535:
        counts = counts.astype(np.uint16)
    else:
        counts = counts.astype(np.int32)

    # Задача #DATA-7: метка среза — КОНЕЦ интервала, поэтому старт интервала i = метка среза i-1,
    # а для i = 0 — момент старта записи t0. Иначе первый срез уезжал бы на свою длительность.
    prev_ms = np.concatenate(([t0_ms], ts_ms[:-1])) if n_slices > 1 else np.array([t0_ms], dtype=np.int64)
    time_offsets_s = (prev_ms - t0_ms) / 1000.0

    # Защита от немонотонности
    time_offsets_s = np.maximum.accumulate(time_offsets_s)

    # GPS-трек
    gps_track = None
    valid_gps = False
    for i in range(n_slices):
        lat, lon = latitudes[i], longitudes[i]
        if lat != 0 or lon != 0:
            valid_gps = True
            break

    if valid_gps:
        gps_track = np.zeros((n_slices, 2), dtype=np.float64)
        for i in range(n_slices):
            lat, lon = latitudes[i], longitudes[i]
            if lat == 0 and lon == 0:
                gps_track[i] = [np.nan, np.nan]
            else:
                gps_track[i] = [lat, lon]

    # ISO-время
    t0_iso = None
    try:
        match = re.search(r"([+-]\d{4})", lines[1])
        tz_offset_str = match.group(1) if match else None
        if tz_offset_str:
            offset_hours = int(tz_offset_str[1:3])
            offset_minutes = int(tz_offset_str[3:5])
            sign = 1 if tz_offset_str[0] == "+" else -1
            tz = timezone(timedelta(hours=sign * offset_hours, minutes=sign * offset_minutes))
        else:
            tz = timezone.utc

        dt = datetime.fromtimestamp(t0_ms / 1000.0, tz=tz)
        if tz_offset_str:
            # Зона записи известна — метка в ней, смещение с двоеточием (его понимает datetime.fromisoformat).
            t0_iso = re.sub(r"([+-])(\d{2})(\d{2})$", r"\1\2:\3", dt.strftime("%Y-%m-%dT%H:%M:%S%z"))
        else:
            # Зоны в файле нет — UTC в том же виде, что у остальных загрузчиков (aswf/rcspg): «…Z».
            t0_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass

    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=time_offsets_s,
        real_time_s=durations_s,
        live_time_s=durations_s,
        t0_iso=t0_iso,
        source_path=str(path),
        baseline=baseline,
        gps_track=gps_track
    )
