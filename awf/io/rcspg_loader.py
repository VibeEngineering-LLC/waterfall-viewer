from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
from awf.model.spectrogram import Calibration, Spectrogram


def _epoch_ms_to_iso(ms) -> str | None:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(float(ms)/1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return None


def load_rcspg(path, *, max_slices: int | None = None) -> Spectrogram:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    spectrums = doc.get("spectrums") or []
    if max_slices is not None:
        spectrums = spectrums[:max_slices]
    if not spectrums:
        raise ValueError(f"RCSPG: в файле нет спектров (spectrums): {path}")

    n_channels = int(doc.get("channelCount") or 0)
    if n_channels <= 0:
        # Определяем из максимальной длины pulses
        max_len = max(len(s.get("pulses", [])) for s in spectrums)
        if max_len <= 0:
            raise ValueError(f"RCSPG: не удалось определить число каналов: {path}")
        n_channels = max_len

    n_slices = len(spectrums)

    # Определяем dtype по максимальному значению
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

    # База времени
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
        source_path=str(path)
    )
