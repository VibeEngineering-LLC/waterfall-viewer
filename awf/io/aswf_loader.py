from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import struct
import json
import warnings
import zlib
import numpy as np
from awf.model.spectrogram import Calibration, Spectrogram

_DTYPE_SIZE = {"uint16": 2, "uint32": 4, "float32": 4}
_DTYPE_FMT  = {"uint16": "<H", "uint32": "<I", "float32": "<f"}


def _epoch_s_to_iso(sec) -> str | None:
    if sec is None:
        return None
    try:
        return datetime.fromtimestamp(float(sec), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return None


def _read_scalar(buf: bytes, off: int, dtype: str):
    fmt = _DTYPE_FMT.get(dtype)
    if fmt is None:
        raise ValueError(f"ASWF: неизвестный dtype {dtype!r}")
    return struct.unpack_from(fmt, buf, off)[0]


def _rle_decode_row(buf: bytes, pos: int, n_channels: int):
    """Декодирует один RLE-спектр v3. Возвращает (tuple_counts, new_pos)."""
    spec: list[int] = []
    while len(spec) < n_channels:
        if pos + 2 > len(buf):
            raise ValueError(f"ASWF: RLE: EOF при декодировании спектра (pos={pos})")
        v = struct.unpack_from("<H", buf, pos)[0]
        pos += 2
        if v < 0x8000:
            spec.append(v)
        elif v < 0xFFFF:
            spec.extend([0] * (v & 0x7FFF))
        else:
            raise ValueError("ASWF: RLE: зарезервированное значение 0xFFFF")
    return tuple(spec[:n_channels]), pos


def _verify_row_crc32(raw_rows: np.ndarray, covers: int, stored_crc: np.ndarray) -> dict:
    """Задача #DATA-1a: пер-строчная проверка CRC32 (ASWF v4). CRC покрывает первые `covers`
    байт строки (spectrum..dose_rate), zlib-совместим (init/финал XOR 0xFFFFFFFF, poly EDB88320).
    Возвращает отчёт: checked/ok/bad + индексы битых строк (до 64) + статус."""
    n = int(raw_rows.shape[0])
    bad: list[int] = []
    for i in range(n):
        calc = zlib.crc32(raw_rows[i, :covers].tobytes()) & 0xFFFFFFFF
        if calc != int(stored_crc[i]):
            bad.append(i)
    return {"algo": "crc32", "checked": n, "ok": n - len(bad), "bad": len(bad),
            "bad_rows": bad[:64], "status": "ok" if not bad else "corrupt"}


def load_aswf(path, *, max_slices: int | None = None) -> Spectrogram:
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError(f"ASWF: файл обрезан (< 8 байт): {path}")
        if head[:4] != b"ASWF":
            raise ValueError(f"ASWF: неверная сигнатура файла: {path}")
        header_len = struct.unpack("<I", head[4:8])[0]

        raw_header = f.read(header_len)
        try:
            hdr = json.loads(raw_header.split(b"\x00")[0].strip().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"ASWF: повреждённый/обрезанный заголовок {path}: {exc}") from exc

        n_channels = int(hdr.get("channels") or 0)
        if n_channels <= 0:
            raise ValueError(f"ASWF: неверное число каналов: {n_channels}")
        version  = int(hdr.get("version") or 1)
        interval = float(hdr.get("interval_sec") or 0.0)
        counts_bytes = n_channels * 2

        # --- Задача #197: baseline section (v3) ---
        baseline_arr = None
        baseline_bytes = 0
        if version >= 3 and "baseline" in hdr:
            bi = hdr["baseline"]
            # #ASWF4-1: спека ASWF_V4_FORMAT.md — ключ "channels"; "count" — legacy (старые тесты)
            bl_count = int(bi.get("channels", bi.get("count")) or n_channels)
            baseline_bytes = bl_count * 4
            bl_raw = f.read(baseline_bytes)
            if len(bl_raw) < baseline_bytes:
                warnings.warn(f"ASWF: {path}: baseline section обрезан ({len(bl_raw)} < {baseline_bytes} байт)")
                baseline_bytes = len(bl_raw)
            else:
                baseline_arr = np.frombuffer(bl_raw, dtype="<u4").astype(np.int64)

        f.seek(0, 2)
        file_size = f.tell()
        data_off  = 8 + header_len + baseline_bytes

        # --- row_fields (v3) vs row_time (v2) vs v1 ---
        use_fields = version >= 3 and "row_fields" in hdr
        compressed = use_fields and bool(hdr.get("compressed", False))

        if use_fields:
            all_fields = {fd["name"]: fd for fd in hdr["row_fields"]}
            # non-spectrum fields sorted by offset
            nonspec = sorted(
                [(fd["name"], fd) for fd in hdr["row_fields"] if fd["name"] != "spectrum"],
                key=lambda x: int(x[1]["offset"]))
            row_stride = int(hdr.get("row_stride") or 0)
        else:
            all_fields = {}
            nonspec    = []
            row_stride = int(hdr.get("row_stride") or counts_bytes)

        if not compressed and row_stride < counts_bytes:
            raise ValueError(f"ASWF: row_stride={row_stride} < counts_bytes={counts_bytes}")

        # --- row count ---
        if compressed:
            n_rows_file = None
        else:
            n_rows_file = (file_size - data_off) // row_stride
            _tail = (file_size - data_off) % row_stride
            if _tail:
                warnings.warn(f"ASWF: {path}: {_tail} байт неполного последнего интервала отброшены")

        saved_rows = hdr.get("saved_rows")
        if saved_rows is None or int(saved_rows) <= 0:
            n_rows = n_rows_file
        else:
            n_rows = int(saved_rows)
            if n_rows_file is not None:
                n_rows = min(n_rows, n_rows_file)

        if max_slices is not None and n_rows is not None:
            n_rows = min(n_rows, max_slices)

        f.seek(data_off)
        payload = f.read()

    # Задача #DATA-1: отчёт целостности (per-row CRC32 v4); None если контроля нет в файле.
    integrity_report = None

    # ========================= UNCOMPRESSED =========================
    if not compressed:
        if not n_rows or n_rows < 1:
            raise ValueError(f"ASWF: нет строк данных: {path}")
        raw_rows = (np.frombuffer(payload[:n_rows * row_stride], dtype=np.uint8)
                    .reshape(n_rows, row_stride))
        counts = (np.ascontiguousarray(raw_rows[:, :counts_bytes])
                  .view("<u2").reshape(n_rows, n_channels).astype(np.uint16, copy=True))

        # duration
        if use_fields and "duration" in all_fields:
            fd = all_fields["duration"]
            off = int(fd["offset"]); sz = _DTYPE_SIZE.get(fd.get("dtype", "uint16"), 2)
            npdt = "<u4" if sz == 4 else "<u2"
            real_time_s = (np.ascontiguousarray(raw_rows[:, off:off+sz])
                           .view(npdt).ravel().astype(np.float64))
        else:
            rt_meta = hdr.get("row_time") or None
            if rt_meta and row_stride > counts_bytes:
                off  = int(rt_meta.get("offset") or counts_bytes)
                npdt = "<u4" if str(rt_meta.get("dtype","uint16")) == "uint32" else "<u2"
                sz   = 4 if npdt == "<u4" else 2
                real_time_s = (np.ascontiguousarray(raw_rows[:, off:off+sz])
                               .view(npdt).ravel().astype(np.float64))
            else:
                real_time_s = np.full(n_rows, interval if interval > 0 else np.nan, dtype=np.float64)

        # timestamp (v3)
        timestamps = None
        if use_fields and "timestamp" in all_fields:
            fd = all_fields["timestamp"]; off = int(fd["offset"])
            timestamps = (np.ascontiguousarray(raw_rows[:, off:off+4])
                          .view("<u4").ravel().astype(np.float64))

        # GPS (v3)
        gps_track = None
        if use_fields and "latitude" in all_fields and "longitude" in all_fields:
            lat_fd = all_fields["latitude"]; lon_fd = all_fields["longitude"]
            lats = (np.ascontiguousarray(raw_rows[:, int(lat_fd["offset"]):int(lat_fd["offset"])+4])
                    .view("<f4").ravel().astype(np.float64))
            lons = (np.ascontiguousarray(raw_rows[:, int(lon_fd["offset"]):int(lon_fd["offset"])+4])
                    .view("<f4").ravel().astype(np.float64))
            gps_track = np.column_stack([lats, lons])

        # dose_rate (v3: имя dose_rate_usv_h; v4: имя dose_rate — принимаем оба)
        dose_rate_usv_h = None
        _dose_name = ("dose_rate_usv_h" if "dose_rate_usv_h" in all_fields
                      else ("dose_rate" if "dose_rate" in all_fields else None))
        if use_fields and _dose_name:
            fd = all_fields[_dose_name]; off = int(fd["offset"])
            dose_rate_usv_h = (np.ascontiguousarray(raw_rows[:, off:off+4])
                               .view("<f4").ravel().astype(np.float64))

        # Задача #DATA-1a: проверка целостности per-row CRC32 (только v4 — поле crc32 в шапке).
        if use_fields and "crc32" in all_fields:
            cfd = all_fields["crc32"]
            covers = int(cfd.get("covers", cfd.get("offset", 0)))
            coff = int(cfd["offset"])
            if 0 < covers <= row_stride - 4 and coff + 4 <= row_stride:
                stored_crc = np.ascontiguousarray(raw_rows[:, coff:coff+4]).view("<u4").ravel()
                integrity_report = _verify_row_crc32(raw_rows, covers, stored_crc)
                integrity_report["version"] = version

    # ========================== COMPRESSED ==========================
    else:
        rows_counts: list = []
        field_vals: dict[str, list] = {name: [] for name, _ in nonspec}
        pos = 0
        while pos < len(payload):
            if max_slices is not None and len(rows_counts) >= max_slices:
                break
            spec_tuple, pos = _rle_decode_row(payload, pos, n_channels)
            rows_counts.append(spec_tuple)
            for name, fd in nonspec:
                dtype = fd.get("dtype", "uint16")
                field_vals[name].append(_read_scalar(payload, pos, dtype))
                pos += _DTYPE_SIZE.get(dtype, 2)

        n_rows = len(rows_counts)
        if n_rows < 1:
            raise ValueError(f"ASWF: нет строк данных (compressed): {path}")
        counts = np.array(rows_counts, dtype=np.uint16)

        dur_list = field_vals.get("duration")
        real_time_s = (np.array(dur_list, dtype=np.float64) if dur_list
                       else np.full(n_rows, interval if interval > 0 else np.nan, dtype=np.float64))

        ts_list = field_vals.get("timestamp")
        timestamps = np.array(ts_list, dtype=np.float64) if ts_list else None

        lat_list = field_vals.get("latitude")
        lon_list = field_vals.get("longitude")
        gps_track = (np.column_stack([lat_list, lon_list])
                     if lat_list and lon_list else None)

        dose_list = field_vals.get("dose_rate_usv_h") or field_vals.get("dose_rate")
        dose_rate_usv_h = np.array(dose_list, dtype=np.float64) if dose_list else None

        # Задача #DATA-1a: CRC покрывает сырую (несжатую) раскладку строки; в RLE-режиме
        # исходные байты не восстанавливаем — помечаем контроль как пропущенный.
        if "crc32" in field_vals:
            integrity_report = {"algo": "crc32", "checked": 0, "ok": 0, "bad": 0,
                                "bad_rows": [], "status": "skipped_compressed", "version": version}

    # ========================= TIME AXIS ===========================
    real_time_s_adj = real_time_s.copy()
    if interval > 0:
        real_time_s_adj[real_time_s_adj == 0.0] = interval

    time_offsets_s = np.zeros(n_rows, dtype=np.float64)
    if n_rows > 1:
        time_offsets_s[1:] = np.cumsum(real_time_s_adj[:-1])

    # override with per-row absolute timestamps where valid (v3)
    if timestamps is not None:
        started_at = float(hdr.get("started_at") or 0.0)
        if started_at > 0:
            valid = (timestamps > 0) & np.isfinite(timestamps)
            if valid.any():
                time_offsets_s[valid] = timestamps[valid] - started_at

    live_time_s = real_time_s_adj.copy()

    # ====================== CALIBRATION / t0 =======================
    # Задача #DATA-1b/1c: межсегментные контроли (seg_seq/total_at_open) в сшитом файле вьюера
    # проверить нельзя (одна шапка) — прикладываем как информацию к отчёту, если есть.
    if integrity_report is not None:
        if hdr.get("seg_seq") is not None:
            integrity_report["seg_seq"] = int(hdr["seg_seq"])
        if hdr.get("total_at_open") is not None:
            integrity_report["total_at_open"] = int(hdr["total_at_open"])

    cal = hdr.get("calibration")
    calibration = (Calibration(coeffs=np.asarray(cal, dtype=np.float64)) if cal
                   else Calibration(coeffs=np.array([0.0, 1.0], dtype=np.float64)))
    t0_iso = _epoch_s_to_iso(hdr.get("started_at"))

    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=time_offsets_s,
        real_time_s=real_time_s,
        live_time_s=live_time_s,
        t0_iso=t0_iso,
        source_path=str(path),
        baseline=baseline_arr,
        dose_rate_usv_h=dose_rate_usv_h,
        gps_track=gps_track,
        integrity_report=integrity_report,
    )