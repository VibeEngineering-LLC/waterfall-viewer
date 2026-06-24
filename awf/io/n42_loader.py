from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from lxml import etree
from awf.model.spectrogram import Calibration, Spectrogram

_NS_STRIP = re.compile(r"^\{[^}]*\}")
def _local(tag) -> str:
    # имя тега без namespace; tag может быть не-строкой (комментарии lxml) -> вернуть "".
    if not isinstance(tag, str): return ""
    return _NS_STRIP.sub("", tag)

_DUR_RE = re.compile(
    r"^P(?:(?P<d>\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)?$")

def parse_iso_duration(text: str | None) -> Optional[float]:
    # "PT10S"->10.0 ; "PT1M30S"->90.0 ; "P1DT2H"->93600.0 ; пусто/невалидно -> None.
    if not text: return None
    m = _DUR_RE.match(text.strip())
    if not m: return None
    d = float(m.group("d") or 0); h = float(m.group("h") or 0)
    mi = float(m.group("m") or 0); s = float(m.group("s") or 0)
    total = d * 86400 + h * 3600 + mi * 60 + s
    return total

def _parse_datetime(text: str | None) -> Optional[datetime]:
    # ISO-8601; поддержать хвостовой 'Z' -> '+00:00'. Невалидно -> None.
    if not text: return None
    t = text.strip()
    if t.endswith("Z"): t = t[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(t)
    except ValueError:
        return None

def decode_counted_zeroes_scalar(tokens) -> list[int]:
    # Эталонная реализация: явный конечный автомат слева-направо. Корректна для ЛЮБОГО входа,
    # включая count==0 (пустой пробег). Используется в тестах как независимый verifier.
    out: list[int] = []
    i = 0; n = len(tokens)
    while i < n:
        v = int(tokens[i])
        if v == 0:
            if i + 1 >= n:
                # некорректный хвостовой одиночный ноль -> трактуем как один ноль (не падать)
                out.append(0); i += 1
            else:
                count = int(tokens[i + 1])
                if count > 0:
                    out.extend([0] * count)
                i += 2
        else:
            out.append(v); i += 1
    return out

def decode_counted_zeroes_vec(tokens: np.ndarray) -> np.ndarray:
    # Векторная реализация (быстрая, для продакшен-загрузки больших файлов).
    # ПРЕДУСЛОВИЕ: корректный CountedZeroes, где каждый ноль — маркер, за ним count>=1
    #              (последовательность "0 0" = пустой пробег НЕ поддерживается векторным путём;
    #               в реальных данных N42 не встречается). Для произвольного входа используйте scalar.
    # tokens — 1D целочисленный np.ndarray.
    t = np.asarray(tokens)
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    t = t.astype(np.int64, copy=False)
    is_zero = (t == 0)
    consumed = np.zeros(n, dtype=bool)
    consumed[1:] = is_zero[:-1]            # токен i — это count, если предыдущий был маркер-ноль
    marker = is_zero & ~consumed           # маркеры = нули, которые сами не являются count
    counts = np.ones(n, dtype=np.int64)
    nxt = np.zeros(n, dtype=np.int64)
    nxt[:-1] = t[1:]
    counts[marker] = nxt[marker]           # маркер порождает 'count' нулей (значение токена==0)
    counts[consumed] = 0                   # сами count-токены ничего не порождают
    return np.repeat(t, counts)

def _decode_channel_text(text: str) -> np.ndarray:
    # Текст ChannelData -> 1D int64 массив декодированных отсчётов (через векторный декодер).
    if not text or not text.strip():
        return np.zeros(0, dtype=np.int64)
    toks = np.fromstring(text, dtype=np.float64, sep=" ").astype(np.int64)
    return decode_counted_zeroes_vec(toks)

def load_n42(path, *, max_slices: int | None = None) -> Spectrogram:
    """Загрузить waterfall-спектрограмму N42.42. path: str | Path. max_slices: ограничить число
    срезов (None — все). Поднимает ValueError, если в файле не найдено ни одного ChannelData."""
    path = Path(path)
    cal_coeffs_text: str | None = None
    start_times: list[str] = []
    real_times: list[str] = []
    live_times: list[str] = []
    spectra: list[np.ndarray] = []

    # Потоковый разбор. recover=True — терпеть мелкие синтаксические дефекты.
    context = etree.iterparse(str(path), events=("end",), recover=True, huge_tree=True)
    for _event, elem in context:
        tag = _local(elem.tag)
        if tag == "CoefficientValues":
            if cal_coeffs_text is None:
                cal_coeffs_text = (elem.text or "").strip()
        elif tag == "StartDateTime":
            start_times.append((elem.text or "").strip())
        elif tag == "RealTimeDuration":
            real_times.append((elem.text or "").strip())
        elif tag == "LiveTimeDuration":
            live_times.append((elem.text or "").strip())
        elif tag == "ChannelData":
            spectra.append(_decode_channel_text(elem.text or ""))
            if max_slices is not None and len(spectra) >= max_slices:
                # достигнут лимит — очистить и прекратить
                elem.clear()
                break
        # Освобождение памяти: после закрытия крупных контейнеров чистим поддерево и
        # удаляем уже обработанных предыдущих сиблингов (классический паттерн lxml).
        if tag in ("RadMeasurement", "Spectrum", "RadInstrumentInformation",
                   "EnergyCalibration", "DerivedData"):
            elem.clear()
            prev = elem.getprevious()
            parent = elem.getparent()
            while prev is not None and parent is not None:
                del parent[0]
                prev = elem.getprevious()
    del context

    if not spectra:
        raise ValueError(f"N42: в файле не найдено ChannelData: {path}")

    counts = _stack_spectra(spectra)
    n_slices = counts.shape[0]
    time_offsets = _build_time_axis(start_times, real_times, n_slices)
    real_arr = _durations_to_array(real_times, n_slices)
    live_arr = _durations_to_array(live_times, n_slices)
    if cal_coeffs_text:
        calibration = Calibration.from_coeff_string(cal_coeffs_text)
    else:
        # калибровки нет — единичная (энергия == номер канала), чтобы вьюер работал
        calibration = Calibration(coeffs=np.array([0.0, 1.0], dtype=np.float64))
    t0_iso = start_times[0] if start_times else None
    return Spectrogram(
        counts=counts, calibration=calibration, time_offsets_s=time_offsets,
        real_time_s=real_arr, live_time_s=live_arr, t0_iso=t0_iso, source_path=str(path))

def _stack_spectra(spectra: list[np.ndarray]) -> np.ndarray:
    # Все срезы дополнить нулями до максимальной длины и собрать в 2D.
    # dtype: uint16 если глобальный максимум <= 65535, иначе int32.
    max_len = max(s.size for s in spectra)
    gmax = 0
    for s in spectra:
        if s.size:
            m = int(s.max())
            if m > gmax: gmax = m
    dtype = np.uint16 if gmax <= 65535 else np.int32
    out = np.zeros((len(spectra), max_len), dtype=dtype)
    for k, s in enumerate(spectra):
        if s.size:
            out[k, :s.size] = s
    return out

def _build_time_axis(start_times: list[str], real_times: list[str], n_slices: int) -> np.ndarray:
    # Секунды от начала записи для каждого среза.
    # Основной путь: разности StartDateTime[i]-StartDateTime[0]. Запасной: кумулятивная сумма
    # RealTimeDuration (или 1.0с-шаг, если и их нет).
    dts = [_parse_datetime(s) for s in start_times[:n_slices]]
    if len(dts) == n_slices and all(d is not None for d in dts):
        t0 = dts[0]
        return np.array([(d - t0).total_seconds() for d in dts], dtype=np.float64)
    # запасной путь по длительностям
    durs = [parse_iso_duration(r) for r in real_times[:n_slices]]
    step_ok = len(durs) == n_slices and all(x is not None for x in durs)
    offs = np.zeros(n_slices, dtype=np.float64)
    acc = 0.0
    for i in range(n_slices):
        offs[i] = acc
        acc += (durs[i] if step_ok else 1.0)
    return offs

def _durations_to_array(durs_text: list[str], n_slices: int) -> np.ndarray:
    # Длительности в секунды; недостающие/невалидные -> np.nan. Выровнять длину к n_slices.
    out = np.full(n_slices, np.nan, dtype=np.float64)
    for i in range(min(n_slices, len(durs_text))):
        v = parse_iso_duration(durs_text[i])
        if v is not None:
            out[i] = v
    return out
