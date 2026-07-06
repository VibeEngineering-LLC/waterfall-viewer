from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
import numpy as np
from awf.model.spectrogram import Calibration, Spectrogram

def _localname(el) -> str:
    return el.tag.rsplit("}", 1)[-1]

def _find_local(scope, name) -> ET.Element | None:
    for el in scope.iter():
        if _localname(el).lower() == name.lower():
            return el
    return None

def _findall_local(scope, name) -> list[ET.Element]:
    return [el for el in scope.iter() if _localname(el).lower() == name.lower()]

def _text_float(el) -> float | None:
    if el is None or not el.text:
        return None
    try:
        return float(el.text.strip())
    except (ValueError, TypeError):
        return None

def _text_int(el) -> int | None:
    if el is None or not el.text:
        return None
    try:
        return int(el.text.strip())
    except (ValueError, TypeError):
        return None

def _first_text(scope, names) -> str | None:
    for name in names:
        el = _find_local(scope, name)
        if el is not None and el.text:
            return el.text.strip()
    return None

def _parse_iso(s) -> str | None:
    if not s:
        return None
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return s
    except ValueError:
        return None

def _parse_calibration(scope) -> Calibration:
    coeffs = []
    
    # Попробуем найти <Coefficients><Coefficient>...</Coefficient></Coefficients>
    coeff_elements = _findall_local(scope, "Coefficient")
    if coeff_elements:
        for el in coeff_elements:
            val = _text_float(el)
            if val is not None:
                coeffs.append(val)
    
    # Если не нашли, попробуем <PolynomialCoefficients> или <CalibrationCoefficients>
    if not coeffs:
        for name in ["PolynomialCoefficients", "CalibrationCoefficients"]:
            el = _find_local(scope, name)
            if el is not None and el.text:
                try:
                    vals = [float(x) for x in el.text.strip().split()]
                    coeffs.extend(vals)
                    break
                except (ValueError, TypeError):
                    continue
    
    # Если всё ещё не нашли — используем fallback
    if not coeffs:
        coeffs = [0.0, 1.0]
    
    return Calibration(coeffs=np.array(coeffs, dtype=np.float64))

def _parse_counts(scope) -> list[int]:
    counts = []
    
    # Источник 1: <DataPoint> элементы
    data_points = _findall_local(scope, "DataPoint")
    if data_points:
        for dp in data_points:
            try:
                val = int(dp.text.strip())
                counts.append(val)
            except (ValueError, TypeError):
                continue
        return counts
    
    # Источник 2: <Channel> внутри <Spectrum> или <Channels>
    channels = []
    for name in ["Spectrum", "Channels"]:
        spectrum_el = _find_local(scope, name)
        if spectrum_el is not None:
            channel_els = _findall_local(spectrum_el, "Channel")
            if channel_els:
                # Проверим наличие атрибутов n или index
                indexed_channels = []
                for ch in channel_els:
                    idx = ch.get("n") or ch.get("index")
                    try:
                        idx = int(idx)
                        indexed_channels.append((idx, ch))
                    except (TypeError, ValueError):
                        continue
                if indexed_channels:
                    indexed_channels.sort(key=lambda x: x[0])
                    for _, ch in indexed_channels:
                        try:
                            val = int(ch.text.strip())
                            counts.append(val)
                        except (ValueError, TypeError):
                            continue
                else:
                    # Без индексов — по порядку
                    for ch in channel_els:
                        try:
                            val = int(ch.text.strip())
                            counts.append(val)
                        except (ValueError, TypeError):
                            continue
                return counts
    
    # Источник 3: <Spectrum> или <SpectrumData> с текстом
    spectrum_el = _find_local(scope, "Spectrum")
    if spectrum_el is None:
        spectrum_el = _find_local(scope, "SpectrumData")
    
    if spectrum_el is not None and spectrum_el.text:
        try:
            # Попробуем разделить по запятой или пробелам
            text = spectrum_el.text.strip()
            parts = [x for x in text.split() if x]
            counts = [int(x) for x in parts]
        except (ValueError, TypeError):
            pass
    
    return counts

def load_becqmoni(path) -> Spectrogram:
    path = Path(path)
    try:
        tree = ET.parse(path)
    except ParseError as exc:
        raise ValueError(f"BecqMoni: не XML: {path}: {exc}") from exc
    root = tree.getroot()
    
    # Найти EnergySpectrum или использовать корневой элемент
    energy_spectrum = _find_local(root, "EnergySpectrum")
    scope = energy_spectrum if energy_spectrum is not None else root
    
    # Тайминги
    measurement_time_el = _find_local(scope, "MeasurementTime")
    real_time_s = _text_float(measurement_time_el)
    
    real_time_el = _find_local(scope, "RealTime")
    if real_time_s is None:
        real_time_s = _text_float(real_time_el)
    
    live_time_el = _find_local(scope, "LiveTime")
    live_time_s = _text_float(live_time_el)
    
    if live_time_s is None:
        live_time_s = real_time_s
    
    # Если оба отсутствуют — NaN
    if real_time_s is None:
        real_time_s = np.nan
    if live_time_s is None:
        live_time_s = np.nan
    
    # Калибровка
    calibration = _parse_calibration(scope)
    
    # Время начала
    t0_iso = _parse_iso(_first_text(root, ["StartTime", "SampleDateTime", "AcquisitionStartTime"]))
    
    # Отсчёты
    counts_list = _parse_counts(scope)
    if not counts_list:
        raise ValueError(f"BecqMoni: нет спектра: {path}")
    
    # Проверка на отрицательные значения
    negative_count = sum(1 for x in counts_list if x < 0)
    if negative_count > 0:
        warnings.warn(f"Найдено {negative_count} отрицательных отсчётов, будут обнулены")
        counts_list = [max(0, x) for x in counts_list]
    
    # Обработка NumberOfChannels
    n_declared = _text_int(_find_local(scope, "NumberOfChannels"))
    if n_declared is not None:
        if n_declared < len(counts_list):
            warnings.warn(f"Спектр обрезан до {n_declared} каналов")
            counts_list = counts_list[:n_declared]
        elif n_declared > len(counts_list):
            warnings.warn(f"Спектр дополнен нулями до {n_declared} каналов")
            counts_list.extend([0] * (n_declared - len(counts_list)))
    
    # Преобразование в массив
    counts_1d = np.array(counts_list, dtype=np.int32)
    if counts_1d.max() <= 65535:
        counts_1d = counts_1d.astype(np.uint16)
    
    n_channels = len(counts_1d)
    counts = counts_1d.reshape(1, n_channels)
    
    # Возвращаем Spectrogram
    return Spectrogram(
        counts=counts,
        calibration=calibration,
        time_offsets_s=[0.0],
        real_time_s=[real_time_s],
        live_time_s=[live_time_s],
        t0_iso=t0_iso,
        source_path=str(path)
    )
