"""Задача #217 — экспорт в InterSpec/ANSI N42.42-2012 .n42.

Формат ANSI N42.42-2012 (RadInstrumentData) — XML со следующими обязательными
блоками для одного спектра: <RadInstrumentInformation>, <EnergyCalibration>,
<RadMeasurement>/<Spectrum>/<ChannelData>. InterSpec, PeakEasy и Sandia
GADRAS читают именно эту форму. Реализован минимальный, но валидный ANSI N42.

Референс: ANSI N42.42-2012 Annex D (пример spectrum-only); InterSpec Test
Suite `spectrum_examples/N42-2012/`; PeakEasy `n42_writer.cpp` (Sandia).

Замечание: energy calibration в N42.42-2012 передаётся через
<EnergyCalibration> с полиномиальными коэффициентами в порядке
[c0, c1, c2, ...] (E = sum c_i * ch^i) — совпадает с нашим порядком.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import numpy as np


def _iso8601_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso8601_duration(seconds: float) -> str:
    s = max(0.0, float(seconds))
    return f"PT{s:.6f}S"


def write_n42(path, counts, *, live_time_s: float, real_time_s: float,
              calibration=None, mea_time: Optional[datetime] = None,
              instrument_model: str = "AtomSpectra",
              instrument_manufacturer: str = "AtomFast") -> Path:
    """Экспортировать одномерный спектр в ANSI N42.42-2012 .n42."""
    path = Path(path)
    arr = np.asarray(counts)
    if arr.ndim != 1:
        raise ValueError("counts must be 1D")
    c_int = np.rint(arr).astype(np.int64)

    if mea_time is None:
        mea_time = datetime.now(timezone.utc)

    coeffs = list(calibration) if calibration is not None else [0.0, 1.0]

    channel_str = " ".join(str(int(v)) for v in c_int.tolist())
    cal_str = " ".join(f"{float(x):.9g}" for x in coeffs)

    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<RadInstrumentData xmlns="http://physics.nist.gov/N42/2011/N42"'
               ' n42DocUUID="waterfall-viewer-export"'
               ' n42DocDateTime="' + _iso8601_utc(mea_time) + '">')
    xml.append('  <RadInstrumentInformation id="Instr-1">')
    xml.append(f'    <RadInstrumentManufacturerName>{escape(instrument_manufacturer)}</RadInstrumentManufacturerName>')
    xml.append(f'    <RadInstrumentModelName>{escape(instrument_model)}</RadInstrumentModelName>')
    xml.append('    <RadInstrumentClassCode>Spectroscopic Personal Radiation Detector</RadInstrumentClassCode>')
    xml.append('    <RadInstrumentVersion>')
    xml.append('      <RadInstrumentComponentName>Software</RadInstrumentComponentName>')
    xml.append('      <RadInstrumentComponentVersion>waterfall-viewer</RadInstrumentComponentVersion>')
    xml.append('    </RadInstrumentVersion>')
    xml.append('  </RadInstrumentInformation>')
    xml.append('  <EnergyCalibration id="EC-1">')
    xml.append(f'    <CoefficientValues>{cal_str}</CoefficientValues>')
    xml.append('  </EnergyCalibration>')
    xml.append(f'  <RadMeasurement id="M-1">')
    xml.append('    <MeasurementClassCode>Foreground</MeasurementClassCode>')
    xml.append(f'    <StartDateTime>{_iso8601_utc(mea_time)}</StartDateTime>')
    xml.append(f'    <RealTimeDuration>{_iso8601_duration(real_time_s)}</RealTimeDuration>')
    xml.append('    <Spectrum id="Spec-1" radDetectorInformationReference="Det-1"'
               ' energyCalibrationReference="EC-1">')
    xml.append(f'      <LiveTimeDuration>{_iso8601_duration(live_time_s)}</LiveTimeDuration>')
    xml.append(f'      <ChannelData compressionCode="None">{channel_str}</ChannelData>')
    xml.append('    </Spectrum>')
    xml.append('  </RadMeasurement>')
    xml.append('</RadInstrumentData>')

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(xml) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
