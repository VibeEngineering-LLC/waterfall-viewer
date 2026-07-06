"""
Export 1D spectrum to BecqMoni XML format.

References:
- BecqMoni ResultDataFile FormatVersion=120920
- Project issue #226 (refactor of #217)
- BUG-BQ1: identity calibration fallback for empty calibration

Exports a 1D energy spectrum in XML format compatible with BecqMoni and
its corresponding loader `awf/io/becqmoni_loader.py`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
import numpy as np


FORMAT_VERSION = "120920"


def write_becqmoni_xml(path, counts, *, live_time_s: float, real_time_s: float,
                      calibration=None, mea_time: Optional[datetime] = None,
                      sample_name: str = "waterfall-viewer export") -> Path:
    path = Path(path)
    arr = np.asarray(counts)
    if arr.ndim != 1:
        raise ValueError("counts must be 1D")
    c_int = np.rint(arr).astype(np.int64)
    c_int = np.clip(c_int, 0, None)
    total = int(c_int.sum())

    if not calibration:
        coefs = [0.0, 1.0]
    else:
        coefs = [float(x) for x in calibration]
    polynomial_order = len(coefs) - 1

    if mea_time is None:
        mea_time = datetime.now(timezone.utc)
    mea_time_str = mea_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_dt = mea_time + timedelta(seconds=float(real_time_s))
    end_dt_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

    root = ET.Element("ResultDataFile")
    ET.SubElement(root, "FormatVersion").text = FORMAT_VERSION

    result_data_list = ET.SubElement(root, "ResultDataList")
    result_data = ET.SubElement(result_data_list, "ResultData")

    sample_info = ET.SubElement(result_data, "SampleInfo")
    ET.SubElement(sample_info, "Name").text = sample_name

    device_config_ref = ET.SubElement(result_data, "DeviceConfigReference")
    ET.SubElement(device_config_ref, "Name").text = "waterfall-viewer"

    ET.SubElement(result_data, "StartTime").text = mea_time_str
    ET.SubElement(result_data, "EndTime").text = end_dt_str

    energy_spectrum = ET.SubElement(result_data, "EnergySpectrum")

    ET.SubElement(energy_spectrum, "NumberOfChannels").text = str(len(c_int))
    ET.SubElement(energy_spectrum, "ChannelPitch").text = "1"

    energy_calibration = ET.SubElement(energy_spectrum, "EnergyCalibration")
    ET.SubElement(energy_calibration, "PolynomialOrder").text = str(polynomial_order)
    coeffs_elem = ET.SubElement(energy_calibration, "Coefficients")

    for c in coefs:
        coeff_elem = ET.SubElement(coeffs_elem, "Coefficient")
        coeff_elem.text = f"{c:.10g}"

    ET.SubElement(energy_spectrum, "ValidPulseCount").text = str(total)
    ET.SubElement(energy_spectrum, "TotalPulseCount").text = str(total)
    ET.SubElement(energy_spectrum, "MeasurementTime").text = f"{float(real_time_s):.4f}"
    ET.SubElement(energy_spectrum, "LiveTime").text = f"{float(live_time_s):.6f}"
    ET.SubElement(energy_spectrum, "NumberOfSamples").text = "0"

    spectrum = ET.SubElement(energy_spectrum, "Spectrum")
    for v in c_int:
        data_point = ET.SubElement(spectrum, "DataPoint")
        data_point.text = str(int(v))

    ET.SubElement(result_data, "Visible").text = "true"

    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    # Atomic write
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(xml_bytes)
    tmp_path.replace(path)

    return path
