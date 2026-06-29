# -*- coding: utf-8 -*-
"""Задача #120: проверка автокалибровки FWHM(E) на РЕАЛЬНОМ .aswf-файле.

Path-agnostic — путь к файлу из argv[1], в коде НЕ зашит (файл LOCAL-ONLY).
Печатает source калиброванной модели и FWHM(пробной энергии) калиброванной vs дефолтной.

Запуск:
    python scripts/task120_fwhm_calib_check.py <path-to.aswf> [probe_keV]
"""
import sys

import numpy as np

from awf.io.aswf_loader import load_aswf
from awf.analysis.peaks import auto_calibrate_fwhm_model, default_fwhm_model


def main(argv):
    if len(argv) < 2:
        print("usage: python scripts/task120_fwhm_calib_check.py <path.aswf> [probe_keV]")
        return 2
    path = argv[1]
    probe = float(argv[2]) if len(argv) > 2 else 186.0
    sg = load_aswf(path)
    counts = np.asarray(sg.total_spectrum(), dtype=np.float64)
    energies = np.asarray(sg.energies(), dtype=np.float64)
    model = auto_calibrate_fwhm_model(counts, energies)
    default = default_fwhm_model()
    auto_fwhm = float(model(probe))
    def_fwhm = float(default(probe))
    print("source        :", model.source)
    print("b (auto)      :", round(model.b, 4))
    print("b (default)   :", round(default.b, 4))
    print(f"FWHM({probe:.0f}) auto    : {auto_fwhm:.2f} keV")
    print(f"FWHM({probe:.0f}) default : {def_fwhm:.2f} keV")
    print(f"ratio auto/def: {auto_fwhm/def_fwhm:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
