"""Задача #113 — проверка детекции транзиентных пиков на РЕАЛЬНОМ файле (LOCAL-ONLY).

Путь к файлу — ТОЛЬКО из argv[1] (НЕ хардкодить: имя файла содержит IP = PII, репо публичный).
Запуск:
    cd <project>
    QT_QPA_PLATFORM=offscreen PYTHONPATH=. PYTHONIOENCODING=utf-8         python scripts/task113_transient_check.py "<путь к .aswf>"

Печатает: найден ли пик в окне ~180-192 кэВ (оконная значимость, окно срезов), общее число
добавленных транзиентов (контроль ложных), их энергии, фактическую оконную значимость ~186 кэВ.
Все числа — из реального прогона. Изотоп не называется (терминология «~186 кэВ»).
"""
import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from PySide6 import QtWidgets
from awf.io.aswf_loader import load_aswf
from awf.io.rcspg_loader import load_rcspg
from awf.io.n42_loader import load_n42
from awf.ui.view3d import Waterfall3DView, _TRANSIENT_SIGMA_MARGIN, _MAX_ENERGY_KEV
from awf.analysis.peaks import (
    auto_calibrate_fwhm_model, fwhm_channels_from_model,
    find_peaks, find_transient_peaks, _transient_windows,
)

WIN_LO, WIN_HI = 180.0, 192.0   # окно интереса вокруг ~186 кэВ


def _load(path):
    pl = path.lower()
    if pl.endswith(".aswf"):
        return load_aswf(path)
    if pl.endswith(".rcspg"):
        return load_rcspg(path)
    if pl.endswith(".n42"):
        return load_n42(path)
    raise SystemExit("Неизвестный формат: " + path)


def _window_significance_186(arr, energies, widths, sigma_thr, min_slices=40,
                             n_windows=12, overlap=0.5):
    """Повтор оконного скана с трассировкой: для каждого окна, где найден пик в [WIN_LO,WIN_HI],
    вернуть (s, e, energy, significance). Числа из реального find_peaks по окну."""
    n_slices, n_channels = arr.shape
    w = max(int(min_slices), int(round(n_slices / max(1, n_windows))))
    w = min(w, n_slices)
    step = max(1, int(round(w * (1.0 - overlap))))
    hits = []
    for s, e in _transient_windows(n_slices, w, step):
        wspec = arr[s:e].sum(axis=0)
        for pk in find_peaks(wspec, widths, sigma_threshold=sigma_thr, energies=energies):
            if WIN_LO <= float(pk.energy) <= WIN_HI:
                hits.append((s, e, float(pk.energy), float(pk.significance)))
    return w, step, hits


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Использование: python scripts/task113_transient_check.py <путь к файлу>")
    path = sys.argv[1]
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sg = _load(path)
    print("Файл загружен: n_slices=%d, n_channels=%d" % (sg.n_slices, sg.n_channels))

    v = Waterfall3DView()
    v.set_spectrogram(sg, max_time=400, max_chan=512)

    counts = np.asarray(sg.total_spectrum(), dtype=np.float64)
    energies = np.asarray(sg.energies(), dtype=np.float64)
    model = auto_calibrate_fwhm_model(counts, energies)
    widths = fwhm_channels_from_model(model, energies)
    print("FWHM-модель: source=%s" % model.source)

    sigma = v._peak_sigma                       # дефолт 3.0
    thr = sigma + _TRANSIENT_SIGMA_MARGIN
    print("σ(дефолт)=%.1f, _TRANSIENT_SIGMA_MARGIN=%.1f -> порог окна=%.1f" % (sigma, _TRANSIENT_SIGMA_MARGIN, thr))

    integral = find_peaks(counts, widths, sigma_threshold=sigma, energies=energies)
    integral_in_win = [p for p in integral if WIN_LO <= float(p.energy) <= WIN_HI]
    print("Интегральных пиков всего: %d; в окне ~180-192: %d" % (len(integral), len(integral_in_win)))

    raw_2d = np.asarray(sg.counts, dtype=np.float64)
    transient = find_transient_peaks(raw_2d, energies, widths, integral,
                                     transient_sigma_threshold=thr)
    print("Транзиентных (новых) добавлено: %d" % len(transient))
    print("Энергии транзиентов (кэВ): %s" % ", ".join("%.1f" % p.energy for p in transient))

    in_win = [p for p in transient if WIN_LO <= float(p.energy) <= WIN_HI]
    if in_win:
        print("НАЙДЕН транзиент в окне ~180-192 кэВ:")
        for p in in_win:
            print("   E=%.2f кэВ, оконная significance(итог)=%.2fσ" % (p.energy, p.significance))
    else:
        print("В окне ~180-192 кэВ транзиент НЕ добавлен (проверь маржу/интеграл).")

    # Трассировка по окнам: где именно и с какой значимостью виден ~186 кэВ.
    w, step, hits = _window_significance_186(raw_2d, energies, widths, thr)
    print("Окна: w=%d срезов, step=%d. Срабатываний в окне ~180-192: %d" % (w, step, len(hits)))
    for s, e, eV, sig in hits:
        print("   срезы [%d:%d]: E=%.2f кэВ, оконная significance=%.2fσ" % (s, e, eV, sig))

    # Итог _found_peaks (merged), как увидит таблица/гребни.
    merged = v._found_peaks()
    merged_in_win = [p for p in merged if WIN_LO <= float(p.energy) <= WIN_HI]
    print("_found_peaks() всего: %d; в окне ~180-192: %d (<=%.0f кэВ фильтр #119)"
          % (len(merged), len(merged_in_win), _MAX_ENERGY_KEV))


if __name__ == "__main__":
    main()
