"""Тесты математики фона/вычитания (Задача #96): диапазон-фон, файл-фон с интерполяцией
по спектральной плотности, знаковый вычет в отсчётах (без клипа к 0, Задача #134 —
интеграл по времени сокращается точно, когда фон = самому спектру)."""
import numpy as np
import pytest

from awf.model.spectrogram import Calibration, Spectrogram
from awf.model.background import (background_from_range, background_from_spectrogram,
                                  subtract_background)


def _sg(counts, coeffs=(0.0, 1.0), live=None, real=None):
    counts = np.asarray(counts, dtype=np.float64)
    ns = counts.shape[0]
    live = np.ones(ns) if live is None else np.asarray(live, dtype=np.float64)
    real = live if real is None else np.asarray(real, dtype=np.float64)
    t = np.arange(ns, dtype=np.float64)
    return Spectrogram(counts=counts, calibration=Calibration(coeffs=list(coeffs)),
                       time_offsets_s=t, real_time_s=real, live_time_s=live)


def test_range_mean_cps_uniform():
    # 4 среза по 4 канала, live_time=1 c -> фон = средняя скорость = Σcounts/Σlt
    counts = np.array([[10, 0, 2, 0]] * 4, dtype=np.float64)
    bg = background_from_range(_sg(counts), 0, 4)
    assert np.allclose(bg, [10.0, 0.0, 2.0, 0.0])   # 40/4 = 10 cps и т.д.


def test_range_live_time_weighting():
    # фон взвешен по живому времени: bg[ch] = Σcounts / Σlive_time, не среднее по срезам
    counts = np.array([[6, 0], [6, 0]], dtype=np.float64)
    bg = background_from_range(_sg(counts, live=[1.0, 3.0]), 0, 2)
    assert np.allclose(bg, [12.0 / 4.0, 0.0])       # 3 cps (не 6 cps)


def test_range_subset_window():
    counts = np.array([[100, 0], [2, 0], [4, 0]], dtype=np.float64)
    bg = background_from_range(_sg(counts), 1, 3)    # только срезы 1..2
    assert np.allclose(bg, [3.0, 0.0])               # (2+4)/2


def test_range_empty_raises():
    with pytest.raises(ValueError):
        background_from_range(_sg(np.array([[1.0, 1.0]])), 1, 1)


def test_subtract_scales_by_live_time():
    # вычет в отсчётах: counts - bg_cps*live_time (масштаб по живому времени среза)
    sg = _sg(np.array([[10, 5], [10, 5]], dtype=np.float64), live=[1.0, 2.0])
    out = subtract_background(sg, np.array([1.0, 0.0]))
    assert np.allclose(out.counts, [[9, 5], [8, 5]])  # 10-1*1=9 ; 10-1*2=8
    assert out.n_slices == 2 and out.n_channels == 2


def test_subtract_keeps_signed_net():
    # Задача #134: остаток ЗНАКОВЫЙ (без клипа к 0). Раньше 3-5 клипалось к 0 — это
    # асимметрично разрушало сокращение при фон=спектр; теперь хранится -2.
    sg = _sg(np.array([[3, 1]], dtype=np.float64), live=[1.0])
    out = subtract_background(sg, np.array([5.0, 0.0]))
    assert np.allclose(out.counts, [[-2.0, 1.0]])     # 3-5=-2 (знаковый, НЕ клип к 0)


def test_self_subtraction_integrates_to_zero():
    # Задача #134 (корень бага): фон = самому спектру => интегральный (суммарный по времени)
    # спектр остатка должен сокращаться в ноль поканально. Многосрезовые данные с временно́й
    # вариацией: положительные и отрицательные срезовые остатки гасят друг друга точно.
    counts = np.array([[10, 1], [2, 9], [6, 5], [0, 3]], dtype=np.float64)
    sg = _sg(counts, live=[1.0, 2.0, 1.0, 3.0])
    bg_cps = background_from_range(sg, 0, sg.n_slices)      # фон = весь файл
    out = subtract_background(sg, bg_cps)
    integrated = out.counts.sum(axis=0)                     # суммарный спектр среза
    assert np.allclose(integrated, [0.0, 0.0], atol=1e-9)   # точное сокращение
    assert (out.counts < 0).any()                           # именно знаковость это обеспечивает


def test_self_subtraction_via_file_path_integrates_to_zero():
    # Задача #134: тот же инвариант, но фон получен через background_from_spectrogram
    # (ветка «фон = отдельный файл», совпадающая калибровка => поканальный cps).
    counts = np.array([[8, 0, 4], [0, 6, 4], [4, 2, 4]], dtype=np.float64)
    sg = _sg(counts, live=[1.0, 2.0, 1.0])
    bg_cps = background_from_spectrogram(sg, sg)            # фон = тот же файл
    out = subtract_background(sg, bg_cps)
    assert np.allclose(out.counts.sum(axis=0), [0.0, 0.0, 0.0], atol=1e-9)


def test_overlay_equals_spectrum_when_bg_is_whole_file():
    # Задача #134: наложение фона (overlay) при фон=весь файл совпадает с самим спектром.
    # Overlay в режиме counts = bg_cps * Σlive_time; спектр (counts) = Σcounts по времени.
    # bg_cps = Σcounts/Σlt => bg_cps*Σlt == Σcounts тождественно (нет ложного остатка).
    counts = np.array([[10, 1], [2, 9], [6, 5], [0, 3]], dtype=np.float64)
    sg = _sg(counts, live=[1.0, 2.0, 1.0, 3.0])
    lt_total = float(np.asarray(sg.live_time_s).sum())
    bg_cps = background_from_range(sg, 0, sg.n_slices)
    overlay_counts = bg_cps * lt_total
    assert np.allclose(overlay_counts, counts.sum(axis=0))


def test_subtract_length_mismatch_raises():
    sg = _sg(np.array([[1.0, 1.0]]))
    with pytest.raises(ValueError):
        subtract_background(sg, np.array([1.0, 2.0, 3.0]))


def test_file_same_calibration_reduces_to_cps():
    # отдельный файл с той же калибровкой -> поканальный cps (Σcounts/Σlt) без изменений
    bg_sg = _sg(np.array([[4, 8, 0, 2], [4, 0, 0, 2]], dtype=np.float64), live=[1.0, 1.0])
    target = _sg(np.zeros((3, 4), dtype=np.float64))
    bg = background_from_spectrogram(bg_sg, target)
    assert np.allclose(bg, [4.0, 4.0, 0.0, 2.0])      # [8,8,0,4]/2


def test_file_different_calibration_density_interp():
    # грубые каналы фона (2 кэВ/канал) -> мелкие каналы цели (1 кэВ/канал): плотность сохраняется,
    # за диапазоном энергий фона -> 0 (без экстраполяции)
    bg_sg = _sg(np.array([[4, 4, 4]], dtype=np.float64), coeffs=(0.0, 2.0), live=[1.0])
    target = _sg(np.zeros((1, 6), dtype=np.float64), coeffs=(0.0, 1.0))
    bg = background_from_spectrogram(bg_sg, target)
    assert np.allclose(bg, [2.0, 2.0, 2.0, 2.0, 2.0, 0.0])


def test_file_zero_live_time_raises():
    bg_sg = _sg(np.array([[1.0, 1.0]]), live=[0.0])
    target = _sg(np.zeros((1, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        background_from_spectrogram(bg_sg, target)