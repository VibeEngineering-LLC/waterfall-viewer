# Задача #156: нормализация по эффективности регистрации ε(E)
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from awf.analysis.efficiency import (EfficiencyCurve, default_gamma1s,
                                     load_efficiency_curve, apply_efficiency)
from awf.model.spectrogram import Spectrogram, Calibration


def _sg(counts, coeffs=(0.0, 1.0)):
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.shape[0]
    t = np.arange(n, dtype=np.float64)
    real = np.ones(n, dtype=np.float64)
    live = np.ones(n, dtype=np.float64)
    return Spectrogram(counts=counts, calibration=Calibration(coeffs=list(coeffs)),
                       time_offsets_s=t, real_time_s=real, live_time_s=live)

# ---------- кривая по умолчанию (Гамма-1С) ----------

def test_default_curve_16_points_and_anchor_values():
    c = default_gamma1s()
    assert len(c.energies) == len(c.values) == 16
    # опорные измеренные точки: максимум на 121.782 кэВ, край на 1408.006
    assert c.efficiency(121.782) == pytest.approx(1.8015e-3, rel=1e-6)
    assert c.efficiency(1408.006) == pytest.approx(1.674492e-4, rel=1e-6)
    # энергии строго возрастают
    assert all(a < b for a, b in zip(c.energies, c.energies[1:]))


def test_flat_extrapolation_outside_range():
    c = default_gamma1s()
    assert c.efficiency(10.0) == pytest.approx(c.efficiency(59.537), rel=1e-9)
    assert c.efficiency(3000.0) == pytest.approx(c.efficiency(1408.006), rel=1e-9)


def test_loglog_interpolation_between_points():
    # между двумя точками лог-лог интерполяция даёт геометрию степенной прямой
    c = EfficiencyCurve(name="t", energies=(100.0, 1000.0), values=(1e-3, 1e-4))
    # середина в лог-осях: E = sqrt(100*1000), ε = sqrt(1e-3*1e-4)
    assert c.efficiency(np.sqrt(100.0 * 1000.0)) == pytest.approx(
        np.sqrt(1e-3 * 1e-4), rel=1e-9)


def test_factors_unity_at_curve_max_and_grow():
    c = default_gamma1s()
    f = c.factors(np.array([121.782, 661.66, 1408.006]))
    assert f[0] == pytest.approx(1.0, rel=1e-9)
    # 1.8015e-3 / 4.8809e-4 и 1.8015e-3 / 1.674492e-4 — из измеренных точек
    assert f[1] == pytest.approx(1.8015e-3 / 4.8809e-4, rel=1e-6)
    assert f[2] == pytest.approx(1.8015e-3 / 1.674492e-4, rel=1e-6)
    assert f[0] < f[1] < f[2]

# ---------- apply_efficiency ----------

def test_apply_efficiency_scales_channels_and_keeps_metadata():
    curve = EfficiencyCurve(name="t", energies=(1.0, 4.0), values=(1e-2, 1e-3))
    sg = _sg(np.ones((3, 5)))          # E каналов = 0..4 кэВ (коэфф. 0,1)
    out = apply_efficiency(sg, curve)
    f = curve.factors(np.asarray(sg.energies(), dtype=np.float64))
    assert np.allclose(np.asarray(out.counts), f[None, :])
    # все временные срезы отмасштабированы одинаково
    assert np.allclose(np.asarray(out.counts)[0], np.asarray(out.counts)[2])
    # метаданные исходной спектрограммы сохранены
    assert np.allclose(out.calibration.coeffs, sg.calibration.coeffs)
    assert np.allclose(out.time_offsets_s, sg.time_offsets_s)
    assert np.allclose(out.live_time_s, sg.live_time_s)


def test_apply_efficiency_unity_at_max_channel():
    sg = _sg(np.full((2, 4), 7.0), coeffs=(0.0, 100.0))   # E = 0/100/200/300
    curve = EfficiencyCurve(name="t", energies=(100.0, 300.0), values=(2e-3, 1e-3))
    out = apply_efficiency(sg, curve)
    # канал на максимуме кривой (100 кэВ) не изменён
    assert np.asarray(out.counts)[0, 1] == pytest.approx(7.0, rel=1e-9)
    # канал 300 кэВ усилен в ε_ref/ε = 2 раза
    assert np.asarray(out.counts)[0, 3] == pytest.approx(14.0, rel=1e-9)

# ---------- загрузка кривой из файла ----------

_EFR_SAMPLE = """[Гамма-1С-тест;Точка-15см;Cs-137]
Detector=Gamma-1S-test
Geometry=Point-15cm
661.66=4.8809E-4,2.1,Cs-137,12345,67,85.1
[Гамма-1С-тест;Точка-15см;Am-241]
59.537=1.579658E-3,3.0,Am-241,54321,89,35.9
"""


def test_load_efr_cp1251(tmp_path):
    p = tmp_path / "test_curve.efr"
    p.write_bytes(_EFR_SAMPLE.encode("cp1251"))
    c = load_efficiency_curve(p)
    assert c.name == "test_curve"
    assert c.energies == (59.537, 661.66)     # отсортировано по энергии
    assert c.values[0] == pytest.approx(1.579658e-3, rel=1e-9)
    assert c.values[1] == pytest.approx(4.8809e-4, rel=1e-9)


def test_load_two_column_text(tmp_path):
    p = tmp_path / "curve.csv"
    p.write_text("# E_keV, eps\n100 2e-3\n300, 1e-3\n500;5e-4\nоборванная строка\n",
                 encoding="utf-8")
    c = load_efficiency_curve(p)
    assert c.energies == (100.0, 300.0, 500.0)
    assert c.values == (2e-3, 1e-3, 5e-4)


def test_load_json_points(tmp_path):
    p = tmp_path / "curve.json"
    p.write_text('{"points": [[661.66, 4.8809e-4], [59.537, 1.579658e-3]]}',
                 encoding="utf-8")
    c = load_efficiency_curve(p)
    assert c.energies == (59.537, 661.66)


def test_load_duplicate_energies_averaged(tmp_path):
    p = tmp_path / "dup.txt"
    p.write_text("100 1e-3\n100 3e-3\n200 5e-4\n", encoding="utf-8")
    c = load_efficiency_curve(p)
    assert c.energies == (100.0, 200.0)
    assert c.values[0] == pytest.approx(2e-3, rel=1e-9)   # среднее дублей


def test_load_less_than_two_points_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("100 1e-3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_efficiency_curve(p)

# ---------- MainWindow: тумблер меню «Анализ» ----------

@pytest.fixture(scope="module")
def app():
    from PyQt5 import QtWidgets
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def test_mainwindow_eff_toggle_normalizes_waterfall(app):
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    w._on_loaded(_sg(np.full((4, 8), 5.0), coeffs=(0.0, 200.0)))
    orig = np.asarray(w._slices._sg.counts).copy()
    assert w._act_eff_norm.isChecked() is False        # по умолчанию выключено
    w._act_eff_norm.setChecked(True)                   # тумблер меню «Анализ»
    normed = np.asarray(w._slices._sg.counts)
    f = w._eff_curve.factors(np.asarray(w._sg.energies(), dtype=np.float64))
    assert np.allclose(normed, orig * f[None, :])      # 3D/2D/срезы — нормализованы
    assert np.allclose(np.asarray(w._heatmap._sg.counts), normed)
    assert float(np.asarray(w._sg.counts).sum()) == pytest.approx(orig.sum())
    w._act_eff_norm.setChecked(False)                  # выключение -> исходный водопад
    assert np.allclose(np.asarray(w._slices._sg.counts), orig)
    w.close()


def test_mainwindow_eff_curve_survives_reload_and_info_updates(app, tmp_path):
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    w._on_loaded(_sg(np.ones((3, 6))))
    assert "Гамма-1С" in w._eff_info_label.text()        # дефолтная кривая в инфо-пункте
    p = tmp_path / "my_det.txt"
    p.write_text("100 2e-3\n1000 2e-4\n", encoding="utf-8")
    w._eff_curve = load_efficiency_curve(p)            # как _on_eff_load без диалога
    w._update_eff_info()
    assert "my_det" in w._eff_info_label.text()
    w._on_loaded(_sg(np.ones((2, 6))))                 # новый файл
    assert w._eff_curve.name == "my_det"               # кривая — свойство детектора
    w.close()

# ---------- Задача #158: поиск пиков/ID — по ненормализованным данным ----------

def test_158_peak_search_and_poisson_mask_ignore_normalization(app):
    # Currie предполагает var=N; ε-нормализация ломала статистику (ложные пики на ВЭ)
    # и давала двойную ε-коррекцию в идентификации (Th-232/K-40 пропадали).
    from awf.ui.main_window import MainWindow
    w = MainWindow()
    counts = np.random.RandomState(7).poisson(30, size=(6, 64)).astype(np.float64)
    counts[:, 40] += 500.0                              # пик ~2000 кэВ
    w._on_loaded(_sg(counts, coeffs=(0.0, 50.0)))
    e_off = [float(p.energy) for p in w._view3d._found_peaks()]
    zint_off = np.asarray(w._view3d._z_counts_int).copy()
    assert e_off                                        # пик найден
    w._act_eff_norm.setChecked(True)                    # нормализация — только дисплей
    f = w._eff_curve.factors(np.asarray(w._sg.energies(), dtype=np.float64))
    assert np.allclose(np.asarray(w._view3d._sg.counts),
                       np.asarray(w._sg.counts) * f[None, :])
    assert w._view3d._sg_analysis is w._sg              # анализ — сырые данные
    assert [float(p.energy) for p in w._view3d._found_peaks()] == e_off
    assert np.allclose(np.asarray(w._view3d._z_counts_int), zint_off)
    w._view3d.set_spectrogram(w._view3d._sg, 400, 512)  # ре-рендер без analysis_sg
    assert w._view3d._sg_analysis is w._sg              # источник анализа сохранён
    w.close()
