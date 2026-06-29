"""Задача #120: автокалибровка модели разрешения FWHM(E) под реальный детектор.

Дефолт-модель default_fwhm_model() (R=7%@662 кэВ, sqrt-закон) СИСТЕМНО ШИРЕ реальной;
слишком широкое ядро matched-фильтра -G'' занижает значимость Currie L_C и пропускает
узкие пики. auto_calibrate_fwhm_model() строит модель по сильным реальным пикам спектра
(анкеры из first-pass find_peaks), фитит sqrt-закон FWHM=sqrt(b*E), валидирует монотонность,
на бедных спектрах откатывается на default. Проверки — численные, из поведения, не из памяти.

Методология: Knoll «Radiation Detection and Measurement» 4-е изд. гл.10; Gilmore «Practical
Gamma-ray Spectrometry» 2-е изд. гл.2; sqrt-закон = статистический предел R∝1/sqrt(E).
"""
import numpy as np
import pytest

from awf.analysis.peaks import (
    auto_calibrate_fwhm_model,
    default_fwhm_model,
    FwhmModel,
)


# Калибровка ~3 кэВ/канал, 1024 канала. Дефолт b = 0.07^2 * 662 = 3.2438.
_NC = 1024
_E0 = 3.0
_KEV_PER_CH = 3.0
_DEF_B = (0.07 ** 2) * 662.0


def _energies():
    return _E0 + _KEV_PER_CH * np.arange(_NC, dtype=np.float64)


def _narrow_spectrum(true_b, peak_E=(200.0, 500.0, 900.0, 1400.0),
                     heights=(40000.0, 60000.0, 50000.0, 30000.0), bg=800.0):
    """Спектр с пиками, истинная FWHM которых = sqrt(true_b*E). true_b < _DEF_B ⇒
    пики У́ЖЕ дефолт-модели."""
    energies = _energies()
    ch = np.arange(_NC, dtype=np.float64)
    spec = bg + np.zeros(_NC, dtype=np.float64)
    for pe, h in zip(peak_E, heights):
        c0 = (pe - _E0) / _KEV_PER_CH
        fwhm_kev = np.sqrt(true_b * pe)
        sig_ch = (fwhm_kev / _KEV_PER_CH) / 2.3548
        spec = spec + h * np.exp(-0.5 * ((ch - c0) / sig_ch) ** 2)
    return spec, energies


def test_auto_calibrate_returns_auto_source_on_narrow_peaks():
    """Узкие реальные пики (FWHM в 1.4x уже дефолта) → source == 'auto'."""
    true_b = _DEF_B / 1.96   # FWHM в sqrt(1.96)=1.4 раза уже дефолта
    spec, energies = _narrow_spectrum(true_b)
    model = auto_calibrate_fwhm_model(spec, energies)
    assert isinstance(model, FwhmModel)
    assert model.source == "auto"


def test_auto_calibrate_fwhm_close_to_true_and_narrower_than_default():
    """FWHM(E) калиброванной модели близка к истинной и У́ЖЕ дефолта на энергии пика."""
    true_b = _DEF_B / 1.96
    spec, energies = _narrow_spectrum(true_b)
    model = auto_calibrate_fwhm_model(spec, energies)
    default = default_fwhm_model()
    for pe in (200.0, 500.0, 900.0, 1400.0):
        true_fwhm = float(np.sqrt(true_b * pe))
        auto_fwhm = float(model(pe))
        def_fwhm = float(default(pe))
        # близко к истинной (в пределах 8 %)
        assert abs(auto_fwhm - true_fwhm) <= 0.08 * true_fwhm, (pe, auto_fwhm, true_fwhm)
        # заметно уже дефолта (хотя бы на 15 %)
        assert auto_fwhm < 0.85 * def_fwhm, (pe, auto_fwhm, def_fwhm)


def test_auto_calibrate_model_monotonic_increasing():
    """Калиброванная FWHM(E) монотонно не убывает по E."""
    true_b = _DEF_B / 1.96
    spec, energies = _narrow_spectrum(true_b)
    model = auto_calibrate_fwhm_model(spec, energies)
    Es = np.linspace(50.0, 3000.0, 100)
    w = np.asarray(model(Es), dtype=np.float64)
    assert np.all(np.isfinite(w))
    assert np.all(w > 0.0)
    assert np.all(np.diff(w) >= -1e-9)


def test_auto_calibrate_falls_back_to_default_on_poor_spectrum():
    """Бедный спектр (нет пиков выше порога) → откат на default (source == 'default')."""
    energies = _energies()
    rng = np.random.default_rng(0)
    poor = np.clip(100.0 + rng.normal(0.0, 1.0, _NC), 0.0, None)
    model = auto_calibrate_fwhm_model(poor, energies)
    assert model.source == "default"
    # совпадает с дефолтом по коэффициенту
    assert model.b == pytest.approx(default_fwhm_model().b)


def test_auto_calibrate_b_positive():
    """Коэффициент b калиброванной модели строго положителен (sqrt-закон корректен)."""
    true_b = _DEF_B / 1.96
    spec, energies = _narrow_spectrum(true_b)
    model = auto_calibrate_fwhm_model(spec, energies)
    assert model.b > 0.0
    # b калиброван у́же дефолта (узкие пики)
    assert model.b < _DEF_B


def test_auto_calibrate_short_energies_returns_default():
    """energies < 2 точек → default без падения."""
    model = auto_calibrate_fwhm_model(np.array([1.0]), np.array([10.0]))
    assert model.source == "default"
