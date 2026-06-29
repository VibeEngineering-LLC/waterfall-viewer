"""Задача #113: транзиентные (время-локализованные) пики на спектрограмме.

Проверяем find_transient_peaks() — скользящие перекрывающиеся окна по времени,
суммирующие СЫРЫЕ counts, с порогом значимости выше интегрального. Сценарии:
  * транзиент виден окном, но утоплен в интегральном спектре (find_peaks не находит);
  * стабильный пик (во всех срезах) — найден интегралом, в объединённом наборе
    после dedup присутствует один раз (транзиент-скан его отбрасывает);
  * чистый шум → транзиентов не больше одного (контроль ложных срабатываний);
  * монотонность: больший порог → не больше транзиентов;
  * защита от вырожденных входов (пустой/1D/рассинхрон каналов/NaN) → [].

Синтетика детерминирована (фон без поканального Пуассон-шума там, где это влияет
на исход), поэтому результат не зависит от random-seed.
"""
import numpy as np
import pytest

from awf.model.spectrogram import Calibration, Spectrogram
from awf.analysis.peaks import (
    find_peaks, find_transient_peaks,
    default_fwhm_model, fwhm_channels_from_model,
)

# --- параметры синтетики ---
NC = 256                     # каналов
KEV_PER_CH = 2.0
E0 = 1.0
PEAK_CH = 120                # канал синтетического пика
FWHM_CH = 6.0
SIGMA_CH = FWHM_CH / 2.355
E_AT_PEAK = E0 + KEV_PER_CH * PEAK_CH   # = 241.0 кэВ (синтетика, НЕ реальные ~186)
ETOL = 3.0 * KEV_PER_CH                 # допуск совпадения по энергии

_CAL = Calibration(coeffs=[E0, KEV_PER_CH])
_ENERGIES = _CAL.energies(NC)
_WIDTHS = fwhm_channels_from_model(default_fwhm_model(), _ENERGIES)


def _gauss(center, height):
    ch = np.arange(NC, dtype=np.float64)
    return height * np.exp(-((ch - center) ** 2) / (2.0 * SIGMA_CH ** 2))


def _mksg(counts):
    counts = np.asarray(counts, dtype=np.int64)
    ns = counts.shape[0]
    t = np.arange(ns, dtype=np.float64)
    lt = np.ones(ns, dtype=np.float64)
    return Spectrogram(counts=counts, calibration=_CAL,
                       time_offsets_s=t, real_time_s=lt, live_time_s=lt)


def _has(peaks, energy):
    return any(abs(p.energy - energy) <= ETOL for p in peaks)


def _sig_at(peaks, energy):
    hit = [p.significance for p in peaks if abs(p.energy - energy) <= ETOL]
    return max(hit) if hit else 0.0


# === фикстуры ===

def _transient_sg():
    """Пик присутствует только в коротком интервале [250:290) длинной записи (600 срезов).
    В интеграле net утоплен (sqrt(ns/окно)-разбавление), окно совпадает с транзиентом."""
    ns, base, height = 600, 200.0, 12.0
    counts = np.full((ns, NC), base, dtype=np.float64)
    bump = _gauss(PEAK_CH, height)
    counts[250:290] = counts[250:290] + bump
    return _mksg(np.round(counts))


def _stable_sg():
    """Пик во ВСЕХ срезах: найден интегралом; транзиент-скан отбросит его как
    совпадающий с integral_peak → в объединённом наборе он один раз."""
    ns, base, height = 120, 200.0, 40.0
    counts = np.full((ns, NC), base, dtype=np.float64)
    counts = counts + _gauss(PEAK_CH, height)[None, :]
    return _mksg(np.round(counts))


def _noise_sg(seed=12345):
    """Чистый пуассоновский фон без пика."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(200.0, size=(600, NC)).astype(np.float64)
    return _mksg(counts)


def _integral(sg, sigma=3.0):
    return find_peaks(np.asarray(sg.total_spectrum(), dtype=np.float64),
                      _WIDTHS, sigma_threshold=sigma, energies=_ENERGIES)


def _transient(sg, integral, sigma=6.0):
    return find_transient_peaks(np.asarray(sg.counts, dtype=np.float64),
                                _ENERGIES, _WIDTHS, integral,
                                transient_sigma_threshold=sigma)


# === тесты ===

def test_transient_found_while_integral_misses():
    """Главный сценарий: транзиент находит ОКОННЫЙ скан, а интегральный — нет."""
    sg = _transient_sg()
    integ = _integral(sg)
    tr = _transient(sg, integ)
    assert not _has(integ, E_AT_PEAK), "интегральный спектр НЕ должен видеть транзиент"
    assert _has(tr, E_AT_PEAK), "оконный скан ОБЯЗАН найти транзиент"
    assert _sig_at(tr, E_AT_PEAK) >= 6.0      # выше порога окна с запасом


def test_stable_peak_deduped_in_merged_set():
    """Стабильный пик найден интегралом; в merged-наборе встречается один раз
    (транзиент-скан отбрасывает совпадающие с integral_peaks)."""
    sg = _stable_sg()
    integ = _integral(sg)
    tr = _transient(sg, integ)
    merged = integ + tr
    n_at = sum(1 for p in merged if abs(p.energy - E_AT_PEAK) <= ETOL)
    assert _has(integ, E_AT_PEAK)             # интеграл его видит
    assert n_at == 1, f"стабильный пик должен быть в merged ровно 1 раз, а не {n_at}"


def test_pure_noise_yields_at_most_one_transient():
    """Контроль ложных: на чистом шуме транзиентов не больше одного."""
    sg = _noise_sg()
    integ = _integral(sg)
    tr = _transient(sg, integ)
    assert len(tr) <= 1, f"на чистом шуме ожидается ≤1 транзиент, получено {len(tr)}"


def test_threshold_monotonic_for_transients():
    """Больший порог значимости → НЕ больше транзиентов (монотонность)."""
    sg = _transient_sg()
    integ = _integral(sg)
    n_low = len(_transient(sg, integ, sigma=6.0))
    n_high = len(_transient(sg, integ, sigma=20.0))
    assert n_high <= n_low


def test_degenerate_inputs_return_empty():
    """Защита: пустой / 1D / рассинхрон каналов / NaN-only → []."""
    # пустой
    assert find_transient_peaks(np.zeros((0, NC)), _ENERGIES, _WIDTHS, []) == []
    # 1D (не матрица время×энергия)
    assert find_transient_peaks(np.zeros(NC), _ENERGIES, _WIDTHS, []) == []
    # рассинхрон числа каналов с осью энергий
    bad = np.zeros((100, NC + 7), dtype=np.float64)
    assert find_transient_peaks(bad, _ENERGIES, _WIDTHS, []) == []
    # слишком мало срезов (< min_slices)
    few = np.full((10, NC), 200.0)
    assert find_transient_peaks(few, _ENERGIES, _WIDTHS, []) == []
    # NaN-only матрица достаточного размера → не падает, []
    nans = np.full((120, NC), np.nan)
    assert find_transient_peaks(nans, _ENERGIES, _WIDTHS, []) == []


# === интеграция с view3d (#113 встроен в _found_peaks) ===

@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def test_view3d_found_peaks_includes_transient(app):
    """_found_peaks() на транзиентной спектрограмме возвращает пик ~канала 120,
    которого нет в чисто интегральном результате."""
    from awf.ui.view3d import Waterfall3DView
    sg = _transient_sg()
    v = Waterfall3DView()
    v.set_spectrogram(sg, max_time=400, max_chan=512)
    peaks = v._found_peaks()
    assert _has(peaks, E_AT_PEAK), "транзиент должен попасть в _found_peaks()"


def test_view3d_monotonic_with_sigma(app):
    """Монотонность через UI: больший σ → не больше пиков (интеграл+транзиент)."""
    from awf.ui.view3d import Waterfall3DView
    sg = _transient_sg()
    v = Waterfall3DView()
    v.set_spectrogram(sg, max_time=400, max_chan=512)
    n_low = len(v._found_peaks())
    v.set_peak_sigma(8.0)
    n_high = len(v._found_peaks())
    assert n_high <= n_low


def test_view3d_short_record_no_transient_scan(app):
    """Короткая запись (< min_slices) — транзиент-скан не запускается, поведение
    как у чистого интеграла (см. test_peaks_ui: ns=10 → ровно интегральные пики)."""
    from awf.ui.view3d import Waterfall3DView
    ns = 10
    base = np.full((ns, NC), 50.0)
    base = base + _gauss(PEAK_CH, 30.0)[None, :]
    sg = _mksg(np.round(base))
    v = Waterfall3DView()
    v.set_spectrogram(sg, max_time=400, max_chan=512)
    # не падает; число пиков = интегральное (транзиент-скан вернул [])
    integ = _integral(sg)
    peaks = v._found_peaks()
    assert len(peaks) == len([p for p in integ if p.energy <= 3000.0])
