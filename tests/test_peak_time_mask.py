import numpy as np
import pytest
from awf.analysis import peak_time_mask
from awf.analysis.peaks import peak_time_mask as _direct  # тот же объект из модуля

NT = 40        # временных слоёв
NC = 64        # каналов LOD
PEAK_CH = 30   # канал пика
BG = 5.0       # ровный фон на канал


def _column_peak(present_slices, amp=200.0, bg=BG, nt=NT, nc=NC):
    """Матрица (nt, nc): ровный фон bg; в канале PEAK_CH добавлена амплитуда amp
    ТОЛЬКО в слоях из present_slices (итерабельный набор индексов)."""
    z = np.full((nt, nc), bg, dtype=np.float64)
    for i in present_slices:
        z[i, PEAK_CH] += amp
    return z


def test_peak_present_only_in_band():
    z = _column_peak(range(10, 20))
    mask = peak_time_mask(z, PEAK_CH)
    assert mask.dtype == bool
    assert mask.shape == (NT,)
    assert mask[10:20].all()
    # #112-доработка: временно́е гаусс-сглаживание (σ=2) уширяет ступеньку присутствия на
    # ≤ полуокна (±~2 бина у каждого фронта). Сама зона присутствия 10:20 цела, разлёт
    # симметричен и ограничен; допуск поднят с 2 до полуокна*2, опираясь на то что блок
    # остаётся ОДНИМ сегментом (не покрывает всю ось — это проверяет two_separate).
    outside = np.concatenate([mask[:10], mask[20:]])
    assert outside.sum() <= 4
    assert not mask[:6].any() and not mask[24:].any()  # разлёт строго локален у фронтов


def test_peak_present_in_all_slices():
    z = _column_peak(range(NT))
    mask = peak_time_mask(z, PEAK_CH)
    assert mask.sum() >= NT - 2


def test_pure_noise_column_all_false():
    rng = np.random.default_rng(0)
    z = np.full((NT, NC), BG, dtype=np.float64)
    z[:, PEAK_CH] += rng.normal(0.0, 0.5, size=NT)   # колебания ±~0.5 вокруг фона
    mask = peak_time_mask(z, PEAK_CH)
    assert mask.sum() <= 2     # почти всё False


def test_two_separate_appearances_not_merged():
    z = _column_peak(list(range(5, 10)) + list(range(25, 30)))
    mask = peak_time_mask(z, PEAK_CH)
    assert mask[5:10].all()
    assert mask[25:30].all()
    assert not mask[15:20].any()


def test_nt_less_than_two_no_exception():
    z = _column_peak([0], nt=1)
    mask = peak_time_mask(z, PEAK_CH)
    assert mask.shape == (1,)
    assert mask.dtype == bool
    assert mask[0] == True


def test_zero_time_rows_returns_empty():
    z = np.empty((0, NC), dtype=np.float64)
    mask = peak_time_mask(z, PEAK_CH)
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_channel_at_left_edge_uses_available_shoulder():
    z = np.full((NT, NC), BG, dtype=np.float64)
    z[10:20, 0] += 200.0
    mask = peak_time_mask(z, 0)
    assert mask[10:20].all()
    # допуск как в only_in_band: σ-сглаживание уширяет фронты на ≤ полуокна (см. обоснование)
    outside = np.concatenate([mask[:10], mask[20:]])
    assert outside.sum() <= 4


def test_channel_at_right_edge_uses_available_shoulder():
    z = np.full((NT, NC), BG, dtype=np.float64)
    z[10:20, NC-1] += 200.0
    mask = peak_time_mask(z, NC-1)
    assert mask[10:20].all()
    # допуск как в only_in_band: σ-сглаживание уширяет фронты на ≤ полуокна (см. обоснование)
    outside = np.concatenate([mask[:10], mask[20:]])
    assert outside.sum() <= 4


def test_channel_out_of_range_clipped():
    z = _column_peak([10, 11, 12])
    mask = peak_time_mask(z, channel=999)
    assert mask.dtype == bool
    assert len(mask) == NT


def test_nan_inf_filtered_no_exception():
    z = _column_peak(range(10, 20))
    z[0, PEAK_CH] = np.nan
    z[1, 5] = np.inf
    mask = peak_time_mask(z, PEAK_CH)
    assert mask.dtype == bool
    assert mask[10:20].all()


def test_robust_to_counts_vs_cps_scaling():
    z = _column_peak(range(10, 20))
    m_counts = peak_time_mask(z, PEAK_CH)
    m_cps = peak_time_mask(z * 0.013, PEAK_CH)   # произвольный масштаб
    assert np.array_equal(m_counts, m_cps)


def test_nc_less_than_two_returns_all_true():
    z = np.full((NT, 1), BG)
    mask = peak_time_mask(z, 0)
    assert mask.all()
    assert len(mask) == NT
    assert mask.dtype == bool


def _segments(mask):
    """Число непрерывных True-сегментов (как _draw_ridge_segments через np.diff)."""
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return int((d == 1).sum())


def test_weak_stable_noisy_line_is_nearly_continuous():
    """#112-доработка (ключевой инвариант): слабая, но СТАБИЛЬНАЯ линия, присутствующая
    всё измерение, при пер-бин пуассоновском шуме должна давать ПОЧТИ СПЛОШНУЮ маску
    (мало сегментов, высокое покрытие), а не десятки штрихов."""
    rng = np.random.default_rng(7)
    nt, nc, ch = 200, 64, 30
    bg = 30.0
    z = rng.poisson(bg, size=(nt, nc)).astype(np.float64)   # шумовой фон везде
    z[:, ch] = rng.poisson(bg + 12.0, size=nt).astype(np.float64)  # слабый стабильный пик
    mask = peak_time_mask(z, ch)
    assert mask.mean() >= 0.7, f"покрытие {mask.mean():.2f} < 0.7 — линия раздроблена"
    assert _segments(mask) <= 3, f"{_segments(mask)} сегментов > 3 — пунктир-каша"


def test_single_bin_spike_suppressed():
    """#112-доработка: одиночный пер-бин выброс УРОВНЯ ШУМА (1 бин, ~2.5σ) подавляется
    (гейт присутствия + opening). Мощный одиночный бин — это уже реальный краткий
    транзиент и законно выживает; здесь проверяем именно шумовой выброс."""
    rng = np.random.default_rng(3)
    nt, nc, ch = 80, 64, 30
    bg = 50.0
    z = rng.poisson(bg, size=(nt, nc)).astype(np.float64)
    z[:, ch] = rng.poisson(bg, size=nt).astype(np.float64)   # канал = чистый фон, без линии
    z[40, ch] += np.sqrt(bg) * 2.5                            # одиночный выброс ~2.5σ
    mask = peak_time_mask(z, ch)
    assert mask.sum() == 0, "одиночный шумовой выброс не должен формировать сегмент"


def test_real_transient_stays_a_single_mid_segment():
    """#112-доработка (НЕЛЬЗЯ ОСЛАБЛЯТЬ): настоящий транзиент «появился-ушёл» в середине
    оси → ОТДЕЛЬНЫЙ сегмент в середине, НЕ покрывает всю ось."""
    nt, nc, ch = 120, 64, 30
    z = np.full((nt, nc), 5.0, dtype=np.float64)
    z[50:70, ch] += 300.0               # транзиент: срезы 50..69
    mask = peak_time_mask(z, ch)
    assert _segments(mask) == 1, "транзиент должен быть одним сегментом"
    assert mask[55:65].all(), "ядро транзиента покрыто"
    assert not mask[:40].any(), "до появления — пусто (не вся ось)"
    assert not mask[80:].any(), "после ухода — пусто (не вся ось)"
    assert mask.mean() < 0.5, "транзиент не покрывает всю ось времени"


def test_export_is_same_object():
    assert peak_time_mask is _direct
