"""Тесты математики фона/вычитания (Задача #96): диапазон-фон, файл-фон с интерполяцией
по спектральной плотности, знаковый вычет в отсчётах (без клипа к 0, Задача #134 —
интеграл по времени сокращается точно, когда фон = самому спектру)."""
import numpy as np
import pytest

from awf.model.spectrogram import Calibration, Spectrogram
from awf.model.background import (background_from_range, background_from_spectrogram,
                                  subtract_background, gate_significant_net,
                                  background_window_like, tile_background_block)


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


# ---------- #147: сырой блок -> поячеечный вычет (самовычет = точный ноль в каждой ячейке) ----------

def test_self_subtraction_raw_block_exact_zero_per_cell():
    # #147: bg_raw = тот же файл -> counts − bg_counts·(lt/bg_lt) — ноль ПОЯЧЕЕЧНО,
    # а не только в интеграле по времени (усреднённый bg_cps оставляет пуассонов остаток).
    counts = np.array([[10, 1], [2, 9], [6, 5], [0, 3]], dtype=np.float64)
    sg = _sg(counts, live=[1.0, 2.0, 1.0, 3.0])
    bg_cps = background_from_range(sg, 0, sg.n_slices)
    raw = (counts.copy(), np.array([1.0, 2.0, 1.0, 3.0]))
    out = subtract_background(sg, bg_cps, raw)
    assert np.allclose(out.counts, 0.0, atol=1e-12)          # точный ноль в каждой ячейке
    assert (subtract_background(sg, bg_cps).counts != 0).any()   # усреднённый путь — не ноль


def test_subtract_raw_block_shape_mismatch_falls_back():
    # #147: блок под-диапазона (форма != sg.counts) -> прежний путь через средний bg_cps
    sg = _sg(np.array([[10, 5], [10, 5]], dtype=np.float64), live=[1.0, 2.0])
    raw = (np.array([[10.0, 5.0]]), np.array([1.0]))
    out = subtract_background(sg, np.array([1.0, 0.0]), raw)
    assert np.allclose(out.counts, [[9, 5], [8, 5]])         # как без raw (см. тест выше)


# ---------- #136: значимостное гашение нетто (самовычет читается как ноль, как в BecqMoni) ----------

def test_gate_zeros_subthreshold_noise():
    # Задача #136: нетто в пределах счётной статистики (|net| < kσ, σ≈√(bg·lt)) гасится в ноль.
    # base=40, отклонения ±3 при σ≈√40≈6.3, 3σ≈19 => весь нетто-шум уходит в 0 (самовычет=ноль).
    # Столбцы отклонений суммируются в 0 => bg=ровно 40/канал, net=отклонение.
    base = np.full((4, 3), 40.0)
    dev = np.array([[3, -2, 1], [-3, 2, -1], [2, -3, 3], [-2, 3, -3]], dtype=np.float64)
    sg = _sg(base + dev, live=[1.0, 1.0, 1.0, 1.0])
    bg = background_from_range(sg, 0, sg.n_slices)          # ≈40 cps/канал
    sub = subtract_background(sg, bg)
    gated = gate_significant_net(sub, bg, k=3.0)
    assert np.count_nonzero(gated.counts) == 0             # весь нетто-шум погашен в ноль


def test_gate_keeps_significant_net():
    # Задача #136: значимый нетто-пик (|net| ≫ kσ) переживает гашение; фоновая ячейка (net=0) гаснет.
    base = np.full((3, 3), 40.0)
    base[1, 1] += 200.0                                     # яркий транзиент: срез 1, канал 1
    sg = _sg(base, live=[1.0, 1.0, 1.0])
    bg = background_from_range(sg, 0, sg.n_slices)
    sub = subtract_background(sg, bg)
    gated = gate_significant_net(sub, bg, k=3.0)
    assert gated.counts[1, 1] > 0.0                        # значимый нетто пережил гашение
    assert gated.counts[0, 0] == 0.0                       # канал 0 стационарен (net=0) => 0


def test_gate_returns_new_spectrogram_without_mutating():
    # гашение отдаёт НОВУЮ спектрограмму; исходный знаковый вычет (модель среза, #134) не трогаем.
    sg = _sg(np.array([[10, 1], [2, 9]], dtype=np.float64), live=[1.0, 1.0])
    bg = background_from_range(sg, 0, sg.n_slices)
    sub = subtract_background(sg, bg)
    before = sub.counts.copy()
    gated = gate_significant_net(sub, bg, k=3.0)
    assert gated is not sub
    assert np.array_equal(sub.counts, before)              # исходный вычет не мутирован


def test_gate_length_mismatch_raises():
    sg = _sg(np.array([[1.0, 1.0]]))
    sub = subtract_background(sg, np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        gate_significant_net(sub, np.array([1.0, 2.0, 3.0]))
# ---------- #137: посегментно усреднённый нетто (образец и фон сглажены одинаково) ----------

def _seg(t_lo, t_hi):
    # лёгкий TimeSegment: region_averaged_net читает только .t_lo/.t_hi
    from awf.analysis.segment import TimeSegment
    return TimeSegment(t_lo, t_hi, 0.0, 0.0, 0.0, 0)


def test_region_averaged_self_subtract_is_exact_zero():
    # Задача #137: фон = весь файл, один участок стабильности на всю запись => образец,
    # усреднённый по участку, тождественно равен фону => нетто ≡ 0 поканально (точный ноль,
    # без гейта 3σ). Именно это оператор проверял в BecqMoni: вычет одинаковых спектров = ноль.
    from awf.model.background import region_averaged_net
    counts = np.array([[10, 1], [2, 9], [6, 5], [0, 3]], dtype=np.float64)
    sg = _sg(counts, live=[1.0, 2.0, 1.0, 3.0])
    bg = background_from_range(sg, 0, sg.n_slices)
    out = region_averaged_net(sg, bg, [_seg(0, sg.n_slices)])
    assert np.allclose(out.counts, 0.0, atol=1e-9)          # самовычет = точный ноль везде


def test_region_averaged_preserves_per_region_integral():
    # Задача #137: усреднение перераспределяет отсчёты РАВНОМЕРНО (по live-time) внутри участка,
    # но интеграл нетто по участку сохраняется точно = Σ(counts − bg·lt) — знаковая модель #134.
    from awf.model.background import region_averaged_net
    counts = np.array([[10, 1], [2, 9], [6, 5], [0, 3]], dtype=np.float64)
    sg = _sg(counts, live=[1.0, 2.0, 1.0, 3.0])
    bg = background_from_range(sg, 0, sg.n_slices)
    out = region_averaged_net(sg, bg, [_seg(0, 2), _seg(2, 4)])   # два участка
    signed = subtract_background(sg, bg)                          # эталон интеграла (#134)
    for lo, hi in [(0, 2), (2, 4)]:
        assert np.allclose(out.counts[lo:hi].sum(axis=0),
                           signed.counts[lo:hi].sum(axis=0), atol=1e-9)


def test_region_averaged_is_smooth_within_region():
    # Задача #137: внутри участка все срезы показывают ОДНУ форму спектра (net_rate·lt) —
    # образец сглажен так же, как фон; убирает картину «фон гладкий, образец шумный».
    from awf.model.background import region_averaged_net
    counts = np.array([[10, 1], [2, 9], [6, 5]], dtype=np.float64)
    sg = _sg(counts, live=[1.0, 2.0, 3.0])
    bg = np.array([1.0, 1.0])                                # произвольный фон (не весь файл): net≠0
    out = region_averaged_net(sg, bg, [_seg(0, 3)])
    rate = out.counts / np.asarray(sg.live_time_s)[:, None]  # cps/канал каждого среза
    assert np.allclose(rate[0], rate[1]) and np.allclose(rate[1], rate[2])   # одинаковая форма
    assert not np.allclose(rate[0], 0.0)                     # форма нетривиальна (net_rate≠0)


def test_region_averaged_length_mismatch_raises():
    from awf.model.background import region_averaged_net
    sg = _sg(np.array([[1.0, 1.0]]))
    with pytest.raises(ValueError):
        region_averaged_net(sg, np.array([1.0, 2.0, 3.0]), [_seg(0, 1)])


# ---------- #139: сырой фон, «лохматый как образец» (совпадение статистики по live-time) ----------

def test_bg_window_full_range_equals_raw_sum():
    # tgt = полное живое время фона => Σ сырых отсчётов (полный пуассонов шум, «лохматый»)
    counts = np.array([[10, 0], [2, 8], [6, 4]], dtype=np.float64)
    win = background_window_like(counts, np.array([1.0, 1.0, 1.0]), 3.0)
    assert np.allclose(win, [18.0, 12.0])              # Σ по времени, без усреднения


def test_bg_window_scales_to_target_live_time():
    # tgt=1 покрывается первым срезом; масштаб ровно к tgt (коэффициент 1.0)
    counts = np.array([[10, 0], [4, 8]], dtype=np.float64)
    win = background_window_like(counts, np.array([1.0, 1.0]), 1.0)
    assert np.allclose(win, [10.0, 0.0])               # первый срез, без усреднения


def test_bg_window_matches_sample_window():
    # ключевой инвариант #139: фон=образцу и окно совпадает => оверлей == сырому образцу
    counts = np.array([[10, 3], [2, 9], [6, 5], [0, 7]], dtype=np.float64)
    lt = np.array([1.0, 1.0, 1.0, 1.0])
    smp = counts[0:2].sum(axis=0)                       # «образец»: сырое окно [0:2]
    win = background_window_like(counts[0:2], lt[0:2], 2.0)
    assert np.allclose(win, smp)                        # совпали (фон не сглажен)


def test_bg_window_zero_target_returns_raw_sum():
    counts = np.array([[5, 1], [3, 2]], dtype=np.float64)
    win = background_window_like(counts, np.array([1.0, 1.0]), 0.0)
    assert np.allclose(win, [8.0, 3.0])                # tgt<=0 => Σ сырых (fallback)


def test_bg_window_bad_shape_raises():
    with pytest.raises(ValueError):
        background_window_like(np.array([1.0, 2.0]), np.array([1.0]), 1.0)


def test_tile_full_range_phase0_is_identity():
    # Задача #149: блок = вся шкала, phase=0 -> тождественная копия
    counts = np.arange(12.0).reshape(4, 3)
    lt = np.array([1.0, 2.0, 3.0, 4.0])
    tc, tl = tile_background_block(counts, lt, 4, phase=0)
    assert np.array_equal(tc, counts) and np.array_equal(tl, lt)


def test_tile_subrange_clones_itself_inside_and_cycles_outside():
    # Задача #149: участок [lo:hi) клонирует сам себя (самовычет=0, #147), снаружи — циклически
    counts = np.arange(20.0).reshape(10, 2)
    lo, hi = 3, 6
    tc, tl = tile_background_block(counts[lo:hi], np.arange(10.0)[lo:hi], 10, phase=lo)
    assert np.array_equal(tc[lo:hi], counts[lo:hi])     # внутри — сами себя
    assert np.array_equal(tc[0], counts[lo])            # (0-3)%3=0 -> первый срез блока
    assert np.array_equal(tc[6], counts[lo])            # (6-3)%3=0 -> цикл после участка
    assert tc.shape == (10, 2) and tl.shape == (10,)


def test_tile_bad_shape_raises():
    with pytest.raises(ValueError):
        tile_background_block(np.zeros((0, 3)), np.zeros(0), 5)   # пустой блок
    with pytest.raises(ValueError):
        tile_background_block(np.zeros((2, 3)), np.zeros(3), 5)   # lt не по срезам


def test_tile_subtract_bg_region_is_exact_zero():
    # Задача #149: вычет тайлированного фона обнуляет участок [lo:hi) точно, поканально
    rng = np.random.default_rng(5)
    sg = _sg(rng.poisson(9.0, (8, 4)).astype(np.float64))
    lo, hi = 2, 5
    raw = tile_background_block(sg.counts[lo:hi], sg.live_time_s[lo:hi], 8, phase=lo)
    bg_cps = background_from_range(sg, lo, hi)
    net = subtract_background(sg, bg_cps, bg_raw=raw)
    assert np.all(net.counts[lo:hi] == 0.0)             # самовычет участка — точный ноль
    outside = np.delete(net.counts, np.s_[lo:hi], axis=0)
    assert np.any(outside != 0.0)                       # вне участка — реальная разность

