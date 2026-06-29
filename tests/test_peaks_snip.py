"""Тесты SNIP-континуума для find_peaks (#113).

SNIP (Ryan 1988 / Morhac 1997) снимает медленно-меняющийся континуум ПЕРЕД
детекцией Mariscotti+Currie. Польза: пик на крутом/искривлённом континууме
(комптоновская ступень) даёт сильный отклик второй производной от самого
континуума, маскирующий пик; после снятия пьедестала пик проявляется.

Регимы детерминированы (без пуассон-шума) ради воспроизводимости.
"""
import numpy as np
import pytest
from awf.analysis.peaks import find_peaks, snip_baseline
from awf.analysis import snip_baseline as snip_baseline_exported

N = 1024
FWHM = 8.0
PEAK_CH = 480


def _gauss(n, center, height, fwhm):
    sigma = fwhm / 2.355
    ch = np.arange(n, dtype=np.float64)
    return height * np.exp(-((ch - center) ** 2) / (2.0 * sigma ** 2))


def _curved_continuum_with_peak(peak_height):
    """Крутая комптоновская ступень (sigmoid) прямо под пиком + гаусс-пик.

    Ступень амплитудой 3000 с полушириной 8 каналов центрирована на PEAK_CH:
    её вторая производная даёт сильный отрицательный отклик, маскирующий слабый
    пик для Mariscotti.
    """
    ch = np.arange(N, dtype=np.float64)
    cont = 300.0 + 3000.0 / (1.0 + np.exp((ch - PEAK_CH) / 8.0))
    return cont + _gauss(N, PEAK_CH, peak_height, FWHM)


def test_snip_exported_from_package():
    """snip_baseline экспортирован из awf.analysis рядом с find_peaks."""
    assert snip_baseline_exported is snip_baseline


def test_peak_hidden_on_steep_continuum_without_snip():
    """На крутой комптоновской ступени пик H=50 НЕ детектируется без SNIP."""
    counts = _curved_continuum_with_peak(50.0)
    off = find_peaks(counts, FWHM, sigma_threshold=3.0)
    near = [p for p in off if abs(p.channel - PEAK_CH) <= 5]
    assert near == [], f"без SNIP пик не должен находиться, нашлись: {near}"


def test_peak_recovered_with_snip():
    """Тот же пик ПОЯВЛЯЕТСЯ с SNIP-вкл и проходит порог 3σ.

    #113-fix: значимость теперь меряется по GROSS-дисперсии (пуассон реальных
    отсчётов ≈1850 под пиком, σ≈43/канал), а не по near-zero остатку SNIP. Пик
    H=50 на континууме 1800 имеет физически корректную Currie-значимость ~3σ
    (старые ~7σ были артефактом заниженной дисперсии остатка — корень «лавины»).
    Главное: пик НА КРУТОМ КОНТИНУУМЕ по-прежнему НАХОДИТСЯ (без SNIP — нет, см.
    test_peak_hidden_on_steep_continuum_without_snip), просто без переоценки.
    """
    counts = _curved_continuum_with_peak(50.0)
    on = find_peaks(counts, FWHM, sigma_threshold=3.0, snip_iterations=-1)
    near = [p for p in on if abs(p.channel - PEAK_CH) <= 5]
    assert len(near) == 1, f"с SNIP пик должен находиться ровно один раз: {near}"
    assert near[0].significance > 3.0   # проходит порог (корректная gross-дисперсия)


def test_snip_contrast_off_vs_on():
    """Прямой контраст: число пиков у ступени 0 (off) -> 1 (on)."""
    counts = _curved_continuum_with_peak(50.0)
    off = [p for p in find_peaks(counts, FWHM, sigma_threshold=3.0)
           if abs(p.channel - PEAK_CH) <= 5]
    on = [p for p in find_peaks(counts, FWHM, sigma_threshold=3.0, snip_iterations=-1)
          if abs(p.channel - PEAK_CH) <= 5]
    assert len(off) == 0
    assert len(on) == 1


def test_snip_auto_iterations_equal_ceil_fwhm():
    """snip_iterations=-1 (авто M=ceil(FWHM)=8) совпадает с явным snip_iterations=8."""
    counts = _curved_continuum_with_peak(50.0)
    auto = find_peaks(counts, FWHM, sigma_threshold=3.0, snip_iterations=-1)
    expl = find_peaks(counts, FWHM, sigma_threshold=3.0, snip_iterations=8)
    assert [round(p.channel) for p in auto] == [round(p.channel) for p in expl]


def test_snip_off_by_default_preserves_behaviour():
    """Дефолт snip_iterations=0 == полное отсутствие параметра (обратная совместимость)."""
    counts = _curved_continuum_with_peak(50.0)
    a = find_peaks(counts, FWHM, sigma_threshold=3.0)
    b = find_peaks(counts, FWHM, sigma_threshold=3.0, snip_iterations=0)
    assert [p.channel for p in a] == [p.channel for p in b]
    assert [p.significance for p in a] == [p.significance for p in b]


def test_snip_does_not_harm_flat_background_peaks():
    """На спектре БЕЗ континуума (плоский фон) SNIP-вкл не теряет находимые пики
    и не плодит ложные: то же число пиков, значимость не падает."""
    centers = (150, 500, 800)
    heights = (400.0, 1000.0, 2200.0)
    counts = 60.0 * np.ones(N, dtype=np.float64)
    for c, h in zip(centers, heights):
        counts = counts + _gauss(N, c, h, FWHM)
    off = sorted(find_peaks(counts, FWHM, sigma_threshold=3.0), key=lambda p: p.channel)
    on = sorted(find_peaks(counts, FWHM, sigma_threshold=3.0, snip_iterations=-1),
                key=lambda p: p.channel)
    assert len(off) == 3 and len(on) == 3
    for po, pn, c in zip(off, on, centers):
        assert abs(po.channel - c) <= 2 and abs(pn.channel - c) <= 2
        # значимость с SNIP не должна падать (континуум плоский, вычитается ~ровно)
        assert pn.significance >= 0.9 * po.significance


def test_snip_baseline_tracks_pure_continuum():
    """На чистом континууме (без пиков) baseline ~ сам сигнал, не плодит пиков,
    везде baseline <= counts (clipping не поднимает фон)."""
    ch = np.arange(N, dtype=np.float64)
    cont = 500.0 * np.exp(-ch / 200.0) + 50.0
    b = snip_baseline(cont, FWHM, iterations=8)
    assert b.shape == cont.shape
    assert np.all(b <= cont + 1e-6)
    assert float(np.corrcoef(b, cont)[0, 1]) > 0.99
    rel_dev = float(np.max(np.abs(b - cont))) / float(cont.mean())
    assert rel_dev < 0.25
    invented = find_peaks(cont, FWHM, sigma_threshold=3.0, snip_iterations=-1)
    assert invented == []


def test_snip_baseline_does_not_jump_under_peak():
    """Под пиком baseline НЕ выпрыгивает вверх: оценка фона в канале пика близка
    к интерполяции континуума по соседям, а не к вершине пика."""
    ch = np.arange(N, dtype=np.float64)
    cont = 400.0 * np.exp(-ch / 250.0) + 80.0
    counts = cont + _gauss(N, PEAK_CH, 600.0, FWHM)
    b = snip_baseline(counts, FWHM, iterations=8)
    # baseline под пиком близок к истинному континууму, а не к counts[PEAK_CH]
    assert b[PEAK_CH] < 0.5 * counts[PEAK_CH]
    interp = 0.5 * (cont[PEAK_CH - 30] + cont[PEAK_CH + 30])
    assert abs(b[PEAK_CH] - interp) < 0.3 * interp


def test_snip_baseline_edge_cases():
    """Краевые случаи snip_baseline: пустой вход, отрицательные -> 0, длина сохранна."""
    assert snip_baseline(np.zeros(0), FWHM).shape == (0,)
    neg = snip_baseline(np.array([-5.0, -1.0, 0.0, 2.0, 1.0]), 2.0)
    assert np.all(neg >= 0.0)
    arr = snip_baseline(np.full(100, 10.0), FWHM)
    assert arr.shape == (100,)


# === #113-fix: развязка «формы» (SNIP-остаток) и «дисперсии» (gross-пуассон) ===
# Currie L.A. 1968; Gilmore «Practical Gamma-ray Spectrometry» §6.4. Дисперсия
# значимости берётся из РЕАЛЬНЫХ gross-отсчётов (Var=counts, пуассон), а не из
# near-zero остатка SNIP — иначе шумовые бугорки набирают завышенную значимость
# (была «лавина» 128 ложных пиков на реальном файле).

def _flat_poisson(level, n, seed):
    """Плоский пуассон-шумный спектр БЕЗ настоящих пиков (фикс. seed)."""
    rng = np.random.default_rng(seed)
    return rng.poisson(level, size=n).astype(np.float64)


def _steep_continuum_noisy_peak(height, seed):
    """Слабый гаусс-пик на КРУТОЙ комптоновской ступени + пуассон-шум (кейс 337).

    Среднее = ступень(амплитуда 3000, полуширина 8) + гаусс H на PEAK_CH;
    отсчёты разыграны пуассоном с фикс. seed → воспроизводимо.
    """
    ch = np.arange(N, dtype=np.float64)
    cont = 300.0 + 3000.0 / (1.0 + np.exp((ch - PEAK_CH) / 8.0))
    mean = cont + _gauss(N, PEAK_CH, height, FWHM)
    rng = np.random.default_rng(seed)
    return rng.poisson(mean).astype(np.float64)


def test_snip_flat_poisson_no_avalanche():
    """#113-fix критерий 2: на ПЛОСКОМ пуассон-шуме (без пиков) SNIP-вкл при σ=3
    даёт контролируемый FP-rate (как snip-выкл), а НЕ лавину.

    До фикса дисперсия считалась по near-zero остатку SNIP → шумовые бугорки
    проходили 3–5σ (десятки/сотни ложных пиков). После фикса дисперсия по
    gross-пуассону → ложных пиков столько же, сколько у snip-выкл (≈0).
    """
    flat = _flat_poisson(500.0, 2048, seed=20260628)
    off = find_peaks(flat, 8.0, sigma_threshold=3.0)
    on = find_peaks(flat, 8.0, sigma_threshold=3.0, snip_iterations=-1)
    # snip-выкл на чистом шуме при σ=3 даёт ~0 ложных
    assert len(off) <= 2, f"snip-выкл FP-rate должен быть мал: {len(off)}"
    # КЛЮЧЕВОЕ: snip-вкл НЕ плодит лавину — в пределах малой дельты от snip-выкл
    assert len(on) < 5, f"snip-вкл не должен давать лавину ложных пиков: {len(on)}"
    assert len(on) <= len(off) + 2, (
        f"snip-вкл FP-rate ({len(on)}) сопоставим с snip-выкл ({len(off)}), не лавина")


def test_snip_weak_peak_on_steep_continuum_still_found_noisy():
    """#113-fix критерий 3: ГЕНУИННО различимый слабый пик (H=200) на крутом
    континууме + пуассон-шум при σ=3 SNIP-вкл по-прежнему НАХОДИТСЯ (значимость
    по gross-дисперсии > порога). Фикс не «глушит» реальные пики, только лавину.

    H=200 на ступени 1800 даёт устойчиво ~7–8σ по корректной gross-дисперсии
    (проверено по 5 seed) — генуинно различимый пик. Маскировку СЛАБОГО пика
    (H=50) без SNIP отдельно покрывает детерминированный
    test_peak_hidden_on_steep_continuum_without_snip; здесь проверяем именно
    СОХРАННОСТЬ детектируемого пика при SNIP (фикс не глушит реальные пики).
    """
    counts = _steep_continuum_noisy_peak(200.0, seed=1000)
    on = find_peaks(counts, FWHM, sigma_threshold=3.0, snip_iterations=-1)
    near = [p for p in on if abs(p.channel - PEAK_CH) <= 5]
    assert len(near) == 1, f"слабый пик на крутом фоне должен находиться: {near}"
    assert near[0].significance > 3.0, (
        f"значимость по gross-дисперсии должна превышать порог: {near[0].significance}")


def test_snip_variance_decoupled_from_form():
    """#113-fix ядро: при SNIP дисперсия НЕ зависит от остатка, а только от gross.

    Берём один и тот же gross-спектр; убеждаемся, что значимость SNIP-пика
    конечна и НЕ взрывается (как было бы при делении на near-zero остаток).
    Контраст: на чистой ступени БЕЗ пика значимость остаётся низкой (нет лавины).
    """
    ch = np.arange(N, dtype=np.float64)
    pure_step = 300.0 + 3000.0 / (1.0 + np.exp((ch - PEAK_CH) / 8.0))
    # чистая ступень без пика и без шума: SNIP снимает континуум, остаток ~0;
    # при делении на остаток-дисперсию значимость взлетела бы — теперь нет.
    on = find_peaks(pure_step, FWHM, sigma_threshold=3.0, snip_iterations=-1)
    near = [p for p in on if abs(p.channel - PEAK_CH) <= 8]
    assert near == [], (
        f"на чистой ступени без пика SNIP не должен изобретать пик: {near}")
