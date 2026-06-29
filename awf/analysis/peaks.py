"""Поиск фотопиков: фильтр Mariscotti (−G'') + критерий значимости Currie L_C (Задача 2).

Порт SpectraVibe peaks/search.py mariscotti_search() (стр. 261–450). Qt-free, чистый numpy.
Работает в пространстве каналов; энергия присваивается опционально через калибровку.

Методология: Mariscotti M.A., NIM 50 (1967) 309; Currie L.A., Anal. Chem. 40 (1968) 586;
Gilmore & Joss «Practical Gamma-ray Spectrometry» 3-е изд., §9.3.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union
import numpy as np
from awf.analysis.types import FoundPeak

# Задача #114: ширина matched-фильтра может задаваться скаляром (как было),
# функцией от номера канала ИЛИ массивом длины n_channels (per-channel FWHM).
# Массив нужен для широких сцинтилляционных пиков с FWHM(E), растущей по энергии.
FwhmSpec = Union[float, Callable[[int], float], Sequence[float], np.ndarray]
SigmaSpec = Union[float, Callable[[int], float], Sequence[float], np.ndarray]


# ===========================================================================
# Задача #114: модель энергетического разрешения сцинтиллятора FWHM(E).
# Дефолт R(662 кэВ)=7% (CsI(Tl)/NaI(Tl); Knoll «Radiation Detection and
# Measurement» 4-е изд. гл.10; Gilmore «Practical Gamma-ray Spectrometry»
# 2-е изд. гл.2). Статистика: R=FWHM/E ∝ 1/√E ⇒ FWHM ∝ √E.
# FWHM(E)=√(a+b·E+c·E²); √-закон = a=c=0, b=0.07²·662. estimate_fwhm_model
# автофитит a,b(,c) по чистым изолированным пикам, валидирует монотонность,
# при провале откатывается на default_fwhm_model(). fwhm_channels_from_model
# строит per-channel массив ширины matched-фильтра для find_peaks.
# ===========================================================================
_DEFAULT_R_AT_662 = 0.07
_DEFAULT_REF_E = 662.0


@dataclass(frozen=True)
class FwhmModel:
    """Модель разрешения FWHM(E)[кэВ] = √(a + b·E + c·E²). Монотонно растёт по E.
    source: 'auto' (фит по спектру) либо 'default' (7%@662 кэВ, √-закон). Вызов
    векторизован: model(E)→FWHM в кэВ (clip под корнем к 0)."""
    a: float
    b: float
    c: float = 0.0
    source: str = "default"

    def __call__(self, E):
        E = np.asarray(E, dtype=np.float64)
        return np.sqrt(np.clip(self.a + self.b * E + self.c * E * E, 0.0, None))


def default_fwhm_model():
    """Физический дефолт: R(662 кэВ)=7%, √-закон FWHM∝√E (статистический предел).
    FWHM(662)=0.07·662≈46.3 кэВ; R(E)=7%·√(662/E). Откат при невозможном/ненадёжном
    автофите. Источник: Knoll 4-е изд. гл.10; Gilmore 2-е изд. гл.2."""
    b = (_DEFAULT_R_AT_662 ** 2) * _DEFAULT_REF_E
    return FwhmModel(a=0.0, b=b, c=0.0, source="default")


def fwhm_model_keV(E, a, b, c=0.0):
    """Прямая формула FWHM(E)[кэВ]=√(a+b·E+c·E²) без построения FwhmModel.
    E — скаляр или массив; под корнем clip к 0. Удобно для явных коэффициентов."""
    E = np.asarray(E, dtype=np.float64)
    return np.sqrt(np.clip(a + b * E + c * E * E, 0.0, None))


def _is_monotonic_increasing_model(model, e_lo, e_hi, npts=64):
    """Проверка: FWHM(E) конечна, положительна и не убывает на [max(e_lo,1),e_hi]
    (npts точек). Защищает от автофита с отрицательным наклоном (FWHM падает по E)."""
    if e_hi <= e_lo:
        return True
    Es = np.linspace(max(e_lo, 1.0), e_hi, npts)
    w = np.asarray(model(Es), dtype=np.float64)
    if not np.all(np.isfinite(w)):
        return False
    if np.any(w <= 0.0):
        return False
    return bool(np.all(np.diff(w) >= -1e-9))


def _fit_gauss_fwhm_keV(counts, energies, e0, *, search_keV=25.0, half_keV=None):
    """Фит «гаусс+линейный фон» в ПРОСТРАНСТВЕ ЭНЕРГИЙ у анкера e0[кэВ].
    Локализует максимум в ±search_keV, фитит окно ±half_keV (по умолч. 1.5·10%·e0)
    через scipy curve_fit; sigma сразу в кэВ → FWHM=2.3548·|sigma|.
    Возвращает (mu_кэВ, FWHM_кэВ) либо None (нет scipy / край / провал фита /
    неположит. амплитуда/sigma / mu вне окна). Анти-галлюцинация: число из фита, не из памяти."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None
    counts = np.asarray(counts, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    n = counts.size
    c0 = int(np.searchsorted(energies, e0))
    if c0 <= 0 or c0 >= n - 1:
        return None
    local_disp = float(energies[c0 + 1] - energies[c0])
    if local_disp <= 0:
        return None
    search_ch = max(3, int(round(search_keV / local_disp)))
    s_lo = max(0, c0 - search_ch)
    s_hi = min(n, c0 + search_ch + 1)
    if s_hi - s_lo < 5:
        return None
    c_peak = s_lo + int(np.argmax(counts[s_lo:s_hi]))
    if half_keV is None:
        half_keV = 1.5 * (0.10 * max(float(e0), 1.0))
    half_ch = max(8, int(round(half_keV / local_disp)))
    f_lo = max(0, c_peak - half_ch)
    f_hi = min(n, c_peak + half_ch + 1)
    x = energies[f_lo:f_hi]
    y = counts[f_lo:f_hi]
    if x.size < 7:
        return None
    mu0 = float(energies[c_peak])
    A0 = float(y.max() - 0.5 * (y[0] + y[-1]))
    sig0 = max(0.05 * float(e0), local_disp * 3.0)
    b00 = 0.5 * (y[0] + y[-1])
    b10 = (y[-1] - y[0]) / max(x[-1] - x[0], 1e-9)

    def g(xx, A, mu, sigma, b0, b1):
        return A * np.exp(-0.5 * ((xx - mu) / sigma) ** 2) + b0 + b1 * (xx - mu)

    try:
        popt, _ = curve_fit(g, x, y, p0=[max(A0, 1.0), mu0, sig0, b00, b10], maxfev=20000)
    except Exception:
        return None
    A, mu, sigma, _, _ = popt
    sigma = abs(float(sigma))
    if A <= 0 or sigma <= 0 or not (x[0] <= mu <= x[-1]):
        return None
    fwhm_keV = 2.3548 * sigma
    if fwhm_keV <= 0:
        return None
    return (float(mu), float(fwhm_keV))


def estimate_fwhm_model(counts, energies, *, anchor_energies=(583.0, 662.0),
                        search_keV=25.0, min_anchors=2, degree=2):
    """Автооценка FWHM(E) по спектру: фит гаусса у каждого анкера → пары (E,FWHM),
    МНК на W²=a+b·E (2 пар.) либо a+b·E+c·E² (3 пар. при degree≥3 и ≥3 анкеров).
    Валидация монотонности на всём диапазоне; немонотонная/<min_anchors/нет scipy →
    default_fwhm_model(). Не бросает исключение. Анкеры 583/662 кэВ — главные."""
    energies = np.asarray(energies, dtype=np.float64)
    if energies.size < 2:
        return default_fwhm_model()
    Es, Ws = [], []
    for e0 in anchor_energies:
        r = _fit_gauss_fwhm_keV(counts, energies, e0, search_keV=search_keV)
        if r is not None:
            Es.append(r[0])
            Ws.append(r[1])
    if len(Es) < int(min_anchors):
        return default_fwhm_model()
    E = np.asarray(Es, dtype=np.float64)
    W2 = np.asarray(Ws, dtype=np.float64) ** 2
    use_cubic = (int(degree) >= 3 and E.size >= 3)
    try:
        if use_cubic:
            M = np.vstack([np.ones_like(E), E, E * E]).T
            sol, *_ = np.linalg.lstsq(M, W2, rcond=None)
            a, b, c = float(sol[0]), float(sol[1]), float(sol[2])
        else:
            M = np.vstack([np.ones_like(E), E]).T
            sol, *_ = np.linalg.lstsq(M, W2, rcond=None)
            a, b, c = float(sol[0]), float(sol[1]), 0.0
    except Exception:
        return default_fwhm_model()
    model = FwhmModel(a=a, b=b, c=c, source="auto")
    lo = float(np.min(energies))
    hi = float(np.max(energies))
    if not _is_monotonic_increasing_model(model, lo, hi):
        return default_fwhm_model()
    return model


def auto_calibrate_fwhm_model(counts, energies, *, sigma_threshold=8.0,
                              max_anchors=8, min_anchors=2, max_energy_kev=3000.0):
    """Автокалибровка модели разрешения FWHM(E) под РЕАЛЬНЫЙ детектор (Задача #120).

    В отличие от estimate_fwhm_model (фикс-анкеры 583/662 кэВ, в реальных файлах
    часто отсутствуют -> откат на default), анкеры берутся из самого спектра: сильные
    чистые пики, найденные первым проходом find_peaks по дефолт-модели. Реальное
    разрешение сцинтиллятора часто УЖЕ физического дефолта (R=7%@662 кэВ); слишком
    широкое ядро matched-фильтра -G'' занижает значимость Currie L_C и пропускает узкие
    пики. Калибровка возвращает ширину фильтра к измеренной (Задача #120).

    Алгоритм:
      1) first-pass: ширины из default_fwhm_model() -> find_peaks(sigma_threshold) даёт
         только СИЛЬНЫЕ чистые пики (высокий sigma — порог по умолчанию 8.0 sigma).
      2) фильтр energy <= max_energy_kev; сортировка по significance убыв.; до max_anchors
         анкеров, требуя изоляции (новый анкер дальше ~2*его измеренной FWHM от уже взятых).
      3) для каждого анкера фит реальной FWHM через _fit_gauss_fwhm_keV(e0=pk.energy);
         пары (E, FWHM_кэВ), None отбрасываются; анкер ШИРЕ дефолт-модели на своей
         энергии отбрасывается как corrupted-фит (мультиплет/низкая статистика).
      4) пар < min_anchors -> default_fwhm_model() (сохранить поведение на бедных спектрах).
      5) фит sqrt-закона той же формы, что у дефолта (a=0, c=0): взвешенный LSQ через начало
         координат b = Sum(E_i*W_i^2) / Sum(E_i^2); защита от b<=0.
      6) валидация _is_monotonic_increasing_model на диапазоне energies; провал/b<=0 ->
         default_fwhm_model().

    sqrt-закон FWHM∝sqrt(E) — статистический предел R∝1/sqrt(E) (Knoll «Radiation Detection
    and Measurement» 4-е изд. гл.10; Gilmore «Practical Gamma-ray Spectrometry» 2-е изд.
    гл.2). Числа из методологии Задачи #120, не из памяти; FWHM каждого анкера — из фита.

    Args:
        counts: 1D-массив отсчётов суммарного спектра (full resolution).
        energies: массив энергий [кэВ], длина = длина counts.
        sigma_threshold: порог значимости first-pass (сильные пики, дефолт 8.0 sigma).
        max_anchors: макс. число анкеров для фита (дефолт 8).
        min_anchors: мин. число успешных фитов; меньше -> default (дефолт 2).
        max_energy_kev: верхний предел энергии анкера [кэВ] (дефолт 3000).

    Returns:
        FwhmModel: source="auto" при успехе, иначе default_fwhm_model() (source="default").
    """
    counts = np.asarray(counts, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    if energies.size < 2:
        return default_fwhm_model()
    # 1) first-pass по дефолт-модели — только сильные чистые пики.
    widths0 = fwhm_channels_from_model(default_fwhm_model(), energies)
    strong = find_peaks(counts, widths0, sigma_threshold=sigma_threshold, energies=energies)
    if not strong:
        return default_fwhm_model()
    # 2) фильтр по энергии, сортировка по значимости убыв., отбор изолированных анкеров.
    cand = [pk for pk in strong if float(pk.energy) <= max_energy_kev]
    cand.sort(key=lambda p: float(p.significance), reverse=True)
    default = default_fwhm_model()
    Es, Ws = [], []
    for pk in cand:
        if len(Es) >= int(max_anchors):
            break
        e0 = float(pk.energy)
        r = _fit_gauss_fwhm_keV(counts, energies, e0)
        if r is None:
            continue
        mu, fwhm_keV = r
        # Отбраковка corrupted-фита: анкер ШИРЕ дефолт-модели на своей энергии —
        # фит подцепил мультиплет/сосед или поплыл на низкой статистике. Мы калибруем
        # именно потому, что детектор У́ЖЕ дефолта (R=7%@662 кэВ); анкер шире дефолта
        # физически не может быть тем разрешением, что калибруем — отбрасываем, иначе
        # через-нулевой LSQ (вес ∝ E·W²) задирает b и модель становится ШИРЕ дефолта
        # (на реальном файле так раздувался b из-за пары 924/2604 кэВ). Задача #120.
        if fwhm_keV > float(default(mu)):
            continue
        # изоляция: новый анкер дальше ~2*его измеренной FWHM от уже принятых
        if any(abs(mu - e_prev) < 2.0 * fwhm_keV for e_prev in Es):
            continue
        Es.append(mu)
        Ws.append(fwhm_keV)
    # 4) недостаточно анкеров -> откат на default (бедный спектр).
    if len(Es) < int(min_anchors):
        return default_fwhm_model()
    # 5) взвешенный LSQ sqrt-закона через начало координат: W^2 = b*E => b = Sum(E*W^2)/Sum(E^2).
    E = np.asarray(Es, dtype=np.float64)
    W2 = np.asarray(Ws, dtype=np.float64) ** 2
    denom = float(np.sum(E * E))
    if denom <= 0.0:
        return default_fwhm_model()
    b = float(np.sum(E * W2) / denom)
    if not np.isfinite(b) or b <= 0.0:
        return default_fwhm_model()
    model = FwhmModel(a=0.0, b=b, c=0.0, source="auto")
    # 6) валидация монотонности на диапазоне energies.
    lo = float(np.min(energies))
    hi = float(np.max(energies))
    if not _is_monotonic_increasing_model(model, lo, hi):
        return default_fwhm_model()
    return model


def fwhm_channels_from_model(model_or_coeffs, energies, *, floor=1.0):
    """Per-channel массив ширины matched-фильтра [каналов] для find_peaks.
    Принимает FwhmModel / callable(E)→кэВ / кортеж (a,b[,c]). Конверсия кэВ→каналы
    через локальную дисперсию калибровки np.gradient(energies) [кэВ/канал]:
    width_ch = FWHM_кэВ / (кэВ/канал). NaN/inf и нераспознанный вход → floor.
    Результат — длины energies.size, ≥ floor; передаётся в find_peaks(fwhm_channels=...)."""
    energies = np.asarray(energies, dtype=np.float64)
    try:
        if isinstance(model_or_coeffs, FwhmModel):
            model = model_or_coeffs
        elif callable(model_or_coeffs):
            model = model_or_coeffs
        elif isinstance(model_or_coeffs, (tuple, list)) and len(model_or_coeffs) in (2, 3):
            a = float(model_or_coeffs[0])
            b = float(model_or_coeffs[1])
            c = float(model_or_coeffs[2]) if len(model_or_coeffs) == 3 else 0.0
            model = FwhmModel(a=a, b=b, c=c)
        else:
            return np.full(energies.size, floor, dtype=np.float64)
        fwhm_keV = np.asarray(model(energies), dtype=np.float64)
        disp = np.maximum(np.gradient(energies), 1e-9)
        ch = fwhm_keV / disp
        ch = np.nan_to_num(ch, nan=floor, posinf=floor, neginf=floor)
        return np.clip(ch, floor, None).astype(np.float64)
    except Exception:
        return np.full(energies.size, floor, dtype=np.float64)


# Задача #112: окно «плеч» вокруг канала пика для оценки фона.
# В детекторе полноразмерный FWHM = 8 каналов на 8192-сетке (±FWHM ≈ ±8 кан).
# В LOD-пространстве сетка прорежена (nc << 8192), пик занимает ~1 канал; берём
# небольшую отбивку gap от центра, чтобы плечи не цепляли сам пик, и узкую полосу.
_PTM_SHOULDER_GAP = 1     # каналов-отбивки от центра до начала плеча (хвост пика)
_PTM_SHOULDER_WIDTH = 2   # ширина каждой боковой полосы в каналах (усреднение фона)

# Задача #112 (доработка): устойчивость маски к пер-бин пуассоновскому шуму.
# Модель шума: net_i = column_i − baseline_i — реализация пуассоновского счёта за бин;
# дисперсия net на бин ≈ column_i + baseline_i (counts), для слабой стабильной линии
# сравнима с самим сигналом → одиночный бин тонет под порогом, хотя источник присутствует
# ВСЁ измерение. Усреднение W бинов снижает шум как 1/√W, не размывая медленный транзиент.
_PTM_SMOOTH_SIGMA = 2.0   # σ гауссова сглаживания net по оси времени (бины)
_PTM_CLOSE_LEN = 4        # макс. длина разрыва (бины), заращиваемого binary closing
_PTM_MIN_LEN = 3          # мин. длина сегмента (бины); короче — шумовой выброс (opening)
_PTM_SNR_K = 3.0          # порог значимости: net_smooth ≥ med + K·(1.4826·MAD) оси времени


def _shoulder_baseline(z: np.ndarray, channel: int) -> np.ndarray:
    """
    Фон по слоям из боковых полос вокруг channel; обрезка по краям,
    fallback на доступное плечо (как peakmap.window_series:80-89).
    Возвращает 1D длины nt.
    """
    nt, nc = z.shape
    g = _PTM_SHOULDER_GAP
    w = max(1, _PTM_SHOULDER_WIDTH)
    l0, l1 = max(0, channel - g - w), max(0, channel - g)
    r0, r1 = min(nc, channel + g + 1), min(nc, channel + g + 1 + w)
    left = z[:, l0:l1].mean(axis=1) if l1 > l0 else None
    right = z[:, r0:r1].mean(axis=1) if r1 > r0 else None
    if left is None and right is None:
        return np.zeros(nt, dtype=np.float64)
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return 0.5 * (left + right)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Нормированное гауссово ядро 1D, обрезанное на ±3σ (минимум 1 отсчёт по краю)."""
    if sigma <= 0.0:
        return np.ones(1, dtype=np.float64)
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    s = float(k.sum())
    return k / s if s > 0 else np.ones(1, dtype=np.float64)


def _smooth_time(sig: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссово сглаживание 1D по оси времени с reflect-паддингом краёв (не zero —
    у границ сигнал не проседает к нулю и не отрезает присутствие пика на первых/
    последних бинах). Ядро ужимается, если шире паддинга (короткие оси)."""
    sig = sig.astype(np.float64, copy=False)
    n = sig.size
    if n < 2 or sigma <= 0.0:
        return sig.astype(np.float64, copy=True)
    k = _gaussian_kernel(sigma)
    r = k.size // 2
    if r >= n:
        r = max(1, n - 1)
        c = k.size // 2
        k = k[max(0, c - r): c + r + 1]
        k = k / float(k.sum())
        r = k.size // 2
    padded = np.pad(sig, r, mode="reflect")
    return np.convolve(padded, k, mode="valid")


def _morph(mask: np.ndarray, w: int, dilate: bool) -> np.ndarray:
    """Дилатация (dilate=True, OR-окно) или эрозия (dilate=False, AND-окно) булевой
    1D-маски скользящим окном ширины w (нечётной) c reflect-паддингом краёв."""
    n = mask.size
    if w <= 1 or n == 0:
        return mask.astype(bool, copy=True)
    r = w // 2
    pad = np.pad(mask.astype(np.int64), r, mode="reflect")
    csum = np.convolve(pad, np.ones(2 * r + 1, dtype=np.int64), mode="valid")
    return csum > 0 if dilate else csum >= (2 * r + 1)


def _binary_close_open(mask: np.ndarray, close_len: int, min_len: int) -> np.ndarray:
    """closing(close_len) затем opening(min_len) булевой 1D-маски.
    closing = erode∘dilate: заращивает разрывы длиной ≤ close_len между True-блоками.
    opening = dilate∘erode: убирает изолированные True-сегменты длиной < min_len.
    Транзиент длиной ≥ min_len переживает; границы смещаются ≤ полуокна операций."""
    m = mask.astype(bool, copy=True)
    if close_len >= 2:
        wc = close_len if close_len % 2 == 1 else close_len + 1
        m = _morph(_morph(m, wc, dilate=True), wc, dilate=False)
    if min_len >= 2:
        wo = min_len if min_len % 2 == 1 else min_len + 1
        m = _morph(_morph(m, wo, dilate=False), wo, dilate=True)
    return m


def peak_time_mask(
    z_counts: np.ndarray,
    channel: int,
    *,
    rel_threshold: float = 0.15,        # лёгкий относит. пол (доля от max сглаж. net)
    noise_factor: float = _PTM_SNR_K,   # K порога значимости над MAD-шумом оси времени
    min_peak_over_bg: float = 1.0,      # Currie-подобный пол: пик ощутимо выше фона столбца
    smooth_sigma: Optional[float] = None,  # σ гаусс-сглаживания net (None → авто от nt)
    close_len: Optional[int] = None,    # заращивать разрывы ≤ close_len (None → авто от nt)
    min_len: int = _PTM_MIN_LEN,        # убрать сегменты < min_len (binary opening)
) -> np.ndarray:
    """
    Маска присутствия фотопика по времени для обрезки 3D-хребта (Задача #112).
    ФОН = среднее по плечам channel±Δ (примитив из peakmap.window_series), обрезка по
    краям, fallback на плечо. net_i = z[i,channel] − baseline_i (clip к 0).

    Модель шума. net_i — реализация ПУАССОНОВСКОГО счёта за бин; дисперсия net на бин
    ≈ column_i + baseline_i (counts). Для стабильной СЛАБОЙ линии (источник присутствует
    всё измерение) истинная скорость почти постоянна, но пер-бин шум сравним с сигналом →
    одиночные бины проваливаются под порог, наивная маска дробит гребень на десятки штрихов.

    Метод (отличает «слабый но стабильный» от «появился-ушёл» и от одиночного выброса):
    (1) сглаживание net по ВРЕМЕНИ гауссом σ=smooth_sigma бинов — шум давится как 1/√W,
        медленный транзиент не размывается (reflect-края, присутствие у границ не режется);
    (2) Currie-подобный АБСОЛЮТНЫЙ порог над ФОНОМ (нулём net, baseline уже вычтен):
        net_s ≥ noise_factor·σ_noise, где σ_noise=1.4826·центрированный MAD нижней половины
        сглаж. net (фоновая/спадовая зона) — чистый шум, а не уровень сигнала. Устойчивее
        хрупкого относит. rel_threshold·max(net); rel_threshold оставлен лёгким относит.полом;
    (3) морфология: binary closing заращивает разрывы ≤ close_len (стабильная линия →
        почти сплошная маска), opening убирает сегменты < min_len (одиночный выброс гаснет).
    Транзиент длиной ≥ min_len переживает обе операции (границы сдвиг ≤ полуокна).
    Currie-подобный колоночный гейт: net_max < min_peak_over_bg·фон → маска вся False.
    Робастно к counts/cps (все условия масштабно-инвариантны).
    Граничные случаи: nt<1→пусто, nt==1 или nc<2→все True, channel клипуется, NaN/inf→0.
    """
    z = np.asarray(z_counts, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z_counts должен быть 2D (nt, nc)")
    nt, nc = z.shape
    if nt < 1:
        return np.zeros(0, dtype=bool)
    if nt == 1 or nc < 2:
        return np.ones(nt, dtype=bool)  # присутствие не определимо — не прячем хребет
    ch = int(np.clip(int(channel), 0, nc - 1))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    column = z[:, ch]
    baseline = _shoulder_baseline(z, ch)
    net = np.clip(column - baseline, 0.0, None)
    net_max = float(net.max())
    if net_max <= 0.0:
        return np.zeros(nt, dtype=bool)  # нигде нет превышения над фоном
    # Колоночный гейт присутствия (анти-шум, масштаб-инвариант): пиковая нетто-
    # амплитуда столбца должна превысить min_peak_over_bg * уровень_фона. Без него
    # rel_threshold*max(net) пропускает чистый шум, где max(net) сам — лишь шумовой
    # всплеск (см. диагностику #112: net_max/bg ~0.15-0.8 для шума, >=6 для пика).
    # Currie-подобно: сигнал считается пиком, лишь если ощутимо выше континуума.
    bg_level = max(float(np.median(baseline)), 1e-12)
    if net_max < min_peak_over_bg * bg_level:
        return np.zeros(nt, dtype=bool)  # пик не отделим от фона столбца
    # Окна сглаживания/закрытия масштабируются с длиной оси времени nt (None=авто): на
    # длинной оси пер-бин шум тот же, но «непрерывная» линия требует пропорционально шире
    # сглаживание/closing, иначе покрытие падает с ростом nt (см. диагностику MT=300 vs 500).
    # Доли подобраны на реальном файле: σ≈nt/70, close≈nt/18; транзиент (короткая ось)
    # остаётся локальным (малые окна). Полы _PTM_SMOOTH_SIGMA/_PTM_CLOSE_LEN держат
    # минимум на коротких синтетических осях.
    sm = float(smooth_sigma) if smooth_sigma is not None else max(_PTM_SMOOTH_SIGMA, nt / 70.0)
    cl = int(close_len) if close_len is not None else max(_PTM_CLOSE_LEN, int(round(nt / 18.0)))
    # (1) Временно́е гаусс-сглаживание net: давит пер-бин пуассоновский шум как 1/√W,
    # сохраняя медленные изменения (транзиент шире окна не размывается).
    net_s = _smooth_time(net, sm)
    net_s_max = float(net_s.max())
    if net_s_max <= 0.0:
        return np.zeros(nt, dtype=bool)
    # (2) Currie-подобный АБСОЛЮТНЫЙ порог значимости над ФОНОМ (нулём net), не над
    # медианой сигнала — иначе стабильная линия (net_s≈const) всегда проваливается.
    # σ_noise — ЧИСТЫЙ шум (центрированный MAD по нижней половине net_s = фоновая/спадовая
    # зона; если её нет — линия стабильна, шум мал). Масштаб-инвариантно к counts↔cps (всё
    # линейно по net, см. test_robust_to_counts_vs_cps). Порог над НУЛЁМ net (=над фоном
    # плеч): net_s ≥ noise_factor·σ_noise. Стабильная линия net_s≫σ_noise ВЕЗДЕ → сплошная
    # маска; транзиент net_s≈0 вне блока, ≫σ_noise в блоке → отдельный сегмент.
    med_s = float(np.median(net_s))
    low = net_s[net_s <= med_s]                              # фоновая/спадовая часть
    if low.size < 2:
        low = net_s
    sigma_noise = 1.4826 * float(np.median(np.abs(low - np.median(low))))
    sigma_noise = max(sigma_noise, 1e-9 * net_s_max)         # защита от нулевого MAD
    thr_abs = noise_factor * sigma_noise
    cond_abs = net_s >= thr_abs
    cond_rel = net_s >= rel_threshold * net_s_max  # лёгкий относит. пол (доля от max)
    raw = cond_abs & cond_rel
    # (3) Морфология: заращиваем короткие разрывы стабильной линии, убираем шум-выбросы.
    return _binary_close_open(raw, cl, min_len)


# ===========================================================================
# Задача #113: поиск ТРАНЗИЕНТНЫХ (время-локализованных) фотопиков.
# Пик может быть невидим в СУММАРНОМ по времени спектре (значимость < порога,
# утоплен в шуме интеграла), но РЕАЛЕН и значим в узком временно́м окне срезов.
# Источник интегрального спектра единственный (#110/#111) -> такой пик не попадает
# ни в таблицу, ни в гребни 3D. Решение: скользящие ПЕРЕКРЫВАЮЩИЕСЯ окна по оси
# времени, в каждом окне суммируем СЫРЫЕ counts (как total_spectrum) и гоняем
# find_peaks с ПОВЫШЕННЫМ порогом transient_sigma_threshold (по окну меньше
# статистики -> выше пьедестал шума, без запаса наберём ложные). Те же
# калиброванные ширины #120 (разрешение детектора не зависит от времени).
# Currie/Пуассон требуют сырых counts (не cps) -> окна суммируются по counts_2d.
# ===========================================================================
def _fwhm_keV_at_energy(energy, energies, fwhm_channels):
    """FWHM[кэВ] на энергии energy: ширина в каналах * локальная дисперсия [кэВ/канал].
    Провенанс: обратная к fwhm_channels_from_model (там ch=FWHM_кэВ/disp,
    disp=|grad(energies)|). Берём ширину канала, ближайшего к energy, * disp там."""
    energies = np.asarray(energies, dtype=np.float64)
    fw = np.asarray(fwhm_channels, dtype=np.float64)
    n = energies.size
    if n < 2 or fw.size != n:
        return 1.0
    ch = int(np.clip(np.searchsorted(energies, float(energy)), 0, n - 1))
    disp = np.maximum(np.abs(np.gradient(energies)), 1e-9)
    return float(max(fw[ch] * disp[ch], 1e-9))


def _transient_windows(n_slices, w, step):
    """Список (s, e) ПЕРЕКРЫВАЮЩИХСЯ окон [s, s+w) с шагом step; последнее прижато
    к концу (s=n_slices-w); дедуп идентичных пар. w/step берутся уже клипованными."""
    if n_slices <= 0 or w <= 0:
        return []
    w = int(min(w, n_slices))
    step = int(max(1, step))
    wins = []
    s = 0
    while s + w < n_slices:
        wins.append((s, s + w))
        s += step
    wins.append((n_slices - w, n_slices))   # последнее окно прижать к концу
    seen = set()
    out = []
    for se in wins:
        if se not in seen:
            seen.add(se)
            out.append(se)
    return out


def find_transient_peaks(
    counts_2d,
    energies,
    fwhm_channels,
    integral_peaks,
    *,
    transient_sigma_threshold: float = 5.0,
    n_windows: int = 12,
    overlap: float = 0.5,
    merge_energy_factor: float = 1.0,
    max_energy_kev: float = 3000.0,
    min_slices: int = 40,
) -> list[FoundPeak]:
    """Поиск время-локализованных (транзиентных) фотопиков по окнам времени (Задача #113).

    Пик, утопленный в шуме СУММАРНОГО спектра, но значимый в узком окне срезов,
    не находит find_peaks по интегралу. Здесь спектрограмма сканируется
    скользящими ПЕРЕКРЫВАЮЩИМИСЯ окнами по времени: в каждом окне суммируются
    СЫРЫЕ counts (как total_spectrum — Currie/Пуассон требуют сырых отсчётов, не
    cps) и гоняется find_peaks с ПОВЫШЕННЫМ порогом transient_sigma_threshold
    (на окне меньше статистики -> выше шумовой пьедестал; без запаса набрались бы
    ложные). Ширины matched-фильтра — те же калиброванные #120 (разрешение
    детектора от времени не зависит).

    Возвращаются ТОЛЬКО НОВЫЕ пики: совпавшие по энергии с любым из integral_peaks
    (в пределах FWHM(E)) отбрасываются — они уже в таблице из интеграла.

    Args:
        counts_2d: сырые counts формы (n_slices, n_channels).
        energies: массив энергий [кэВ], длина n_channels.
        fwhm_channels: per-channel ширина matched-фильтра [каналов] (#120), длина n_channels.
        integral_peaks: список FoundPeak интегрального спектра (для отсева «не новых»).
        transient_sigma_threshold: порог значимости Currie в окне (дефолт 5.0).
        n_windows: целевое число окон (ширина окна ~ n_slices/n_windows).
        overlap: доля перекрытия соседних окон [0..1) (дефолт 0.5).
        merge_energy_factor: дедуп оконных пиков ближе merge_energy_factor*FWHM(E).
        max_energy_kev: верхний предел энергии пика [кэВ].
        min_slices: меньше срезов -> [] (бедная статистика/синтетика).

    Returns:
        list[FoundPeak]: только транзиентные (новые) пики, отсортированы по энергии.
    """
    arr = np.asarray(counts_2d, dtype=np.float64)
    if arr.ndim != 2:
        return []
    n_slices, n_channels = arr.shape
    energies = np.asarray(energies, dtype=np.float64)
    fw = np.asarray(fwhm_channels, dtype=np.float64)
    # Защитное программирование: бедная статистика, рассинхрон длин, NaN.
    if n_slices < int(min_slices) or n_channels < 50:
        return []
    if energies.size != n_channels or fw.size != n_channels:
        return []
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(energies)):
        return []
    # Ширина/шаг скользящих окон. w >= min_slices, чтобы окно не было беднее порога
    # бедной статистики самой функции; шаг по доле перекрытия overlap.
    w = max(int(min_slices), int(round(n_slices / max(1, int(n_windows)))))
    w = min(w, n_slices)
    step = max(1, int(round(w * (1.0 - float(overlap)))))
    windows = _transient_windows(n_slices, w, step)
    if not windows:
        return []
    # Сбор оконных пиков. Каждый помечается окном (s,e) — для отсева совпадений с
    # integral_peaks и дедупа делаем в одном энергетическом пространстве.
    collected = []   # list[FoundPeak]
    for s, e in windows:
        wspec = arr[s:e].sum(axis=0)            # сырые counts окна (как total_spectrum)
        wp = find_peaks(wspec, fw, sigma_threshold=transient_sigma_threshold,
                        energies=energies)
        for pk in wp:
            if float(pk.energy) <= float(max_energy_kev):
                collected.append(pk)
    if not collected:
        return []
    # Дедуп ВНУТРИ оконных: пики ближе merge_energy_factor*FWHM(E) — один источник
    # (источник присутствует в нескольких перекрывающихся окнах). Оставляем с МАКС
    # significance. Жадно по убыванию значимости (как _enforce_separation_adaptive).
    collected.sort(key=lambda p: float(p.significance), reverse=True)
    merged = []
    for pk in collected:
        e0 = float(pk.energy)
        tol = float(merge_energy_factor) * _fwhm_keV_at_energy(e0, energies, fw)
        if any(abs(e0 - float(m.energy)) < tol for m in merged):
            continue
        merged.append(pk)
    # Отсев «не новых»: совпал по энергии с интегральным пиком (в пределах FWHM(E)) ->
    # он уже в таблице из интеграла, транзиентом не считаем.
    integral_E = [float(p.energy) for p in (integral_peaks or [])]
    transient = []
    for pk in merged:
        e0 = float(pk.energy)
        tol = float(merge_energy_factor) * _fwhm_keV_at_energy(e0, energies, fw)
        if any(abs(e0 - eI) < tol for eI in integral_E):
            continue
        transient.append(pk)
    transient.sort(key=lambda p: float(p.energy))
    return transient


def snip_baseline(
    counts,
    fwhm_channels: float,
    *,
    iterations: Optional[int] = None,
) -> np.ndarray:
    """SNIP-оценка континуума (медленно-меняющегося фона) 1D-спектра.

    SNIP = Statistics-sensitive Non-linear Iterative Peak-clipping —
    Ryan C.G. et al., NIM B34 (1988) 396; LLS-формулировка и связь окна с FWHM —
    Morhac M. et al., NIM A401 (1997) 113. Реализация чистая (Qt-free numpy).

    Алгоритм:
      1) Прямое LLS-преобразование (log-log-sqrt), сжимающее динамический диапазон
         и стабилизирующее дисперсию пуассоновского счёта:
             v = log( log( sqrt(counts + 1) + 1 ) + 1 ).
      2) Итеративный clipping с РАСТУЩИМ полуокном m = 1..M:
             v[i] <- min( v[i], 0.5 * (v[i-m] + v[i+m]) ).
         За M проходов срезаются структуры у́же ~2M каналов (пики), остаётся
         гладкий континуум. Края — зеркальный паддинг (reflect), фон у границ
         не задирается.
      3) Обратное LLS-преобразование возвращает фон в шкалу отсчётов:
             b = (exp(exp(v) - 1) - 1) ** 2 - 1   (точная инверсия шага 1).

    Выбор M (полуширина окна = число итераций), если iterations=None:
        M = ceil(fwhm_channels)
    Обоснование: clipping за M проходов гарантированно срезает любой пик
    полушириной <= M каналов. Фотопик ширины FWHM укладывается в окно ±M при
    M >= FWHM (Morhac §2: ширина окна clipping ~ FWHM пиков); берём верхнюю
    оценку M = ceil(FWHM) — надёжно срезать слегка уширенные/комптоновские
    структуры. Меньшее M недосрезает пик (фон выпрыгивает под пиком), большее
    M начинает срезать сам континуум на изгибах.

    Args:
        counts: 1D-массив отсчётов (отрицательные приводятся к 0).
        fwhm_channels: характерная FWHM пика в каналах (для выбора окна M).
        iterations: явное число итераций/максимальное полуокно M (override).
                    None -> M = ceil(fwhm_channels).

    Returns:
        np.ndarray float64 той же длины — оценка континуума в шкале отсчётов,
        поэлементно <= исходных counts (clipping не поднимает фон).
    """
    c = np.asarray(counts, dtype=np.float64)
    n = c.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(c, 0.0, None)
    if iterations is None:
        m_max = int(math.ceil(max(1.0, float(fwhm_channels))))
    else:
        m_max = int(iterations)
    if m_max < 1:
        return c.copy()
    # 1) Прямое LLS-преобразование.
    v = np.log(np.log(np.sqrt(c + 1.0) + 1.0) + 1.0)
    # 2) Итеративный clipping, m = 1..M. Зеркальный паддинг краёв (reflect).
    for m in range(1, m_max + 1):
        if m >= n:
            break
        padded = np.pad(v, m, mode="reflect")
        left = padded[:n]            # v[i-m]
        right = padded[2 * m:]       # v[i+m]
        avg = 0.5 * (left + right)
        v = np.minimum(v, avg)
    # 3) Обратное LLS-преобразование (точная инверсия шага 1).
    b = np.exp(np.exp(v) - 1.0) - 1.0
    b = b * b - 1.0
    np.clip(b, 0.0, None, out=b)
    return b


def _gaussian_second_derivative_kernel(fwhm_channels: float, half_width: int) -> np.ndarray:
    """Порт search.py:103–126. Ядро второй производной гауссиана."""
    sigma = fwhm_channels / 2.355
    x = np.arange(-half_width, half_width + 1, dtype=np.float64)
    g = ((x / sigma) ** 2 - 1.0) * np.exp(-(x / sigma) ** 2 / 2.0)
    g = g - g.mean()                       # убрать DC (вторая производная интегрируется в 0)
    norm = math.sqrt(float((g * g).sum())) # единичная L2-норма
    if norm > 0:
        g = g / norm
    return g


def _coerce_per_channel(spec, n_channels: int, *, floor: float, name: str) -> np.ndarray:
    """Привести array-like spec к float64-массиву длины n_channels с полом floor.
    NaN/inf → floor; длина обязана совпасть с числом каналов."""
    arr = np.asarray(spec, dtype=np.float64).ravel()
    if arr.size != n_channels:
        raise ValueError(
            f"{name}: массив длины {arr.size} не совпадает с числом каналов {n_channels}")
    arr = np.nan_to_num(arr, nan=floor, posinf=floor, neginf=floor)
    return np.maximum(arr, floor)


def _is_array_like(spec) -> bool:
    """True для последовательностей/ndarray (НЕ скаляр, НЕ callable, НЕ строка)."""
    if callable(spec) or isinstance(spec, (str, bytes)):
        return False
    if isinstance(spec, np.ndarray):
        return spec.ndim >= 1
    return hasattr(spec, "__len__")


def _build_fwhm_array(fwhm_spec: FwhmSpec, n_channels: int) -> np.ndarray:
    """Порт search.py:133–150 + Задача #114 (per-channel массив).

    callable → per-channel max(1.0, float(fwhm_spec(ch)));
    массив длины n_channels → поэлементно max(1.0, float(·));
    скаляр → np.full(n, max(1.0, float(fwhm_spec))) (БИТ-В-БИТ как было).
    """
    if callable(fwhm_spec):
        fwhm_arr = np.array([max(1.0, float(fwhm_spec(ch))) for ch in range(n_channels)], dtype=np.float64)
    elif _is_array_like(fwhm_spec):
        fwhm_arr = _coerce_per_channel(fwhm_spec, n_channels, floor=1.0, name="fwhm_channels")
    else:
        fwhm_arr = np.full(n_channels, max(1.0, float(fwhm_spec)), dtype=np.float64)
    return fwhm_arr


def _build_sigma_threshold_array(sigma_spec: SigmaSpec, n_channels: int) -> np.ndarray:
    """Порт search.py:153–161 + Задача #114 (per-channel массив).

    callable → per-channel float(sigma_spec(ch));
    массив длины n_channels → как есть (порог Currie на канал, пол −inf);
    скаляр → np.full(n, float(sigma_spec)) (БИТ-В-БИТ как было).
    """
    if callable(sigma_spec):
        sigma_arr = np.array([float(sigma_spec(ch)) for ch in range(n_channels)], dtype=np.float64)
    elif _is_array_like(sigma_spec):
        sigma_arr = _coerce_per_channel(sigma_spec, n_channels, floor=-np.inf, name="sigma_threshold")
    else:
        sigma_arr = np.full(n_channels, float(sigma_spec), dtype=np.float64)
    return sigma_arr


def _partition_into_bands(fwhm_arr: np.ndarray, max_ratio: float = 1.2) -> list:
    """Порт search.py:164–200. Растить полосу пока max_fwhm/min_fwhm <= max_ratio; при превышении — закрыть полосу,
    репрезентативная FWHM = геометрическое среднее sqrt(band_lo * band_hi)."""
    n = len(fwhm_arr)
    if n == 0:
        return []
    bands = []
    start = 0
    band_lo = float(fwhm_arr[0])
    band_hi = float(fwhm_arr[0])
    for i in range(1, n):
        v = float(fwhm_arr[i])
        new_lo = min(band_lo, v)
        new_hi = max(band_hi, v)
        if new_hi / new_lo > max_ratio:
            bands.append((start, i, math.sqrt(band_lo * band_hi)))
            start = i
            band_lo = v
            band_hi = v
        else:
            band_lo = new_lo
            band_hi = new_hi
    bands.append((start, n, math.sqrt(band_lo * band_hi)))
    return bands


def _band_filter(counts, band_start, band_end, band_fwhm, n_channels,
                 variance_counts=None) -> np.ndarray:
    """Порт search.py:203–254. Полуширина ядра (BUG-56).

    Развязка «формы» и «дисперсии» (#113-fix). Значимость = отклик фильтра,
    делённый на его СКО (Currie L.A., Anal.Chem.40(1968)586; Gilmore «Practical
    Gamma-ray Spectrometry» 2-е изд. §6.4 «detection limits»). Для линейной
    свёртки response = Σ_j k[j]·c[i−j] распространение пуассоновской ошибки даёт
        σ_resp²(i) = Σ_j k[j]² · Var(c[i−j]) = (k² ∗ Var_counts)(i),
    где Var(c) = c — дисперсия РЕАЛЬНЫХ измеренных gross-отсчётов (пуассон).
    Поэтому ФОРМА отклика берётся из `counts` (после SNIP — остаток без
    пьедестала), а ДИСПЕРСИЯ в знаменателе — из `variance_counts` (исходный
    gross-спектр ДО вычитания). Вычитание континуума меняет оценку фона B, но
    НЕ меняет статистику отсчётов → дисперсию занижать нельзя (иначе шумовые
    бугорки остатка набирают завышенную значимость — корень «лавины» ложных
    пиков). При variance_counts=None дисперсия по `counts` (прежнее поведение,
    бит-в-бит для SNIP-выкл).
    """
    if band_fwhm >= 15.0:
        half_width = max(int(math.ceil(1.0 * band_fwhm)), 3)
    else:
        half_width = max(int(math.ceil(1.5 * band_fwhm)), 3)
    kernel = _gaussian_second_derivative_kernel(band_fwhm, half_width)
    seg_start = max(0, band_start - half_width)
    seg_end = min(n_channels, band_end + half_width)
    seg = counts[seg_start:seg_end]
    response = -np.convolve(seg, kernel, mode="same")
    kernel_sq = kernel ** 2
    var_seg = seg if variance_counts is None else variance_counts[seg_start:seg_end]
    var = np.convolve(np.maximum(var_seg, 1.0), kernel_sq, mode="same")
    sigma_resp = np.sqrt(np.maximum(var, 1e-12))
    sig = response / sigma_resp
    central_offset = band_start - seg_start
    central_len = band_end - band_start
    return sig[central_offset:central_offset + central_len]


def _local_maxima_masked(signal: np.ndarray, mask: np.ndarray) -> list:
    """Порт search.py:457–489. Строгий локальный максимум signal[i] > signal[i-1] и signal[j] > signal[j+1]
    с обработкой плато (берётся левый индекс плато), только где mask[i]."""
    n = signal.size
    if n < 3:
        return []
    out = []
    i = 1
    while i < n - 1:
        if not mask[i]:
            i += 1; continue
        if not (signal[i] > signal[i - 1]):
            i += 1; continue
        j = i
        while j < n - 1 and signal[j + 1] == signal[i]:
            j += 1
        if (j < n - 1) and (signal[j] > signal[j + 1]):
            out.append(i)
        i = j + 1
    return out


def _enforce_separation_adaptive(candidates, significance, fwhm_arr, min_separation_factor) -> list:
    """Порт search.py:492–520. Жадный отбор по убыванию значимости; кандидат принимается если до каждого уже принятого
    abs(ch - a) >= min_separation_factor * max(fwhm_arr[ch], fwhm_arr[a])."""
    if not candidates:
        return []
    accepted = []
    for ch in sorted(candidates, key=lambda c: significance[c], reverse=True):
        if all(abs(ch - a) >= min_separation_factor * max(fwhm_arr[ch], fwhm_arr[a]) for a in accepted):
            accepted.append(ch)
    return sorted(accepted)


def find_peaks(
    counts_1d,
    fwhm_channels: FwhmSpec,
    *,
    sigma_threshold: SigmaSpec = 3.0,
    spike_min_fwhm_frac: float = 0.3,
    min_separation_factor: float = 1.0,
    band_ratio: float = 1.2,
    energies: Optional[np.ndarray] = None,
    snip_iterations: int = 0,
) -> list[FoundPeak]:
    """Поиск пиков в спектре по методу Mariscotti + Currie L_C.

    Args:
        counts_1d: массив подсчетов (dtype=float64)
        fwhm_channels: ширина matched-фильтра −G'' в каналах. Принимает (Задача #114):
            СКАЛЯР (как прежде, бит-в-бит), функцию от номера канала ИЛИ МАССИВ
            длины n_channels с локальной FWHM на каждом канале. Массив нужен для
            широких сцинтилляционных пиков, у которых FWHM(E) растёт по энергии:
            фиксированная узкая ширина (8 кан) в разы у́же реального пика занижает
            отклик фильтра и значимость Currie → реальные пики пропускаются.
            Построить массив из модели FWHM(E) — fwhm_channels_from_model().
        sigma_threshold: порог значимости (скаляр, функция от номера канала или массив длины n)
        spike_min_fwhm_frac: минимальная доля FWHM для фильтра спайков (>0 — включён)
        min_separation_factor: коэффициент мин. расстояния между пиками
        band_ratio: максимальное отношение FWHM в полосе
        energies: массив энергий (длина = длина counts_1d)
        snip_iterations: вычитание SNIP-континуума ПЕРЕД детекцией (#113).
            0 (дефолт) = ВЫКЛ, прежнее поведение. >0 = явное число итераций
            (полуокно M) snip_baseline; <0 = авто M=ceil(FWHM[0]). Пик на
            крутом/искривлённом континууме (напр. комптоновская ступень)
            детектируется после снятия пьедестала.

    Returns:
        список найденных пиков FoundPeak
    """
    counts = np.asarray(counts_1d, dtype=np.float64)
    n = counts.size
    if n < 50:
        return []
    fwhm_arr = _build_fwhm_array(fwhm_channels, n)
    # #113: опциональное вычитание SNIP-континуума ПЕРЕД детекцией. Детекция,
    # континуум-baseline под пиком и net_height далее идут по спектру без
    # пьедестала — пик на крутом/искривлённом фоне набирает значимость Currie.
    # #113-fix: дисперсия значимости ВСЕГДА по gross-спектру (пуассон реальных
    # отсчётов), а не по near-zero остатку SNIP. Сохраняем gross в variance_gross
    # ДО вычитания и передаём в _band_filter как знаменатель Currie L_C.
    variance_gross = None
    if snip_iterations != 0:
        snip_M = None if snip_iterations < 0 else int(snip_iterations)
        baseline = snip_baseline(counts, float(fwhm_arr[0]), iterations=snip_M)
        variance_gross = counts                       # gross ДО вычитания (дисперсия)
        counts = np.clip(counts - baseline, 0.0, None)  # форма (остаток без фона)
    sigma_arr = _build_sigma_threshold_array(sigma_threshold, n)
    if np.all(fwhm_arr < 1.0):
        return []
    bands = _partition_into_bands(fwhm_arr, max_ratio=band_ratio)
    significance = np.zeros(n)
    for b_start, b_end, b_fwhm in bands:
        significance[b_start:b_end] = _band_filter(
            counts, b_start, b_end, b_fwhm, n, variance_counts=variance_gross)
    em_lo = int(math.ceil(2 * fwhm_arr[0]))
    em_hi = int(math.ceil(2 * fwhm_arr[-1]))
    if em_lo > 0:
        significance[:em_lo] = 0.0
    if em_hi > 0:
        significance[-em_hi:] = 0.0
    above_threshold = significance > sigma_arr
    candidates = _local_maxima_masked(significance, above_threshold)
    if not candidates:
        return []
    candidates = _enforce_separation_adaptive(candidates, significance, fwhm_arr, min_separation_factor)
    peaks = []
    for ch in candidates:
        local_fwhm = float(fwhm_arr[ch])
        sigma_ch = local_fwhm / 2.355
        cont_l_lo = max(0, ch - int(round(3 * local_fwhm)))
        cont_l_hi = max(cont_l_lo + 1, ch - int(round(2 * local_fwhm)))
        cont_r_lo = min(n - 1, ch + int(round(2 * local_fwhm)))
        cont_r_hi = min(n, ch + int(round(3 * local_fwhm)))
        b_left = counts[cont_l_lo:cont_l_hi].mean() if cont_l_hi > cont_l_lo else 0.0
        b_right = counts[cont_r_lo:cont_r_hi].mean() if cont_r_hi > cont_r_lo else 0.0
        # BUG-21: лог-линейная интерполяция baseline в канале пика (точна для экспон. континуума).
        c_left = 0.5 * (cont_l_lo + cont_l_hi - 1)
        c_right = 0.5 * (cont_r_lo + cont_r_hi - 1)
        if c_right > c_left and b_left > 0.0 and b_right > 0.0:
            log_b = math.log(b_left) + (math.log(b_right) - math.log(b_left)) * (ch - c_left) / (c_right - c_left)
            b = float(math.exp(log_b))
        elif c_right > c_left:
            b = float(b_left + (b_right - b_left) * (ch - c_left) / (c_right - c_left))
        else:
            b = 0.5 * (b_left + b_right)
        net_height = max(counts[ch] - b, 0.0)
        if net_height <= 0:
            continue
        # Спайк-фильтр (порт F-139, search.py:413–431)
        if spike_min_fwhm_frac > 0.0 and net_height >= 10.0 and local_fwhm >= 4.0:
            half_level = b + 0.5 * net_height
            search_range = max(2, int(round(2.5 * local_fwhm)))
            rr = ch
            while rr < min(n - 1, ch + search_range) and counts[rr] >= half_level:
                rr += 1
            ll = ch
            while ll > max(0, ch - search_range) and counts[ll] >= half_level:
                ll -= 1
            measured_fwhm = max(1.0, float(rr - ll))
            if measured_fwhm < spike_min_fwhm_frac * local_fwhm:
                continue
        area = 2.507 * sigma_ch * net_height
        energy = float(energies[ch]) if energies is not None else float(ch)
        peak = FoundPeak(
            channel=float(ch),
            energy=energy,
            height=net_height,
            fwhm_channels=local_fwhm,
            significance=float(significance[ch]),
            area_estimate=area,
        )
        peaks.append(peak)
    peaks.sort(key=lambda p: p.channel)
    return peaks
