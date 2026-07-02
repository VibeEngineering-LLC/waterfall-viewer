from __future__ import annotations
import numpy as np

# Режимы Z-шкалы (контраст отображения counts). Ключ -> человекочитаемая подпись для UI.
Z_MODES = (("linear", "Линейная"), ("sqrt", "Корень √"), ("log", "Логарифм log10"))

# Дефолты регулировки контраста (Задача 16). При этих значениях apply_z_scale возвращает
# ровно тот же результат, что и базовая Z-шкала — backward-compat short-circuit.
DEFAULT_GAIN = 1.0
DEFAULT_GAMMA = 1.0
DEFAULT_CLIP = (0.0, 100.0)   # перцентили (нижний, верхний)


def _base_transform(a: np.ndarray, mode: str) -> np.ndarray:
    """Базовая Z-шкала контраста: linear -> как есть; sqrt -> √(max(x,0));
    log -> истинный log10(max(x,floor)/floor) с авто-порогом по данным (Задача #54).
    Любой неизвестный режим трактуется как linear."""
    nn = np.maximum(a, 0.0)
    if mode == "sqrt":
        return np.sqrt(nn)
    if mode == "log":
        # Задача #54: истинный log10 с авто-порогом floor. Масштаб-инвариантно: floor
        # масштабируется вместе с данными, поэтому отношение x/floor (а значит и форма рельефа)
        # одинаково для counts и cps. Прежний log10(1+x) был калиброван под целые counts (x≥1);
        # для cps<1 он вырождался в линейный (≈0.4343·x), теряя контраст и зависая от единиц.
        # Задача #167: floor = 10-й перцентиль pos, не 1-й. После ε-нормировки #156 (×10.76 в
        # ВЭ-хвосте, ×~0.5 в НЭ-полке) распределение положительных значений реформируется:
        # percentile-1 уползает вниз вслед за scale-down НЭ-каналов → log10(x/floor) растёт
        # непропорционально → простыня 3D поднимается, а 0-count каналы остаются на Z=0 →
        # визуально глубокие «провалы вниз» (оператор #167). Percentile-10 устойчивее к таким
        # смещениям (симметрично #166 для окна Срезы).
        pos = nn[nn > 0.0]
        if pos.size == 0:
            return nn
        floor = float(np.percentile(pos, 10.0))
        if not (floor > 0.0):
            floor = float(pos.min())
        return np.log10(np.maximum(nn, floor) / floor)
    return nn


def apply_z_scale(arr, mode: str, *, gain: float = DEFAULT_GAIN,
                  gamma: float = DEFAULT_GAMMA, clip=DEFAULT_CLIP):
    """Преобразовать массив отсчётов для отображения. Возвращает float32.

    Конвейер контраста (Задача 16):
      1. базовая Z-шкала (linear/sqrt/log) -> t;
      2. перцентильное отсечение выбросов: уровни lo_v=pctl(t,clip[0]), hi_v=pctl(t,clip[1]),
         t зажимается в [lo_v, hi_v];
      3. нормировка в [0,1] по [lo_v, hi_v];
      4. гамма-кривая n**gamma (gamma<1 — поднять слабые, >1 — придавить);
      5. усиление gain: clip(n*gain, 0, 1) (gain>1 — высветлить с насыщением);
      6. обратное масштабирование в диапазон [lo_v, hi_v] — значения остаются в «сырых»
         единицах шкалы, поэтому downstream autoLevels сохраняет форму кривой.

    При gain=1, gamma=1, clip=(0,100) возвращается ровно базовая Z-шкала (short-circuit):
    нормировка/денормировка тождественна, форма не меняется — полная обратная совместимость."""
    a = np.asarray(arr, dtype=np.float32)
    t = _base_transform(a, mode).astype(np.float32, copy=False)

    clip_lo, clip_hi = float(clip[0]), float(clip[1])
    is_default = (float(gain) == DEFAULT_GAIN and float(gamma) == DEFAULT_GAMMA
                  and clip_lo == DEFAULT_CLIP[0] and clip_hi == DEFAULT_CLIP[1])
    if is_default or t.size == 0:
        return t

    # уровни отсечения по перцентилям (clip_lo<clip_hi гарантируем перестановкой)
    if clip_lo > clip_hi:
        clip_lo, clip_hi = clip_hi, clip_lo
    lo_v = float(np.percentile(t, clip_lo))
    hi_v = float(np.percentile(t, clip_hi))
    if not (hi_v > lo_v):
        # вырожденный диапазон (константа/нулевой clip) — вернуть зажатую базу без деления на 0
        return np.clip(t, lo_v, hi_v).astype(np.float32, copy=False)

    tc = np.clip(t, lo_v, hi_v)
    n = (tc - lo_v) / (hi_v - lo_v)               # [0,1]
    g = max(float(gamma), 1e-6)
    n = np.power(n, g, dtype=np.float32)          # гамма-кривая
    n = np.clip(n * float(gain), 0.0, 1.0)        # усиление с насыщением
    out = lo_v + n * (hi_v - lo_v)                # обратно в сырой диапазон шкалы
    return out.astype(np.float32, copy=False)


# Обесцвечивание базы выкл. по умолчанию (Задача 18 — подсветка выбранных пиков).
DEFAULT_DESAT = 0.0


def desaturate_rgba(colors, amount: float):
    """Понизить насыщенность RGBA-массива к серому (luma-mix) — для приглушения базы при
    подсветке выбранных пиков (Задача 18). amount∈[0,1]: 0 — без изменений, 1 — серый.
    colors: (...,4) float в [0..1]. Возвращает новый массив той же формы (alpha не трогаем)."""
    a = float(max(0.0, min(1.0, amount)))
    if a <= 0.0:
        return np.asarray(colors, dtype=np.float32)
    c = np.array(colors, dtype=np.float32, copy=True)
    rgb = c[..., :3]
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
    c[..., :3] = rgb * (1.0 - a) + lum * a
    return c
# Регулируемое усреднение спектра выкл. по умолчанию (Замечание IV-R4).
DEFAULT_SMOOTH = 0

# Задача #163: рукоятка «Сглаживание» — дискретный выбор режима вместо непрерывного радиуса.
# 0 = выкл, 1 = SMA (равные веса, smooth_counts), 2 = WMA (треугольные веса, сильнее к центру).
SMOOTH_MODE_OFF, SMOOTH_MODE_SMA, SMOOTH_MODE_WMA = 0, 1, 2
# Фиксированная ширина окна (2*r+1 = 5 каналов) для обоих активных режимов — раньше её задавал
# сам регулятор (0..15); теперь регулятор выбирает только алгоритм, ширина окна не варьируется.
SMOOTH_RADIUS = 2


def smooth_counts(arr, radius, axis: int = -1):
    """Скользящее среднее (box-фильтр) шириной 2*radius+1 вдоль оси axis с краевым
    дополнением 'edge'. Регулируемое усреднение спектра (Замечание IV-R4). radius<=0 —
    массив возвращается без изменений (short-circuit). Возвращает float32 той же формы."""
    r = int(radius)
    a = np.asarray(arr, dtype=np.float32)
    if r <= 0 or a.ndim == 0 or a.shape[axis] < 2:
        return a
    work = np.moveaxis(a, axis, -1).astype(np.float64, copy=False)
    L = work.shape[-1]
    r = min(r, L)
    k = 2 * r + 1
    pad = np.pad(work, [(0, 0)] * (work.ndim - 1) + [(r, r)], mode="edge")
    csum = np.cumsum(pad, axis=-1)
    zero = np.zeros(work.shape[:-1] + (1,), dtype=np.float64)
    csum = np.concatenate([zero, csum], axis=-1)
    win = (csum[..., k:] - csum[..., :-k]) / float(k)
    out = np.moveaxis(win, -1, axis)
    return np.ascontiguousarray(out, dtype=np.float32)


def weighted_moving_average(arr, radius, axis: int = -1):
    """Взвешенное скользящее среднее (Задача #163): треугольные веса radius+1-|смещение|,
    центр окна весит больше краёв (в отличие от smooth_counts — там веса равны). Ширина окна
    2*radius+1, краевое дополнение 'edge'. radius<=0 — массив без изменений. float32."""
    r = int(radius)
    a = np.asarray(arr, dtype=np.float32)
    if r <= 0 or a.ndim == 0 or a.shape[axis] < 2:
        return a
    work = np.moveaxis(a, axis, -1).astype(np.float64, copy=False)
    L = work.shape[-1]
    r = min(r, L)
    weights = (r + 1 - np.abs(np.arange(-r, r + 1))).astype(np.float64)
    pad = np.pad(work, [(0, 0)] * (work.ndim - 1) + [(r, r)], mode="edge")
    out = np.zeros_like(work)
    for i, w in enumerate(weights):
        out += w * pad[..., i:i + L]
    out /= weights.sum()
    return np.ascontiguousarray(np.moveaxis(out, -1, axis), dtype=np.float32)


def smooth_by_mode(arr, mode, axis: int = -1, radius: int = SMOOTH_RADIUS):
    """Применить сглаживание по коду режима рукоятки «Сглаживание» (Задача #163):
    SMOOTH_MODE_OFF -> без изменений, SMOOTH_MODE_SMA -> smooth_counts,
    SMOOTH_MODE_WMA -> weighted_moving_average. Неизвестный код трактуется как выкл."""
    if int(mode) == SMOOTH_MODE_SMA:
        return smooth_counts(arr, radius, axis=axis)
    if int(mode) == SMOOTH_MODE_WMA:
        return weighted_moving_average(arr, radius, axis=axis)
    return np.asarray(arr, dtype=np.float32)