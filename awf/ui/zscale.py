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
    log -> log10(1+max(x,0)). Любой неизвестный режим трактуется как linear."""
    nn = np.maximum(a, 0.0)
    if mode == "sqrt":
        return np.sqrt(nn)
    if mode == "log":
        return np.log10(1.0 + nn)
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