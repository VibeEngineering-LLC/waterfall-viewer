from __future__ import annotations
import numpy as np

# Режимы Z-шкалы (контраст отображения counts). Ключ -> человекочитаемая подпись для UI.
Z_MODES = (("linear", "Линейная"), ("sqrt", "Корень √"), ("log", "Логарифм log10"))

def apply_z_scale(arr, mode: str):
    """Преобразовать массив отсчётов для отображения по выбранной Z-шкале. Возвращает float32.
    linear -> как есть; sqrt -> sqrt(max(x,0)); log -> log10(1+max(x,0)) (защита от <=0)."""
    a = np.asarray(arr, dtype=np.float32)
    if mode == "sqrt":
        return np.sqrt(np.maximum(a, 0.0))
    if mode == "log":
        return np.log10(1.0 + np.maximum(a, 0.0))
    return a  # linear и любой неизвестный режим
