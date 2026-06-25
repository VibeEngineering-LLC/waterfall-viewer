from __future__ import annotations
import numpy as np
import pyqtgraph as pg

# Палитра iZotope Insight: синяя база -> чёрный провал -> оранжевый -> жёлто-белый пик
# (ТЗ-A.5). Позиции в [0,1]: 0 = минимальная интенсивность, 1 = пик.
_INSIGHT_POS = np.array([0.0, 0.18, 0.45, 0.72, 1.0], dtype=np.float64)
_INSIGHT_COLORS = np.array([
    [12, 18, 56, 255],     # тёмно-синий — база слабого сигнала
    [0, 0, 0, 255],        # чёрный — провал контраста
    [196, 88, 12, 255],    # оранжевый
    [255, 168, 32, 255],   # ярко-оранжевый
    [255, 246, 206, 255],  # жёлто-белый — пик
], dtype=np.ubyte)

# Доступные палитры для переключателя UI: ключ -> человекочитаемая подпись.
COLORMAPS = (("insight", "iZotope Insight"), ("inferno", "Inferno"), ("viridis", "Viridis"))
COLORMAP_NAMES = tuple(key for key, _ in COLORMAPS)


def insight_colormap() -> pg.ColorMap:
    """Кастомная палитра в стиле iZotope Insight (оранжевый-на-чёрном + синяя база)."""
    return pg.ColorMap(pos=_INSIGHT_POS, color=_INSIGHT_COLORS)


def get_colormap(name: str) -> pg.ColorMap:
    """Палитра по имени. 'insight' — кастомная; прочие — из pyqtgraph (inferno/viridis/…).
    Неизвестное имя или отсутствие палитры в pyqtgraph -> fallback на Insight."""
    if name == "insight":
        return insight_colormap()
    try:
        cm = pg.colormap.get(name)
    except Exception:
        cm = None
    return cm if cm is not None else insight_colormap()