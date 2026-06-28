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

# Палитра Parula (matlab по умолчанию). В pyqtgraph/matplotlib отсутствует (лицензия MathWorks),
# поэтому задаём приближение контрольными точками: индиго -> синий -> бирюзовый -> жёлто-зелёный -> жёлтый.
_PARULA_POS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
_PARULA_COLORS = np.array([
    [62, 38, 168, 255],    # индиго — низ
    [33, 121, 228, 255],   # синий
    [33, 178, 159, 255],   # бирюзовый
    [160, 191, 76, 255],   # жёлто-зелёный
    [249, 251, 20, 255],   # жёлтый — пик
], dtype=np.ubyte)

# Доступные палитры для UI: (ключ, человекочитаемая подпись, краткое описание).
# Порядок = порядок строк в окне «Цветовая палитра» (#102). Insight — фирменная, дефолт приложения.
# Описания — со скриншота-макета оператора (2026-06-28). Ключи: insight/parula — кастомные ниже;
# inferno/magma/plasma/viridis/cividis/turbo берутся из pyqtgraph напрямую; jet/hot/ocean/cubehelix/
# Spectral/cool/gray — из источника matplotlib (см. get_colormap).
COLORMAPS = (
    ("insight",   "iZotope Insight", "фирменная: синяя база → чёрный → оранжевый"),
    ("inferno",   "Inferno",         "тёмная, контрастная"),
    ("magma",     "Magma",           "мягкая, фиолетово-розовая"),
    ("plasma",    "Plasma",          "фиолет → жёлтый, без чёрного"),
    ("viridis",   "Viridis",         "перцептивная, друг-к-другу слепых"),
    ("cividis",   "Cividis",         "для дальтоников, синь→жёлт"),
    ("turbo",     "Turbo",           "яркая, плавный синий→красн"),
    ("jet",       "Jet",             "классика matlab"),
    ("hot",       "Hot",             "чёрный→красн→жёлт→белый"),
    ("ocean",     "Ocean",           "чёрный→синий→белый, спокойная"),
    ("parula",    "Parula",          "matlab по умолчанию, синь→жёлт"),
    ("cubehelix", "Cubehelix",       "яркость растёт, ч/б-совместима"),
    ("Spectral",  "Spectral",        "диверг., синий→жёлт→красн"),
    ("cool",      "Cool",            "голубой→пурпурный, яркая"),
    ("gray",      "Grayscale",       "моно, для печати"),
)
COLORMAP_NAMES = tuple(key for key, _label, _desc in COLORMAPS)


def insight_colormap() -> pg.ColorMap:
    """Кастомная палитра в стиле iZotope Insight (оранжевый-на-чёрном + синяя база)."""
    return pg.ColorMap(pos=_INSIGHT_POS, color=_INSIGHT_COLORS)


def parula_colormap() -> pg.ColorMap:
    """Приближение matlab-палитры Parula (в pyqtgraph/matplotlib не входит из-за лицензии)."""
    return pg.ColorMap(pos=_PARULA_POS, color=_PARULA_COLORS)


def get_colormap(name: str) -> pg.ColorMap:
    """Палитра по имени. 'insight'/'parula' — кастомные; остальные берутся из pyqtgraph, при
    отсутствии там — из источника matplotlib (jet/hot/ocean/cubehelix/cool/gray/Spectral). Неизвестное
    имя или полное отсутствие палитры -> fallback на Insight (чтобы UI всегда что-то показал)."""
    if name == "insight":
        return insight_colormap()
    if name == "parula":
        return parula_colormap()
    for source in (None, "matplotlib"):
        try:
            cm = pg.colormap.get(name) if source is None else pg.colormap.get(name, source=source)
        except Exception:
            cm = None
        if cm is not None:
            return cm
    return insight_colormap()