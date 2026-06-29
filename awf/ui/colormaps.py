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

# Задача #122: встроенные контрольные точки палитр (равномерно по [0,1]), снятые с эталонных LUT
# matplotlib (getLookupTable) 2026-06-29. Раньше jet/hot/ocean/cubehelix/Spectral/cool/gray
# резолвились ТОЛЬКО через pg.colormap.get(name, source="matplotlib"); если matplotlib в рантайме
# недоступен (он НЕ объявлен в requirements.txt), все семь молча вырождались в Insight — оператор
# видел одинаковые чёрно-оранжевые превью (дефект #122). Теперь заданы явно и от matplotlib НЕ зависят.
_BUILTIN_COLORS: dict[str, list[list[int]]] = {
    "jet": [[0, 0, 128], [0, 0, 224], [0, 42, 255], [0, 127, 255], [0, 212, 255], [55, 255, 191],
            [123, 255, 123], [192, 255, 54], [255, 229, 0], [255, 151, 0], [255, 72, 0], [224, 0, 0],
            [128, 0, 0]],
    "hot": [[11, 0, 0], [94, 0, 0], [178, 0, 0], [255, 6, 0], [255, 90, 0], [255, 173, 0],
            [255, 255, 3], [255, 255, 129], [255, 255, 255]],
    "ocean": [[0, 128, 0], [0, 95, 21], [0, 64, 42], [0, 31, 64], [0, 0, 85], [0, 31, 106],
              [0, 64, 127], [0, 95, 149], [0, 128, 170], [64, 159, 191], [127, 191, 212],
              [191, 223, 234], [255, 255, 255]],
    "cubehelix": [[0, 0, 0], [24, 16, 40], [24, 46, 72], [21, 83, 75], [43, 111, 57], [96, 122, 47],
                  [159, 121, 73], [202, 123, 132], [212, 144, 198], [199, 178, 236], [194, 216, 242],
                  [216, 241, 239], [255, 255, 255]],
    "Spectral": [[158, 1, 66], [203, 51, 76], [233, 93, 71], [248, 141, 81], [253, 190, 110],
                 [254, 229, 147], [255, 255, 191], [234, 246, 158], [190, 229, 160], [136, 207, 164],
                 [84, 174, 172], [57, 126, 184], [94, 79, 162]],
    "cool": [[0, 255, 255], [255, 0, 255]],
    "gray": [[0, 0, 0], [255, 255, 255]],
}

# Доступные палитры для UI: (ключ, человекочитаемая подпись, краткое описание).
# Порядок = порядок строк в окне «Цветовая палитра» (#102). Insight — фирменная, дефолт приложения.
# Описания — со скриншота-макета оператора (2026-06-28). Ключи: insight/parula — кастомные ниже;
# inferno/magma/plasma/viridis/cividis/turbo берутся из pyqtgraph напрямую; jet/hot/ocean/cubehelix/
# Spectral/cool/gray — из встроенных контрольных точек _BUILTIN_COLORS (#122, без matplotlib).
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


def _evenly_spaced_colormap(colors: list[list[int]]) -> pg.ColorMap:
    """Задача #122: ColorMap из встроенных контрольных точек с равномерными позициями по [0,1]."""
    arr = np.array(colors, dtype=np.ubyte)
    pos = np.linspace(0.0, 1.0, len(arr), dtype=np.float64)
    return pg.ColorMap(pos=pos, color=arr)


def get_colormap(name: str) -> pg.ColorMap:
    """Палитра по имени. 'insight'/'parula' — кастомные; jet/hot/ocean/cubehelix/Spectral/cool/gray —
    из встроенных контрольных точек _BUILTIN_COLORS (#122, без зависимости от matplotlib);
    inferno/magma/plasma/viridis/cividis/turbo — нативные в pyqtgraph. Неизвестное имя или полное
    отсутствие палитры -> fallback на Insight (чтобы UI всегда что-то показал)."""
    if name == "insight":
        return insight_colormap()
    if name == "parula":
        return parula_colormap()
    if name in _BUILTIN_COLORS:
        return _evenly_spaced_colormap(_BUILTIN_COLORS[name])
    for source in (None, "matplotlib"):
        try:
            cm = pg.colormap.get(name) if source is None else pg.colormap.get(name, source=source)
        except Exception:
            cm = None
        if cm is not None:
            return cm
    return insight_colormap()