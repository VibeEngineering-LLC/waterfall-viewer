import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pyqtgraph as pg
from awf.ui.colormaps import (
    get_colormap, insight_colormap, COLORMAPS, COLORMAP_NAMES,
)


def test_registry_shape():
    assert COLORMAP_NAMES[0] == "insight"
    assert len(COLORMAPS) == 15                              # #102: Insight + 14 палитр со скриншота
    assert COLORMAPS[0][:2] == ("insight", "iZotope Insight")
    assert all(len(entry) == 3 for entry in COLORMAPS)       # (ключ, подпись, описание)


def test_registry_has_screenshot_palettes():
    for key in ("inferno", "magma", "plasma", "viridis", "cividis", "turbo", "jet",
                "hot", "ocean", "parula", "cubehelix", "Spectral", "cool", "gray"):
        assert key in COLORMAP_NAMES                         # #102: все палитры макета на месте


def test_added_palettes_resolve_distinct_from_insight():
    # #102: каждая добавленная палитра даёт НЕ-insight карту (нет молчаливого fallback на Insight)
    samples = np.array([0.1, 0.5, 0.9], dtype=float)
    b = insight_colormap().map(samples, mode="float")
    for name in ("magma", "plasma", "cividis", "turbo", "jet", "hot",
                 "ocean", "parula", "cubehelix", "Spectral", "cool", "gray"):
        a = get_colormap(name).map(samples, mode="float")
        assert not np.allclose(a, b), name


def test_get_colormap_each_name():
    for name in COLORMAP_NAMES:
        cm = get_colormap(name)
        assert isinstance(cm, pg.ColorMap)


def test_insight_stops():
    cm = insight_colormap()
    pos = cm.pos
    assert len(pos) == 5
    assert np.isclose(pos[0], 0.0) and np.isclose(pos[-1], 1.0)


def test_insight_low_darker_than_high():
    cm = get_colormap("insight")
    vals = cm.map(np.array([0.0, 1.0], dtype=float), mode="float")
    low_rgb = vals[0, :3].sum()
    high_rgb = vals[1, :3].sum()
    assert high_rgb > low_rgb  # пик ярче базы


def test_unknown_name_falls_back_to_insight():
    cm = get_colormap("nonsense-xyz")
    ins = insight_colormap()
    a = cm.map(np.array([0.0, 0.5, 1.0]), mode="float")
    b = ins.map(np.array([0.0, 0.5, 1.0]), mode="float")
    assert np.allclose(a, b)


def test_inferno_distinct_from_insight():
    inf = get_colormap("inferno").map(np.array([0.5]), mode="float")
    ins = get_colormap("insight").map(np.array([0.5]), mode="float")
    assert not np.allclose(inf, ins)


def test_builtin_palettes_independent_of_matplotlib():
    # #122: jet/hot/ocean/cubehelix/Spectral/cool/gray НЕ должны вырождаться в Insight, даже
    # если matplotlib недоступен в рантайме (он не объявлен в requirements). Блокируем импорт.
    import builtins, sys
    real_import = builtins.__import__
    def block(name, *a, **k):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib blocked (#122 regression)")
        return real_import(name, *a, **k)
    saved = {m: sys.modules[m] for m in list(sys.modules) if m.split(".")[0] == "matplotlib"}
    for mod in saved:
        del sys.modules[mod]
    builtins.__import__ = block
    try:
        ins = insight_colormap().getLookupTable(0.0, 1.0, 32, alpha=False)
        for name in ("jet", "hot", "ocean", "cubehelix", "Spectral", "cool", "gray"):
            lut = get_colormap(name).getLookupTable(0.0, 1.0, 32, alpha=False)
            assert not np.array_equal(lut, ins), name
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


def test_builtin_palette_endpoints():
    # #122: точные эндпоинты встроенных линейных палитр (2-точечные карты).
    cool = get_colormap("cool").getLookupTable(0.0, 1.0, 2, alpha=False)
    assert tuple(cool[0]) == (0, 255, 255) and tuple(cool[-1]) == (255, 0, 255)
    gray = get_colormap("gray").getLookupTable(0.0, 1.0, 2, alpha=False)
    assert tuple(gray[0]) == (0, 0, 0) and tuple(gray[-1]) == (255, 255, 255)