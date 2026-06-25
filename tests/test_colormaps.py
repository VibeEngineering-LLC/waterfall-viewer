import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pyqtgraph as pg
from awf.ui.colormaps import (
    get_colormap, insight_colormap, COLORMAPS, COLORMAP_NAMES,
)


def test_registry_shape():
    assert COLORMAP_NAMES[0] == "insight"
    assert len(COLORMAPS) == 3
    assert dict(COLORMAPS)["insight"] == "iZotope Insight"


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