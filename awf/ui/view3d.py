from __future__ import annotations
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from awf.ui.zscale import apply_z_scale

class Waterfall3DView(gl.GLViewWidget):
    """3D-поверхность спектрограммы. Наследует GLViewWidget => вращение ЛКМ, зум колесом,
    панорама СКМ уже работают. set_spectrogram(sg) строит/заменяет поверхность."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(pg.mkColor(15, 15, 20))
        self.setCameraPosition(distance=300, elevation=35, azimuth=-60)
        self._surface = None          # текущий GLSurfacePlotItem (или None)
        self._sg = None               # последняя спектрограмма (для смены Z-шкалы)
        self._z_mode = "log"          # текущая Z-шкала рельефа/цвета
        self._max_time = 400          # параметры LOD-прорежки последнего рендера
        self._max_chan = 512
        self._grid = gl.GLGridItem()  # опорная сетка под поверхностью
        self._grid.setColor(pg.mkColor(60, 60, 70))
        self.addItem(self._grid)

    def set_spectrogram(self, sg, max_time: int = 400, max_chan: int = 512) -> None:
        """Прорядить через sg.downsample(method='max') и построить цветную поверхность.
        Геометрия в индексном пространстве (X=индекс времени, Y=индекс канала), высота Z и цвет —
        по counts. Реальные единицы (с / кэВ) тут НЕ подписываем (это делает 2D-панель)."""
        self._sg = sg
        self._max_time, self._max_chan = max_time, max_chan
        # 1) LOD-прорежка
        z_counts, t_centers, ch_centers = sg.downsample(max_time, max_chan, method="max")
        z_counts = np.asarray(z_counts, dtype=np.float32)
        nt, nc = z_counts.shape
        # 2) Z-шкала контраста, затем нормировка для высоты и цвета (защита от нулевого максимума)
        z_disp = apply_z_scale(z_counts, self._z_mode)
        zmax = float(z_disp.max()) if z_disp.size else 0.0
        zn = z_disp / zmax if zmax > 0 else z_disp
        # 3) геометрия: X,Y — индексы (растянуты в сопоставимый по осям размер),
        #    высота Z — рельеф (нормированные counts, масштаб ~ четверть большей стороны).
        x = np.arange(nt, dtype=np.float32)
        y = np.arange(nc, dtype=np.float32)
        height_scale = 0.25 * float(max(nt, nc, 1))
        z_surface = (zn * height_scale).astype(np.float32)
        # 4) цвет по нормированной интенсивности (colormap 'inferno'); форма (nt, nc, 4)
        cmap = pg.colormap.get("inferno")
        colors = cmap.map(zn, mode="float").astype(np.float32)  # (nt, nc, 4), RGBA в [0..1]
        # MeshData индексирует цвета ПЛОСКИМ индексом вершины (k = i*nc+j, C-order), поэтому
        # колор-массив обязан быть (nt*nc, 4), а не (nt, nc, 4) — иначе IndexError при отрисовке.
        colors = colors.reshape(nt * nc, 4)
        # 5) пересоздать поверхность
        if self._surface is not None:
            self.removeItem(self._surface)
            self._surface = None
        surf = gl.GLSurfacePlotItem(x=x, y=y, z=z_surface, colors=colors,
                                    shader=None, computeNormals=False, smooth=False)
        # центрируем поверхность в начало координат сдвигом
        surf.translate(-nt / 2.0, -nc / 2.0, 0.0)
        self.addItem(surf)
        self._surface = surf
        # 6) сетку — под поверхностью, размер по большей стороне; камера — отдалить под размер
        span = float(max(nt, nc, 10))
        self._grid.setSize(x=span * 1.2, y=span * 1.2)
        self._grid.setSpacing(x=max(1.0, span / 10.0), y=max(1.0, span / 10.0))
        self.setCameraPosition(distance=span * 1.6)

    def set_z_scale(self, mode: str) -> None:
        """Сменить Z-шкалу рельефа/цвета и перестроить поверхность из той же спектрограммы."""
        self._z_mode = mode
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def clear_surface(self) -> None:
        """Убрать текущую поверхность (например, перед загрузкой нового файла)."""
        if self._surface is not None:
            self.removeItem(self._surface)
            self._surface = None
