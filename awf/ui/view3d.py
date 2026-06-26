from __future__ import annotations
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import GL_DEPTH_TEST, GL_BLEND, GL_ALPHA_TEST, GL_CULL_FACE
from PySide6 import QtCore, QtGui, QtWidgets
from awf.ui.zscale import (apply_z_scale, DEFAULT_GAIN, DEFAULT_GAMMA,
                           DEFAULT_CLIP, desaturate_rgba, smooth_counts)
from awf.ui.colormaps import get_colormap

# Оси секущих плоскостей и их цвета (RGB 0..1). По 2 плоскости (slot 0/1) на ось.
PLANE_AXES = ("time", "energy", "counts")
_AXIS_RGB = {
    "time":   (0.20, 0.85, 0.95),   # бирюзовый — плоскости перпендикулярно оси Времени (X)
    "energy": (0.95, 0.35, 0.85),   # пурпурный — перпендикулярно оси Энергии (Y)
    "counts": (0.95, 0.85, 0.25),   # жёлтый — горизонтальные плоскости уровня Отсчётов (Z)
}
_AXIS_LABEL = {"time": "Время (с)", "energy": "Энергия (кэВ)", "counts": "Отсчёты (выс.)"}
_PLANE_ALPHA = 0.20
_TICK_COLOR = (205, 205, 215)       # цвет подписей делений осей (Задача 14)

# Задача #41: кривую сечения (профиль поверхности на секущей плоскости) рисуем ПОВЕРХ
# непрозрачного рельефа с выключенным depth-тестом. Без этого профиль совпадает по высоте
# с поверхностью, тонет в её «коже» (z-fighting) — на плоскости видна только пустая рамка.
# Сплошной цвет оси (BLEND off), без alpha/cull — как 'opaque', но всегда сверху.
_PROFILE_GL = {GL_DEPTH_TEST: False, GL_BLEND: False,
               GL_ALPHA_TEST: False, GL_CULL_FACE: False}
# Задача #47: профиль на секущей плоскости — НЕ срез по одному индексу, а ПРОЕКЦИЯ суммы
# рельефа в объёме между парой плоскостей оси (_section_projection), вписанная в высоту плоскости.
# Задача #48: цвет профиля = цвет рамки/оси (как рама плоскости), не белый — проекция вынесена
# на высоту плоскости и рисуется поверх (см. _PROFILE_GL), потому уже не тонет в рельефе.
# Задача #46: направление источника света (время, канал, высота), нормированное; рельефное
# затенение по Ламберту — затемняет склоны, обращённые от света, подчёркивая объём пиков.
_LIGHT_DIR = (-0.507, -0.405, 0.760)
_SHADE_AMBIENT = 0.35   # фоновая засветка: даже полностью затенённый склон не уходит в 0


def _surface_shading(z_surface, intensity):
    """Множитель освещённости (nt,nc) рельефа по нормали из градиента высоты (Задача #46).
    intensity<=0 -> None (без затенения); иначе линейная смесь 1↔(ambient..1) по Ламберту."""
    if intensity <= 0.0 or z_surface is None:
        return None
    h = np.asarray(z_surface, dtype=np.float64)
    dz_t = np.gradient(h, axis=0)
    dz_c = np.gradient(h, axis=1)
    inv = 1.0 / np.sqrt(dz_t * dz_t + dz_c * dz_c + 1.0)   # 1/|нормали|, нормаль ~ (-dz_t,-dz_c,1)
    lx, ly, lz = _LIGHT_DIR
    lam = np.clip((-dz_t * lx - dz_c * ly + lz) * inv, 0.0, 1.0)
    shade = _SHADE_AMBIENT + (1.0 - _SHADE_AMBIENT) * lam
    return ((1.0 - intensity) + intensity * shade).astype(np.float32)


def _fmt_count(v: float) -> str:
    """Подпись деления оси счёта: целое для крупных значений, дробное для малых cps (Задача #44)."""
    a = abs(float(v))
    if a >= 10.0:
        return f"{v:.0f}"
    if a >= 1.0:
        return f"{v:.1f}"
    if a > 0.0:
        return f"{v:.2f}"
    return "0"

_ENERGY_TICK_RGBA = (0.62, 0.69, 0.45, 0.95)  # оливковые отрезки-зубцы шкалы энергий (IV-R2)


class Waterfall3DView(gl.GLViewWidget):
    """3D-поверхность спектрограммы. Наследует GLViewWidget => вращение ЛКМ, зум колесом,
    панорама СКМ уже работают. set_spectrogram(sg) строит/заменяет поверхность.

    Задача 13: по 2 подвижные секущие плоскости на каждую ось (Время/Энергия/Отсчёты) с
    профилем сечения. Позиция плоскости задаётся долей frac∈[0,1] оси и пересчитывается в
    реальные единицы (с/кэВ) через центры LOD-бинов downsample (t_centers/ch_centers) —
    поэтому корректна при любой прорежке.

    Задача 14: подписи делений трёх осей реальными единицами — время (с) и энергия (кэВ)
    из центров LOD-бинов, ось счёта (Z) — через эмпирическую монотонную проекцию реальных
    отсчётов в дисплейную высоту рельефа (учитывает текущую Z-шкалу/контраст).

    Задача 15: вертикальные лучи-маркеры на выбранных энергиях нуклидов — от опорной
    плоскости до вершины энергетического столбца рельефа; энергия -> индекс канала через
    ch_centers (LOD-aware), цвет — цвет нуклида из NuclidePanel.

    Задача 18: режим подсветки — база десатурируется (luma-mix), столбцы выбранных энергий
    остаются насыщенными; список подсветки = те же выбранные линии нуклидов.

    Замечание IV-R2: подписи шкалы энергий вынесены на дальнее по времени ребро (X=xmax) и
    снабжены вертикальными отрезками-зубцами (как шкала частот в iZotope Insight).

    Замечание IV-R3: при включённых ОБЕИХ секущих плоскостях оси показывается только объём
    между ними — поверхность обрезается до окна (по индексам для время/энергия, по высоте
    через alpha для счёта); снаружи ничего не рисуется.

    Замечание IV-R4: регулируемое усреднение (скользящее среднее) спектра по энергетической
    оси перед Z-шкалой; радиус задаётся set_smoothing()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(pg.mkColor(15, 15, 20))
        self.setCameraPosition(distance=300, elevation=35, azimuth=-60)
        self._surface = None          # текущий GLSurfacePlotItem (или None)
        self._sg = None               # последняя спектрограмма (для смены Z-шкалы)
        self._z_mode = "log"          # текущая Z-шкала рельефа/цвета
        self._gain = DEFAULT_GAIN      # регулировка контраста (Задача 16)
        self._gamma = DEFAULT_GAMMA
        self._clip = DEFAULT_CLIP
        self._cmap_name = "insight"   # палитра рельефа/цвета (Задача 17)
        self._smooth = 0              # радиус усреднения спектра по энергии (Замечание IV-R4)
        self._unit = "counts"         # единицы рельефа/цвета: counts | cps (Задача #44)
        self._light = 0.0             # интенсивность рельефного затенения 0..1 (Задача #46)
        self._max_time = 400          # параметры LOD-прорежки последнего рендера
        self._max_chan = 512
        self._grid = gl.GLGridItem()  # опорная сетка под поверхностью
        self._grid.setColor(pg.mkColor(60, 60, 70))
        self.addItem(self._grid)

        # --- геометрия последнего рендера (для позиционирования плоскостей) ---
        self._nt = 0
        self._nc = 0
        self._height_scale = 1.0
        self._z_surface = None        # (nt, nc) высоты рельефа (дисплейные)
        self._z_counts = None         # (nt, nc) исходные counts бинов (для оси счёта, Задача 14)
        self._colors_full = None      # (nt, nc, 4) полный RGBA рельефа (до обрезки, IV-R3)
        self._clip_sig = None         # сигнатура текущих окон обрезки (IV-R3)
        self._t_centers = None        # (nt,) реальное время бинов, с
        self._ch_centers = None       # (nc,) реальная энергия бинов, кэВ

        # --- подписи делений осей (Задача 14) ---
        self._axis_items = []          # текущие GLTextItem/GLLinePlotItem (удаляются при перестроении)
        self._axis_labels_visible = True

        # --- лучи энергий нуклидов (Задача 15) ---
        self._energy_lines = []        # list[(energy_keV, color_str, label)]
        self._ray_items = []           # текущие GLLinePlotItem лучей

        # --- подсветка выбранных пиков (Задача 18) ---
        self._highlight_on = False     # режим подсветки: база приглушена, выбранные столбцы ярки
        self._base_desat = 0.65        # сила обесцвечивания базы (luma-mix) при подсветке
        self._hl_halfwidth = 1         # полуширина подсвечиваемой полосы каналов (±каналы)

        # --- секущие плоскости: (axis, slot) -> dict(mesh, line, frac, visible) ---
        self._planes = {}
        for axis in PLANE_AXES:
            for slot in (0, 1):
                # Задача #37: плоскость сечения больше НЕ заливается полупрозрачным
                # цветом — заливка перекрывала данные водопада за ней. mesh оставлен
                # как невидимый держатель состояния (drawFaces/drawEdges=False — ничего
                # не рисует, диагонального ребра quad'а тоже нет); видимый контур сечения
                # рисует отдельный border — замкнутый прямоугольник по 4 углам.
                mesh = gl.GLMeshItem(smooth=False, drawEdges=False, drawFaces=False,
                                     glOptions="translucent")
                mesh.setVisible(False)
                self.addItem(mesh)
                border = gl.GLLinePlotItem(mode="line_strip", antialias=True, width=2.0)
                border.setVisible(False)
                self.addItem(border)
                line = gl.GLLinePlotItem(mode="line_strip", antialias=True, width=2.5)
                line.setGLOptions(_PROFILE_GL)   # Задача #41: профиль всегда поверх рельефа
                # Задача #49: поверхность пересоздаётся каждый рендер (removeItem+addItem) и
                # потому рисуется ПОСЛЕДНЕЙ, перекрывая линию (depth-тест у линии выключен, но
                # решает порядок отрисовки). Больший depthValue => линия сортируется после
                # поверхности и видна снаружи плоскостей, в т.ч. на дальней.
                line.setDepthValue(10)
                line.setVisible(False)
                self.addItem(line)
                self._planes[(axis, slot)] = {
                    "mesh": mesh, "border": border, "line": line,
                    "frac": 0.5, "visible": False}

    def set_spectrogram(self, sg, max_time: int = 400, max_chan: int = 512) -> None:
        """Прорядить через sg.downsample(method='max') и построить цветную поверхность.
        Геометрия в индексном пространстве (X=индекс времени, Y=индекс канала), высота Z и цвет —
        по counts. Реальные единицы (с / кэВ) подписываем делениями осей (Задача 14)."""
        self._sg = sg
        self._max_time, self._max_chan = max_time, max_chan
        # 1) LOD-прорежка; t_centers/ch_centers — реальные с/кэВ для центров бинов. В режиме cps
        #    (Задача #44) прорежаем матрицу скорости (counts/live_time по срезу), а не сами отсчёты.
        src = sg.counts_in_unit(self._unit)
        z_counts, t_centers, ch_centers = sg.downsample(max_time, max_chan, method="max", data=src)
        z_counts = np.asarray(z_counts, dtype=np.float32)
        # 1b) регулируемое усреднение спектра по энергетической оси (axis=1) — Замечание IV-R4
        z_counts = smooth_counts(z_counts, self._smooth, axis=1)
        nt, nc = z_counts.shape
        # 2) Z-шкала контраста, затем нормировка для высоты и цвета (защита от нулевого максимума)
        z_disp = apply_z_scale(z_counts, self._z_mode, gain=self._gain,
                               gamma=self._gamma, clip=self._clip)
        zmax = float(z_disp.max()) if z_disp.size else 0.0
        zn = z_disp / zmax if zmax > 0 else z_disp
        # 3) геометрия: X,Y — индексы; высота Z — рельеф (нормированные counts, масштаб ~ четверть
        #    большей стороны).
        height_scale = 0.25 * float(max(nt, nc, 1))
        z_surface = (zn * height_scale).astype(np.float32)
        # 4) цвет по нормированной интенсивности (палитра Задачи 17); форма (nt, nc, 4)
        cmap = get_colormap(self._cmap_name)
        colors = cmap.map(zn, mode="float").astype(np.float32)  # (nt, nc, 4), RGBA в [0..1]
        # 4b) режим подсветки (Задача 18): базу обесцвечиваем, столбцы выбранных энергий —
        #     оставляем насыщенными. Если ни одна выбранная энергия не попала в диапазон —
        #     базу НЕ глушим (нечего выделять).
        if self._highlight_on and self._energy_lines:
            mask = self._highlight_mask(ch_centers, nc)        # (nc,) bool
            if mask.any():
                desat = desaturate_rgba(colors, self._base_desat)   # (nt, nc, 4)
                colors = np.where(mask[None, :, None], colors, desat).astype(np.float32)
        # 4c) рельефное затенение (Задача #46): RGB×множитель освещённости по нормали; alpha не трогаем
        shade = _surface_shading(z_surface, self._light)
        if shade is not None:
            colors[..., :3] *= shade[..., None]
            np.clip(colors, 0.0, 1.0, out=colors)
        # 5) запомнить полную геометрию и цвет; поверхность строит _rebuild_surface, при
        #    включённых обеих плоскостях оси показывая ТОЛЬКО объём между ними (IV-R3).
        self._nt, self._nc = nt, nc
        self._height_scale = height_scale
        self._z_surface = z_surface
        self._z_counts = z_counts
        self._colors_full = colors
        self._t_centers = np.asarray(t_centers, dtype=np.float64)
        self._ch_centers = np.asarray(ch_centers, dtype=np.float64)
        self._rebuild_surface()
        # 6) сетку — под поверхностью, размер по большей стороне; камера — отдалить под размер
        span = float(max(nt, nc, 10))
        self._grid.setSize(x=span * 1.2, y=span * 1.2)
        self._grid.setSpacing(x=max(1.0, span / 10.0), y=max(1.0, span / 10.0))
        self.setCameraPosition(distance=span * 1.6)
        # 7) переразместить активные секущие плоскости/подписи/лучи
        self._refresh_all_planes()
        self._rebuild_axis_labels()
        self._rebuild_energy_rays()

    def _clip_windows(self):
        """Окна обрезки поверхности по осям; активны ТОЛЬКО когда видимы ОБЕ плоскости оси
        (Замечание IV-R3). Возвращает (i0,i1,j0,j1,z_lo,z_hi,counts_active) в дисплейных
        индексах/высотах. Неактивная пара -> окно = весь диапазон оси."""
        nt, nc = self._nt, self._nc
        i0, i1 = 0, max(0, nt - 1)
        j0, j1 = 0, max(0, nc - 1)
        z_lo, z_hi = -np.inf, np.inf
        counts_active = False
        tv = self._planes[("time", 0)], self._planes[("time", 1)]
        if tv[0]["visible"] and tv[1]["visible"]:
            a = self._frac_to_index(self._t_centers, tv[0]["frac"])
            b = self._frac_to_index(self._t_centers, tv[1]["frac"])
            i0, i1 = min(a, b), max(a, b)
        ev = self._planes[("energy", 0)], self._planes[("energy", 1)]
        if ev[0]["visible"] and ev[1]["visible"]:
            a = self._frac_to_index(self._ch_centers, ev[0]["frac"])
            b = self._frac_to_index(self._ch_centers, ev[1]["frac"])
            j0, j1 = min(a, b), max(a, b)
        cv = self._planes[("counts", 0)], self._planes[("counts", 1)]
        if cv[0]["visible"] and cv[1]["visible"]:
            H = self._height_scale
            a = max(0.0, min(1.0, cv[0]["frac"])) * H
            b = max(0.0, min(1.0, cv[1]["frac"])) * H
            z_lo, z_hi = min(a, b), max(a, b)
            counts_active = True
        return i0, i1, j0, j1, z_lo, z_hi, counts_active

    def _rebuild_surface(self) -> None:
        """(Пере)построить поверхность из полных z/colors. При включённых ОБЕИХ плоскостях оси
        показывается только объём между ними (Замечание IV-R3): время/энергия — обрезка окна
        индексов; счёт — обнуление alpha вне высотного слоя + translucent."""
        if self._z_surface is None or self._colors_full is None:
            return
        nt, nc = self._nt, self._nc
        i0, i1, j0, j1, z_lo, z_hi, counts_active = self._clip_windows()
        self._clip_sig = (i0, i1, j0, j1, z_lo, z_hi, counts_active)
        xs = np.arange(i0, i1 + 1, dtype=np.float32)
        ys = np.arange(j0, j1 + 1, dtype=np.float32)
        zsub = np.ascontiguousarray(self._z_surface[i0:i1 + 1, j0:j1 + 1], dtype=np.float32)
        csub = np.array(self._colors_full[i0:i1 + 1, j0:j1 + 1, :], dtype=np.float32, copy=True)
        if counts_active:
            outside = (zsub < z_lo) | (zsub > z_hi)
            csub[..., 3] = np.where(outside, 0.0, csub[..., 3])
        colors_flat = csub.reshape(-1, 4)
        if self._surface is not None:
            self.removeItem(self._surface)
            self._surface = None
        surf = gl.GLSurfacePlotItem(x=xs, y=ys, z=zsub, colors=colors_flat,
                                    shader=None, computeNormals=False, smooth=False)
        surf.setGLOptions("translucent" if counts_active else "opaque")
        surf.translate(-nt / 2.0, -nc / 2.0, 0.0)
        self.addItem(surf)
        self._surface = surf

    def _maybe_reclip(self) -> None:
        """Пересобрать поверхность, только если сигнатура окон обрезки изменилась (IV-R3)."""
        if self._z_surface is None:
            return
        i0, i1, j0, j1, z_lo, z_hi, ca = self._clip_windows()
        if (i0, i1, j0, j1, z_lo, z_hi, ca) != self._clip_sig:
            self._rebuild_surface()

    def set_smoothing(self, radius: int) -> None:
        """Радиус скользящего среднего по энергии (Замечание IV-R4); ре-рендер из той же sg."""
        self._smooth = max(0, int(radius))
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def set_unit_mode(self, mode: str) -> None:
        """Единицы рельефа/цвета: 'counts' | 'cps' (Задача #44); ре-рендер из той же sg."""
        self._unit = "cps" if mode == "cps" else "counts"
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def set_light_intensity(self, value: float) -> None:
        """Интенсивность рельефного затенения 0..1 (Задача #46): 0 — плоская заливка по палитре,
        1 — полные тени по нормали. Ре-рендер из той же sg."""
        self._light = max(0.0, min(1.0, float(value)))
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def set_z_scale(self, mode: str) -> None:
        """Сменить Z-шкалу рельефа/цвета и перестроить поверхность из той же спектрограммы."""
        self._z_mode = mode
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def set_contrast(self, *, gain: float = None, gamma: float = None,
                     clip=None) -> None:
        """Обновить регулировку контраста (Задача 16) и перестроить поверхность.
        Не переданные параметры сохраняют текущее значение."""
        if gain is not None:
            self._gain = float(gain)
        if gamma is not None:
            self._gamma = float(gamma)
        if clip is not None:
            self._clip = (float(clip[0]), float(clip[1]))
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def set_colormap(self, name: str) -> None:
        """Сменить палитру (Задача 17) и перестроить поверхность из той же спектрограммы."""
        self._cmap_name = name
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def clear_surface(self) -> None:
        """Убрать текущую поверхность (например, перед загрузкой нового файла)."""
        if self._surface is not None:
            self.removeItem(self._surface)
            self._surface = None

    # ---------- градуировка осей (Задача 14) ----------
    @staticmethod
    def _nice_ticks(lo: float, hi: float, n_target: int = 5):
        """Адаптивные «круглые» деления в [lo, hi] (шаг 1/2/5×10^k), ~n_target штук."""
        lo = float(lo); hi = float(hi)
        if not (hi > lo):
            return np.array([], dtype=float)
        raw = (hi - lo) / max(1, n_target)
        mag = 10.0 ** np.floor(np.log10(raw))
        norm = raw / mag
        step = (1.0 if norm < 1.5 else 2.0 if norm < 3.0 else 5.0 if norm < 7.0 else 10.0) * mag
        start = np.ceil(lo / step) * step
        ticks = np.arange(start, hi + 0.5 * step, step)
        return ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]

    def _clear_axis_items(self) -> None:
        for it in self._axis_items:
            self.removeItem(it)
        self._axis_items = []

    def _add_text(self, pos, text: str, font) -> None:
        item = gl.GLTextItem(pos=np.asarray(pos, dtype=float), text=text,
                             color=_TICK_COLOR, font=font)
        self.addItem(item)
        self._axis_items.append(item)

    def _add_line(self, p0, p1, color) -> None:
        """Короткий GL-отрезок (зубец шкалы); хранится в _axis_items для совместной очистки."""
        pts = np.array([p0, p1], dtype=np.float32)
        item = gl.GLLinePlotItem(pos=pts, color=color, width=1.6,
                                 mode="line_strip", antialias=True)
        self.addItem(item)
        self._axis_items.append(item)

    def set_axis_labels_visible(self, visible: bool) -> None:
        """Показать/скрыть подписи делений осей (Задача 14)."""
        self._axis_labels_visible = bool(visible)
        self._rebuild_axis_labels()

    def _rebuild_axis_labels(self) -> None:
        """Перестроить подписи делений трёх осей под текущую геометрию/Z-шкалу."""
        self._clear_axis_items()
        if not self._axis_labels_visible or self._z_surface is None:
            return
        nt, nc = self._nt, self._nc
        if nt == 0 or nc == 0:
            return
        xmin, xmax, ymin, ymax, zmax = self._axis_extent()
        span = float(max(nt, nc, 10))
        pad = 0.05 * span
        font = QtGui.QFont("Helvetica", 10)
        title_font = QtGui.QFont("Helvetica", 11, QtGui.QFont.Bold)
        # ось времени (X) — деления в секундах вдоль переднего ребра Y=ymin, Z=0
        tc = self._t_centers
        if tc is not None and len(tc) >= 2 and tc[-1] > tc[0]:
            idx = np.arange(nt, dtype=float)
            for tv in self._nice_ticks(float(tc[0]), float(tc[-1])):
                wx = float(np.interp(tv, tc, idx)) - nt / 2.0
                self._add_text((wx, ymin - pad, 0.0), f"{tv:.0f}", font)
            self._add_text((xmax + pad, ymin - pad, 0.0), "t, с", title_font)
        # ось энергии (Y) — деления в кэВ на ДАЛЬНЕМ по времени ребре X=xmax, с вертикальными
        # отрезками-зубцами вверх по Z (как шкала частот в iZotope Insight) — Замечание IV-R2
        cc = self._ch_centers
        if cc is not None and len(cc) >= 2 and cc[-1] > cc[0]:
            idx = np.arange(nc, dtype=float)
            tooth = 0.10 * zmax if zmax > 0 else 1.0
            for ev in self._nice_ticks(float(cc[0]), float(cc[-1])):
                wy = float(np.interp(ev, cc, idx)) - nc / 2.0
                self._add_text((xmax + pad, wy, 0.0), f"{ev:.0f}", font)
                self._add_line((xmax, wy, 0.0), (xmax, wy, tooth), _ENERGY_TICK_RGBA)
            self._add_text((xmax + pad, ymax + pad, 0.0), "E, кэВ", title_font)
        # ось счёта (Z) — деления реальных отсчётов вдоль вертикали угла (xmin,ymin);
        # высоту берём эмпирически из монотонной пары (counts->height), учитывает Z-шкалу/контраст
        if self._z_counts is not None and self._z_counts.size and zmax > 0:
            cflat = np.asarray(self._z_counts, dtype=float).ravel()
            hflat = np.asarray(self._z_surface, dtype=float).ravel()
            order = np.argsort(cflat)
            cs = cflat[order]; hs = hflat[order]
            cmax = float(cs[-1])
            if cmax > 0:
                ztitle = "N, отсч/с" if self._unit == "cps" else "N, отсч."
                for cv in self._nice_ticks(0.0, cmax):
                    wz = float(np.interp(cv, cs, hs))
                    self._add_text((xmin - pad, ymin - pad, wz), _fmt_count(cv), font)
                self._add_text((xmin - pad, ymin - pad, zmax + pad), ztitle, title_font)

    # ---------- вертикальные лучи энергий (Задача 15) ----------
    def set_energy_lines(self, lines) -> None:
        """Задать энергии-маркеры (список (энергия_кэВ, цвет, метка)) и перестроить лучи.
        Источник — выбранные линии нуклидов из NuclidePanel.linesChanged. В режиме подсветки
        (Задача 18) меняется и раскраска поверхности, поэтому делаем полный ре-рендер."""
        self._energy_lines = list(lines) if lines else []
        if self._highlight_on and self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)
        else:
            self._rebuild_energy_rays()

    def _clear_ray_items(self) -> None:
        for it in self._ray_items:
            self.removeItem(it)
        self._ray_items = []

    def _rebuild_energy_rays(self) -> None:
        """Вертикальный луч на каждой энергии-маркере: от опорной плоскости (Z=0) до вершины
        столбца энергии в рельефе. Энергия -> индекс канала через ch_centers (LOD-aware)."""
        self._clear_ray_items()
        if self._z_surface is None or not self._energy_lines:
            return
        nc = self._nc
        cc = self._ch_centers
        if cc is None or len(cc) < 2 or nc == 0:
            return
        xmin, xmax, ymin, ymax, zmax = self._axis_extent()
        emin = float(cc[0]); emax = float(cc[-1])
        idx = np.arange(nc, dtype=float)
        stub = 0.05 * zmax if zmax > 0 else 1.0
        for energy, color, _label in self._energy_lines:
            e = float(energy)
            if e < emin or e > emax:
                continue   # энергия вне диапазона спектра — луч не рисуем
            jf = float(np.interp(e, cc, idx))
            py = jf - nc / 2.0
            jcol = int(min(nc - 1, max(0, round(jf))))
            h_top = float(self._z_surface[:, jcol].max())
            if h_top < stub:
                h_top = stub    # нет сигнала на энергии — короткий заметный штырь
            try:
                rgba = pg.mkColor(color).getRgbF()
            except Exception:
                rgba = (1.0, 1.0, 1.0, 1.0)
            pos = np.array([[xmin, py, 0.0], [xmin, py, h_top]], dtype=np.float32)
            ray = gl.GLLinePlotItem(pos=pos, color=rgba, width=2.5,
                                    mode="line_strip", antialias=True)
            self.addItem(ray)
            self._ray_items.append(ray)

    # ---------- подсветка выбранных пиков (Задача 18) ----------
    def _highlight_mask(self, centers, nc: int):
        """Булева маска (nc,) подсвечиваемых столбцов-каналов из выбранных энергий нуклидов.
        Энергия -> индекс канала через centers (LOD-aware), полоса шириной ±_hl_halfwidth."""
        mask = np.zeros(nc, dtype=bool)
        if centers is None or len(centers) < 2 or nc == 0 or not self._energy_lines:
            return mask
        emin = float(centers[0]); emax = float(centers[-1])
        idx = np.arange(nc, dtype=float)
        hw = self._hl_halfwidth
        for energy, _color, _label in self._energy_lines:
            e = float(energy)
            if e < emin or e > emax:
                continue
            j = int(round(float(np.interp(e, centers, idx))))
            lo = max(0, j - hw); hi = min(nc, j + hw + 1)
            mask[lo:hi] = True
        return mask

    def set_highlight_enabled(self, on: bool) -> None:
        """Режим подсветки выбранных пиков (Задача 18): база десатурируется, столбцы выбранных
        энергий остаются насыщенными. При выключении возвращается полноцветная база."""
        self._highlight_on = bool(on)
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    # ---------- секущие плоскости (Задача 13) ----------
    def _axis_extent(self):
        """Габариты сцены в центрированных мировых координатах: (Xmin,Xmax,Ymin,Ymax,Zmax)."""
        nt, nc, H = self._nt, self._nc, self._height_scale
        return (-nt / 2.0, nt / 2.0, -nc / 2.0, nc / 2.0, H)

    def _frac_to_index(self, centers, frac):
        """Доля frac∈[0,1] реального диапазона -> ближайший дисплейный индекс бина (LOD-aware)."""
        if centers is None or len(centers) == 0:
            return 0
        lo = float(centers[0]); hi = float(centers[-1])
        target = lo + max(0.0, min(1.0, frac)) * (hi - lo)
        return int(np.argmin(np.abs(np.asarray(centers) - target)))

    def plane_value(self, axis: str, frac: float):
        """Реальное значение позиции плоскости для подписи: (value, unit_label)."""
        frac = max(0.0, min(1.0, float(frac)))
        if axis == "time":
            if self._t_centers is None or len(self._t_centers) == 0:
                return (0.0, "с")
            i = self._frac_to_index(self._t_centers, frac)
            return (float(self._t_centers[i]), "с")
        if axis == "energy":
            if self._ch_centers is None or len(self._ch_centers) == 0:
                return (0.0, "кэВ")
            j = self._frac_to_index(self._ch_centers, frac)
            return (float(self._ch_centers[j]), "кэВ")
        # counts: доля высоты рельефа -> приблизительный уровень реальных отсчётов/скорости
        peak = 0.0
        if self._z_surface is not None and self._z_surface.size and self._height_scale > 0:
            # высота нормирована: frac высоты == frac пикового дисплейного уровня
            peak = float(self._sg.counts_in_unit(self._unit).max()) if self._sg is not None else 0.0
        unit = "отсч/с (≈)" if self._unit == "cps" else "отсч. (≈)"
        return (frac * peak, unit)

    def active_plane_values(self):
        """Реальные значения видимых секущих плоскостей по слотам (Задачи #38/#39):
        {axis: [val_or_None, val_or_None]} в с / кэВ / отсч.; скрытая плоскость -> None.
        Источник синхронизации дока срезов и 2D-карты с сечениями 3D."""
        out = {}
        for axis in PLANE_AXES:
            vals = []
            for slot in (0, 1):
                entry = self._planes[(axis, slot)]
                if entry["visible"]:
                    value, _ = self.plane_value(axis, entry["frac"])
                    vals.append(value)
                else:
                    vals.append(None)
            out[axis] = vals
        return out

    def set_plane(self, axis: str, slot: int, frac: float, visible: bool) -> None:
        """Поставить плоскость (axis, slot) на долю frac оси и показать/скрыть её вместе с профилем.
        При включении/перемещении пары плоскостей оси пересобираем обрезку поверхности (IV-R3)."""
        key = (axis, slot)
        if key not in self._planes:
            return
        entry = self._planes[key]
        entry["frac"] = max(0.0, min(1.0, float(frac)))
        entry["visible"] = bool(visible)
        # Задача #47: окно проекции зависит от ОБЕИХ плоскостей оси — обновляем профили обоих
        # слотов (сумма между плоскостями меняется при движении/включении любой из пары).
        self._apply_plane(axis, 0)
        self._apply_plane(axis, 1)
        self._maybe_reclip()

    def _refresh_all_planes(self) -> None:
        for axis in PLANE_AXES:
            for slot in (0, 1):
                self._apply_plane(axis, slot)

    def _section_projection(self, axis: str):
        """Профиль на секущей плоскости (Задача #47): проекция-сумма рельефа в объёме между
        парой плоскостей оси, вписанная в высоту плоскости [0, zmax]. Окно суммирования — как у
        обрезки (IV-R3): обе плоскости оси видимы -> между ними, иначе вся ось. time -> сумма по
        времени (кривая по энергии, nc); energy -> сумма по энергии (кривая по времени, nt).

        Задача #50: суммируем СЫРЫЕ counts (z_counts), затем применяем Z-шкалу к ИНТЕГРАЛУ —
        log(Σ), а НЕ Σ log(дисплейных высот). Иначе суммирование уже log-сжатых высот
        переоценивает широкие полки и теряет узкие/транзиентные пики (рельеф по method='max'
        показывает такой пик башней, а кривая суммы — нет: пик «не считается»)."""
        i0, i1, j0, j1, _zlo, _zhi, _ca = self._clip_windows()
        if axis == "time":
            s = self._z_counts[i0:i1 + 1, :].sum(axis=0).astype(np.float64)   # интеграл по времени
        else:
            s = self._z_counts[:, j0:j1 + 1].sum(axis=1).astype(np.float64)   # интеграл по энергии
        d = apply_z_scale(s, self._z_mode, gain=self._gain, gamma=self._gamma, clip=self._clip)
        d = np.asarray(d, dtype=np.float64)
        m = float(d.max()) if d.size else 0.0
        if m > 0.0:
            d = d / m * self._height_scale          # вписать проекцию в высоту плоскости
        return d.astype(np.float32)

    def _apply_plane(self, axis: str, slot: int) -> None:
        entry = self._planes[(axis, slot)]
        mesh, border, line = entry["mesh"], entry["border"], entry["line"]
        if not entry["visible"] or self._z_surface is None:
            mesh.setVisible(False)
            border.setVisible(False)
            line.setVisible(False)
            return
        frac = entry["frac"]
        xmin, xmax, ymin, ymax, zmax = self._axis_extent()
        r, g, b = _AXIS_RGB[axis]
        nt, nc = self._nt, self._nc

        if axis == "time":
            i = self._frac_to_index(self._t_centers, frac)
            px = float(i) - nt / 2.0
            verts = np.array([[px, ymin, 0.0], [px, ymax, 0.0],
                              [px, ymax, zmax], [px, ymin, zmax]], dtype=np.float32)
            prof = np.column_stack([                       # Задача #47: проекция суммы по времени
                np.full(nc, px, dtype=np.float32),
                np.arange(nc, dtype=np.float32) - nc / 2.0,
                self._section_projection("time")])
        elif axis == "energy":
            j = self._frac_to_index(self._ch_centers, frac)
            py = float(j) - nc / 2.0
            verts = np.array([[xmin, py, 0.0], [xmax, py, 0.0],
                              [xmax, py, zmax], [xmin, py, zmax]], dtype=np.float32)
            prof = np.column_stack([                       # Задача #47: проекция суммы по энергии
                np.arange(nt, dtype=np.float32) - nt / 2.0,
                np.full(nt, py, dtype=np.float32),
                self._section_projection("energy")])
        else:  # counts — горизонтальная плоскость уровня Z; профиль-контур не строим
            pz = frac * zmax
            verts = np.array([[xmin, ymin, pz], [xmax, ymin, pz],
                              [xmax, ymax, pz], [xmin, ymax, pz]], dtype=np.float32)
            prof = None

        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh.setMeshData(vertexes=verts, faces=faces,
                         color=(r, g, b, _PLANE_ALPHA))
        mesh.setVisible(True)
        # Задача #37: видимый контур сечения — замкнутый прямоугольник по 4 углам
        # quad'а (без диагонали и без заливки). Цвет оси, непрозрачный.
        loop = np.vstack([verts, verts[0:1]]).astype(np.float32)
        border.setData(pos=loop, color=(r, g, b, 1.0), width=2.0)
        border.setVisible(True)
        if prof is not None:
            # Задачи #47/#48: на плоскости — проекция суммы между плоскостями (высота prof.z из
            # _section_projection), цвет = цвет рамки/оси (r,g,b); поверх рельефа (см. _PROFILE_GL).
            line.setData(pos=prof, color=(r, g, b, 1.0), width=3.0)
            line.setVisible(True)
        else:
            line.setVisible(False)


class SectionControls(QtWidgets.QWidget):
    """Док «Сечения»: по 2 ряда на каждую ось (Время/Энергия/Отсчёты) — чекбокс видимости,
    слайдер позиции (0..1000 -> доля 0..1) и подпись реального значения. Излучает planeChanged."""

    # axis: str, slot: int, frac: float, visible: bool
    planeChanged = QtCore.Signal(str, int, float, bool)

    _SLIDER_MAX = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = {}  # (axis, slot) -> dict(check, slider, label)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Секущие плоскости (2 на ось)"))
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)
        row = 0
        for axis in PLANE_AXES:
            grid.addWidget(QtWidgets.QLabel(_AXIS_LABEL[axis]), row, 0, 1, 4)
            row += 1
            for slot in (0, 1):
                check = QtWidgets.QCheckBox(f"#{slot + 1}")
                slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                slider.setRange(0, self._SLIDER_MAX)
                slider.setValue(int(self._SLIDER_MAX * (0.33 if slot == 0 else 0.66)))
                vlabel = QtWidgets.QLabel("—")
                vlabel.setMinimumWidth(90)
                grid.addWidget(check, row, 0)
                grid.addWidget(slider, row, 1)
                grid.addWidget(vlabel, row, 2)
                self._rows[(axis, slot)] = {
                    "check": check, "slider": slider, "label": vlabel}
                check.toggled.connect(lambda _=False, a=axis, s=slot: self._emit(a, s))
                slider.valueChanged.connect(lambda _=0, a=axis, s=slot: self._emit(a, s))
                row += 1
        layout.addStretch(1)

    def _frac(self, axis: str, slot: int) -> float:
        return self._rows[(axis, slot)]["slider"].value() / float(self._SLIDER_MAX)

    def _emit(self, axis: str, slot: int) -> None:
        r = self._rows[(axis, slot)]
        self.planeChanged.emit(axis, slot, self._frac(axis, slot), r["check"].isChecked())

    def set_value_label(self, axis: str, slot: int, text: str) -> None:
        self._rows[(axis, slot)]["label"].setText(text)

    def emit_all(self) -> None:
        """Переиспустить planeChanged для всех рядов (после загрузки файла — обновить подписи/позиции)."""
        for axis in PLANE_AXES:
            for slot in (0, 1):
                self._emit(axis, slot)