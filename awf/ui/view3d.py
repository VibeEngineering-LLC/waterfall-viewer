from __future__ import annotations
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import GL_DEPTH_TEST, GL_BLEND, GL_ALPHA_TEST, GL_CULL_FACE
from PySide6 import QtCore, QtGui, QtWidgets
from awf.ui.zscale import (apply_z_scale, DEFAULT_GAIN, DEFAULT_GAMMA,
                           DEFAULT_CLIP, desaturate_rgba, smooth_by_mode,
                           SMOOTH_MODE_SMA, SMOOTH_MODE_WMA)
from awf.ui.colormaps import get_colormap
from awf.ui.i18n import tr                 # Задача #169: локализация панели сечений
from awf.model.background import background_window_like   # Задача #140: сырое фоновое окно простыни
from awf.ui.knobs import Knob          # Задача #59: панель сечений в том же knob-стиле
from awf.analysis.peaks import (            # Задача #110/#114/#112/#120: поиск фотопиков на 3D-водопаде
    find_peaks, peak_time_mask,
    peak_presence_mask,                    # Задача #160: сумма окна пика + Currie-гейт по бинам
    default_fwhm_model, fwhm_channels_from_model,
    auto_calibrate_fwhm_model,             # Задача #120: автокалибровка FWHM(E) под детектор
    find_transient_peaks,                  # Задача #113: транзиентные (время-локализованные) пики
)

# Оси секущих плоскостей и их цвета (RGB 0..1). По 2 плоскости (slot 0/1) на ось.
PLANE_AXES = ("time", "energy", "counts")
_AXIS_RGB = {
    "time":   (0.20, 0.85, 0.95),   # бирюзовый — плоскости перпендикулярно оси Времени (X)
    "energy": (0.95, 0.35, 0.85),   # пурпурный — перпендикулярно оси Энергии (Y)
    "counts": (0.95, 0.85, 0.25),   # жёлтый — горизонтальные плоскости уровня Отсчётов (Z)
}
_AXIS_LABEL = {"time": "Время (с)", "energy": "Энергия (кэВ)", "counts": "Отсчёты (выс.)"}
_PLANE_ALPHA = 0.20
# Задача #61: грани секущих плоскостей (контур-рамка border) приглушены — полупрозрачные,
# «не такие яркие». Цвет оси сохраняется, понижается только alpha (rgb×alpha-блендинг).
_BORDER_ALPHA = 0.30
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


def _floor_shift_linear(z_disp):
    """Задача #168 (итер.3): поканальный floor-сдвиг `max(x - floor_col, 0)`. Глубина «иглы»
    0-count бина = локальный уровень «1 отсчёт» (ε-фактор #156, ×10.76 за 1408 кэВ) — зависит
    от энергии, поэтому глобальный скаляр итер.2 (percentile-10 = 0.36 < 0.43 уровня ВЭ-полосы
    на реальном файле) иглы не срезал. floor_col = min положительного в колонке (= локальный
    квант 1 отсчёта в «дырявых» колонках) с потолком cap = max кванта по колонкам с 0-бинами:
    низкоэнергетический континуум (min >> квант) не проседает, его форма сохраняется. 0 и
    1 отсчёт садятся на Z=0 при любой энергии, настоящие пики (≥2 отсчётов) поднимаются.
    Пустые временные строки (хвост файла) не считаются «дырками». Sqrt/log не трогаем:
    у log свой floor из #167, sqrt сжимает динамику."""
    if z_disp.ndim != 2 or not np.any(z_disp > 0.0):
        return z_disp
    zz = z_disp[(z_disp > 0.0).any(axis=1)]            # только строки-времена с сигналом
    masked = np.where(zz > 0.0, zz, np.inf)
    min_pos = masked.min(axis=0)                       # (nc,) min положительного колонки
    holey = (zz <= 0.0).any(axis=0) & np.isfinite(min_pos)
    fin = np.isfinite(min_pos)
    cap = float(min_pos[holey].max()) if holey.any() else float(min_pos[fin].min())
    floor_col = np.where(fin, np.minimum(min_pos, cap), 0.0)
    return np.maximum(z_disp - floor_col[None, :], 0.0).astype(z_disp.dtype, copy=False)


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

# Задача #68/#63: координатная сетка — линии на делениях шкал, поле обрамлено пустой клеткой.
_GRID_RGBA = (0.28, 0.28, 0.33, 0.55)         # тонкие линии сетки на делениях шкал
_GRID_BORDER_RGBA = (0.46, 0.47, 0.54, 0.9)   # рамка поля (на 1 клетку шире данных) — ярче сетки
# Задача #64: единицы оси времени и их множитель к секундам (выбор сек/мин/часы в тулбаре).
_TIME_UNIT_SCALE = {"с": 1.0, "мин": 60.0, "ч": 3600.0}
# Задача #71/#72: предпочтительные «человеческие» шаги делений осей. Время — 15 минут (для длинных
# водопадов), энергия — 200 кэВ. Применяются, если на диапазоне выходит разумное число делений;
# иначе (короткая запись / узкий диапазон) — откат на авто-деления `_nice_ticks`.
_TIME_STEP_15MIN_S = 900.0     # 15 мин в секундах (#71)
_ENERGY_STEP_KEV = 200.0       # 200 кэВ (#72)
_PREF_STEP_MIN_TICKS = 3       # меньше — диапазон мал, идём на _nice_ticks
_PREF_STEP_MAX_TICKS = 40      # больше — деления слишком частые, идём на _nice_ticks
# Задача #76: «подложка» = плоское дно рельефа (ячейки с нормированной высотой zn≈0, в палитре —
# фиолетовый минимум). При отключении подложки такие ячейки делаем прозрачными (alpha=0): остаётся
# только рельеф-«всплески», сплошной фиолетовый прямоугольник базы исчезает (виден фон).
_FLOOR_FRAC = 0.02             # доля высоты рельефа, ниже которой ячейка считается «дном»
# Задача #78: верхний предел энергии для вывода и сетки 3D-водопада. Каналы с энергией выше
# отсекаются ПОСЛЕ LOD-прорежки — и рельеф, и деления оси энергии обрезаются согласованно.
_MAX_ENERGY_KEV = 3000.0
# Задача #153: нижний предел энергии ПОИСКА ПИКОВ (рельеф/сетка не обрезаются). Ниже ~50 кэВ —
# рентгеновский/шумовой край сцинтиллятора: «пики» 23.6/37.1 кэВ на реальном файле — не гамма-линии.
_MIN_ENERGY_KEV = 50.0

# Задача #145: цвета «однотонного»/«каркасного» стиля простыней; конвенция окна срезов:
# образец — голубой, фон — оранжевый.
_SHEET_RGB_SMP = (0.45, 0.65, 0.95)
_SHEET_RGB_BG = (1.0, 0.55, 0.15)
# Задача #140: критуровень Currie для простыни (#135, _BG_SHEET_K) отменён решением оператора
# «простыня фона и образец должны строиться одинаково» — простыня строится как рельеф, без надбавок.
# Задача #110: выделение фотопиков на 3D-водопаде. Каждый пик — зелёная линия по гребню рельефа
# вдоль оси времени (или её части, #112) на энергии пика (на ПОЛЕ спектрограммы, не у ребра);
# occlusion #95 прячет линию за высоким рельефом.
# Задача #114: ширина фильтра теперь per-channel из default_fwhm_model() (R=7%@662 кэВ, √-закон).
# PEAK_FWHM_CHANNELS=8.0 оставлена мёртвой константой для совместимости тестов #110 (check importer).
PEAK_FWHM_CHANNELS = 8.0                    # мёртвая константа — #114 использует FWHM(E)-модель
_PEAK_SIGMA_DEFAULT = 3.0                    # Задача #114: дефолтный порог значимости σ Currie L_C
_PEAK_RAY_RGBA = (0.20, 0.86, 0.20, 1.0)    # зелёный, как маркеры пиков #108
# Задача #124: подсветка выбранного пика (клик по строке в панели «Найденные пики») —
# яркая малиновая линия, заведомо отличимая и от зелёных хребтов, и от палитр рельефа.
_PEAK_HILITE_RGBA = (1.0, 0.20, 0.90, 1.0)
# Задача #113: транзиентный (оконный) скан жёстче интегрального. На окне срезов
# меньше статистики -> выше пьедестал шума Currie; без запаса набрались бы ложные.
# Порог окна = self._peak_sigma + маржа; привязка к _peak_sigma ОБЯЗАТЕЛЬНА для
# монотонности (больше σ -> не больше пиков). Маржа откалибрована на реальном файле
# (869 срезов × 8192 кан, scripts/task113_transient_check.py): пик ~186 кэВ в окне
# [684:756] имеет оконную значимость ~41.5σ, при дефолте σ=3.0 (порог окна 6.0)
# находится с большим запасом, добавленных транзиентов немного (6 на этом файле).
_TRANSIENT_SIGMA_MARGIN = 3.0


# Задача #174: карта режимов сглаживания по t (7 ступеней: 0=выкл, 1-3=SMA, 4-6=WMA)
_T_SMOOTH_MAP = {
    1: (SMOOTH_MODE_SMA, 2), 2: (SMOOTH_MODE_SMA, 4), 3: (SMOOTH_MODE_SMA, 8),
    4: (SMOOTH_MODE_WMA, 2), 5: (SMOOTH_MODE_WMA, 4), 6: (SMOOTH_MODE_WMA, 8),
}


def _extrap_lin(block, r_out, direction):
    n = block.shape[0]
    if n < 2:
        edge = block[0] if direction < 0 else block[-1]
        return np.broadcast_to(edge, (r_out, block.shape[1])).copy()
    x = np.arange(n, dtype=np.float64)
    xm = x.mean()
    xc = x - xm
    denom = float((xc ** 2).sum())
    bm = block.mean(axis=0)
    slope = (xc[:, None] * (block - bm)).sum(axis=0) / denom
    intercept = bm - slope * xm
    if direction > 0:
        xe = np.arange(n, n + r_out, dtype=np.float64)
    else:
        xe = np.arange(-r_out, 0, dtype=np.float64)
    return np.clip(slope[None, :] * xe[:, None] + intercept[None, :], 0.0, None)


def _smooth_by_segs(arr, mode, radius, t_centers, seg_bounds_sec):
    t = np.asarray(t_centers, dtype=np.float64)
    out = arr.copy()
    segs = np.asarray(seg_bounds_sec, dtype=np.float64)
    if len(segs) == 0:
        return smooth_by_mode(arr, mode, axis=0, radius=radius)
    assignment = np.zeros(len(t), dtype=np.int64)
    for bi in range(len(t)):
        tc = t[bi]
        dists = np.maximum(0.0, tc - segs[:, 1]) + np.maximum(0.0, segs[:, 0] - tc)
        assignment[bi] = int(np.argmin(dists))
    for si in range(len(segs)):
        idx = np.where(assignment == si)[0]
        n = len(idx)
        if n < 2:
            continue
        seg = arr[idx]
        r = min(2 * radius, n - 1)
        fw = n
        left_pad = _extrap_lin(seg[:fw], r, -1)
        right_pad = _extrap_lin(seg[-fw:], r, +1)
        padded = np.concatenate([left_pad, seg, right_pad], axis=0)
        smoothed_padded = smooth_by_mode(padded, mode, axis=0, radius=radius)
        out[idx] = smoothed_padded[r: r + n]
    return out


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
        self.setCameraPosition(distance=300, elevation=30, azimuth=-45)  # #205: 0,0 наверху (center задаётся при загрузке)
        self._surface = None          # текущий GLSurfacePlotItem (или None)
        self._surface_on = True       # Задача #143: тумблер видимости рельефа-«простыни образца»
        self._surface_style = "palette"  # Задача #145: стиль простыни образца (palette|solid|wire)
        self._bg_style = "palette"       # Задача #145: стиль простыни фона (palette|solid|wire)
        self._sg = None               # последняя спектрограмма (для смены Z-шкалы)
        self._sg_analysis = None      # Задача #158: источник для пиков/ID (без ε-нормализации)
        self._z_mode = "log"          # текущая Z-шкала рельефа/цвета
        self._gain = DEFAULT_GAIN      # регулировка контраста (Задача 16)
        self._gamma = DEFAULT_GAMMA
        self._clip = DEFAULT_CLIP
        self._cmap_name = "insight"   # палитра рельефа/цвета (Задача 17)
        self._smooth = 0              # радиус усреднения спектра по энергии (Замечание IV-R4)
        self._t_smooth = 0            # Задача #172: режим сглаживания по оси времени (0/SMA/WMA)
        self._t_smooth_by_seg = False  # применять по сегментам независимо
        self._seg_bounds_sec = []     # [(t_start_s, t_end_s), …] текущих сегментов
        self._unit = "cps"            # единицы рельефа/цвета: counts | cps (Задача #44; дефолт cps — #53)
        self._light = 0.0             # интенсивность рельефного затенения 0..1 (Задача #46)
        self._max_time = 400          # параметры LOD-прорежки последнего рендера
        self._max_chan = 512
        # Задача #68: прежняя равномерная GLGridItem заменена координатной сеткой на делениях
        # шкал (_rebuild_grid). Объект оставлен скрытым, чтобы не плодить ссылок; не рисуется.
        self._grid = gl.GLGridItem()
        self._grid.setColor(pg.mkColor(60, 60, 70))
        self._grid.setVisible(False)
        self.addItem(self._grid)
        self._grid_items = []         # линии координатной сетки/рамки (Задача #63/#68)
        self._time_unit = "с"         # единицы оси времени: с | мин | ч (Задача #64)
        self._floor_visible = False   # Задача #76: видна ли «подложка»; #150: дефолт — скрыта
        # Задача #98: «фоновая простыня» — полупрозрачная поверхность на высоте рельефа, отвечающей
        # фону bg(энергия), постоянная во времени; показывает уровень фона над/под водопадом.
        self._bg_cps_full = None      # поканальный фон cps (полное разрешение), None — нет фона
        self._bg_sheet_on = False     # видна ли простыня (пункт «Наложение фона», #96/#98)
        self._bg_sheet = None         # GLSurfacePlotItem простыни (или None)
        self._bg_sheet_raw = None     # Задача #140: (counts_блок, lt_блок) сырого фонового окна | None

        # --- геометрия последнего рендера (для позиционирования плоскостей) ---
        self._nt = 0
        self._nc = 0
        self._height_scale = 1.0
        self._zmax_sample = 0.0       # zmax образца до floor-сдвига (Задача P-3, reuse в _add_bg_sheet)
        self._z_surface = None        # (nt, nc) высоты рельефа (дисплейные)
        self._z_counts = None         # (nt, nc) исходные counts бинов (max-LOD, для оси счёта/рельефа)
        self._z_counts_sum = None     # (nt, nc) sum-LOD: интеграл counts в бине (Задача #52 — спектр окна)
        self._z_counts_int = None     # (nt, nc) sum-LOD СЫРЫХ counts (Задача #152 — пуассонова маска гребней)
        self._colors_full = None      # (nt, nc, 4) полный RGBA рельефа (до обрезки, IV-R3)
        self._clip_sig = None         # сигнатура текущих окон обрезки (IV-R3)
        self._t_centers = None        # (nt,) реальное время бинов, с
        self._ch_centers = None       # (nc,) реальная энергия бинов, кэВ

        # --- подписи делений осей (Задача 14) ---
        self._axis_items = []          # текущие GLTextItem/GLLinePlotItem (удаляются при перестроении)
        self._axis_labels_visible = True
        self._label_side_sig = None    # Задача #77: (sx,sy) — на каком крае оси стоят подписи (по взгляду)

        # --- лучи энергий нуклидов (Задача 15) ---
        self._energy_lines = []        # list[(energy_keV, color_str, label)]
        self._ray_items = []           # текущие GLLinePlotItem лучей
        self._plane_nuclide_items = [] # маркеры нуклидов на гранях плоскостей Времени (Задача #67)

        # --- выделение найденных фотопиков на рельефе (Задача #110/#114/#112) ---
        self._peaks_on = False         # включён ли поиск пиков (зелёные хребты на 3D)
        self._peak_ridge_items = []    # GLLinePlotItem линий-хребтов по гребню рельефа на пиках
        self._peak_sigma = _PEAK_SIGMA_DEFAULT   # Задача #114: порог значимости σ; сеттер = set_peak_sigma
        # Задача #124: пер-пиковая видимость и подсветка по энергии-ключу (round(E,3)).
        # Ключ по энергии связывает строку панели и гребень без хрупкой привязки к индексу:
        # _found_peaks() детерминирован при фикс. σ/данных, потому энергии стабильны между
        # вызовами «заполнить панель» и «перестроить гребни».
        self._peak_hidden_keys = set()   # энергии-ключи пиков со снятым чекбоксом (скрыты)
        self._peak_highlight_key = None  # энергия-ключ подсвеченного пика (или None)

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
                # Задача #61: профиль/проекция на плоскости убраны — элемент остаётся
                # держателем (всегда скрыт), GL-настройки ниже сохранены на случай возврата.
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

    def set_spectrogram(self, sg, max_time: int | None = None,
                        max_chan: int | None = None, analysis_sg=None) -> None:
        """Прорядить через sg.downsample(method='max') и построить цветную поверхность.
        Геометрия в индексном пространстве (X=индекс времени, Y=индекс канала), высота Z и цвет —
        по counts. Реальные единицы (с / кэВ) подписываем делениями осей (Задача 14).
        Задача #158: analysis_sg — аналитический источник (вычет фона БЕЗ ε-нормализации #156,
        та же форма/калибровка, что sg) для поиска пиков и пуассоновой маски #152."""
        # Задача #92: отличить загрузку НОВОГО спектра от ре-рендера того же (рукоятки Регулировок,
        # смена палитры/Z-шкалы/единиц передают self._sg). Идентичность объекта — надёжный признак:
        # сеттеры ре-рендера зовут set_spectrogram(self._sg, ...), новый файл — другой объект.
        is_new = sg is not self._sg
        self._sg = sg
        # Задача #158: ре-рендеры (analysis_sg=None, sg is self._sg) сохраняют прежний
        # аналитический источник; новый объект без analysis_sg — сам sg (прямые вызовы/тесты).
        if analysis_sg is not None:
            self._sg_analysis = analysis_sg
        elif is_new or self._sg_analysis is None:
            self._sg_analysis = sg
        # Задача #56: None -> сохранить текущие max_time/max_chan (ширина выборки по времени из
        # рукоятки переживает загрузку файла, как _gain/_smooth/_light); число -> установить новое.
        if max_time is not None:
            self._max_time = int(max_time)
        if max_chan is not None:
            self._max_chan = int(max_chan)
        max_time, max_chan = self._max_time, self._max_chan
        # 1) LOD-прорежка; t_centers/ch_centers — реальные с/кэВ для центров бинов. В режиме cps
        #    (Задача #44) прорежаем матрицу скорости (counts/live_time по срезу), а не сами отсчёты.
        src = sg.counts_in_unit(self._unit)
        z_counts, t_centers, ch_centers = sg.downsample(max_time, max_chan, method="max", data=src)
        z_counts = np.asarray(z_counts, dtype=np.float32)
        # Задача #52: профиль на плоскости = спектр верхнего-правого окна (sum по окну, НЕ max).
        # Отдельная sum-LOD на ТЕХ ЖЕ бинах: интеграл counts в каждом (бин времени × бин канала).
        # Сумма по окну времени == sg.sum_spectrum, прорежённому к nc каналам — точно как окно.
        z_sum, _ts, _cs = sg.downsample(max_time, max_chan, method="sum", data=src)
        self._z_counts_sum = np.asarray(z_sum, dtype=np.float64)
        # Задача #152: сырые counts (sum-LOD, без cps-нормировки и сглаживания) — для
        # пуассоновой маски присутствия гребней (σ≈√N осмысленна только в counts).
        # Задача #158: из аналитического источника (без ε-нормализации #156): умножение на
        # ε_ref/ε(E) ломает пуассонову статистику (var растёт как f², а маска считает var=N).
        asg = self._sg_analysis if self._sg_analysis is not None else sg
        z_int, _ti, _ci = asg.downsample(max_time, max_chan, method="sum")
        z_int = np.asarray(z_int, dtype=np.float64)
        # Задача #78: ограничить вывод и сетку 3D по энергии _MAX_ENERGY_KEV (3000 кэВ). Каналы
        # выше порога отсекаем здесь, на прорежённых данных: ch_centers задаёт и рельеф, и деления
        # оси энергии, поэтому срез прefix'а (калибровка канал→энергия монотонно возрастает) обрезает
        # данные и градуировку согласованно. Если файл и так в пределах порога — срез не нужен.
        ch_centers = np.asarray(ch_centers, dtype=np.float64)
        keep = int(np.count_nonzero(ch_centers <= _MAX_ENERGY_KEV))
        if 0 < keep < ch_centers.size:
            z_counts = z_counts[:, :keep]
            self._z_counts_sum = self._z_counts_sum[:, :keep]
            z_int = z_int[:, :keep]
            ch_centers = ch_centers[:keep]
        self._z_counts_int = z_int
        # #192: подавление одиночных временных игл (артефакт LOD method="max" — одиночный
        # граничный срез с аномальным счётом захватывается как max всего LOD-бина и стоит
        # иглой над соседними бинами). Трёхточечная медиана по оси t: если бин > 6×median
        # (prev, self, next) → замена медианой. Два смежных ненулевых бина = настоящий
        # источник, median = высокое → не подавляется. Граничные строки (строка 0: prev=self,
        # строка nt-1: next=self) при равенстве median==self → никогда не подавляются.
        if z_counts.shape[0] >= 3:
            _pr = np.concatenate([z_counts[:1], z_counts[:-1]], axis=0)
            _nx = np.concatenate([z_counts[1:], z_counts[-1:]], axis=0)
            _med3 = np.median(np.stack([_pr, z_counts, _nx], axis=0), axis=0)
            _spike = z_counts > (6.0 * np.maximum(_med3, 1e-10))
            if _spike.any():
                z_counts = np.where(_spike, _med3, z_counts).astype(np.float32)
        # #193: подавление одиночных энергетических игл на правом краю спектра. Аналог #192
        # но по оси E (axis=1). Порог 10× консервативнее 6× по t: реальные фотопики NaI
        # span ≥6 LOD-бинов по E → median высокий → не давится. Одиночный граничный канал
        # с аномальным счётом → median(left, spike, 0) = left → подавляется.
        if z_counts.shape[1] >= 3:
            _epr = np.concatenate([z_counts[:, :1], z_counts[:, :-1]], axis=1)
            _enx = np.concatenate([z_counts[:, 1:], z_counts[:, -1:]], axis=1)
            _emed3 = np.median(np.stack([_epr, z_counts, _enx], axis=0), axis=0)
            _espike = z_counts > (10.0 * np.maximum(_emed3, 1e-10))
            if _espike.any():
                z_counts = np.where(_espike, _emed3, z_counts).astype(np.float32)
        # 1b) сглаживание спектра по энергетической оси (axis=1) — Замечание IV-R4 / #163
        z_counts = smooth_by_mode(z_counts, self._smooth, axis=1)
        # 1c) сглаживание по оси времени (axis=0) — Задача #172/#174
        if self._t_smooth:
            _ts = _T_SMOOTH_MAP.get(self._t_smooth)
            if _ts:
                _algo, _radius = _ts
                if self._t_smooth_by_seg and self._seg_bounds_sec:
                    z_counts = _smooth_by_segs(z_counts, _algo, _radius, t_centers, self._seg_bounds_sec)
                else:
                    z_counts = smooth_by_mode(z_counts, _algo, axis=0, radius=_radius)
        nt, nc = z_counts.shape
        # 2) Z-шкала контраста, затем нормировка для высоты и цвета (защита от нулевого максимума)
        z_disp = apply_z_scale(z_counts, self._z_mode, gain=self._gain,
                               gamma=self._gamma, clip=self._clip)
        # Задача P-3: запомнить zmax образца ДО floor-сдвига — тот же расчёт, что делал
        # _add_bg_sheet повторным вызовом apply_z_scale(self._z_counts, ...) только ради max().
        self._zmax_sample = float(z_disp.max()) if z_disp.size else 0.0
        # Задача #168 (итер.3): поканальный floor-сдвиг для лин режима — обрезаем «отрицательные
        # пики» (провалы 0-count бинов на фоне поднятой ε-нормировкой #156 простыни). floor_col =
        # локальный уровень «1 отсчёт» колонки (с потолком по «дырявым» колонкам); 0 и 1 отсчёт
        # садятся ровно на Z=0 на любой энергии, пики поднимаются. Симметрично #167 log-floor.
        if self._z_mode == "linear":
            z_disp = _floor_shift_linear(z_disp)
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
        # 6) координатная сетка на делениях шкал (Задача #63/#68); камера — отдалить под размер.
        # Задача #92: кадрируем камеру ТОЛЬКО при загрузке нового спектра. Ре-рендеры от рукояток
        # Регулировок (set_contrast/set_smoothing/set_light_intensity/set_time_bins/set_z_scale/
        # set_colormap/set_unit_mode передают тот же self._sg) сохраняют зум/панораму пользователя.
        if is_new:
            self._seg_bounds_sec = []  # #178-fix1a: старые сегменты не переносить на новый файл
            span = float(max(nt, nc, 10))
            # Задача #205: azimuth=45 → камера в (+X,+Y) квадранте → (0,0) = дальний = наверху экрана
            self.setCameraPosition(
                distance=span * 1.6,
                azimuth=45,
            )
        self._rebuild_grid()
        # 7) переразместить активные секущие плоскости/подписи/маркеры нуклидов на плоскостях
        self._refresh_all_planes()
        self._rebuild_axis_labels()
        self._rebuild_energy_rays()       # Задача #85: лишь снимает старые рёберные лучи
        self._rebuild_plane_nuclides()    # маркеры изотопов — только на секущих плоскостях
        self._rebuild_bg_sheet()          # Задача #98: «фоновая простыня» под текущий рельеф
        self._rebuild_peak_ridges()       # Задача #110: выделение фотопиков по гребню рельефа

    def _clip_windows(self):
        """Окна обрезки поверхности по осям. Задача #84: каждая ВИДИМАЯ плоскость режет
        односторонне от своего края оси до текущей позиции — слот 0 задаёт нижнюю границу
        (от минимума оси), слот 1 — верхнюю (от максимума); пройденная полоса скрывается.
        Видимы обе плоскости оси -> окно между ними (обратно совместимо с IV-R3). Возвращает
        (i0,i1,j0,j1,z_lo,z_hi,counts_active) в дисплейных индексах/высотах; невидимый слот ->
        его граница = край оси (весь диапазон)."""
        nt, nc = self._nt, self._nc
        i0, i1 = 0, max(0, nt - 1)
        j0, j1 = 0, max(0, nc - 1)
        z_lo, z_hi = -np.inf, np.inf
        counts_active = False
        # Задача #84: слот 0 -> нижняя граница (режет от минимума оси), слот 1 -> верхняя (от максимума)
        tv = self._planes[("time", 0)], self._planes[("time", 1)]
        if tv[0]["visible"]:
            i0 = self._frac_to_index(self._t_centers, tv[0]["frac"])
        if tv[1]["visible"]:
            i1 = self._frac_to_index(self._t_centers, tv[1]["frac"])
        ev = self._planes[("energy", 0)], self._planes[("energy", 1)]
        if ev[0]["visible"]:
            j0 = self._frac_to_index(self._ch_centers, ev[0]["frac"])
        if ev[1]["visible"]:
            j1 = self._frac_to_index(self._ch_centers, ev[1]["frac"])
        H = self._height_scale
        cv = self._planes[("counts", 0)], self._planes[("counts", 1)]
        if cv[0]["visible"]:
            z_lo = max(0.0, min(1.0, cv[0]["frac"])) * H
            counts_active = True
        if cv[1]["visible"]:
            z_hi = max(0.0, min(1.0, cv[1]["frac"])) * H
            counts_active = True
        return i0, i1, j0, j1, z_lo, z_hi, counts_active

    def _rebuild_surface(self) -> None:
        """(Пере)построить поверхность из полных z/colors. При включённых ОБЕИХ плоскостях оси
        показывается только объём между ними (Замечание IV-R3): время/энергия — обрезка окна
        индексов; счёт — обнуление alpha вне высотного слоя + translucent."""
        if self._z_surface is None or self._colors_full is None:
            return
        # Задача #143: тумблер «Простыня образца» на тулбаре «Вид» скрывает рельеф.
        if not self._surface_on:
            if self._surface is not None:
                self.removeItem(self._surface)
                self._surface = None
            return
        nt, nc = self._nt, self._nc
        i0, i1, j0, j1, z_lo, z_hi, counts_active = self._clip_windows()
        self._clip_sig = (i0, i1, j0, j1, z_lo, z_hi, counts_active)
        # Задача #84: встречные плоскости пересеклись (срезы перекрыли друг друга) -> данных
        # в окне не остаётся; убираем поверхность, чтобы не строить пустой GLSurfacePlotItem.
        if i0 > i1 or j0 > j1:
            if self._surface is not None:
                self.removeItem(self._surface)
                self._surface = None
            return
        xs = np.arange(i0, i1 + 1, dtype=np.float32)
        ys = np.arange(j0, j1 + 1, dtype=np.float32)
        zsub = np.ascontiguousarray(self._z_surface[i0:i1 + 1, j0:j1 + 1], dtype=np.float32)
        csub = np.array(self._colors_full[i0:i1 + 1, j0:j1 + 1, :], dtype=np.float32, copy=True)
        if counts_active:
            outside = (zsub < z_lo) | (zsub > z_hi)
            csub[..., 3] = np.where(outside, 0.0, csub[..., 3])
        # Задача #76: отключённая подложка -> ячейки «дна» (высота ниже _FLOOR_FRAC от рельефа)
        # становятся прозрачными; остаётся только рельеф, фиолетовый прямоугольник базы исчезает.
        if not self._floor_visible:
            floor = zsub <= _FLOOR_FRAC * float(self._height_scale)
            csub[..., 3] = np.where(floor, 0.0, csub[..., 3])
            # Задача #146: у «каркаса» рёбра не имеют per-vertex alpha — дно вырезается NaN-ом
            if self._surface_style == "wire":
                zsub = zsub.copy()
                zsub[floor] = np.nan
        # Задача #145: стиль «однотонный» — перекрас в цвет образца (голубой), полупрозрачно.
        if self._surface_style == "solid":
            self._apply_solid_style(csub, _SHEET_RGB_SMP, 0.45)
        colors_flat = csub.reshape(-1, 4)
        if self._surface is not None:
            self.removeItem(self._surface)
            self._surface = None
        surf = gl.GLSurfacePlotItem(x=xs, y=ys, z=zsub, colors=colors_flat,
                                    shader=None, computeNormals=False, smooth=False,
                                    **self._sheet_style_kwargs(self._surface_style, _SHEET_RGB_SMP))
        # translucent нужен, когда alpha значимы: обрезка по счёту (IV-R3), скрытое дно (#76)
        # или нестандартный стиль (#145); иначе opaque (быстрее, без сортировки прозрачности).
        translucent = (counts_active or not self._floor_visible
                       or self._surface_style != "palette")
        surf.setGLOptions("translucent" if translucent else "opaque")
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
            self._rebuild_peak_ridges()   # Задача #121: гребни следуют за окном обрезки

    def set_floor_visible(self, visible: bool) -> None:
        """Задача #76: показать/скрыть «подложку» (плоское дно рельефа, фиолетовый прямоугольник).
        Меняет только alpha ячеек дна — достаточно пересобрать поверхность, без ре-LOD."""
        visible = bool(visible)
        if visible == self._floor_visible:
            return
        self._floor_visible = visible
        if self._z_surface is not None:
            self._rebuild_surface()
        self._rebuild_bg_sheet()   # Задача #146: дно простыни фона следует за тем же тумблером

    def set_background_sheet(self, bg_cps, raw=None) -> None:
        """Задача #98: задать поканальный фон (cps, полное разрешение) для «простыни» на 3D;
        None — снять. Видимость управляет set_background_sheet_visible («Наложение фона»).
        Задача #140: raw = (counts_блок, lt_блок) сырого фонового окна (range-источник) —
        простыня строится из него тем же способом, что рельеф; None — гладкий bg_cps."""
        self._bg_cps_full = (None if bg_cps is None
                             else np.asarray(bg_cps, dtype=np.float64).ravel())
        self._bg_sheet_raw = None if raw is None else (np.asarray(raw[0], dtype=np.float64),
                                                       np.asarray(raw[1], dtype=np.float64).ravel())
        self._rebuild_bg_sheet()

    def set_surface_visible(self, on: bool) -> None:
        """Задача #143: показать/скрыть «простыню образца» (основной 3D-рельеф)."""
        self._surface_on = bool(on)
        self._rebuild_surface()

    def set_surface_style(self, style: str) -> None:
        """Задача #145: стиль простыни образца: 'palette' | 'solid' | 'wire'."""
        self._surface_style = style
        self._rebuild_surface()

    def set_bg_sheet_style(self, style: str) -> None:
        """Задача #145: стиль простыни фона: 'palette' | 'solid' | 'wire'."""
        self._bg_style = style
        self._rebuild_bg_sheet()

    @staticmethod
    def _sheet_style_kwargs(style: str, rgb) -> dict:
        """Задача #145: kwargs GLSurfacePlotItem для стиля 'wire' — каркас без граней."""
        if style == "wire":
            return {"drawEdges": True, "drawFaces": False, "edgeColor": (*rgb, 0.85)}
        return {}

    @staticmethod
    def _apply_solid_style(colors, rgb, alpha: float) -> None:
        """Задача #145: перекрасить массив цветов (..., 4) в однотонный rgb; прозрачные
        ячейки (alpha==0 — обрезка по счёту/скрытое дно) остаются прозрачными."""
        colors[..., 0], colors[..., 1], colors[..., 2] = rgb
        colors[..., 3] = np.where(colors[..., 3] > 0.0, alpha, 0.0)

    def set_background_sheet_visible(self, on: bool) -> None:
        """Задача #98: показать/скрыть «фоновую простыню» на 3D-водопаде."""
        self._bg_sheet_on = bool(on)
        self._rebuild_bg_sheet()

    def _rebuild_bg_sheet(self) -> None:
        """Задача #98/#140: «простыня» фона — полупрозрачный лист, ПОСТРОЕННЫЙ ТЕМ ЖЕ СПОСОБОМ,
        что рельеф-образец (решение оператора #140: «они должны строиться одинаково»): сырое
        фоновое окно -> тот же LOD/сглаживание -> та же карта value->height. Постоянна по времени;
        при фон=образец идёт сквозь рельеф в пределах пуассонова шума, а не над ним."""
        if self._bg_sheet is not None:
            self.removeItem(self._bg_sheet); self._bg_sheet = None
        bg = None if self._bg_cps_full is None else np.asarray(self._bg_cps_full, dtype=np.float64)
        if (not self._bg_sheet_on or bg is None or self._z_surface is None
                or self._z_counts is None or self._sg is None or self._nc == 0):
            return
        e_full = np.asarray(self._sg.energies(), dtype=np.float64)
        if e_full.size != bg.size:
            return
        # Задача #144: простыня — полноценный рельеф ТЕМ ЖЕ пайплайном, что рельеф-образец
        # (LOD max → обрезка по _MAX_ENERGY_KEV → smooth → apply_z_scale → нормировка на zmax
        # образца → cmap). Если сырой блок фона совпадает по форме с целевой матрицей (фон=тот же
        # файл или совместимый) — гоним счётную матрицу фона в текущих единицах через тот же
        # sg.downsample; иначе fallback: константное поле поканального фона.
        bg_field = self._bg_sheet_field(bg)
        if bg_field is None:
            return
        z_bg, _t, ch_bg = self._sg.downsample(self._max_time, self._max_chan,
                                              method="max", data=bg_field)
        z_bg = np.asarray(z_bg, dtype=np.float64)
        keep = int(np.count_nonzero(np.asarray(ch_bg, dtype=np.float64) <= _MAX_ENERGY_KEV))
        if 0 < keep < z_bg.shape[1]:
            z_bg = z_bg[:, :keep]
        z_bg = smooth_by_mode(z_bg, self._smooth, axis=1)[:, :self._nc]
        if z_bg.shape[1] < self._nc:
            return
        self._add_bg_sheet(z_bg)

    def _bg_sheet_field(self, bg_cps):
        """Задача #144: полноразмерная матрица счётов фона в текущих единицах для sg.downsample.
        При совпадении формы с counts образца — берём сырой блок как есть (counts / lt при cps,
        counts при counts). Иначе fallback: константа поканального фона размножается по времени."""
        target_shape = self._sg.counts.shape
        if (self._bg_sheet_raw is not None
                and self._bg_sheet_raw[0].shape == target_shape):
            bg_counts = np.asarray(self._bg_sheet_raw[0], dtype=np.float64)
            if self._unit == "cps":
                lt_arr = np.asarray(self._bg_sheet_raw[1], dtype=np.float64)
                safe = np.where(lt_arr > 0.0, lt_arr, np.inf)
                return bg_counts / safe[:, None]
            return bg_counts
        lt_arr = np.asarray(self._sg.live_time_s, dtype=np.float64)
        pos = lt_arr[lt_arr > 0.0]
        lt_ref = float(np.median(pos)) if pos.size else 1.0
        val = np.asarray(bg_cps, dtype=np.float64)
        val = val if self._unit == "cps" else val * lt_ref
        return np.ones(target_shape, dtype=np.float64) * val[None, :]

    def _add_bg_sheet(self, z_bg) -> None:
        """Задача #144: цветной полупрозрачный рельеф простыни (палитра как у образца, alpha=0.55).
        Высоты нормируются на zmax ОБРАЗЦА (одна высотная система с рельефом), при фон=образец
        простыня ложится точно поверх рельефа."""
        z_disp = apply_z_scale(np.asarray(z_bg, dtype=np.float64), self._z_mode,
                               gain=self._gain, gamma=self._gamma, clip=self._clip)
        # Задача P-3: zmax образца уже посчитан в set_spectrogram (self._zmax_sample) —
        # не гонять apply_z_scale по self._z_counts ещё раз только ради .max().
        zmax = self._zmax_sample
        zn = z_disp / zmax if zmax > 0 else z_disp
        z = (zn * self._height_scale).astype(np.float32)
        colors = get_colormap(self._cmap_name).map(zn, mode="float").astype(np.float32)
        colors[..., 3] = 0.55
        # Задача #145: стиль «однотонный» — перекрас в цвет фона (оранжевый, конвенция срезов).
        if self._bg_style == "solid":
            self._apply_solid_style(colors, _SHEET_RGB_BG, 0.55)
        # Задача #146: «Подложка» гасит дно и у простыни фона (критерий как у рельефа, #76);
        # у «каркаса» рёбра без per-vertex alpha — ячейки дна вырезаются NaN-ом.
        if not self._floor_visible:
            floor = zn <= _FLOOR_FRAC
            colors[..., 3] = np.where(floor, 0.0, colors[..., 3])
            if self._bg_style == "wire":
                z = z.copy()
                z[floor] = np.nan
        nt, nc = z.shape
        xs = np.arange(nt, dtype=np.float32)
        ys = np.arange(nc, dtype=np.float32)
        sheet = gl.GLSurfacePlotItem(x=xs, y=ys, z=z, colors=colors.reshape(-1, 4),
                                     shader=None, computeNormals=False, smooth=False,
                                     **self._sheet_style_kwargs(self._bg_style, _SHEET_RGB_BG))
        sheet.setGLOptions("translucent"); sheet.setDepthValue(5)
        sheet.translate(-self._nt / 2.0, -self._nc / 2.0, 0.0)
        self.addItem(sheet); self._bg_sheet = sheet

    def set_smoothing(self, mode: int) -> None:
        """Режим сглаживания по энергии (Задача #163): 0/SMA/WMA (см. zscale.SMOOTH_MODE_*);
        ре-рендер из той же sg."""
        self._smooth = max(0, min(2, int(mode)))
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    @property
    def has_segments(self) -> bool:
        """Задача #174: True если сегменты уже вычислены (для авто-триггера сегментации)."""
        return bool(self._seg_bounds_sec)

    def set_t_smoothing(self, mode: int, by_seg: bool = False) -> None:
        """Задача #172/#174: режим сглаживания по оси времени (0..6) + флаг «по сегментам»."""
        self._t_smooth = max(0, min(6, int(mode)))
        self._t_smooth_by_seg = bool(by_seg)
        if self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def set_segment_bounds(self, segs) -> None:
        """Задача #172: задать временны́е границы сегментов (список TimeSegment или (t0, t1) с).
        Ре-рендер только если активно временно́е сглаживание по сегментам."""
        self._seg_bounds_sec = [(float(s.t_start_s), float(s.t_end_s))
                                if hasattr(s, 't_start_s') else (float(s[0]), float(s[1]))
                                for s in segs]
        if self._t_smooth and self._t_smooth_by_seg and self._sg is not None:
            self.set_spectrogram(self._sg, self._max_time, self._max_chan)

    def set_time_bins(self, max_time: int) -> None:
        """Задача #56: ширина выборки по времени = число временны́х бинов LOD-прорежки.
        Больше бинов (у́же выборка) — детальнее/«растянуто» по времени; меньше (шире выборка) —
        грубее/«сжато». Ре-рендер из той же sg; max_chan не трогаем."""
        self._max_time = max(1, int(max_time))
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

    @staticmethod
    def _step_ticks(lo: float, hi: float, step: float):
        """Деления фиксированным шагом `step`, кратные ему, в пределах [lo, hi] (Задача #71/#72)."""
        lo = float(lo); hi = float(hi)
        if step <= 0 or not (hi > lo):
            return np.array([], dtype=float)
        first = np.ceil(lo / step - 1e-9) * step
        n = int(np.floor((hi - first) / step + 1e-9)) + 1
        if n < 1:
            return np.array([], dtype=float)
        return first + np.arange(n) * step

    @classmethod
    def _pick_ticks(cls, lo: float, hi: float, pref_step: float):
        """Деления с предпочтительным шагом `pref_step` (#71/#72); если их число вне разумного
        диапазона (узкий/огромный диапазон) — откат на адаптивные `_nice_ticks`."""
        t = cls._step_ticks(lo, hi, pref_step)
        if _PREF_STEP_MIN_TICKS <= t.size <= _PREF_STEP_MAX_TICKS:
            return t
        return cls._nice_ticks(lo, hi)

    def _clear_axis_items(self) -> None:
        for it in self._axis_items:
            self.removeItem(it)
        self._axis_items = []

    def _add_text(self, pos, text: str, font) -> None:
        item = gl.GLTextItem(pos=np.asarray(pos, dtype=float), text=text,
                             color=_TICK_COLOR, font=font)
        self.addItem(item)
        self._axis_items.append(item)

    def _time_ticks(self):
        """Деления оси времени (Задача #64/#71): (disp_values, world_x, unit). Для длинных
        водопадов — фиксированный шаг 15 минут (#71); для коротких записей — авто-деления
        `_nice_ticks` в выбранной единице (с/мин/ч). Значения подписей — в выбранной единице."""
        tc = self._t_centers
        if tc is None or len(tc) < 2 or tc[-1] <= tc[0] or self._nt == 0:
            return (np.array([]), np.array([]), self._time_unit)
        scale = _TIME_UNIT_SCALE.get(self._time_unit, 1.0)
        idx = np.arange(self._nt, dtype=float)
        sv = self._step_ticks(float(tc[0]), float(tc[-1]), _TIME_STEP_15MIN_S)   # 15-мин деления (#71)
        if _PREF_STEP_MIN_TICKS <= sv.size <= _PREF_STEP_MAX_TICKS:
            dv = sv / scale
        else:                       # короткая запись -> авто-деления в выбранной единице (#64)
            dv = self._nice_ticks(float(tc[0]) / scale, float(tc[-1]) / scale)
        wx = np.interp(dv * scale, tc, idx) - self._nt / 2.0
        return (dv, wx, self._time_unit)

    def _energy_ticks(self):
        """Деления оси энергии (Задача #66/#72): (values, world_y) в центрированных мировых
        координатах. Для широкого спектра — фиксированный шаг 200 кэВ (#72); для узкого
        диапазона — авто-деления `_nice_ticks` (внутри `_pick_ticks`)."""
        cc = self._ch_centers
        if cc is None or len(cc) < 2 or cc[-1] <= cc[0] or self._nc == 0:
            return (np.array([]), np.array([]))
        idx = np.arange(self._nc, dtype=float)
        ev = self._pick_ticks(float(cc[0]), float(cc[-1]), _ENERGY_STEP_KEV)   # 200-кэВ деления (#72)
        wy = np.interp(ev, cc, idx) - self._nc / 2.0
        return (ev, wy)

    def _add_grid_line(self, p0, p1, color, width=1.0) -> None:
        """Линия координатной сетки/рамки (Задача #63/#68); хранится в _grid_items.
        Задача #157: glOptions='translucent' => depth-тест ВКЛ (+блендинг: у цветов сетки
        альфа 0.55/0.9), рельеф перекрывает линии сетки позади себя. Дефолт GLLinePlotItem —
        'additive' (depth-тест ВЫКЛ) => сетка просвечивала сквозь рельеф (корень как в #95)."""
        pts = np.array([p0, p1], dtype=np.float32)
        item = gl.GLLinePlotItem(pos=pts, color=color, width=width,
                                 mode="line_strip", antialias=True,
                                 glOptions="translucent")
        self.addItem(item)
        self._grid_items.append(item)

    def set_time_unit(self, unit: str) -> None:
        """Единицы оси времени: 'с' | 'мин' | 'ч' (Задача #64). Перестроить сетку и подписи."""
        if unit not in _TIME_UNIT_SCALE:
            return
        self._time_unit = unit
        self._rebuild_grid()
        self._rebuild_axis_labels()

    def _rebuild_grid(self) -> None:
        """Координатная сетка (Задача #63/#68): линии на круглых делениях шкал t/E; поле
        обрамлено отступом в полклетки со всех сторон (#70), рамка поля ярче линий сетки."""
        for it in self._grid_items:
            self.removeItem(it)
        self._grid_items = []
        if not self._axis_labels_visible or self._z_surface is None:
            return
        if self._nt == 0 or self._nc == 0:
            return
        xmin, xmax, ymin, ymax, _zmax = self._axis_extent()
        _dv, wx, _u = self._time_ticks()
        _ev, wy = self._energy_ticks()
        cellx = float(np.median(np.diff(wx))) if wx.size >= 2 else (xmax - xmin) / 5.0
        celly = float(np.median(np.diff(wy))) if wy.size >= 2 else (ymax - ymin) / 5.0
        mx, my = 0.5 * cellx, 0.5 * celly   # Задача #70: отступ поля = полклетки (было 1 клетка, #63)
        gx0, gx1 = xmin - mx, xmax + mx
        gy0, gy1 = ymin - my, ymax + my
        self._draw_grid_lines(wx, wy, gx0, gx1, gy0, gy1)

    def _draw_grid_lines(self, wx, wy, gx0, gx1, gy0, gy1) -> None:
        """Линии сетки на делениях шкал + яркая рамка поля на полклетки шире данных (#63/#70)."""
        for x in np.asarray(wx, float):
            self._add_grid_line((x, gy0, 0.0), (x, gy1, 0.0), _GRID_RGBA)
        for y in np.asarray(wy, float):
            self._add_grid_line((gx0, y, 0.0), (gx1, y, 0.0), _GRID_RGBA)
        corners = [(gx0, gy0), (gx1, gy0), (gx1, gy1), (gx0, gy1), (gx0, gy0)]
        for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
            self._add_grid_line((x0, y0, 0.0), (x1, y1, 0.0), _GRID_BORDER_RGBA, width=1.8)

    def set_axis_labels_visible(self, visible: bool) -> None:
        """Показать/скрыть подписи делений осей и координатную сетку (Задача 14/#63)."""
        self._axis_labels_visible = bool(visible)
        self._rebuild_axis_labels()
        self._rebuild_grid()

    def _rebuild_axis_labels(self) -> None:
        """Подписи делений осей времени и энергии (Задача #64/#66): значение + единица на каждой
        клетке. Вертикальную шкалу счёта (Z) не строим (Задача #65). Зубцы шкалы энергий убраны
        (Задача #80)."""
        self._clear_axis_items()
        if not self._axis_labels_visible or self._z_surface is None:
            return
        nt, nc = self._nt, self._nc
        if nt == 0 or nc == 0:
            return
        xmin, xmax, ymin, ymax, _zmax = self._axis_extent()   # zmax не нужен: зубцов нет (#80)
        pad = 0.05 * float(max(nt, nc, 10))
        font = QtGui.QFont("Helvetica", 7)   # Задача #73: подписи делений были крупны (10) — уменьшено
        # Задача #77: подписи держим на ближнем к зрителю крае оси (по знаку XY-направления камеры),
        # чтобы они не уходили за рельеф; при пересечении квадранта край меняется (_maybe_reorient_labels).
        sx, sy = self._viewer_sides()
        self._label_side_sig = (sx, sy)
        y_time = (ymax if sy > 0 else ymin) + sy * pad     # край подписи времени (вынос наружу)
        x_en = xmax if sx > 0 else xmin                     # край оси энергии (зубцы и подписи)
        x_en_lab = x_en + sx * pad
        # ось времени (X): значение в выбранной единице + единица на каждом делении (Задача #64/#66).
        # Задача #183: единица оси через tr() для RU↔EN.
        dv, wx, unit = self._time_ticks()
        unit_lbl = tr(unit)
        for tv, x in zip(dv, wx):
            self._add_text((float(x), y_time, 0.0), f"{tv:g} {unit_lbl}", font)
        # ось энергии (Y): значение в кэВ + единица на каждом делении (Задача #66).
        # Задача #80: вертикальные оливковые зубцы-отрезки (IV-R2/#77) убраны — только подписи.
        keV_lbl = tr("кэВ")
        ev, wy = self._energy_ticks()
        for en, y in zip(ev, wy):
            self._add_text((x_en_lab, float(y), 0.0), f"{en:g} {keV_lbl}", font)

    def _viewer_sides(self):
        """Задача #77: знаки X/Y-направления «на зрителя» (камера → центр сцены, центр в 0,0).
        +1 — ближний край оси у +X/+Y (xmax/ymax), -1 — у -X/-Y (xmin/ymin)."""
        try:
            cam = self.cameraPosition()
            cx, cy = float(cam.x()), float(cam.y())
        except Exception:
            cx, cy = 1.0, -1.0    # дефолт камеры (azimuth=-60): energy→xmax, time→ymin
        sx = 1 if cx >= 0 else -1
        sy = 1 if cy >= 0 else -1
        return sx, sy

    def _maybe_reorient_labels(self) -> None:
        """Задача #77: если поворот камеры сменил ближний край (квадрант) — перевесить подписи
        осей на сторону взгляда. Перестраиваем только при смене стороны (редко, не каждый кадр)."""
        if not self._axis_labels_visible or self._z_surface is None:
            return
        if self._viewer_sides() != self._label_side_sig:
            self._rebuild_axis_labels()

    def mouseMoveEvent(self, ev):
        """Вращение камеры мышью (базовое поведение GLViewWidget) + перевешивание подписей осей
        на ближнюю к зрителю сторону при смене квадранта (Задача #77)."""
        super().mouseMoveEvent(ev)
        self._maybe_reorient_labels()

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
            self._rebuild_plane_nuclides()   # Задача #67/#69: маркеры на плоскостях Времени

    def _clear_ray_items(self) -> None:
        for it in self._ray_items:
            self.removeItem(it)
        self._ray_items = []

    def _rebuild_energy_rays(self) -> None:
        """Задача #85: маркеры изотопов отображаются ТОЛЬКО на секущих плоскостях
        (`_rebuild_plane_nuclides`). Прежние рёберные лучи у края данных (Задача 15/IV-R2)
        больше не рисуются — метод лишь снимает ранее построенные лучи, чтобы при
        перестроениях не оставалось «висящих» маркеров вне плоскостей."""
        self._clear_ray_items()

    def _clear_plane_nuclides(self) -> None:
        for it in self._plane_nuclide_items:
            self.removeItem(it)
        self._plane_nuclide_items = []

    def _rebuild_plane_nuclides(self) -> None:
        """Маркеры выбранных гамма-линий нуклидов на гранях видимых плоскостей Времени
        (Задача #67): цветной вертикальный отрезок на позиции энергии каждой линии. Высота
        ∝ интенсивности (Задача #69) — ярчайшая линия = полная высота zmax, остальные ниже,
        так на грани читается соотношение линий семейства."""
        self._clear_plane_nuclides()
        if self._z_surface is None or not self._energy_lines:
            return
        cc, nc = self._ch_centers, self._nc
        if cc is None or len(cc) < 2 or nc == 0:
            return
        _xn, _xx, _yn, _yx, zmax = self._axis_extent()
        if zmax <= 0:
            return
        emin, emax = float(cc[0]), float(cc[-1])
        idx = np.arange(nc, dtype=float)
        imax = self._max_line_intensity()
        for slot in (0, 1):
            entry = self._planes.get(("time", slot))
            if entry is None or not entry["visible"]:
                continue
            i = self._frac_to_index(self._t_centers, entry["frac"])
            px = float(i) - self._nt / 2.0
            self._draw_plane_nuclide_lines(px, cc, idx, emin, emax, zmax, imax)

    def _max_line_intensity(self) -> float:
        """Максимальная интенсивность среди выбранных линий (нормировка высот #69).
        Линии без интенсивности (3-кортежи) игнорируются; нет данных -> 0.0."""
        vals = [float(ln[3]) for ln in self._energy_lines
                if len(ln) > 3 and ln[3] is not None and float(ln[3]) > 0]
        return max(vals) if vals else 0.0

    def _draw_plane_nuclide_lines(self, px, cc, idx, emin, emax, zmax, imax) -> None:
        """Вертикальные маркеры выбранных линий на грани плоскости Времени (x=px): py —
        энергия->мировой Y; высота h ∝ интенсивности (доля imax), полная при её отсутствии."""
        nc = self._nc
        for ln in self._energy_lines:
            e = float(ln[0])
            if e < emin or e > emax:
                continue
            py = float(np.interp(e, cc, idx)) - nc / 2.0
            self._add_plane_nuclide_line(px, py, ln, zmax, imax)

    def _add_plane_nuclide_line(self, px, py, ln, zmax, imax) -> None:
        """Один маркер на грани: высота ∝ интенсивности (#69) либо полная (#67, 3-кортеж)."""
        inten = float(ln[3]) if len(ln) > 3 and ln[3] is not None else None
        if inten is not None and imax > 0:
            h = max(0.04, inten / imax) * zmax    # доля высоты по интенсивности (#69)
        else:
            h = zmax                              # нет интенсивности -> полная высота (#67)
        try:
            rgba = pg.mkColor(ln[1]).getRgbF()
        except Exception:
            rgba = (1.0, 1.0, 1.0, 1.0)
        pos = np.array([[px, py, 0.0], [px, py, h]], dtype=np.float32)
        # Задача #95: glOptions='opaque' => depth-тест ВКЛ, рельеф перекрывает маркеры, оказавшиеся
        # позади массива. Дефолт GLLinePlotItem — 'additive' (depth-тест ВЫКЛ) => маркеры просвечивали
        # сквозь рельеф. 'opaque' даёт корректное перекрытие при любом порядке отрисовки (blend не нужен,
        # потому antialias=False — сглаживание линии без blend всё равно не применяется).
        item = gl.GLLinePlotItem(pos=pos, color=rgba, width=3.0,
                                 mode="line_strip", antialias=False, glOptions="opaque")
        self.addItem(item)
        self._plane_nuclide_items.append(item)

    # ---------- выделение найденных фотопиков на рельефе (Задача #110/#114/#112) ----------
    def set_peak_search(self, on: bool) -> None:
        """Задача #110: вкл/выкл выделение фотопиков (Mariscotti+Currie) на 3D-водопаде."""
        self._peaks_on = bool(on)
        self._rebuild_peak_ridges()

    def set_peak_sigma(self, sigma: float) -> None:
        """Задача #114/#111: сменить порог значимости σ → пересчитать пики и гребни+панель."""
        self._peak_sigma = float(sigma)
        if self._peaks_on and self._sg is not None:
            self._rebuild_peak_ridges()

    @staticmethod
    def _peak_key(energy) -> float:
        """Задача #124: ключ пика по энергии — round до 3 знаков (кэВ). Стабилен между
        детерминированными вызовами _found_peaks() при фиксированных σ/данных."""
        return round(float(energy), 3)

    def set_peak_visible(self, energy, visible: bool) -> None:
        """Задача #124: показать/скрыть гребень конкретного пика (чекбокс в панели «Найденные
        пики»). Ключ по энергии; перестраивает гребни, если поиск пиков включён."""
        key = self._peak_key(energy)
        if visible:
            self._peak_hidden_keys.discard(key)
        else:
            self._peak_hidden_keys.add(key)
        if self._peaks_on and self._sg is not None:
            self._rebuild_peak_ridges()

    def set_peak_highlight(self, energy) -> None:
        """Задача #124: подсветить (выделить малиновым) гребень пика по энергии — реакция на
        клик по строке панели. energy=None снимает подсветку. Перестраивает гребни."""
        self._peak_highlight_key = None if energy is None else self._peak_key(energy)
        if self._peaks_on and self._sg is not None:
            self._rebuild_peak_ridges()

    def clear_peak_overrides(self) -> None:
        """Задача #124: сбросить пер-пиковые скрытия и подсветку (новый набор пиков)."""
        self._peak_hidden_keys = set()
        self._peak_highlight_key = None

    def _clear_peak_ridges(self) -> None:
        for it in self._peak_ridge_items:
            self.removeItem(it)
        self._peak_ridge_items = []

    def _found_peaks(self) -> list:
        """Задача #120/#114: FoundPeak по АВТОКАЛИБРОВАННОЙ FWHM(E)-модели и σ.
        Модель строится auto_calibrate_fwhm_model() по сильным реальным пикам спектра
        (откат на default_fwhm_model R=7%@662 кэВ на бедных спектрах). Дефолт-модель
        системно шире реальной → широкое ядро −G'' занижало значимость Currie L_C и
        пропускало узкие пики; калибровка сужает ядро под детектор (Задача #120).
        Сырые counts СУММАРНОГО спектра (full resolution). Задача #113: к интегральным
        пикам подмешиваются ТРАНЗИЕНТНЫЕ (значимы лишь в узком окне срезов, утоплены в
        интеграле). Источник истины для #110 и #111.
        Задача #158: данные — _sg_analysis (без ε-нормализации #156): Currie предполагает
        var=N, нормализация давала ложные пики >1700 кэВ и двойную ε-коррекцию в ID."""
        if self._sg is None:
            return []
        sg = self._sg_analysis if self._sg_analysis is not None else self._sg
        # #190/F11: кэш — пики не зависят от положения секущих плоскостей/регулировок вида.
        # #225: id(sg.calibration) в ключе — при перекалибровке (#215) sg тот же объект,
        # но energies() и энергии пиков меняются; без этого таблица пиков и ID не обновятся.
        _ck = (id(sg), id(sg.calibration), self._peak_sigma)
        if getattr(self, '_peaks_cache_key', None) == _ck:
            return self._peaks_cache
        counts = np.asarray(sg.total_spectrum(), dtype=np.float64)
        energies = np.asarray(sg.energies(), dtype=np.float64)
        # Задача #120: автокалибровка модели разрешения под реальный детектор (один раз —
        # ширины едины для интегрального и транзиентного скана: разрешение от времени не зависит).
        model = auto_calibrate_fwhm_model(counts, energies)
        widths = fwhm_channels_from_model(model, energies)
        # Интегральный скан (как было): пики, видимые в СУММАРНОМ по времени спектре.
        integral = find_peaks(counts, widths, sigma_threshold=self._peak_sigma, energies=energies)
        # Задача #113: транзиентные (время-локализованные) пики — невидимы в интеграле
        # (утоплены в шуме), но значимы в узком окне срезов (напр. ~186 кэВ на реальном
        # файле). Сырые counts 2D (как total_spectrum), те же калиброванные ширины #120,
        # порог окна = self._peak_sigma + _TRANSIENT_SIGMA_MARGIN (привязка к _peak_sigma
        # держит монотонность чувствительности: больше σ -> не больше пиков).
        raw_2d = np.asarray(sg.counts, dtype=np.float64)   # #158: тоже аналитический источник
        transient = find_transient_peaks(
            raw_2d, energies, widths, integral,
            transient_sigma_threshold=self._peak_sigma + _TRANSIENT_SIGMA_MARGIN,
        )
        peaks = integral + transient
        # Задача #119: не искать/не показывать пики за пределом отображения (#78 — 3000 кэВ).
        # Поиск идёт по полному спектру, но выше _MAX_ENERGY_KEV ни рельеф, ни сетка не
        # рисуются — значит пик там не нужен (в таблице всплывал «мусорный» 3345 кэВ).
        # Задача #153: снизу — граница _MIN_ENERGY_KEV (50 кэВ), отсекает рентгеновский край.
        result = [pk for pk in peaks
                  if _MIN_ENERGY_KEV <= float(pk.energy) <= _MAX_ENERGY_KEV]
        self._peaks_cache_key = _ck
        self._peaks_cache = result
        return result

    def _found_peak_energies(self) -> list:
        """Задача #114: производное от _found_peaks() — только энергии (кэВ)."""
        return [float(pk.energy) for pk in self._found_peaks()]

    def _rebuild_peak_ridges(self) -> None:
        """Задача #110/#112: выделение пиков из _found_peaks(), гребень — только по срезам
        с присутствием пика (peak_time_mask). Задача #121: гребни обрезаны секущими плоскостями."""
        self._clear_peak_ridges()
        cc, nt, nc = self._ch_centers, self._nt, self._nc
        if not self._peaks_on or self._z_surface is None or cc is None or len(cc) < 2:
            return
        if nt < 1 or nc < 1 or self._z_counts is None:
            return
        i0, i1, j0, j1, z_lo, z_hi, counts_active = self._clip_windows()  # Задача #121
        if i0 > i1 or j0 > j1:
            return
        emin, emax = float(cc[0]), float(cc[-1])
        idx = np.arange(nc, dtype=float)
        for pk in self._found_peaks():
            e = float(pk.energy)
            if e < emin or e > emax:
                continue
            key = self._peak_key(e)                       # Задача #124
            if key in self._peak_hidden_keys:             # чекбокс снят — гребень скрыт
                continue
            jc = max(0, min(nc - 1, int(round(float(np.interp(e, cc, idx))))))
            if jc < j0 or jc > j1:
                continue
            emphasize = (self._peak_highlight_key is not None
                         and key == self._peak_highlight_key)   # клик по строке
            self._add_peak_ridge(jc, i0, i1, z_lo, z_hi, counts_active, emphasize)

    def _add_peak_ridge(self, jc: int, i0: int = 0, i1: int | None = None,
                        z_lo: float = -np.inf, z_hi: float = np.inf,
                        counts_active: bool = False, emphasize: bool = False) -> None:
        """Задача #110/#112: гребень на канале jc только в зоне присутствия пика (#112).
        peak_time_mask → непрерывные True-сегменты → отдельные GLLinePlotItem. Задача #121:
        window ограничивает гребень окном времени [i0,i1] и (при активной) плоскостью счёта."""
        nt, nc = self._nt, self._nc
        if i1 is None:
            i1 = nt - 1
        lift = 0.01 * float(self._height_scale)
        # Задача #152: маска присутствия — по СЫРЫМ counts (sum-LOD) с пуассоновым гейтом
        # net_s ≥ K·√(baseline_s): cps-матрица (масштабно-инвариантные пороги) пропускала
        # почти пустые столбцы (следовые отсчёты) — длинные гребни на пустом поле.
        zsrc = self._z_counts_int if self._z_counts_int is not None else self._z_counts
        poi = self._z_counts_int is not None
        # Окно пика для гейта: FWHM в LOD-каналах ~ nc/64 (256→±4, 512→±8) — пик размазан
        # на несколько LOD-каналов, одноканальный гейт терял кросс-канальную значимость.
        hw = max(2, int(round(nc / 64.0))) if poi else 0
        mask = peak_time_mask(zsrc, jc, poisson=poi, peak_hw=hw)
        if poi:
            # Задача #160: слабая ШИРОКАЯ линия — одноканальная строгая маска пуста или
            # почти пуста (на канал приходится малая доля счётов пика, пер-бин гейт #152
            # не имеет статистической мощи: K-40 1461 → 7/400 бинов), хотя линия
            # интегрально значима (она в таблице пиков: K-40, равновесные линии Th-232
            # 463/727/969/1588, escape 2095.5) — зона 1461–2614 оставалась без гребней.
            # ОБЪЕДИНЯЕМ со суммой окна каналов jc±hw: статистика всей ширины пика,
            # Currie-гейт на каждый бин времени сохраняет локализацию (транзиент не
            # тянется на пустое поле — анти-#112/#126/#129), морфология убивает
            # одиночные шум-биты (контрольные пустые каналы 1800/2400/2900 кэВ
            # реального файла → 0 бинов, гребни там не появляются).
            mask = mask | peak_presence_mask(zsrc, jc, peak_hw=hw)
        if not mask.any():
            # Задача #126: строгая маска присутствия (#112) обнулилась — пик слабый/
            # интегрально-значимый/транзиентный, не прошёл Currie-гейты. РАНЬШЕ гребень
            # тогда рисовался на ВСЮ ось времени (продлевался на пустое место). Берём
            # РЕЛАКСИРОВАННУЮ маску: колоночный Currie-гейт снят (min_peak_over_bg=0),
            # но абсолютный порог НЕ снят до 0 (Задача #129 — рецидив #112/#126): при
            # noise_factor=0.0 порог вырождался в чистый floor (после фикса #129 в
            # peak_time_mask), и на реальном пуассоновском шуме маска снова покрывала
            # почти всю ось. noise_factor=6.0 подобран численно (10 seed-ов синтетич.
            # шумного транзиента): даёт локализованный гребень вместо покрытия всей оси,
            # не регрессирует детерминированный сценарий #126 (там σ_noise=0 → фактор
            # роли не играет).
            mask = peak_time_mask(zsrc, jc, noise_factor=6.0, min_peak_over_bg=0.0,
                                  abs_floor=True, poisson=poi, peak_hw=hw)
        xs_all = np.arange(nt, dtype=np.float64) - nt / 2.0
        y_val = float(jc) - nc / 2.0
        zs_raw = self._z_surface[:, jc].astype(np.float64)
        zs_all = zs_raw + lift
        window = np.zeros(nt, dtype=bool)        # Задача #121
        window[max(0, i0):i1 + 1] = True
        if counts_active:
            window &= (zs_raw >= z_lo) & (zs_raw <= z_hi)
        self._draw_ridge_segments(xs_all, y_val, zs_all, mask, window, emphasize)

    def _draw_ridge_segments(self, xs_all, y_val, zs_all, mask, window=None,
                             emphasize: bool = False) -> None:
        """Задача #112/#126: нарисовать сегменты гребня только там где mask==True.
        Маску (строгую #112 или релаксированную #126) формирует _add_peak_ridge. Прежний
        фолбэк «при пустой маске рисовать ВЕСЬ гребень» убран (#126) — именно он и продлевал
        линию пика на пустые срезы; пустая маска здесь → гребень не рисуется (в столбце нет
        подъёма над фоном — рисовать нечего). Каждый непрерывный True-участок → GLLinePlotItem."""
        if window is not None:
            mask = mask & window   # Задача #121: ограничить активными секущими плоскостями
        # Задача #124: подсвеченный пик — малиновый и толще, прочие — зелёные как раньше.
        color = _PEAK_HILITE_RGBA if emphasize else _PEAK_RAY_RGBA
        width = 6.0 if emphasize else 3.0
        changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < 1:
                continue
            xs = xs_all[s:e].astype(np.float32)
            ys = np.full(e - s, y_val, dtype=np.float32)
            zs = zs_all[s:e].astype(np.float32)
            pos = np.column_stack([xs, ys, zs]).astype(np.float32)
            item = gl.GLLinePlotItem(pos=pos, color=color, width=width,
                                     mode="line_strip", antialias=False,
                                     glOptions="opaque")
            self.addItem(item)
            self._peak_ridge_items.append(item)

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
        for ln in self._energy_lines:
            energy = ln[0]   # 3- или 4-кортеж (Задача #69)
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

    def time_plane_range(self):
        """Задача #148: диапазон времени (t0, t1) в секундах между секущими плоскостями Времени
        для выбора фонового участка. Невидимый слот -> край оси (семантика #84); ни одна
        плоскость Времени не видима -> None (выбор плоскостями не задан).
        Задача #151: LOD-бины рельефа шире сырых срезов — возвращаем КРАЯ крайних бинов
        в сырых срезах (разбиение как в sg.downsample), не центры: иначе сырые срезы внешней
        половины краевого бина выпадают из выборки фона и после вычета остаются «гребнем»
        у плоскости сечения."""
        if self._t_centers is None or len(self._t_centers) == 0 or self._sg is None:
            return None
        p0, p1 = self._planes[("time", 0)], self._planes[("time", 1)]
        if not (p0["visible"] or p1["visible"]):
            return None
        nt = len(self._t_centers)
        ns = int(self._sg.n_slices)
        edges = np.unique(np.linspace(0, ns, nt + 1).astype(np.int64))
        t_raw = np.asarray(self._sg.time_offsets_s, dtype=np.float64)
        i0 = self._frac_to_index(self._t_centers, p0["frac"]) if p0["visible"] else 0
        i1 = self._frac_to_index(self._t_centers, p1["frac"]) if p1["visible"] else nt - 1
        lo, hi = min(i0, i1), max(i0, i1)
        return (float(t_raw[edges[lo]]), float(t_raw[edges[hi + 1] - 1]))

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
        if axis == "time":
            self._rebuild_plane_nuclides()   # Задача #67: маркеры зависят от видимых плоскостей Времени

    def _refresh_all_planes(self) -> None:
        for axis in PLANE_AXES:
            for slot in (0, 1):
                self._apply_plane(axis, slot)

    def _section_projection(self, axis: str):
        """Задача #61: больше НЕ вызывается — профиль/проекция на плоскости убраны; метод
        сохранён на случай возврата фичи.
        Профиль на секущей плоскости (Задача #47): проекция-сумма рельефа в объёме между
        парой плоскостей оси, вписанная в высоту плоскости [0, zmax]. Окно суммирования — как у
        обрезки (IV-R3): обе плоскости оси видимы -> между ними, иначе вся ось. time -> сумма по
        времени (кривая по энергии, nc); energy -> сумма по энергии (кривая по времени, nt).

        Задача #52: профиль = спектр верхнего-правого окна «Срезы/Сечения/Выборки». Окно строит
        sg.sum_spectrum(t_lo,t_hi) — ИНТЕГРАЛ сырых counts по окну времени. Поэтому суммируем
        sum-LOD (`_z_counts_sum`, прорежка method='sum'), а НЕ max-LOD `_z_counts`: сумма sum-бинов
        по окну == sum_spectrum, прорежённому к nc каналам. Суммирование max-бинов (как было в #50)
        раздувало фон и расходилось с окном; истинная сумма совпадает с окном (устойчивая полка с
        бо́льшим интегралом может быть выше короткого транзиента — ровно как в окне)."""
        i0, i1, j0, j1, _zlo, _zhi, _ca = self._clip_windows()
        if axis == "time":
            s = self._z_counts_sum[i0:i1 + 1, :].sum(axis=0).astype(np.float64)  # интеграл по времени
        else:
            s = self._z_counts_sum[:, j0:j1 + 1].sum(axis=1).astype(np.float64)  # интеграл по энергии
        d = apply_z_scale(s, self._z_mode, gain=self._gain, gamma=self._gamma, clip=self._clip)
        d = np.asarray(d, dtype=np.float64)
        m = float(d.max()) if d.size else 0.0
        if m > 0.0:
            d = d / m * self._height_scale          # вписать проекцию в высоту плоскости
        return d.astype(np.float32)

    def _apply_plane(self, axis: str, slot: int) -> None:
        entry = self._planes[(axis, slot)]
        mesh, border, line = entry["mesh"], entry["border"], entry["line"]
        line.setVisible(False)          # Задача #61: профиль/проекция на плоскости убраны
        if not entry["visible"] or self._z_surface is None:
            mesh.setVisible(False)
            border.setVisible(False)
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
        elif axis == "energy":
            j = self._frac_to_index(self._ch_centers, frac)
            py = float(j) - nc / 2.0
            verts = np.array([[xmin, py, 0.0], [xmax, py, 0.0],
                              [xmax, py, zmax], [xmin, py, zmax]], dtype=np.float32)
        else:  # counts — горизонтальная плоскость уровня Z
            pz = frac * zmax
            verts = np.array([[xmin, ymin, pz], [xmax, ymin, pz],
                              [xmax, ymax, pz], [xmin, ymax, pz]], dtype=np.float32)

        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh.setMeshData(vertexes=verts, faces=faces,
                         color=(r, g, b, _PLANE_ALPHA))
        mesh.setVisible(True)
        # Задача #37: видимый контур сечения — прямоугольник по 4 углам quad'а (без заливки).
        # Задача #61: грань приглушена — полупрозрачный дим-контур (_BORDER_ALPHA), «не такая
        # яркая»; профиль/проекция на плоскости убраны (line скрыт в начале метода).
        loop = np.vstack([verts, verts[0:1]]).astype(np.float32)
        border.setData(pos=loop, color=(r, g, b, _BORDER_ALPHA), width=1.5)
        border.setVisible(True)


class SectionControls(QtWidgets.QWidget):
    """Док «Сечения» в knob-стиле панели регулировок (Задача #59): по 2 ряда на ось
    (Время/Энергия/Отсчёты) — горизонтальный движок `Knob` позиции (0..1000 → доля 0..1),
    подпись реального значения и стилизованная кнопка вкл/выкл. Излучает planeChanged.
    По умолчанию все слоты выключены (движки погашены, плоскости не рисуются)."""

    # axis: str, slot: int, frac: float, visible: bool
    planeChanged = QtCore.Signal(str, int, float, bool)

    _SLIDER_MAX = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("adjustPanel")        # тот же фон/QSS, что у панели регулировок (#59)
        self._rows = {}  # (axis, slot) -> dict(check, slider, label)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        cap = QtWidgets.QLabel(tr("Секущие плоскости (2 на ось)"))
        cap.setObjectName("knobTitle")
        self._caption = cap                      # Задача #169: ссылки для retranslate()
        self._axis_heads = {}
        layout.addWidget(cap)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(4)
        layout.addLayout(grid)
        row = 0
        for axis in PLANE_AXES:
            head = QtWidgets.QLabel(tr(_AXIS_LABEL[axis]))
            head.setObjectName("knobTitle")
            self._axis_heads[axis] = head
            grid.addWidget(head, row, 0, 1, 4)
            row += 1
            for slot in (0, 1):
                self._build_slot(grid, axis, slot, row)
                row += 1
        layout.addStretch(1)

    def _build_slot(self, grid, axis, slot, row):
        """Задача #59: ряд слота — #N | движок | значение | вкл/выкл."""
        title = QtWidgets.QLabel(f"#{slot + 1}")
        title.setObjectName("knobTitle")
        title.setFixedWidth(28)
        slider = Knob(0, self._SLIDER_MAX,
                      0 if slot == 0 else self._SLIDER_MAX)   # #60: #1 — мин, #2 — макс
        slider.setEnabled(False)                 # по умолчанию слот выкл (#59)
        vlabel = QtWidgets.QLabel("—")
        vlabel.setObjectName("knobValue")
        vlabel.setMinimumWidth(80)
        vlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        check = QtWidgets.QToolButton()
        check.setCheckable(True)
        check.setText(tr("выкл"))                # дефолт — все выкл (#59)
        check.setObjectName("knobToggle")
        for col, wdg in ((0, title), (1, slider), (2, vlabel), (3, check)):
            grid.addWidget(wdg, row, col)
        self._rows[(axis, slot)] = {"check": check, "slider": slider, "label": vlabel}
        check.toggled.connect(
            lambda on, b=check, s=slider, a=axis, sl=slot: self._on_slot_toggle(on, b, s, a, sl))
        slider.valueChanged.connect(lambda _=0, a=axis, s=slot: self._emit(a, s))

    def _on_slot_toggle(self, on, btn, slider, axis, slot) -> None:
        """Переключение слота (#59): текст вкл/выкл, гашение движка, переиздание planeChanged."""
        btn.setText(tr("вкл") if on else tr("выкл"))
        slider.setEnabled(bool(on))
        self._emit(axis, slot)

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

    def retranslate(self) -> None:
        """Задача #169: подписи панели сечений на текущем языке."""
        self._caption.setText(tr("Секущие плоскости (2 на ось)"))
        for axis, head in self._axis_heads.items():
            head.setText(tr(_AXIS_LABEL[axis]))
        for r in self._rows.values():
            chk = r["check"]
            chk.setText(tr("вкл") if chk.isChecked() else tr("выкл"))