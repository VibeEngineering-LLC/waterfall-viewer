"""
Фон и вычитание фона для waterfall-спектрограмм (Задача #96).

Фон хранится как поканальная скорость счёта (cps на канал), выровненная по энергетической
оси целевой спектрограммы. Два источника:
  * диапазон времени текущего измерения — средняя скорость по выбранным срезам;
  * отдельный файл — суммарная скорость; при иной калибровке интерполируется по спектральной
    плотности (cps/кэВ) и пересчитывается на ширину каналов целевой спектрограммы.

Вычитание идёт на уровне отсчётов: counts[t,ch] - bg_cps[ch]*live_time[t]. Остаток ЗНАКОВЫЙ
(без клипа к 0, Задача #134) — только так интеграл по времени сокращается точно: когда фон =
самому спектру, Σ_t (counts − bg_cps·lt) = total_counts − bg_cps·total_lt = 0 поканально.
Отрицательные ячейки гасятся к нулю уже на этапе ОТОБРАЖЕНИЯ (zscale._base_transform →
np.maximum(a,0)), поэтому 3D/2D-виды выглядят как прежде, а суммарный (интегральный) спектр
среза корректно уходит в ноль.
"""
from __future__ import annotations

import numpy as np

from awf.model.spectrogram import Spectrogram


def background_from_range(sg, t_lo: int, t_hi: int) -> np.ndarray:
    """Поканальный фон (cps) как средняя скорость по срезам [t_lo:t_hi] текущей спектрограммы:
    bg[ch] = sum(counts[t_lo:t_hi, ch]) / sum(live_time[t_lo:t_hi]). Калибровка та же — без
    интерполяции. Длина результата = sg.n_channels."""
    lo = max(0, int(t_lo))
    hi = min(sg.n_slices, int(t_hi))
    if hi <= lo:
        raise ValueError("background_from_range: пустой диапазон времени")
    gross = sg.counts[lo:hi, :].sum(axis=0, dtype=np.float64)
    lt = float(np.asarray(sg.live_time_s, dtype=np.float64)[lo:hi].sum())
    if lt <= 0.0:
        raise ValueError("background_from_range: нулевое живое время окна")
    return gross / lt


def background_window_like(bg_counts, bg_lt, target_lt) -> np.ndarray:
    """Задача #139: сырой фоновый спектр той же статистики, что окно образца. Суммируем сырые
    отсчёты фонового блока за живое время target_lt (окно образца) → совпадающий пуассонов шум."""
    counts = np.asarray(bg_counts, dtype=np.float64)
    lt = np.asarray(bg_lt, dtype=np.float64).ravel()
    if counts.ndim != 2 or counts.shape[0] == 0 or lt.size != counts.shape[0]:
        raise ValueError("background_window_like: неверная форма фонового блока")
    tgt = float(target_lt or 0.0)
    total = float(lt.sum())
    if tgt <= 0.0 or total <= 0.0:
        return counts.sum(axis=0)
    if tgt >= total:
        return counts.sum(axis=0) * (tgt / total)
    k = min(int(np.searchsorted(np.cumsum(lt), tgt, side="left")) + 1, counts.shape[0])
    win = counts[:k, :].sum(axis=0)
    win_lt = float(lt[:k].sum())
    return win * (tgt / win_lt) if win_lt > 0.0 else win


def tile_background_block(bg_counts, bg_lt, n_slices: int, phase: int = 0):
    """Задача #149: транслировать (циклически скопировать) сырой фоновый блок на всю временную
    шкалу из n_slices срезов. Фон аппроксимируется КОПИРОВАНИЕМ сырых срезов, не усреднением
    (директива #138): срез t получает клон bg-среза (t − phase) mod m. phase — позиция начала
    фонового участка в целевой шкале: срезы ВНУТРИ участка получают клонами самих себя, поэтому
    самовычет фонового участка остаётся точным нулём (#147). Возвращает (counts_full, lt_full)."""
    counts = np.asarray(bg_counts, dtype=np.float64)
    lt = np.asarray(bg_lt, dtype=np.float64).ravel()
    if counts.ndim != 2 or counts.shape[0] == 0 or lt.size != counts.shape[0]:
        raise ValueError("tile_background_block: неверная форма фонового блока")
    idx = (np.arange(int(n_slices)) - int(phase)) % counts.shape[0]
    return counts[idx], lt[idx]


def _channel_widths(energies) -> np.ndarray:
    """Ширина каналов по энергии (кэВ/канал) через градиент монотонной шкалы энергий."""
    e = np.asarray(energies, dtype=np.float64)
    if e.size < 2:
        return np.ones_like(e)
    return np.abs(np.gradient(e))


def background_from_spectrogram(bg_sg, target_sg) -> np.ndarray:
    """Поканальный фон (cps) из отдельной спектрограммы, выровненный по энергии target_sg.
    Считаем спектральную плотность скорости (cps/кэВ) фон-файла, интерполируем по энергии на
    каналы цели и умножаем на ширину каналов цели. При совпадающей калибровке сводится к
    поканальному cps. Каналы вне диапазона фон-файла -> 0 (без экстраполяции). Длина = цель."""
    e_bg = np.asarray(bg_sg.energies(), dtype=np.float64)
    e_t = np.asarray(target_sg.energies(), dtype=np.float64)
    lt_bg = float(np.asarray(bg_sg.live_time_s, dtype=np.float64).sum())
    if lt_bg <= 0.0:
        raise ValueError("background_from_spectrogram: нулевое живое время фон-файла")
    bg_cps = bg_sg.counts.sum(axis=0, dtype=np.float64) / lt_bg     # cps на канал фон-файла
    dens_bg = bg_cps / _channel_widths(e_bg)                        # cps/кэВ
    xp, fp = e_bg, dens_bg
    if xp.size >= 2 and xp[0] > xp[-1]:                             # шкала по убыванию -> развернуть
        xp, fp = xp[::-1], fp[::-1]
    dens_t = np.interp(e_t, xp, fp, left=0.0, right=0.0)
    return dens_t * _channel_widths(e_t)


def subtract_background(sg, bg_cps, bg_raw=None) -> Spectrogram:
    """Новая спектрограмма с вычтенным фоном: counts[t,ch] - bg_cps[ch]*live_time[t] (Задача #96).
    Остаток ЗНАКОВЫЙ, без клипа к 0 (Задача #134): поканальный клип асимметрично разрушал
    сокращение и оставлял ложный ~18 % положительный остаток при фон=спектр (должно быть 0).
    Без клипа Σ_t остаток = total_counts − bg_cps·total_lt = 0 поканально при фон=весь файл.
    Калибровка/временные оси сохраняются; отрицательные ячейки гасятся к 0 на отображении
    (zscale._base_transform), поэтому 3D/2D-виды не меняются, а интегральный спектр уходит в 0.

    Задача #147: bg_raw=(counts_блок, lt_блок) той же формы, что sg.counts (фон = тот же файл
    или весь диапазон), — вычет поячеечный: counts − bg_counts·(lt/bg_lt). При самовычете
    остаток ТОЧНЫЙ ноль в каждой ячейке (как совпадающие простыни #139/#141), а не только в
    интеграле по времени; усреднённый bg_cps не размывает пуассонов шум образца. Иная форма
    блока (под-диапазон, чужая сетка) — прежний путь через средний bg_cps."""
    bg = np.asarray(bg_cps, dtype=np.float64).ravel()
    if bg.size != sg.n_channels:
        raise ValueError("subtract_background: длина bg_cps != числу каналов")
    lt = np.asarray(sg.live_time_s, dtype=np.float64)
    if bg_raw is not None and np.asarray(bg_raw[0]).shape == sg.counts.shape:
        bg_counts = np.asarray(bg_raw[0], dtype=np.float64)
        bg_lt = np.asarray(bg_raw[1], dtype=np.float64).ravel()
        safe = np.where(bg_lt > 0.0, bg_lt, np.inf)
        sub = sg.counts.astype(np.float64) - bg_counts * (lt / safe)[:, None]
    else:
        sub = sg.counts.astype(np.float64) - bg[None, :] * lt[:, None]
    return Spectrogram(counts=sub, calibration=sg.calibration,
                       time_offsets_s=sg.time_offsets_s, real_time_s=sg.real_time_s,
                       live_time_s=sg.live_time_s, t0_iso=sg.t0_iso, source_path=sg.source_path)


def gate_significant_net(sub_sg, bg_cps, k: float = 3.0) -> Spectrogram:
    """Значимостное гашение нетто для ОТОБРАЖЕНИЯ 3D/2D-водопада (Задача #136): ячейки,
    статистически неотличимые от нуля (|net| < k·σ, σ≈√(bg·lt) — критический уровень Currie),
    обнуляются. При фон=весь файл нетто каждого среза — пуассонов счётный шум √N (сокращается
    только в интеграле по времени), поэтому гаснет ~весь водопад: самовычет читается как ноль,
    как вычет одинаковых спектров в BecqMoni. Реальный значимый нетто-пик (|net| ≫ kσ) переживает.
    Это ОТДЕЛЬНАЯ копия для 3D/2D; знаковую модель среза (#134, интегральный спектр = точный ноль)
    не трогаем. Возвращает новую Spectrogram; вход не мутируется."""
    net = np.asarray(sub_sg.counts, dtype=np.float64)
    bg = np.asarray(bg_cps, dtype=np.float64).ravel()
    if bg.size != sub_sg.n_channels:
        raise ValueError("gate_significant_net: длина bg_cps != числу каналов")
    lt = np.asarray(sub_sg.live_time_s, dtype=np.float64)
    sigma = np.sqrt(np.maximum(bg[None, :] * lt[:, None], 1.0))
    gated = np.where(np.abs(net) >= float(k) * sigma, net, 0.0)
    return Spectrogram(counts=gated, calibration=sub_sg.calibration,
                       time_offsets_s=sub_sg.time_offsets_s, real_time_s=sub_sg.real_time_s,
                       live_time_s=sub_sg.live_time_s, t0_iso=sub_sg.t0_iso,
                       source_path=sub_sg.source_path)

def region_averaged_net(sg, bg_cps, segments) -> Spectrogram:
    """Задача #137: посегментно усреднённый нетто для 3D/2D-водопада (режим «усреднённый вычет»).

    Корень #137: на водопаде образец (одиночный срез) — пуассонов счётный шум √N, а фон bg_cps·lt
    сглажен (среднее по всему файлу); «способ отображения разный», отсюда и вычет разный. Здесь
    образец усредняется ТАК ЖЕ, как фон, но внутри каждого УЧАСТКА СТАБИЛЬНОСТИ (segment_by_time,
    #131): avg_smp_i[ch] = Σ_{t∈Ri} counts / Σ_{t∈Ri} lt (cps/канал). Нетто участка
    net_rate_i = avg_smp_i − bg_cps (обе стороны — скорости, «одинаковый способ»); каждый срез
    t∈Ri получает net_rate_i·lt[t] — образец и фон сглажены одинаково.

    Свойства: (1) шум усреднён честно, поэтому гейт 3σ (#136) в этом режиме не нужен; (2) интеграл
    нетто по КАЖДОМУ участку сохраняется точно: Σ_{Ri} out = Σ_{Ri}(counts − bg·lt) — знаковая
    модель среза (#134) не нарушена; (3) при фон=весь файл и одном стационарном участке
    avg_smp≡bg_cps ⇒ out≡0 поканально: самовычет одинаковых спектров = точный ноль (как BecqMoni).
    Возвращает новую Spectrogram; вход не мутируется."""
    bg = np.asarray(bg_cps, dtype=np.float64).ravel()
    if bg.size != sg.n_channels:
        raise ValueError("region_averaged_net: длина bg_cps != числу каналов")
    counts = np.asarray(sg.counts, dtype=np.float64)
    lt = np.asarray(sg.live_time_s, dtype=np.float64)
    out = np.zeros_like(counts)
    for seg in segments:
        lo = max(0, int(seg.t_lo)); hi = min(sg.n_slices, int(seg.t_hi))
        if hi <= lo:
            continue
        lt_sum = float(lt[lo:hi].sum())
        if lt_sum <= 0.0:
            continue
        avg_smp = counts[lo:hi, :].sum(axis=0) / lt_sum       # cps/канал: образец, усреднённый по участку
        net_rate = avg_smp - bg                                # cps/канал: нетто (образец − фон)
        out[lo:hi, :] = net_rate[None, :] * lt[lo:hi, None]    # обратно в отсчёты по live-time среза
    return Spectrogram(counts=out, calibration=sg.calibration,
                       time_offsets_s=sg.time_offsets_s, real_time_s=sg.real_time_s,
                       live_time_s=sg.live_time_s, t0_iso=sg.t0_iso, source_path=sg.source_path)

