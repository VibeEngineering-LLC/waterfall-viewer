SYSTEM / OUTPUT CONTRACT (читай дословно):
- Сгенерируй ОДИН файл Python: содержимое модуля `awf/model/spectrogram.py`.
- Выводи ТОЛЬКО сырой код Python. БЕЗ markdown-ограждений (```), БЕЗ пояснений, БЕЗ текста до/после.
- Первая строка вывода — строковый docstring модуля (тройные кавычки).
- Комментарии и docstrings — на русском. Имена идентификаторов — на английском.
- Только стандартная библиотека + numpy. НЕ импортируй lxml, pandas, scipy, matplotlib.
- Python 3.12, стиль: `from __future__ import annotations`, type hints, dataclasses.
- НЕ оставляй TODO/pass-заглушек. Каждая функция полностью реализована.

НАЗНАЧЕНИЕ МОДУЛЯ:
Численное ядро вьюера waterfall-спектрограмм гамма-спектрометра. Только данные и математика
(срезы, сечения, выборки, LOD-даунсэмплинг). Без чтения файлов и без GUI.

================================================================================
КЛАСС 1: Calibration  (dataclass, frozen=True)
================================================================================
Энергетическая калибровка: полином по индексу канала.
  E(ch) = coeffs[0] + coeffs[1]*ch + coeffs[2]*ch^2 + ... (возрастающий порядок степеней)

Поля:
  coeffs: np.ndarray  — 1D float64, коэффициенты полинома в порядке возрастания степени.

__post_init__:
  - привести coeffs к np.asarray(coeffs, dtype=np.float64).ravel();
    т.к. dataclass frozen — присваивать через object.__setattr__(self, "coeffs", ...).
  - если массив пуст — поднять ValueError("Calibration: пустые коэффициенты").

Методы:
  energy_of_channel(self, ch) -> np.ndarray | float
      ch может быть скаляром (int/float) или array-like.
      Реализация: import numpy.polynomial.polynomial as P; return P.polyval(np.asarray(ch, dtype=np.float64), self.coeffs)
      (polyval из numpy.polynomial использует ВОЗРАСТАЮЩИЙ порядок степеней — это правильно для наших coeffs.)
      Если на входе был скаляр (np.ndim(ch)==0) — вернуть float(результат).

  energies(self, n_channels: int) -> np.ndarray
      Вернуть 1D float64 массив длины n_channels: энергии для каналов 0..n_channels-1.
      ch = np.arange(n_channels, dtype=np.float64); return self.energy_of_channel(ch) (как np.ndarray).
      Если n_channels <= 0 — поднять ValueError.

  channel_of_energy(self, energy, n_channels: int) -> np.ndarray | float
      Обратное преобразование энергия->индекс канала через поиск по сетке энергий.
      grid = self.energies(n_channels)  # длина n_channels
      ВНИМАНИЕ: grid должен быть монотонно неубывающим, иначе searchsorted некорректен.
        mono = bool(np.all(np.diff(grid) >= 0))
        если не mono — отсортировать: order = np.argsort(grid); grid_sorted = grid[order];
        иначе order = None, grid_sorted = grid.
      e = np.asarray(energy, dtype=np.float64)
      idx = np.searchsorted(grid_sorted, e)  # индексы вставки
      idx = np.clip(idx, 0, n_channels - 1).astype(np.int64)
      если order is not None: idx = order[idx]
      Если на входе скаляр — вернуть int(idx); иначе вернуть idx (np.int64 массив).

  @classmethod from_coeff_string(cls, text: str) -> "Calibration"
      Парсит строку вида "3.52 0.38 2.6e-05 ..." (разделители — пробелы).
      vals = [float(x) for x in text.split()]; если пусто — ValueError.
      return cls(coeffs=np.asarray(vals, dtype=np.float64)).

================================================================================
КЛАСС 2: Spectrogram  (обычный класс, НЕ frozen)
================================================================================
Хранит матрицу counts[time_index, channel_index] (2D) и временные оси. Канальная ось неявная: 0..n_channels-1.

Конструктор __init__(self, *,
        counts: np.ndarray,             # 2D, shape (n_slices, n_channels), целочисленный dtype
        calibration: Calibration,
        time_offsets_s: np.ndarray,     # 1D float64 (n_slices,), секунды от начала записи (StartDateTime[i]-StartDateTime[0])
        real_time_s: np.ndarray,        # 1D float64 (n_slices,), может содержать np.nan
        live_time_s: np.ndarray,        # 1D float64 (n_slices,), может содержать np.nan
        t0_iso: str | None = None,      # ISO-строка абсолютного StartDateTime[0] (для подписи оси), либо None
        source_path: str | None = None):
  Валидация (поднимать ValueError с понятным русским текстом):
    - counts.ndim == 2;
    - time_offsets_s/real_time_s/live_time_s — 1D и их длина == counts.shape[0];
    - n_slices = counts.shape[0] >= 1 и n_channels = counts.shape[1] >= 1.
  Привести оси к float64 (np.asarray(..., dtype=np.float64)). counts оставить как есть (не копировать).
  Сохранить все поля как атрибуты.

Свойства (@property):
  n_slices  -> int  (counts.shape[0])
  n_channels -> int (counts.shape[1])

Методы анализа:

  energies(self) -> np.ndarray
      return self.calibration.energies(self.n_channels)   # длина n_channels

  energy_spectrum(self, i: int) -> np.ndarray
      Один временной срез = спектр в момент i. Вернуть counts[i] (1D длины n_channels).
      i валидировать: -n_slices <= i < n_slices (разрешить отрицательную индексацию питона).
      При выходе за границы — IndexError.

  channel_time_series(self, j: int) -> np.ndarray
      Сечение по одному каналу: как канал j меняется во времени. Вернуть counts[:, j] (1D длины n_slices).
      Валидация j аналогично.

  band_time_series(self, ch_lo: int, ch_hi: int) -> np.ndarray
      Временной профиль суммы по диапазону каналов [ch_lo, ch_hi) (полуинтервал, hi не включается).
      Нормализовать: lo, hi = sorted((ch_lo, ch_hi)); lo=max(0,lo); hi=min(n_channels,hi); если hi<=lo -> ValueError.
      return counts[:, lo:hi].sum(axis=1, dtype=np.int64)   # 1D длины n_slices, dtype int64 (защита от переполнения)

  energy_band_time_series(self, e_lo: float, e_hi: float) -> np.ndarray
      То же, но границы заданы в кэВ. Перевести энергии в индексы каналов через calibration.channel_of_energy,
      затем band_time_series.
      lo_e, hi_e = sorted((e_lo, e_hi))
      ch_lo = int(self.calibration.channel_of_energy(lo_e, self.n_channels))
      ch_hi = int(self.calibration.channel_of_energy(hi_e, self.n_channels)) + 1  # +1 чтобы верхняя граница вошла
      return self.band_time_series(ch_lo, ch_hi)

  sum_spectrum(self, t_lo: int | None = None, t_hi: int | None = None) -> np.ndarray
      Интегральный спектр (сумма по времени) по выборке срезов [t_lo, t_hi) (полуинтервал).
      По умолчанию (None,None) — по всем срезам.
      lo = 0 if t_lo is None else max(0, t_lo); hi = n_slices if t_hi is None else min(n_slices, t_hi)
      если hi<=lo -> ValueError.
      return counts[lo:hi, :].sum(axis=0, dtype=np.int64)   # 1D длины n_channels

  total_spectrum(self) -> np.ndarray
      return self.sum_spectrum(None, None)

  roi_sum(self, t_lo: int, t_hi: int, ch_lo: int, ch_hi: int) -> int
      Прямоугольная выборка: суммарное число отсчётов в окне [t_lo,t_hi) x [ch_lo,ch_hi).
      Нормализовать обе пары как выше (sorted + clip к границам), при пустом окне -> ValueError.
      return int(counts[t_lo:t_hi, ch_lo:ch_hi].sum(dtype=np.int64))

  downsample(self, max_time: int, max_chan: int, method: str = "max")
        -> tuple[np.ndarray, np.ndarray, np.ndarray]
      LOD-агрегация для отрисовки на GPU при больших записях (n_slices может быть десятки тысяч).
      Возвращает (counts_ds, t_centers, ch_centers):
        counts_ds  — 2D float64, shape (nt, nc), nt<=max_time, nc<=max_chan;
        t_centers  — 1D float64 длины nt: центр каждого временного бина в секундах (из time_offsets_s);
        ch_centers — 1D float64 длины nc: центр каждого канального бина в кэВ (из energies()).
      method: "max" (сохраняет пики — по умолчанию) или "sum" (сохраняет суммарные отсчёты). Иначе ValueError.

      АЛГОРИТМ (реализуй ДОСЛОВНО, это численно-критичная часть):
        ns, ncyh = self.n_slices, self.n_channels
        nt = min(int(max_time), ns); nc = min(int(max_chan), ncyh)
        nt = max(1, nt); nc = max(1, nc)
        # Границы бинов — целочисленные индексы, строго возрастающие:
        t_edges = np.unique(np.linspace(0, ns, nt + 1).astype(np.int64))
        ch_edges = np.unique(np.linspace(0, ncyh, nc + 1).astype(np.int64))
        # из-за np.unique фактическое число бинов = len(edges)-1 (может стать меньше при крошечных размерах) — это ок.
        t_starts = t_edges[:-1]; ch_starts = ch_edges[:-1]
        data = self.counts.astype(np.float64, copy=False)
        if method == "max":
            red = np.maximum
        elif method == "sum":
            red = np.add
        else:
            raise ValueError(...)
        # Двухпроходная агрегация через reduceat: сначала по оси времени (0), потом по каналам (1).
        step1 = red.reduceat(data, t_starts, axis=0)        # shape (len(t_starts), ncyh)
        counts_ds = red.reduceat(step1, ch_starts, axis=1)  # shape (len(t_starts), len(ch_starts))
        # Центры бинов:
        t_off = self.time_offsets_s
        en = self.energies()
        t_centers = np.empty(len(t_starts), dtype=np.float64)
        ch_centers = np.empty(len(ch_starts), dtype=np.float64)
        for k in range(len(t_starts)):
            a = int(t_edges[k]); b = int(t_edges[k + 1])
            t_centers[k] = float(t_off[a:b].mean()) if b > a else float(t_off[min(a, ns - 1)])
        for k in range(len(ch_starts)):
            a = int(ch_edges[k]); b = int(ch_edges[k + 1])
            ch_centers[k] = float(en[a:b].mean()) if b > a else float(en[min(a, ncyh - 1)])
        return counts_ds, t_centers, ch_centers

      ВАЖНО про reduceat: np.maximum.reduceat(a, idx, axis) для каждого i агрегирует срез
      a[idx[i]:idx[i+1]] вдоль оси (последний бин — до конца). Поскольку t_starts/ch_starts получены
      из np.unique(...) — они строго возрастают, поэтому квирк reduceat с равными индексами НЕ возникает.

ЗАМЕЧАНИЕ ПО КОРРЕКТНОСТИ:
  - Все суммы делать с dtype=np.int64 (counts uint16 — иначе переполнение при суммировании по тысячам срезов).
  - Нигде не модифицировать self.counts на месте; astype(..., copy=False) допускает no-op при float, но
    counts целочисленный, так что astype вернёт копию (это намеренно, исходник не трогаем).
