# План внедрения: спектроскопия и аналитика waterfall (v2+)

Документ описывает поэтапное развитие `atomspectra-waterfall-viewer` от текущего просмотрщика
(3D/2D + подсветка нуклидов) до анализатора в духе iZotope Insight для гамма-спектрометрии.

**Принцип:** MVP-first, поэтапно. Каждая фаза самодостаточна и даёт работающий инкремент.
**GPS / пространственная ось — отложены** (решение оператора 2026-06-25): см. §«Отложено».

Источник переносимых алгоритмов — проект **SpectraVibe** (`gamma-spectrum-analysis`,
`D:\GoogleDrive\Дозиметрия\ИИ\1 Скилы\0_Work\gamma-spectrum-analysis`). Все ссылки `scripts/gamma/...`
ниже — относительно его корня. SpectraVibe **только читаем как референс**, не модифицируем.

---

## 0. Архитектурная отправная точка

### Что уже есть (точки расширения)

| Слой | Файл | Что предоставляет | Куда расширяем |
|---|---|---|---|
| Модель | `awf/model/spectrogram.py` | `Calibration` (полином ascending, `numpy.polynomial.polyval`), `Spectrogram` (counts[t,ch], срезы/сечения/ROI-суммы, `downsample`) | новые численные методы: пики, фон, площадь, MDA, FWHM |
| Загрузчики | `awf/io/{n42,aswf,rcspg}_loader.py` | counts + калибровка + временные оси | извлечение GPS-трека (отложено); IAEA-fetcher |
| Библиотека | `awf/io/nuclide_lib.py` (`GammaLine`, `Nuclide`), `awf/data/nuclides.json` (21 нуклид) | парсер LSRM, JSON round-trip | категории нуклидов; IAEA-загрузка; матчинг по энергии |
| 3D | `awf/ui/view3d.py` (`GLViewWidget` + `GLSurfacePlotItem`, colormap `inferno`, LOD) | поверхность Время×Энергия×Отсчёты | сечения-плоскости, оси с подписями, энергетические лучи, цветовая схема |
| 2D | `awf/ui/panels.py` (`HeatmapPanel` + `RectROI`, `SlicePanel`) | карта + ROI, спектр + временной ряд + маркеры нуклидов | фон-зона, найденные пики, контуры |
| Контраст | `awf/ui/zscale.py` (`apply_z_scale`: linear/sqrt/log10) | Z-шкала 2D/3D | усиление/гамма/клиппинг, десатурация |
| Окно | `awf/ui/main_window.py` (QThread-загрузка) | диспетчер форматов, статус | новые доки/тулбары/действия |

### Центральный архитектурный факт

Модель данных **двумерная: `время × энергия`**. Пространственной (координатной/GPS) оси нет
(`awf/model/spectrogram.py` — в `Spectrogram` только `counts`, `time_offsets_s`, `calibration`).
Поэтому фичи ТЗ, требующие оси координат — **Volume Rendering** (Y=координата), **Peak Map**
(координата→активность), **3D terrain + GPS** — переносятся в отложенную Фазу 5. Где можно,
**ось времени служит осью сканирования** (градиент по времени, кластеризация временных срезов).

### Сквозные правила реализации

- **IRON MODE:** любой код Python ≥25 строк — через спецификацию `scripts/ollama/_spec_*.md` →
  кодогенерация Ollama `qwen3-coder:30b` → валидация (`py_compile` + pytest). Мелкие правки <25 строк
  и `.md`/`.json` — напрямую. `.delegation_guard_off` в этом проекте **не создавать**.
- **Численное ядро — без Qt:** все методы анализа кладём в `awf/model/` (или новый `awf/analysis/`),
  чтобы покрыть тестами без графики (как уже сделано для `spectrogram.py`).
- **Тест-first для портов:** опорные значения — независимый оракул (синтетический спектр с
  известными пиками), как в существующих тестах загрузчиков.
- **Анти-галлюцинация:** при портировании сверять каждую формулу с исходной строкой SpectraVibe.
- **Порядок коэффициентов калибровки:** вьюер и SpectraVibe оба ascending (low→high) — совместимо,
  но при каждом порте проверять порядок явным тестом.

---

## 1. Карта фич ТЗ → фазы

| # | Фича ТЗ | Фаза | Точка расширения | Метод из SpectraVibe |
|---|---|---|---|---|
| 9 | Поиск пиков | **1** | `awf/analysis/peaks.py` (новый) | `peaks/search.py` `mariscotti_search()` |
| 7 | Зона фона по времени + аппроксимация на всю запись | **1** | `Spectrogram` + `panels.py` | `peaks/area.py` Cowell, `background_options.py` |
| 8 | Вычитание фонового массива | **1** | `Spectrogram.subtract_background()` | — (своя, тривиальная) |
| 4 | Регулировка интенсивности/контраста | **1** | `awf/ui/zscale.py` | — (расширение apply_z_scale) |
| 10 | Библиотека МАГАТЭ (категории, кор./долгоживущие) | **2** | `awf/io/iaea_fetcher.py` (новый), `nuclides.json` | `data/iaea_fetcher.py`, `targeted_libraries.py`, `nuclide_library.py:199` |
| — | Умная идентификация под найденные пики | **2** | `awf/analysis/identify.py` (новый) | `identification/identify.py`, `proportionality.py` |
| 1 | Сечения-плоскости (2 на ось) | **3** | `awf/ui/view3d.py` | — (UI) |
| 2 | Градуировка осей (подписи) | **3** | `awf/ui/view3d.py` (`GLTextItem`) | — (UI) |
| 3 | Вертикальные энергетические лучи до линии спектра | **3** | `awf/ui/view3d.py` (`GLLinePlotItem`) | — (UI, как референс Insight) |
| 5 | Цветовая схема iZotope Insight | **3** | `awf/ui/view3d.py`, `panels.py` (`pg.ColorMap`) | — (UI) |
| 6 | Десатурация массива + подсветка выбранных пиков | **3** | `zscale.py` + overlay | — (UI) |
| — | Slice Analysis (спектр в точке / профиль по маршруту времени) | **4** | `panels.py` (частично есть) | — |
| — | Contour Plot (изолинии) | **4** | `panels.py` (`pg.IsocurveItem`) | — |
| — | Анализ градиента d(счёт)/d(время) | **4** | `awf/analysis/gradient.py` | — (numpy.gradient) |
| — | Деконволюция мультиплетов | **4** | `awf/analysis/deconvolve.py` | `peaks/deconvolve.py` `multi_gaussian_fit()` (scipy) |
| — | PCA / t-SNE / UMAP по временным срезам | **4** | `awf/analysis/decomposition.py` | — (scikit-learn) |
| — | Спектральная кластеризация (K-Means/DBSCAN/HDBSCAN) | **4** | `awf/analysis/cluster.py` | — (scikit-learn) |
| — | MDA по ISO 11929 (нетто/предел обнаружения) | **2–4** | `awf/analysis/mda.py` | `identification/mda.py:101` |
| — | FWHM(E) модель разрешения | **2** | `awf/analysis/fwhm.py` | `calibration/fwhm_fit.py` |
| — | Volume Rendering (Y=координата) | **5 (отлож.)** | требует прострой оси | — |
| — | Peak Map (координата→активность) | **5 (отлож.)** | требует прострой оси | — |
| — | 3D terrain + спектры + GPS | **5 (отлож.)** | требует GPS-трека | извлечение из `.rcspg`/`.n42` |

---

## 2. Фаза 1 — MVP: спектроскопическое ядро + контраст

**Цель:** дать измерительную ценность поверх существующих видов — найти пики, оценить и вычесть фон,
гибко настроить контраст. Без новых тяжёлых зависимостей (**только numpy + stdlib**).

### 2.1 Поиск пиков — `awf/analysis/peaks.py` (новый)

- **Порт:** `scripts/gamma/peaks/search.py` `mariscotti_search()` (стр. 261–450) — фильтр второй
  производной Гауссиана (Mariscotti 1967) + критерий значимости Currie L_C. **Чистый numpy.**
- **Сигнатура:** `find_peaks(counts_1d, fwhm_channels, sigma_threshold=3.0) -> list[FoundPeak]`
  (`FoundPeak`: channel, energy, height, fwhm_channels, significance, area_estimate).
- **Применение к waterfall:** искать пики в спектре выбранного среза/окна (`Spectrogram.sum_spectrum`)
  и в `total_spectrum`. Опционально — по каждому срезу для будущей peak-карты по времени (Фаза 4).
- **Предв. площадь:** `2.507·σ·height` (`search.py:432`).

### 2.2 Фон/континуум — `awf/analysis/continuum.py` (новый)

- **Порт A (Cowell):** `scripts/gamma/peaks/area.py:77–238` `cowell_area()` — полиномиальная подгонка
  крыльев ROI (`numpy.polyfit/polyval`), нетто = gross − baseline. **Чистый numpy.**
- **Порт B (сглаженная ступень):** `scripts/gamma/peaks/background_options.py:40–89` `smoothed_step_bg()`
  (Будыка 7.29): `B(E)=bg_r+(bg_l−bg_r)·0.5·erfc((E−E₀)/(σ√2))+slope·(E−E₀)`. Зависимость
  `scipy.special.erfc` имеет fallback `math.erfc` — **берём fallback, scipy не вводим в Фазе 1.**

### 2.3 Зона фона по времени + вычитание (ТЗ 7, 8)

- **`Spectrogram.background_spectrum(t_lo, t_hi)`** — усреднённый спектр фонового окна времени
  (нормировка на live-time). Аппроксимация «на всю запись» = вычесть этот опорный спектр из каждого среза.
- **`Spectrogram.subtract_background(bg_spectrum, scale='live_time')`** → новый `Spectrogram`
  с counts_net = clip(counts − bg·k_t, 0, …), где k_t масштабирует фон по времени среза. Тривиально, своя.
- **UI:** в `HeatmapPanel` — второй ROI (или режим) «выделить фон по времени»; в тулбаре — переключатель
  «вычесть фон». Результат отражается в `SlicePanel` (нетто-спектр) и на карте.

### 2.4 Контраст/интенсивность (ТЗ 4) — `awf/ui/zscale.py`

- Расширить `apply_z_scale`: добавить **gain** (множитель), **gamma** (степенная коррекция),
  **percentile-clip** (нижний/верхний перцентиль для отсечки выбросов перед нормировкой).
- UI: слайдеры в тулбаре «Вид» (gain/gamma/clip), живой апдейт 2D и 3D. <25 строк правок — напрямую.

### 2.5 Тест-план Фазы 1

- `tests/test_peaks.py`: синтетический спектр (известные гауссианы на ровном/наклонном фоне) →
  `find_peaks` находит ровно их, центры в пределах ±0.5 канала, значимость монотонна по высоте,
  спайк-фильтр отсекает узкие выбросы.
- `tests/test_continuum.py`: Cowell на линейном фоне даёт нетто≈0 для чистого фона и площадь
  гауссианы в пределах 3% от аналитической; `smoothed_step_bg` непрерывна и монотонна.
- `tests/test_background_subtract.py`: `subtract_background` зануляет идентичный фон, не уводит в минус,
  правильно масштабирует по live-time.

**Новые зависимости Фазы 1: нет** (numpy + stdlib).

---

## 3. Фаза 2 — Библиотека МАГАТЭ + умная идентификация (ТЗ 10)

**Цель:** заменить статичные 21 нуклид на расширяемую библиотеку с загрузкой из IAEA и категориями;
подсказывать нуклиды под найденные пики.

### 3.1 IAEA-fetcher — `awf/io/iaea_fetcher.py` (новый)

- **Порт 1-в-1:** `scripts/gamma/data/iaea_fetcher.py` — `fetch_iaea_gamma_lines(nuclide)` →
  `https://www-nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides=<n>&rad_types=g`, парсинг CSV
  (ENSDF), **атомарный** кэш (`os.replace`), офлайн-воспроизводимость. **Только stdlib (`urllib`+`csv`).**
- **Безопасность:** перенести `_normalize_nuclide_name` с fail-loud (SEC-02, `iaea_fetcher.py:73`) —
  защита от URL-инъекции/path-traversal в имени нуклида.
- **Сеть — по требованию пользователя** (кнопка «Добавить нуклид из IAEA»), не автоматически; кэш
  кладём в `awf/data/iaea_cache/` (в `.gitignore`).
- **Конверсия:** `merge_iaea_into_internal()` (`iaea_fetcher.py:304`) → `{"lines": [[E,I,dI],…]}` —
  ровно формат нашего `nuclides.json`.
- ⚠ **Сетевой риск (§7 доктрины):** запрос с жёстким `timeout` (есть, 30 с) и в фоне (QThread) —
  IAEA отдаёт 403 на дефолтный UA (UA уже выставлен в порте). Не блокировать UI.

### 3.2 Категории нуклидов (ТЗ 10) — `awf/data/nuclides.json` + `nuclide_lib.py`

- Добавить в каждую запись поле `category` ∈ {`natural`, `technogenic`, `medical`, `fission`} и
  `lifetime` ∈ {`short`, `long`} (порог по T½, напр. 60 сут).
- **Основа таксономии — пресеты SpectraVibe** `scripts/gamma/data/targeted_libraries.py`:
  `ENVIRONMENTAL`(NORM/космогенные/техногенный fallout) → natural; `NPP` fission products
  (I-131/132, Cs-134/137, Ru-103/106, Ba-140…) → fission; актиниды/активация → technogenic;
  I-125/I-131/Tc-99m/Ga-67… → medical. Перебукетизировать в наши 4 категории.
- В `Nuclide`/`GammaLine` добавить поля `category`, `lifetime` (расширение dataclass — <25 строк, напрямую).

### 3.3 Матчинг и идентификация — `awf/analysis/identify.py` (новый)

- **Порт матчинга:** `scripts/gamma/data/nuclide_library.py:199` `lookup_by_energy(E, tol)` — линии
  библиотеки в окне ±tol, сортировка по |Δ|. Окно tol ∝ FWHM(E) (см. 3.4).
- **Порт логики (lite):** `scripts/gamma/identification/identify.py` — 3-шаг (Lsrm §6): detection по
  characteristic line, line-matching, индекс уверенности; `proportionality.py` — проверка
  соотношения площадей пиков против библиотечных интенсивностей (`ratio_tolerance≈3.0`), приоритеты
  редких изотопов. **Чистая логика, numpy + словари.**
- **UI:** на найденные пики (Фаза 1) показывать ранжированных кандидатов; фильтр панели нуклидов
  по категории/времени жизни.

### 3.4 FWHM(E) — `awf/analysis/fwhm.py` (новый)

- **Порт:** `scripts/gamma/calibration/fwhm_fit.py` — сцинтилляционная модель `FWHM(E)=k·√(E+α·E²)`
  (AtomSpectra — сцинтиллятор), floor BUG-37. **Чистый numpy.** Нужен для окна толерантности
  идентификации и для деконволюции (Фаза 4).

### 3.5 Тест-план Фазы 2

- `tests/test_iaea_fetcher.py`: парсинг синтетического CSV (как реальные колонки ENSDF), нормализация
  имён (Th-234/234th/234TH → 234th), fail-loud на инъекции (`?evil=`, `../`), атомарная запись кэша,
  офлайн-чтение из кэша. **Без реальной сети** (мокать `urllib` или только тестировать парсер+кэш).
- `tests/test_categories.py`: каждая запись `nuclides.json` имеет валидную `category`/`lifetime`;
  фильтры возвращают ожидаемые подмножества.
- `tests/test_identify.py`: на синтетических пиках известных нуклидов идентификация ранжирует
  правильного первым; пропорциональность отбраковывает ложное совпадение.

**Новые зависимости Фазы 2: нет** (stdlib `urllib`/`csv` + numpy).

---

## 4. Фаза 3 — Визуальная модернизация (стиль iZotope Insight, ТЗ 1–6)

**Цель:** довести 3D/2D-виды до референса Insight. Только UI (`pyqtgraph`/OpenGL), без анализа.

| ТЗ | Что делаем | Инструмент pyqtgraph |
|---|---|---|
| 1 | Сечения-плоскости: 2 на ось (время, энергия, отсчёты). Плоскость рисует профиль пересечения + полупрозрачный квад | `GLMeshItem`/`GLLinePlotItem` + слайдеры позиции; профиль = строка/столбец `counts` на позиции |
| 2 | Градуировка осей: подписи keV / время / отсчёты с тиками | `GLTextItem` (pyqtgraph ≥0.13) — у `GLAxisItem` нет подписей; добавить тики из калибровки |
| 3 | Вертикальные энергетические лучи до линии спектра (как Insight) | `GLLinePlotItem` — лучи от базовой плоскости до поверхности на выбранных энергиях/линиях нуклидов |
| 4 | (сделано в Фазе 1 — контраст) | `zscale.py` |
| 5 | Цветовая схема Insight: оранжевый-на-чёрном + синяя база | кастомный `pg.ColorMap` (замена `inferno`); фон уже тёмный `(15,15,20)` |
| 6 | Десатурация массива + подсветка выбранных пиков | база — приглушённый colormap; оверлей выделенных каналов пиков в насыщенном цвете (`GLScatterPlotItem`/вторая поверхность) |

- **Зависимость от драйвера:** `GLTextItem`/доп. items требуют тот же OpenGL 2.1+, что и текущий 3D
  (KNOWN_ISSUES §1) — без регрессии для машин без GL (2D-путь остаётся независимым).
- **Тест-план:** smoke-скрипты в `scripts/` (строят виджеты на образце без GUI-взаимодействия) —
  как для текущего 3D. Проверять, что элементы создаются и не падают в headless-инициализации GL.

**Новые зависимости Фазы 3: нет** (pyqtgraph уже включает GL-items; убедиться в версии ≥0.13 для `GLTextItem`).

---

## 5. Фаза 4 — Продвинутая аналитика (по оси времени, без координаты)

**Цель:** аналитические методы из ТЗ, которые работают на матрице `время×энергия` (ось координат не
нужна — её роль играет время). **Вводим тяжёлые зависимости здесь, опционально.**

### 5.1 Без новых зависимостей (numpy/pyqtgraph)

- **Contour Plot (изолинии):** `pg.IsocurveItem` поверх `HeatmapPanel` — изолинии плотности отсчётов.
- **Анализ градиента** `awf/analysis/gradient.py`: `numpy.gradient` по времени/энергии — карта
  d(счёт)/d(время) выделяет резкие изменения активности вдоль записи.
- **Slice Analysis:** спектр в точке (есть в `SlicePanel`); профиль по маршруту времени — расширение.

### 5.2 Деконволюция мультиплетов — `awf/analysis/deconvolve.py` (**scipy**)

- **Порт:** `scripts/gamma/peaks/deconvolve.py` `multi_gaussian_fit()` — фикс. позиции/ширины,
  свободные площади ≥0 + линейный/step континуум; решатель `scipy.optimize.lsq_linear` (trf, bounds),
  fallback `numpy.linalg.lstsq` + клиппинг. **Порт средней сложности** (можно начать с numpy-fallback,
  scipy — для точности).
- **MDA по ISO 11929** `awf/analysis/mda.py`: порт `scripts/gamma/identification/mda.py:101` —
  L_C=k_α·√(2·n_bg), L_D, A_MDA=L_D/(ε·I·t), k_α=1.645. **Чистая математика** (можно и в Фазе 2).

### 5.3 Снижение размерности и кластеризация — **scikit-learn**

- `awf/analysis/decomposition.py`: PCA/t-SNE/UMAP над матрицей временных срезов (строка = спектр среза).
  PCA/t-SNE — `scikit-learn`; UMAP — опциональный `umap-learn`.
- `awf/analysis/cluster.py`: K-Means/DBSCAN (`scikit-learn`); HDBSCAN — `hdbscan` или
  `sklearn.cluster.HDBSCAN` (sklearn ≥1.3). Кластеризация срезов выделяет фазы записи с разным составом.
- **UI:** отдельная вкладка «Аналитика» с 2D-скаттером проекций/кластеров (pyqtgraph), кликом → срез.

### 5.4 Тест-план Фазы 4

- `tests/test_deconvolve.py`: разрешение двух близких гауссиан, площади в пределах допуска; fallback
  без scipy не падает.
- `tests/test_mda.py`: L_C/L_D/A_MDA против ручного расчёта ISO 11929 на известных n_bg/ε/I/t.
- `tests/test_gradient.py`: градиент на ступенчатой записи локализует фронт.
- `tests/test_decomposition.py`/`test_cluster.py`: на синтетике из 2–3 кластеров спектров методы
  разделяют их (с фикс. random_state).

**Новые зависимости Фазы 4:** `scipy` (деконволюция, opt. площадь/step), `scikit-learn`
(PCA/t-SNE/кластеры), опц. `umap-learn`, `hdbscan`. Все — **опциональные extras** (graceful import,
функция отключается при отсутствии), чтобы базовый вьюер оставался лёгким.

---

## 6. Отложено — Фаза 5: пространственная ось / GPS

**Решение оператора 2026-06-25:** «GPS — это следующий этап, на перспективу отложи».

Эти фичи требуют **третьей оси — координаты/трека**, которой в модели сейчас нет:

- **Volume Rendering** (X=энергия, Y=координата, Z=время) — нужна ось координат; рендер через
  VTK/Mayavi/Plotly (тяжёлые зависимости, оценить отдельно).
- **Peak Map** (координата → активность) — карта активности вдоль маршрута.
- **3D terrain + спектры + GPS** — наложение на карту местности.

**Предусловия Фазы 5 (когда вернёмся):**
1. Рефактор модели: добавить в `Spectrogram` опциональный `track` (GPS/координата на срез) — это
   центральное изменение, затрагивает загрузчики и UI.
2. Извлечение трека: `.rcspg` (RadiaCode, JSON — по `.gitignore` содержит координаты/время) и `.n42`
   (может содержать GPS) — загрузчики сейчас координаты **не** парсят.
3. Выбор движка объёмного рендера (оценить VTK vs Plotly vs дооснащение pyqtgraph).

---

## 7. Реестр портируемых методов SpectraVibe (с провенансом)

| Метод | Источник (`scripts/gamma/...`) | Зависимости | Портируемость | Фаза |
|---|---|---|---|---|
| Поиск пиков (Mariscotti + Currie) | `peaks/search.py` `mariscotti_search()` (261–450) | numpy | ✅ легко | 1 |
| Альт. поиск (свёртка/matched-filter) | `peaks/convolution_search.py` | numpy | ✅ легко | 1 (опц.) |
| Фон Cowell (полином крыльев) | `peaks/area.py:77–238` `cowell_area()` | numpy | ✅ легко | 1 |
| Фон сглаж. ступень (Будыка 7.29) | `peaks/background_options.py:40–89` | numpy, erfc (fallback math) | ✅ легко | 1 |
| Асимметричные окна фона | `peaks/background_options.py:107–169` | numpy | ✅ легко | 1 |
| IAEA-fetcher (ENSDF CSV + кэш) | `data/iaea_fetcher.py` | stdlib (urllib/csv) | ✅ легко | 2 |
| Матчинг линий по энергии | `data/nuclide_library.py:199` `lookup_by_energy` | numpy | ✅ легко | 2 |
| Пресеты-таксономия нуклидов | `data/targeted_libraries.py` | — | ✅ легко | 2 |
| Идентификация (Lsrm §6, 3-шаг) | `identification/identify.py` | numpy + библиотека | ✅ легко | 2 |
| Пропорциональность интенсивностей | `identification/proportionality.py` | numpy | ✅ легко | 2 |
| FWHM(E) сцинтиллятор | `calibration/fwhm_fit.py` | numpy | ✅ легко | 2 |
| MDA по ISO 11929 | `identification/mda.py:101–150` | math, numpy | ✅ легко | 2–4 |
| Калибровка энергии (полином ascending) | `calibration/energy_fit.py:51–150` | numpy | ✅ легко (совместима) | 2 (опц.) |
| Деконволюция мультиплетов | `peaks/deconvolve.py` `multi_gaussian_fit()` | scipy.optimize.lsq_linear | ⚠ средне | 4 |
| Гаусс + erfc-step площадь | `peaks/area_step_continuum.py:70+` | scipy.optimize.curve_fit | ⚠ средне | 4 |

---

## 8. Сводка по зависимостям

| Зависимость | Зачем | Когда вводится | Обязательность |
|---|---|---|---|
| numpy, lxml, PySide6, pyqtgraph, PyOpenGL | уже есть | — | базовая |
| (ничего нового) | Фазы 1–3 (пики, фон, IAEA, идентификация, визуал) | — | — |
| `scipy` | деконволюция, точная площадь/step-континуум | Фаза 4 | опциональная (graceful) |
| `scikit-learn` | PCA/t-SNE, K-Means/DBSCAN/HDBSCAN | Фаза 4 | опциональная |
| `umap-learn`, `hdbscan` | UMAP, HDBSCAN | Фаза 4 | опциональная |

**Принцип:** MVP (Фазы 1–3) не требует ничего сверх текущего стека — лёгкая установка сохраняется.
Тяжёлые научные пакеты — только в Фазе 4 и только как опциональные extras.

---

## 9. Порядок работ и чек-поинты

1. **Фаза 1** (ядро+контраст) → зелёные тесты пиков/фона/вычитания → демо на реальном `.aswf`.
2. **Фаза 2** (IAEA+идентификация) → офлайн-тесты fetcher/категорий/идентификации → демо подсказок.
3. **Фаза 3** (визуал Insight) → smoke-скрипты 3D → визуальная сверка с референсом.
4. **Фаза 4** (аналитика) → тесты деконволюции/MDA/кластеров → опциональные extras.
5. **Фаза 5** (GPS) — по отдельному решению, после рефактора модели под ось координат.

После первого зелёного сквозного прогона Фазы 1 — `graphify install` в проекте (§21 доктрины).

Каждая фаза: код ≥25 строк через `scripts/ollama/_spec_*.md` → Ollama-кодоген → валидация;
обновление `README.md` (Дорожная карта), `KNOWN_ISSUES.md`, тест-счётчика; коммит конкретными
путями; push — по команде оператора.
