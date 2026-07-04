"""Задача #182/#183: диалоги «Помощь» и «О программе» верхнего меню.

`show_help(parent)` — модальное окно QTextBrowser с оглавлением-якорями по всем ключевым
функциям приложения. `show_about(parent)` — компактный QMessageBox с версией, ссылкой на
GitHub, лицензией и краткой строкой стека (Python/Qt/pyqtgraph).

Задача #183: полный перевод контента RU↔EN. Выбор языка — по i18n.current_language().
"""
from __future__ import annotations

import sys

from PySide6 import QtCore, QtGui, QtWidgets

from awf import __version__ as _APP_VERSION
from awf.ui import i18n
from awf.ui.i18n import tr


REPO_URL = "https://github.com/VibeEngineering-LLC/waterfall-viewer"


def _stack_line() -> str:
    """Строка стека для «О программе»: версии Python/PySide6/pyqtgraph/numpy. Отсутствующая
    зависимость — tr('н/д')."""
    na = tr("н/д")
    try:
        import PySide6
        qt_ver = PySide6.__version__
    except Exception:
        qt_ver = na
    try:
        import pyqtgraph
        pg_ver = pyqtgraph.__version__
    except Exception:
        pg_ver = na
    try:
        import numpy
        np_ver = numpy.__version__
    except Exception:
        np_ver = na
    py_ver = ".".join(str(x) for x in sys.version_info[:3])
    return f"Python {py_ver} · PySide6 {qt_ver} · pyqtgraph {pg_ver} · numpy {np_ver}"


_ABOUT_RU = (
    "<h2 style='margin:0'>Waterfall Viewer</h2>"
    "<p style='color:#888;margin:2px 0 10px 0'>Версия {ver}</p>"
    "<p>Просмотр и анализ waterfall-спектрограмм гамма-спектрометров "
    "AtomSpectra, RadiaCode и файлов ANSI/IEEE N42.42.</p>"
    "<p><b>Возможности:</b> 3D-водопад «энергия × время × счёт», 2D-карта, "
    "секущие плоскости, поиск фотопиков (Mariscotti + Currie), идентификация "
    "нуклидов по IAEA-ENSDF, авто-сегментация по времени, работа с фоном "
    "(наложение / вычет), нормализация по кривой эффективности регистрации, "
    "аналитика (PCA/t-SNE/UMAP, деконволюция мультиплетов, спектральный "
    "кластер-анализ, градиент по времени).</p>"
    "<p><b>Стек:</b> {stack}</p>"
    "<p><b>Лицензия:</b> см. файл LICENSE в поставке.</p>"
    "<p><b>Репозиторий:</b> <a href='{url}'>{url}</a></p>"
)

_ABOUT_EN = (
    "<h2 style='margin:0'>Waterfall Viewer</h2>"
    "<p style='color:#888;margin:2px 0 10px 0'>Version {ver}</p>"
    "<p>Viewer and analyzer for waterfall spectrograms from AtomSpectra and "
    "RadiaCode gamma spectrometers, plus ANSI/IEEE N42.42 files.</p>"
    "<p><b>Features:</b> 3D waterfall (energy × time × counts), 2D map, "
    "cutting planes, photopeak search (Mariscotti + Currie), nuclide "
    "identification via IAEA-ENSDF, auto time segmentation, background "
    "handling (overlay / subtract), efficiency-curve normalization, "
    "analytics (PCA/t-SNE/UMAP, multiplet deconvolution, spectral "
    "clustering, time-gradient analysis).</p>"
    "<p><b>Stack:</b> {stack}</p>"
    "<p><b>License:</b> see the LICENSE file in the distribution.</p>"
    "<p><b>Repository:</b> <a href='{url}'>{url}</a></p>"
)


def show_about(parent: QtWidgets.QWidget | None = None) -> None:
    """Задача #182/#183: диалог «О программе». Локализованное HTML-содержимое."""
    stack = _stack_line()
    tmpl = _ABOUT_EN if i18n.current_language() == i18n.LANG_EN else _ABOUT_RU
    html = tmpl.format(ver=_APP_VERSION, stack=stack, url=REPO_URL)
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle(tr("О программе"))
    box.setTextFormat(QtCore.Qt.RichText)
    box.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
    box.setIconPixmap(QtGui.QPixmap())
    box.setText(html)
    box.setStandardButtons(QtWidgets.QMessageBox.Ok)
    box.exec()


# ---------- Помощь (большое окно с оглавлением) ----------

_HELP_TOC_RU = [
    ("s1",  "1. Быстрый старт"),
    ("s2",  "2. Поддерживаемые форматы файлов"),
    ("s3",  "3. Главные вкладки"),
    ("s4",  "4. 3D-водопад: управление и оси"),
    ("s5",  "5. Секущие плоскости 3D"),
    ("s6",  "6. Регулировки отображения (Z-шкала, гамма, контраст, сглаживание)"),
    ("s7",  "7. Окно «Срезы / Сечения / Выборки»"),
    ("s8",  "8. Поиск фотопиков (Mariscotti + Currie)"),
    ("s9",  "9. Библиотека нуклидов и идентификация по пикам"),
    ("s10", "10. Сегментация записи по времени"),
    ("s11", "11. Фон: выбор, наложение, вычет"),
    ("s12", "12. Нормализация по эффективности регистрации ε(E)"),
    ("s13", "13. Аналитика (PCA/t-SNE/UMAP, деконволюция, кластер, градиент, peakmap)"),
    ("s14", "14. Меню приложения"),
    ("s15", "15. Известные ограничения и советы"),
]

_HELP_TOC_EN = [
    ("s1",  "1. Quick start"),
    ("s2",  "2. Supported file formats"),
    ("s3",  "3. Main tabs"),
    ("s4",  "4. 3D waterfall: controls and axes"),
    ("s5",  "5. 3D cutting planes"),
    ("s6",  "6. Display adjustments (Z-scale, gamma, contrast, smoothing)"),
    ("s7",  '7. "Slices / Sections / Samples" window'),
    ("s8",  "8. Photopeak search (Mariscotti + Currie)"),
    ("s9",  "9. Nuclide library and peak-based identification"),
    ("s10", "10. Time segmentation of the record"),
    ("s11", "11. Background: selection, overlay, subtraction"),
    ("s12", "12. Efficiency normalization ε(E)"),
    ("s13", "13. Analytics (PCA/t-SNE/UMAP, deconvolution, clustering, gradient, peakmap)"),
    ("s14", "14. Application menu"),
    ("s15", "15. Known limits and tips"),
]


def _help_toc_html(lang: str) -> str:
    toc = _HELP_TOC_EN if lang == i18n.LANG_EN else _HELP_TOC_RU
    title = "Contents" if lang == i18n.LANG_EN else "Оглавление"
    items = "".join(f"<li><a href='#{aid}'>{t}</a></li>" for aid, t in toc)
    return f"<h2 id='toc'>{title}</h2><ol style='line-height:1.55'>{items}</ol>"

_HELP_SECTIONS_RU = r"""
<h2 id="s1">1. Быстрый старт</h2>
<ol>
  <li>Меню <b>Файл → Открыть…</b> (или Ctrl+O) — выбрать спектрограмму
      (<code>.aswf</code>, <code>.n42</code>, <code>.rcspg</code>).</li>
  <li>Файл загрузится в фоне; после загрузки — три вкладки:
      <b>3D Waterfall</b>, <b>2D Карта (Время×Энергия)</b>, <b>Аналитика</b>.</li>
  <li>По правому краю — доки: <b>Регулировки отображения</b>, <b>Сечения (3D)</b>,
      <b>Срезы / Сечения / Выборки</b>, <b>Найденные пики</b>,
      <b>Сегментация по времени</b>, <b>Библиотека нуклидов</b>,
      <b>Идентификация по найденным пикам</b>. Управляются через меню
      <b>Инструменты</b> и <b>Изотопы</b>.</li>
  <li>Тулбар <b>Вид</b> сверху — выбор палитры, единиц оси счёта (counts / cps),
      единиц оси времени (сек / мин / часы), режима Z-шкалы (линейная / корень / логарифм),
      подложки, простыней образца и фона.</li>
</ol>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s2">2. Поддерживаемые форматы файлов</h2>
<ul>
  <li><b>.aswf</b> — родной формат AtomSpectra Waterfall. Поддерживаются раскладки v1
      (фиксированный интервал среза) и v2 (row_stride, per-row row_time, saved_rows=0).</li>
  <li><b>.n42</b> — ANSI/IEEE N42.42 (XML). Читаются <code>Measurement</code>,
      <code>Spectrum</code>, калибровка энергии.</li>
  <li><b>.rcspg</b> — RadiaCode waterfall (табличный текст). Поддерживается оверлей
      «мощность дозы» по калибровке RC-103 (пункт меню <b>Анализ → Мощность дозы</b>).</li>
</ul>
<p>Последний канал (шумовой), если он присутствует в исходнике, отбрасывается
   (см. IV-R5). Отрицательные значения в спектрограмме приводятся к нулю перед
   отображением.</p>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s3">3. Главные вкладки</h2>
<h3>3D Waterfall</h3>
<p>Рельеф «энергия × время × счёт». Ось X — энергия (кэВ), ось Y — время
   (сек / мин / часы), ось Z — интенсивность в выбранных единицах. Ограничение —
   3000 кэВ по оси энергии (шкала и сетка).</p>
<h3>2D Карта (Время×Энергия)</h3>
<p>Плоская тепловая карта того же массива. Поддерживается сетка изолиний
   (регулятор в тулбаре). Клик мыши — выбор среза для дока «Срезы».</p>
<h3>Аналитика</h3>
<p>Отдельная вкладка со сводкой алгоритмов: карта пиков во времени (peakmap),
   PCA/t-SNE/UMAP по срезам, деконволюция мультиплетов, спектральный
   кластер-анализ, градиентный анализ d(счёт)/d(время). См. раздел
   <a href='#s13'>13</a>.</p>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s4">4. 3D-водопад: управление и оси</h2>
<ul>
  <li><b>Вращение камеры</b> — левая кнопка мыши + движение.</li>
  <li><b>Панорама</b> — Shift + левая кнопка.</li>
  <li><b>Зум</b> — колесо мыши.</li>
  <li><b>Оси и сетка</b>. Клетки привязаны к шкалам: энергия — по 200 кэВ,
      время — авто (сек / мин / часы; для больших записей — по 15 минут).
      Размерность клеток выводится по осям. Подписи делений гасятся с той
      стороны, куда смотрит взгляд, чтобы не перекрывать рельеф.</li>
  <li><b>Подложка</b> — фиолетовая база рельефа. Отключается тумблером в тулбаре
      «Вид»; по умолчанию — выключена.</li>
  <li><b>Простыни</b> — контурная поверхность образца и фона. Стиль настраивается
      раздельно (палитра / однотонный / каркас), тумблеры в тулбаре.</li>
  <li><b>Вертикальные линии нуклидов</b> — на плоскостях секущих (см. раздел
      <a href='#s5'>5</a>). Высота линии пропорциональна интенсивности гамма-линии
      нуклида; клиппируются к секущим плоскостям и не просвечивают сквозь рельеф.</li>
</ul>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s5">5. Секущие плоскости 3D</h2>
<p>Дока <b>Сечения (3D)</b> — три пары плоскостей (энергия, время, счёт), каждая
   пара — «низ» и «верх». Плоскости в 3D показывают ТОЛЬКО контур (без заливки),
   а отсекают данные от начального положения до текущего.</p>
<ul>
  <li>Плоскости синхронизированы с 2D-картой и доком «Срезы».</li>
  <li>Проекция суммы отсчётов между плоскостями выводится на самой плоскости
      сечения — как спектр или профиль во времени.</li>
  <li>Двойной клик по регулятору сечения — сброс к дефолту (снимает отсечение).</li>
  <li>Маркеры изотопов отображаются ТОЛЬКО на активных секущих плоскостях
      (задача #85), высота ∝ интенсивности линий семейства (задача #69).</li>
</ul>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s6">6. Регулировки отображения</h2>
<p>Док <b>Регулировки отображения</b> — горизонтальные фейдеры в стиле iZotope
   Insight. Каждая рукоятка — с индивидуальным тумблером «вкл/выкл» и кнопкой
   сброса к дефолту.</p>
<ul>
  <li><b>Z-шкала</b> (тулбар «Вид»): <i>Линейная</i>, <i>Корень √</i>, <i>Логарифм log10</i>.
      Логарифм — истинный <code>log10(x / floor)</code>, где floor —
      масштаб-инвариантный порог (10-й перцентиль положительных отсчётов; при
      обнаружении плоского пуассонова дна — поднимается до <code>p50/2</code>).</li>
  <li><b>Gain</b> — усиление с насыщением (значения выше клиппинга «сгорают»).</li>
  <li><b>Gamma</b> — гамма-кривая: γ &lt; 1 поднимает слабые каналы, γ &gt; 1 придавливает.</li>
  <li><b>Clip (%)</b> — перцентильное отсечение снизу/сверху (борьба с выбросами).</li>
  <li><b>Сглаживание</b> — три режима: <b>Выкл</b> · <b>SMA</b> (равные веса) ·
      <b>WMA</b> (треугольные веса, сильнее к центру). Ширина окна фиксирована — 5 каналов.</li>
  <li><b>Освещение 3D</b> — интенсивность теней рельефа.</li>
  <li><b>Обесцвечивание базы</b> — понижение насыщенности рельефа при подсветке
      выбранных пиков (задача #18).</li>
  <li><b>Временно́й масштаб</b> — растяжение оси времени.</li>
</ul>
<p>При всех выключенных рукоятках возвращается «сырой» рельеф без обработки
   контраста (bypass к дефолтам). Кнопки «Сброс» — в каждом ряду.</p>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s7">7. Окно «Срезы / Сечения / Выборки»</h2>
<p>Верхний график — <b>спектр</b> (энергия × отсчёт) в текущем срезе (или сумме
   срезов между плоскостями). Нижний график — <b>профиль во времени</b>: скорость
   счёта в выбранной энергетической полосе (по клику на 3D/2D) плюс «Мощность
   дозы» для RadiaCode.</p>
<ul>
  <li>Переключатель <b>лог / лин по Y</b> для спектра.</li>
  <li>Единицы <b>counts / cps</b> синхронизированы с тулбаром «Вид».</li>
  <li>Спектр не цепляется мышью, зум колесом — относительно указателя (задача #165).</li>
  <li>Двойной клик по графику — сброс зума/панорамы.</li>
  <li>При включённом наложении фона рисуется вторая кривая той же статистики,
      что образец: сырое фоновое окно с совпадающим живым временем — «оба
      лохматые одинаково» (задача #139).</li>
  <li>Y-низ лог-режима устойчив к ВЭ-выбросам ε-нормировки: floor — 10-й
      перцентиль положительных (задача #166).</li>
</ul>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s8">8. Поиск фотопиков</h2>
<p>Меню <b>Анализ → Поиск пиков</b>. Двухпроходной детектор: сгенерированный
   фильтр Mariscotti по FWHM(E) с авто-калибровкой под реальный детектор
   (задача #120), затем критерий значимости L<sub>C</sub> Currie 3σ. Ищет только
   в диапазоне 50–3000 кэВ (задачи #119/#153).</p>
<ul>
  <li>Пики отмечаются на 3D-водопаде зелёными вертикальными «хребтами»,
      ограниченными зоной физического присутствия (не тянутся вдоль всего
      времени — задачи #112/#126/#129).</li>
  <li>Панель <b>Найденные пики</b> — таблица с колонками энергии, FWHM, высоты
      и площади, единицы указаны в шапке (задача #123). Регулятор чувствительности
      (порог σ) — там же.</li>
  <li>Клик по строке таблицы — подсветка соответствующего хребта на 3D
      (задача #124). Чекбоксы — фильтр отображения.</li>
</ul>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s9">9. Библиотека нуклидов и идентификация по пикам</h2>
<p>Два отдельных окна (задача #173), доступ через меню <b>Изотопы</b>:</p>
<ul>
  <li><b>Библиотека нуклидов</b> — просмотр списка со всех источников: природные,
      техногенные, медицинские, продукты деления. Категоризация по времени
      полураспада (короткие/длинные). Клик — линии семейства подсвечиваются на
      секущих плоскостях 3D.</li>
  <li><b>Идентификация по найденным пикам</b> — сопоставление найденных линий
      (см. раздел <a href='#s8'>8</a>) с базой IAEA-ENSDF; семейства
      сворачиваются к родителям (Th-232, Ra-226 и т.д. — задача #154). Значимые
      линии проверяются по интенсивности (задача #162), чтобы не спутать
      La-138 и т.п.</li>
</ul>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s10">10. Сегментация записи по времени</h2>
<p>Меню <b>Анализ → Сегментация по времени…</b>. Автоматическое разбиение всей
   записи на участки стабильности + посегментная идентификация нуклидов
   (задача #131). Слабые источники (урановое стекло, K-40), которые тонут в
   интегральном спектре по всему файлу, всплывают на своём временном сегменте.</p>
<p>Панель <b>Сегментация по времени</b> — таблица сегментов; клик по строке —
   выделение диапазона на 3D/2D через секущие плоскости.</p>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s11">11. Фон: выбор, наложение, вычет</h2>
<p>Меню <b>Анализ</b>:</p>
<ol>
  <li><b>Выбор фона…</b> — задать поканальный фон. Два источника: (а) диапазон
      срезов текущего файла (можно выбрать секущими плоскостями времени,
      задача #148), или (б) отдельный файл. При совпадающей энергосетке
      берётся сырой поканальный блок; при иной калибровке — гладкая усреднённая
      скорость <code>bg_cps</code>.</li>
  <li><b>Наложение фона</b> — оверлей поверх спектра среза и в 3D-простыне.
      Оверлей в окне срезов рисуется той же статистики, что образец
      (задача #139); простыня фона строится тем же пайплайном, что рельеф
      (задача #140).</li>
  <li><b>Вычет фона</b> — знаковый поканальный вычет: <code>net = counts − bg·live_time</code>
      (задача #134/#138). Отрицательные значения в 3D приводятся к нулю;
      интегральный спектр среза при фон=образец точно уходит в 0.</li>
</ol>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s12">12. Нормализация по эффективности регистрации ε(E)</h2>
<p>Меню <b>Анализ → Нормализация по эффективности</b>. Умножает отсчёты каналов
   на <code>ε_ref / ε(E)</code>. Компенсирует падение эффективности фотопика с
   энергией — истинные высокоэнергетические линии становятся сравнимыми с
   низкоэнергетическими по амплитуде рельефа. Кривая по умолчанию —
   <b>Гамма-1С</b>. Пункт <b>Загрузить кривую эффективности…</b> позволяет
   подгрузить свою кривую (JSON/CSV E [кэВ] → ε).</p>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s13">13. Аналитика</h2>
<p>Вкладка <b>Аналитика</b>:</p>
<ul>
  <li><b>PCA / t-SNE / UMAP</b> — снижение размерности по временны́м срезам;
      кластеризация похожих спектров.</li>
  <li><b>Спектральный кластер-анализ</b> — иерархический кластер по срезам,
      дендрограмма.</li>
  <li><b>Деконволюция мультиплетов</b> — разложение перекрывающихся пиков.</li>
  <li><b>Peakmap</b> — карта отдельных пиков во времени.</li>
  <li><b>Градиентный анализ</b> — d(счёт)/d(время): вспышки/спады активности.</li>
</ul>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s14">14. Меню приложения</h2>
<ul>
  <li><b>Файл</b> — открыть спектрограмму (Ctrl+O), выход.</li>
  <li><b>Изотопы</b> — библиотека нуклидов, идентификация по найденным пикам.</li>
  <li><b>Анализ</b> — выбор фона, наложение, вычет, мощность дозы (RadiaCode),
      поиск пиков, сегментация по времени, нормализация по эффективности.</li>
  <li><b>Инструменты</b> — список окон-доков (Найденные пики / Сегментация /
      Срезы / Сечения / Регулировки). Клик — открывает/скрывает.</li>
  <li><b>Сервис</b> — подменю <b>Язык</b> (Русский / English).</li>
  <li><b>Помощь</b> — это окно.</li>
  <li><b>О программе</b> — версия, стек, лицензия, репозиторий.</li>
</ul>
<p><a href='#toc'>↑ к оглавлению</a></p>

<h2 id="s15">15. Известные ограничения и советы</h2>
<ul>
  <li>Ось энергии 3D-водопада ограничена сверху 3000 кэВ (задача #78) — для
      обычных бытовых гамма-спектров этого хватает; пики выше туда не попадают.</li>
  <li>Расположение окон/доков и выбор языка сохраняются между запусками
      (QSettings, задача #40).</li>
  <li>Для больших файлов (тысячи срезов) 3D может тормозить; попробуйте
      уменьшить оконный масштаб, отключить простыню фона, включить
      сглаживание SMA/WMA.</li>
  <li>Мощность дозы доступна только для RadiaCode <code>.rcspg</code>
      (калибровка RC-103).</li>
  <li>Если файл не читается — проверьте формат (только <code>.aswf</code>,
      <code>.n42</code>, <code>.rcspg</code>); при непонятной ошибке приложите
      файл к issue на GitHub.</li>
</ul>
<p><a href='#toc'>↑ к оглавлению</a></p>
"""

_HELP_SECTIONS_EN = r"""
<h2 id="s1">1. Quick start</h2>
<ol>
  <li><b>File → Open…</b> (or Ctrl+O) — pick a spectrogram
      (<code>.aswf</code>, <code>.n42</code>, <code>.rcspg</code>).</li>
  <li>The file loads in the background; then three tabs appear:
      <b>3D Waterfall</b>, <b>2D Map (Time×Energy)</b>, <b>Analytics</b>.</li>
  <li>Docks on the right: <b>Display adjustments</b>, <b>Sections (3D)</b>,
      <b>Slices / Sections / Samples</b>, <b>Found peaks</b>,
      <b>Time segmentation</b>, <b>Nuclide library</b>,
      <b>Identification from found peaks</b>. Toggled via the
      <b>Tools</b> and <b>Isotopes</b> menus.</li>
  <li>The <b>View</b> toolbar on top — palette, counts axis units (counts / cps),
      time axis units (s / min / h), Z-scale mode (linear / square-root /
      log10), floor, sample and background sheets.</li>
</ol>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s2">2. Supported file formats</h2>
<ul>
  <li><b>.aswf</b> — native AtomSpectra Waterfall format. Both v1 (fixed slice
      interval) and v2 (row_stride, per-row row_time, saved_rows=0) layouts
      are supported.</li>
  <li><b>.n42</b> — ANSI/IEEE N42.42 (XML). <code>Measurement</code>,
      <code>Spectrum</code> and energy calibration are read.</li>
  <li><b>.rcspg</b> — RadiaCode waterfall (tabbed text). A "dose rate" overlay
      via RC-103 calibration is available (menu <b>Analysis → Dose rate</b>).</li>
</ul>
<p>The last (noise) channel, if present in the source, is dropped (see
   IV-R5). Negative values in the spectrogram are clamped to zero before
   display.</p>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s3">3. Main tabs</h2>
<h3>3D Waterfall</h3>
<p>An "energy × time × counts" relief. X axis — energy (keV), Y axis — time
   (s / min / h), Z axis — intensity in the selected units. The energy axis
   is capped at 3000 keV (scale and grid).</p>
<h3>2D Map (Time×Energy)</h3>
<p>A flat heat-map of the same array. Contour lines are supported (control in
   the toolbar). Click — pick a slice for the "Slices" dock.</p>
<h3>Analytics</h3>
<p>A separate tab with a summary of algorithms: peakmap over time,
   PCA/t-SNE/UMAP over slices, multiplet deconvolution, spectral
   clustering, d(counts)/d(time) gradient analysis. See section
   <a href='#s13'>13</a>.</p>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s4">4. 3D waterfall: controls and axes</h2>
<ul>
  <li><b>Rotate camera</b> — left mouse button + drag.</li>
  <li><b>Pan</b> — Shift + left mouse button.</li>
  <li><b>Zoom</b> — mouse wheel.</li>
  <li><b>Axes and grid</b>. Cells are aligned to the scales: energy — every
      200 keV, time — auto (s / min / h; for long records — every 15 min).
      Cell units are printed on the axes. Tick labels are hidden on the side
      the camera looks at, to avoid overlapping the relief.</li>
  <li><b>Floor</b> — the purple base of the relief. Toggled from the "View"
      toolbar; off by default.</li>
  <li><b>Sheets</b> — the contour surface of the sample and the background.
      Style is independent (palette / solid / wireframe), toggles are in the
      toolbar.</li>
  <li><b>Vertical nuclide lines</b> — drawn on the cutting planes (see
      section <a href='#s5'>5</a>). Height is proportional to the gamma-line
      intensity; clipped to the cutting planes and hidden behind the relief.</li>
</ul>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s5">5. 3D cutting planes</h2>
<p>The <b>Sections (3D)</b> dock — three pairs of planes (energy, time,
   counts), each pair has a "low" and a "high" plane. In 3D the planes show
   ONLY the outline (no fill), and they clip data from the initial position
   to the current one.</p>
<ul>
  <li>Planes are synchronized with the 2D map and the "Slices" dock.</li>
  <li>The projection of the total counts between the planes is drawn on the
      cutting plane itself — as a spectrum or a time profile.</li>
  <li>Double-click a section knob — reset to default (removes the clipping).</li>
  <li>Isotope markers are drawn ONLY on the active cutting planes
      (task #85); height ∝ family line intensity (task #69).</li>
</ul>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s6">6. Display adjustments</h2>
<p>The <b>Display adjustments</b> dock — horizontal iZotope Insight-style
   faders. Each knob has its own "on/off" toggle and a reset-to-default
   button.</p>
<ul>
  <li><b>Z-scale</b> ("View" toolbar): <i>Linear</i>, <i>Square root √</i>,
      <i>Logarithm log10</i>. The log mode is a true
      <code>log10(x / floor)</code>, where floor is a scale-invariant threshold
      (10th percentile of positive counts; when a flat Poisson floor is
      detected — bumped up to <code>p50/2</code>).</li>
  <li><b>Gain</b> — gain with saturation (values above the clip "burn out").</li>
  <li><b>Gamma</b> — a gamma curve: γ &lt; 1 lifts weak channels, γ &gt; 1
      squashes them.</li>
  <li><b>Clip (%)</b> — percentile clip from below/above (outlier rejection).</li>
  <li><b>Smoothing</b> — three modes: <b>Off</b> · <b>SMA</b> (equal weights)
      · <b>WMA</b> (triangular weights, stronger toward the center). Window
      width is fixed — 5 channels.</li>
  <li><b>3D lighting</b> — the intensity of the relief shading.</li>
  <li><b>Base desaturation</b> — reduces the saturation of the relief when
      highlighting selected peaks (task #18).</li>
  <li><b>Time scale</b> — stretches the time axis.</li>
</ul>
<p>With every knob disabled the "raw" relief is drawn, without contrast
   processing (bypass to defaults). "Reset" buttons — in every row.</p>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s7">7. "Slices / Sections / Samples" window</h2>
<p>The top plot — the <b>spectrum</b> (energy × counts) in the current slice
   (or the sum of slices between planes). The bottom plot — a <b>time
   profile</b>: count rate in the selected energy band (clicked on 3D/2D)
   plus the "Dose rate" curve for RadiaCode.</p>
<ul>
  <li><b>log / lin Y</b> switch for the spectrum.</li>
  <li>The <b>counts / cps</b> units are synchronized with the "View" toolbar.</li>
  <li>The spectrum does not drag under the mouse; wheel-zoom is centered on
      the cursor (task #165).</li>
  <li>Double-click a plot — reset zoom/pan.</li>
  <li>With the background overlay on, a second curve is drawn of the same
      statistics as the sample: a raw background window with matching live
      time — "both equally noisy" (task #139).</li>
  <li>The log-Y floor is robust to the high-energy outliers of the
      ε-normalization: the floor is the 10th percentile of positives
      (task #166).</li>
</ul>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s8">8. Photopeak search</h2>
<p>Menu <b>Analysis → Peak search</b>. A two-pass detector: a synthesized
   Mariscotti filter using FWHM(E) with auto-calibration for the actual
   detector (task #120), then the Currie L<sub>C</sub> 3σ significance
   criterion. Searches only in 50–3000 keV (tasks #119/#153).</p>
<ul>
  <li>Peaks are marked on the 3D waterfall as green vertical "ridges",
      confined to the physical presence zone (they do not stretch across the
      whole time — tasks #112/#126/#129).</li>
  <li>The <b>Found peaks</b> panel — a table with energy, FWHM, height and
      area columns; units printed in the header (task #123). The
      sensitivity control (σ threshold) is here too.</li>
  <li>Click a table row — the corresponding ridge on the 3D is highlighted
      (task #124). Check-boxes — the display filter.</li>
</ul>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s9">9. Nuclide library and peak-based identification</h2>
<p>Two separate windows (task #173), accessed via the <b>Isotopes</b> menu:</p>
<ul>
  <li><b>Nuclide library</b> — a list from every source: natural, technogenic,
      medical, fission products. Categorized by half-life
      (short-lived/long-lived). Click — the family lines are highlighted on
      the 3D cutting planes.</li>
  <li><b>Identification from found peaks</b> — matches the found lines (see
      section <a href='#s8'>8</a>) against the IAEA-ENSDF database; families
      are collapsed to their parents (Th-232, Ra-226 and so on —
      task #154). Significant lines are cross-checked by intensity
      (task #162), to avoid confusing La-138 and the like.</li>
</ul>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s10">10. Time segmentation of the record</h2>
<p>Menu <b>Analysis → Time segmentation…</b>. Automatic splitting of the
   whole record into stable segments + per-segment nuclide identification
   (task #131). Weak sources (uranium glass, K-40) that drown in the
   whole-file integral spectrum surface within their own time segment.</p>
<p>The <b>Time segmentation</b> panel — a table of segments; click a row —
   highlight the range on the 3D/2D via the cutting planes.</p>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s11">11. Background: selection, overlay, subtraction</h2>
<p>Menu <b>Analysis</b>:</p>
<ol>
  <li><b>Select background…</b> — set the per-channel background. Two
      sources: (a) a slice range of the current file (can be picked via the
      time cutting planes, task #148), or (b) a separate file. When the
      energy grid matches, a raw per-channel block is used; otherwise a
      smoothed averaged rate <code>bg_cps</code>.</li>
  <li><b>Overlay background</b> — an overlay on the slice spectrum and on
      the 3D sheet. In the slice window the overlay is drawn with the same
      statistics as the sample (task #139); the background sheet is built
      by the same pipeline as the relief (task #140).</li>
  <li><b>Subtract background</b> — signed per-channel subtraction:
      <code>net = counts − bg·live_time</code> (tasks #134/#138). Negative
      values in 3D are clamped to zero; when background=sample, the slice
      integral spectrum lands exactly on zero.</li>
</ol>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s12">12. Efficiency normalization ε(E)</h2>
<p>Menu <b>Analysis → Efficiency normalization</b>. Multiplies channel
   counts by <code>ε_ref / ε(E)</code>. Compensates the drop of photopeak
   efficiency with energy — true high-energy lines become comparable to
   low-energy ones in relief amplitude. The default curve —
   <b>Gamma-1S</b>. The menu item <b>Load efficiency curve…</b> lets you
   plug in your own curve (JSON/CSV E [keV] → ε).</p>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s13">13. Analytics</h2>
<p>The <b>Analytics</b> tab:</p>
<ul>
  <li><b>PCA / t-SNE / UMAP</b> — dimensionality reduction over time
      slices; clustering of similar spectra.</li>
  <li><b>Spectral clustering</b> — hierarchical clustering over slices, a
      dendrogram.</li>
  <li><b>Multiplet deconvolution</b> — separation of overlapping peaks.</li>
  <li><b>Peakmap</b> — a map of individual peaks over time.</li>
  <li><b>Gradient analysis</b> — d(counts)/d(time): activity spikes/decays.</li>
</ul>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s14">14. Application menu</h2>
<ul>
  <li><b>File</b> — open a spectrogram (Ctrl+O), quit.</li>
  <li><b>Isotopes</b> — nuclide library, identification from found peaks.</li>
  <li><b>Analysis</b> — background selection, overlay, subtract, dose rate
      (RadiaCode), peak search, time segmentation, efficiency normalization.</li>
  <li><b>Tools</b> — the list of dock windows (Found peaks / Time
      segmentation / Slices / Sections / Adjustments). Click — show/hide.</li>
  <li><b>Service</b> — the <b>Language</b> sub-menu (Russian / English).</li>
  <li><b>Help</b> — this window.</li>
  <li><b>About</b> — version, stack, license, repository.</li>
</ul>
<p><a href='#toc'>↑ back to contents</a></p>

<h2 id="s15">15. Known limits and tips</h2>
<ul>
  <li>The 3D waterfall energy axis is capped at 3000 keV (task #78) — for
      usual household gamma spectra this is enough; higher peaks are not
      shown there.</li>
  <li>Window/dock layout and the language choice are persisted across runs
      (QSettings, task #40).</li>
  <li>For large files (thousands of slices) 3D may lag; try to reduce the
      window scale, turn off the background sheet, enable SMA/WMA
      smoothing.</li>
  <li>The dose rate is available only for RadiaCode <code>.rcspg</code>
      (RC-103 calibration).</li>
  <li>If a file does not open — check the format (only <code>.aswf</code>,
      <code>.n42</code>, <code>.rcspg</code>); on any unclear error, attach
      the file to a GitHub issue.</li>
</ul>
<p><a href='#toc'>↑ back to contents</a></p>
"""


def show_help(parent: QtWidgets.QWidget | None = None) -> None:
    """Задача #182/#183: диалог «Помощь». Модальное окно с QTextBrowser: оглавление-якоря
    вверху + 15 разделов с описанием функционала. Ссылки внутренние (по anchor)
    работают через `setOpenLinks(True)`; внешние (GitHub) открываются в браузере.
    Контент выбирается по i18n.current_language()."""
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(tr("Помощь"))
    dlg.resize(900, 700)
    lay = QtWidgets.QVBoxLayout(dlg)
    lay.setContentsMargins(8, 8, 8, 8)

    browser = QtWidgets.QTextBrowser(dlg)
    browser.setOpenLinks(True)
    browser.setOpenExternalLinks(True)
    lang = i18n.current_language()
    sections = _HELP_SECTIONS_EN if lang == i18n.LANG_EN else _HELP_SECTIONS_RU
    browser.setHtml(_help_toc_html(lang) + sections)
    lay.addWidget(browser, 1)

    btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=dlg)
    btns.rejected.connect(dlg.reject)
    btns.accepted.connect(dlg.accept)
    close_btn = btns.button(QtWidgets.QDialogButtonBox.Close)
    if close_btn is not None:
        close_btn.setText(tr("Закрыть"))
        close_btn.clicked.connect(dlg.accept)
    lay.addWidget(btns)

    dlg.exec()