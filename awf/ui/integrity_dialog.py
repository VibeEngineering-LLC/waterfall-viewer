"""Диалог отчёта о целостности данных (Задача #UI-236): показывается при открытии файла,
текст отчёта — build_integrity_text() (чистая функция, тестируется без Qt), кнопка
«Сохранить отчёт…» пишет .txt в UTF-8.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from awf.ui.i18n import tr


def build_integrity_text(sg) -> str:
    """Собирает многострочный отчёт о целостности данных."""
    lines = []
    
    # Заголовок
    lines.append(tr("=== Отчёт о целостности данных ==="))
    
    # Путь к файлу
    lines.append(f"{tr('Файл')}: {sg.source_path or '—'}")
    
    # Количество срезов и каналов
    lines.append(f"{tr('Срезов')}: {sg.n_slices} × {tr('каналов')}: {sg.n_channels}")
    
    # Общее количество отсчётов
    total_counts = int(np.asarray(sg.counts).sum(dtype=np.int64))
    lines.append(f"{tr('Всего отсчётов')}: {total_counts}")
    
    # Время начала записи
    lines.append(f"t0: {sg.t0_iso or '—'}")
    
    # Длительность записи
    dur = float(np.asarray(sg.real_time_s, dtype=np.float64).sum())
    if not np.isfinite(dur):
        lines.append(f"{tr('Длительность записи')}: —")
    else:
        lines.append(f"{tr('Длительность записи')}: {dur:.0f} {tr('с')}")
    
    # Пустая строка
    lines.append("")
    
    # Блок CRC
    rep = sg.integrity_report
    if rep is None:
        lines.append(tr("Контроль целостности: недоступен (формат без CRC32)"))
    else:
        status = rep["status"]
        if status == "ok":
            lines.append(f"{tr('CRC32: OK')} — {tr('проверено строк')}: {rep['checked']}")
        elif status == "corrupt":
            lines.append(f"⚠ {tr('CRC32: ПОВРЕЖДЕНО СТРОК')}: {rep['bad']} / {rep['checked']}")
            bad_rows = ', '.join(str(i) for i in rep.get('bad_rows', []))
            lines.append(f"{tr('Номера строк')}: {bad_rows}")
        elif status == "skipped_compressed":
            lines.append(tr("CRC32: пропущен (сжатый файл)"))
        
        # Версия формата
        if 'version' in rep:
            lines.append(f"{tr('Версия формата ASWF')}: {rep['version']}")
        
        # Дополнительные данные
        if 'seg_seq' in rep:
            lines.append(f"seg_seq: {rep['seg_seq']}")
        if 'total_at_open' in rep:
            lines.append(f"total_at_open: {rep['total_at_open']}")

    # Задача #DATA-6: пропуски во времени — вероятные потерянные сегменты (кольцевой буфер прибора)
    gaps = getattr(sg, "time_gaps", None)
    if gaps:
        lines.append("")
        gtot = sum(g["gap_s"] for g in gaps)
        miss = sum(g["missing_rows"] for g in gaps)
        lines.append(f"⚠ {tr('Пропуски во времени (возможна потеря сегментов)')}: {len(gaps)}, "
                     f"{tr('всего')} {gtot:.0f} {tr('с')} (~{miss} {tr('строк')})")
        for g in gaps:
            lines.append(f"  {tr('после среза')} {g['after_slice']}: "
                         f"{tr('пропуск')} {g['gap_s']:.0f} {tr('с')} (~{g['missing_rows']} {tr('строк')})")

    # Пустая строка
    lines.append("")
    
    # Наличие данных прибора
    dose = "✓" if sg.dose_rate_usv_h is not None and np.isfinite(sg.dose_rate_usv_h).any() else "—"
    lines.append(f"{tr('Доза')}: {dose}")
    
    temp = "✓" if sg.temperature_c is not None and np.isfinite(sg.temperature_c).any() else "—"
    lines.append(f"{tr('Температура')}: {temp}")
    
    gps = "✓" if sg.gps_track is not None and np.isfinite(sg.gps_track).any() else "—"
    lines.append(f"GPS: {gps}")
    
    baseline = "✓" if getattr(sg, 'baseline', None) is not None else "—"
    lines.append(f"Baseline: {baseline}")
    
    # Пустая строка
    lines.append("")
    
    # Время формирования отчёта
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    lines.append(f"{tr('Сформирован')}: {now}")
    
    return "\n".join(lines)


class IntegrityDialog(QtWidgets.QDialog):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Отчёт о целостности"))
        self.resize(640, 480)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Текстовый редактор
        self._text_edit = QtWidgets.QPlainTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self._text_edit.setPlainText(text)
        layout.addWidget(self._text_edit)
        
        # Кнопки
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        
        self._save_btn = QtWidgets.QPushButton(tr("Сохранить отчёт…"))
        self._save_btn.clicked.connect(self._save)
        button_layout.addWidget(self._save_btn)
        
        close_btn = QtWidgets.QPushButton(tr("Закрыть"))
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        self._text = text
    
    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Сохранить отчёт…"), "integrity_report.txt", "Text (*.txt)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._text)
    
    @staticmethod
    def show_report(parent, sg) -> None:
        """Немодальный показ: не блокирует UI при открытии файла и offscreen-тесты
        (exec() завесил бы прогон main_window-тестов без event-loop)."""
        dlg = IntegrityDialog(build_integrity_text(sg), parent)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()
