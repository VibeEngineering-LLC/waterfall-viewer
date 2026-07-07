"""Вкладка «Live USB» — подключение AtomSpectra напрямую по USB-CDC к PC. Замечание #212 (#PC-1)."""

from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtWidgets
import serial.tools.list_ports

from awf.usb.assembler import HistogramAssembler, SweepResult, CHANNELS
from awf.usb.collector import DeltaCollector, SpectrumSnapshot, WaterfallRow
from awf.usb.device import SerialSpectraDevice
from awf.io.aswf_writer import AswfSegmentedWriter
from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.i18n import tr
from awf.ui import i18n

BAUDRATE: int = 600_000           # фиксированный baudrate AtomSpectra
_POLL_MS: int = 200               # период опроса кадров (мс)
_EMIT_EVERY: int = 5              # emit snapshotReady каждые N свипов

class LiveUsbTab(QtWidgets.QWidget):
    """
    Вкладка Live USB. Подключает AtomSpectra напрямую по COM-порту (pyserial),
    собирает кадры shproto, строит водопад в реальном времени и записывает .aswf.

    Сигналы:
        snapshotReady(object): испускается каждые _EMIT_EVERY свипов с объектом Spectrogram.
            Spectrogram содержит все строки, накопленные с начала текущей сессии.
        statusChanged(str): строка статуса для main_window (CPS, время прибора, drops).
    """

    snapshotReady = QtCore.Signal(object)   # объект Spectrogram
    statusChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # --- внутреннее состояние ---
        self._device: SerialSpectraDevice | None = None
        self._assembler = HistogramAssembler()
        self._collector = DeltaCollector()
        self._writer: AswfSegmentedWriter | None = None
        self._out_dir: Path | None = None

        # живые данные: список WaterfallRow + соответствующие смещения времени
        self._live_rows: list[WaterfallRow] = []          # дельта-строки водопада
        self._live_time_offsets: list[float] = []        # time_offsets_s[i] = начало строки
        self._live_dur: list[float] = []                 # real_time_s[i]
        self._session_start: float = 0.0                 # time.monotonic() начала сессии

        self._sweep_count: int = 0   # счётчик свипов с последнего emit

        # --- UI ---
        self._build_ui()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        
        # Подписка на смену языка
        self._lang_conn = i18n.signals.changed.connect(self._on_lang_changed)

    def _build_ui(self) -> None:
        """Строит UI вкладки."""
        # Вертикальный layout с тремя блоками: подключение, статус, запись

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # --- Блок «Подключение» ---
        conn_grp = QtWidgets.QGroupBox(tr("Подключение"))
        conn_lay = QtWidgets.QHBoxLayout(conn_grp)

        conn_lay.addWidget(QtWidgets.QLabel(tr("Порт:")))
        self._port_combo = QtWidgets.QComboBox()
        self._port_combo.setMinimumWidth(140)
        conn_lay.addWidget(self._port_combo)

        self._refresh_btn = QtWidgets.QPushButton(tr("Обновить"))
        self._refresh_btn.clicked.connect(self._refresh_ports)
        conn_lay.addWidget(self._refresh_btn)

        self._connect_btn = QtWidgets.QPushButton(tr("Подключиться"))
        self._connect_btn.clicked.connect(self._toggle_connect)
        conn_lay.addWidget(self._connect_btn)
        conn_lay.addStretch()
        root.addWidget(conn_grp)

        # --- Блок «Статус» ---
        stat_grp = QtWidgets.QGroupBox(tr("Статус прибора"))
        stat_lay = QtWidgets.QFormLayout(stat_grp)
        self._status_lbl = QtWidgets.QLabel(tr("Не подключён"))
        stat_lay.addRow(tr("Состояние:"), self._status_lbl)
        self._cps_lbl = QtWidgets.QLabel("—")
        stat_lay.addRow("CPS:", self._cps_lbl)
        self._time_lbl = QtWidgets.QLabel("—")
        stat_lay.addRow(tr("Время прибора:"), self._time_lbl)
        self._sweeps_lbl = QtWidgets.QLabel("—")
        stat_lay.addRow(tr("Свипов / дропов:"), self._sweeps_lbl)
        root.addWidget(stat_grp)

        # --- Блок «Запись» ---
        rec_grp = QtWidgets.QGroupBox(tr("Запись .aswf"))
        rec_lay = QtWidgets.QVBoxLayout(rec_grp)

        dir_lay = QtWidgets.QHBoxLayout()
        dir_lay.addWidget(QtWidgets.QLabel(tr("Папка:")))
        self._dir_edit = QtWidgets.QLineEdit()
        self._dir_edit.setPlaceholderText(tr("(не выбрана)"))
        self._dir_edit.setReadOnly(True)
        dir_lay.addWidget(self._dir_edit, stretch=1)
        self._dir_btn = QtWidgets.QPushButton("…")
        self._dir_btn.setFixedWidth(32)
        self._dir_btn.clicked.connect(self._pick_dir)
        dir_lay.addWidget(self._dir_btn)
        rec_lay.addLayout(dir_lay)

        self._rec_btn = QtWidgets.QPushButton(tr("Начать запись"))
        self._rec_btn.setEnabled(False)
        self._rec_btn.clicked.connect(self._toggle_record)
        rec_lay.addWidget(self._rec_btn)

        self._rec_status_lbl = QtWidgets.QLabel(tr("Запись не ведётся"))
        rec_lay.addWidget(self._rec_status_lbl)
        root.addWidget(rec_grp)

        root.addStretch()
        self._refresh_ports()

    # ------------------------------------------------------------------ #
    #  Работа с портами                                                    #
    # ------------------------------------------------------------------ #

    def _refresh_ports(self) -> None:
        """Обновляет список доступных COM-портов."""
        # Сохранить текущий выбор
        current = self._port_combo.currentText()
        self._port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self._port_combo.addItems(ports)
        # Восстановить выбор если порт ещё есть
        idx = self._port_combo.findText(current)
        if idx >= 0:
            self._port_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    #  Подключение / отключение                                            #
    # ------------------------------------------------------------------ #

    def _toggle_connect(self) -> None:
        """Кнопка Подключиться/Отключиться."""
        if self._device is None:
            self._do_connect()
        else:
            self._do_disconnect()

    def _do_connect(self) -> None:
        """Открывает SerialSpectraDevice и запускает таймер опроса."""
        port = self._port_combo.currentText()
        if not port:
            QtWidgets.QMessageBox.warning(self, tr("Нет порта"), tr("Выберите COM-порт."))
            return
        try:
            dev = SerialSpectraDevice(port, BAUDRATE)
            dev.open()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, tr("Ошибка подключения"), str(exc))
            return
        self._device = dev
        self._assembler.reset()
        self._collector.reset()
        self._live_rows.clear()
        self._live_time_offsets.clear()
        self._live_dur.clear()
        self._session_start = time.monotonic()
        self._sweep_count = 0

        self._connect_btn.setText(tr("Отключиться"))
        self._port_combo.setEnabled(False)
        self._refresh_btn.setEnabled(False)
        self._rec_btn.setEnabled(True)
        self._status_lbl.setText(tr("Подключён"))
        self._timer.start()

    def _do_disconnect(self) -> None:
        """Останавливает таймер и закрывает устройство."""
        self._timer.stop()
        if self._writer is not None:
            self._stop_record()
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None

        self._connect_btn.setText(tr("Подключиться"))
        self._port_combo.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._rec_btn.setEnabled(False)
        self._status_lbl.setText(tr("Не подключён"))
        self._cps_lbl.setText("—")
        self._time_lbl.setText("—")
        self._sweeps_lbl.setText("—")

    # ------------------------------------------------------------------ #
    #  Запись                                                              #
    # ------------------------------------------------------------------ #

    def _pick_dir(self) -> None:
        """Выбор папки для записи."""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, tr("Папка для записи .aswf"), str(self._out_dir or Path.home()))
        if d:
            self._out_dir = Path(d)
            self._dir_edit.setText(d)

    def _toggle_record(self) -> None:
        """Кнопка Начать/Остановить запись."""
        if self._writer is None:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self) -> None:
        """Создаёт AswfSegmentedWriter и начинает запись."""
        if self._out_dir is None:
            QtWidgets.QMessageBox.warning(self, tr("Нет папки"), tr("Выберите папку для записи."))
            return
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            # baseline = текущий накопленный спектр (нули если ничего нет)
            if self._live_rows:
                # суммируем все дельта-строки — приближение накопленного спектра
                cum = np.zeros(CHANNELS, dtype=np.uint32)
                for row in self._live_rows:
                    cum += row.counts.astype(np.uint32)
                baseline = cum.astype(np.uint32)
            else:
                baseline = np.zeros(CHANNELS, dtype=np.uint32)
            self._writer = AswfSegmentedWriter(
                self._out_dir,
                started_at=started,
                interval_sec=1.0,
                baseline=baseline,
                calibration=None,
                serial="usb-direct",
                note="live-usb",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, tr("Ошибка записи"), str(exc))
            return
        self._rec_btn.setText(tr("Остановить запись"))
        self._rec_status_lbl.setText(tr("Запись…"))

    def _stop_record(self) -> None:
        """Закрывает AswfSegmentedWriter."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        self._rec_btn.setText(tr("Начать запись"))
        self._rec_status_lbl.setText(tr("Запись остановлена"))

    # ------------------------------------------------------------------ #
    #  Опрос кадров (QTimer)                                               #
    # ------------------------------------------------------------------ #

    def _poll(self) -> None:
        """Вызывается QTimer каждые _POLL_MS мс. Обрабатывает кадры из устройства."""
        if self._device is None:
            return

        frames = self._device.drain_frames()
        for cmd, data in frames:
            result: SweepResult | None = self._assembler.feed(cmd, data)
            if result is None:
                continue
            # полный свип
            snap = SpectrumSnapshot(
                bins=result.bins,
                total_counts=result.total_counts,
                total_time_sec=result.total_time_sec,
            )
            row: WaterfallRow = self._collector.feed(snap)

            # время смещения строки
            t_offset = time.monotonic() - self._session_start
            self._live_time_offsets.append(t_offset)
            self._live_dur.append(float(row.dur_sec) if row.dur_sec > 0 else 1.0)
            self._live_rows.append(row)

            # запись на диск
            if self._writer is not None:
                try:
                    self._writer.write_row(row)
                    self._writer.tick(time.time())
                except Exception as exc:
                    self._stop_record()
                    QtWidgets.QMessageBox.critical(self, tr("Ошибка записи"), str(exc))

            # обновить labels
            cps = result.cps
            t_sec = result.total_time_sec
            self._cps_lbl.setText(str(cps))
            self._time_lbl.setText(f"{t_sec} с")
            self._sweeps_lbl.setText(f"{self._assembler.commits} / {self._assembler.drops}")

            # обновить rec_status
            if self._writer is not None:
                n = len(self._live_rows)
                self._rec_status_lbl.setText(f"{tr('Запись…')} ({n} {tr('строк')})")

            # emit snapshot каждые _EMIT_EVERY свипов
            self._sweep_count += 1
            if self._sweep_count >= _EMIT_EVERY:
                self._sweep_count = 0
                sg = self._build_spectrogram()
                if sg is not None:
                    self.snapshotReady.emit(sg)
                    self.statusChanged.emit(f"Live USB — CPS: {cps}, t={t_sec}s, свипов: {self._assembler.commits}")

    def _build_spectrogram(self) -> Spectrogram | None:
        """Строит Spectrogram из накопленных строк для отображения."""
        if not self._live_rows:
            return None
        counts = np.stack([r.counts for r in self._live_rows], axis=0).astype(np.uint16 if np.max([r.counts.max() for r in self._live_rows]) <= 65535 else np.int32)
        # Не делаем trimmed_channels — main_window сам обрежет
        time_offsets = np.array(self._live_time_offsets, dtype=np.float64)
        real_time = np.array(self._live_dur, dtype=np.float64)
        live_time = real_time.copy()
        return Spectrogram(
            counts=counts,
            calibration=Calibration(np.array([0.0, 1.0], dtype=np.float64)),
            time_offsets_s=time_offsets,
            real_time_s=real_time,
            live_time_s=live_time,
            t0_iso=None,
            source_path="live-usb",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        """При закрытии окна — отключиться."""
        self._do_disconnect()
        super().closeEvent(event)

    def retranslate_ui(self) -> None:
        """Обновляет тексты элементов интерфейса после смены языка."""
        self._port_combo.setPlaceholderText(tr("Порт:"))
        self._refresh_btn.setText(tr("Обновить"))
        self._connect_btn.setText(tr("Подключиться") if self._device is None else tr("Отключиться"))
        self._status_lbl.setText(tr("Не подключён") if self._device is None else tr("Подключён"))
        self._rec_btn.setText(tr("Начать запись") if self._writer is None else tr("Остановить запись"))
        self._rec_status_lbl.setText(tr("Запись не ведётся") if self._writer is None else tr("Запись…"))
        self._dir_edit.setPlaceholderText(tr("(не выбрана)"))
        self.findChild(QtWidgets.QGroupBox, "Подключение").setTitle(tr("Подключение"))
        self.findChild(QtWidgets.QGroupBox, "Статус прибора").setTitle(tr("Статус прибора"))
        self.findChild(QtWidgets.QGroupBox, "Запись .aswf").setTitle(tr("Запись .aswf"))
        self.findChild(QtWidgets.QLabel, "Состояние:").setText(tr("Состояние:"))
        self.findChild(QtWidgets.QLabel, "CPS:").setText("CPS:")
        self.findChild(QtWidgets.QLabel, "Время прибора:").setText(tr("Время прибора:"))
        self.findChild(QtWidgets.QLabel, "Свипов / дропов:").setText(tr("Свипов / дропов:"))
        self.findChild(QtWidgets.QLabel, "Папка:").setText(tr("Папка:"))

    def _on_lang_changed(self, _code: str) -> None:
        """Обработчик смены языка."""
        self.retranslate_ui()
