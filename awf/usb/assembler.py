from __future__ import annotations
import numpy as np
from dataclasses import dataclass

CHANNELS: int = 8192          # каналов спектра AtomSpectra
CMD_HISTOGRAM: int = 0x01     # команда — чанк гистограммы
CMD_STAT: int = 0x04          # команда — статистика прибора

@dataclass(frozen=True)
class SweepResult:
    """Результат полного секундного свипа (8192 каналов)."""
    bins: np.ndarray      # dtype=uint32, shape=(CHANNELS,) — кумулятивные счёты с начала записи
    total_counts: int     # int(bins.sum()) — кумулятивная сумма
    total_time_sec: int   # из CMD_STAT (0 если STAT не приходил до этого свипа)
    cps: int              # импульсов/с из CMD_STAT (0 если нет)

class HistogramAssembler:
    """
    Порт firmware-логики staging-сборки свипов (spectrum.c:s_hist_staging).

    Протокол:
    - CMD_HISTOGRAM (0x01): data[0:2] = uint16_t offset LE + data[2:] = N×uint32_t bins LE.
      offset==0 → старт нового свипа. Зазор (offset != ожидаемый) → свип дропается,
      ждём следующий offset==0. Свип полный когда next_offset >= CHANNELS (8192).
    - CMD_STAT (0x04): data[0:4]=uint32_t total_time_sec LE; data[6:10]=uint32_t cps LE (если len>=10).
      STAT буферируется в s_staged_time/s_staged_cps и публикуется вместе со следующим полным свипом.

    Использование:
        asm = HistogramAssembler()
        for cmd, data in device.drain_frames():
            result = asm.feed(cmd, data)
            if result is not None:
                # полный свип: result.bins, result.total_counts, result.total_time_sec, result.cps
    """

    def __init__(self) -> None:
        self._staging = np.zeros(CHANNELS, dtype=np.uint32)
        self._next: int | None = None   # ожидаемый offset; None = свип не активен
        self._ok: bool = False           # нет зазора с offset==0
        self._staged_time: int = 0
        self._staged_cps: int = 0
        self._commits: int = 0           # успешных свипов
        self._drops: int = 0             # дропнутых свипов

    def feed(self, cmd: int, data: bytes) -> SweepResult | None:
        """Скормить кадр (cmd, data). Вернуть SweepResult если свип завершён, иначе None.
        cmd==CMD_STAT: парсит STAT, сохраняет staged_time/staged_cps, возвращает None.
        cmd==CMD_HISTOGRAM: парсит чанк, при полном свипе возвращает SweepResult.
        Другие cmd: игнорировать, вернуть None.
        """
        if cmd == CMD_STAT:
            self._feed_stat(data)
            return None
        elif cmd == CMD_HISTOGRAM:
            return self._feed_hist(data)
        else:
            return None

    def reset(self) -> None:
        """Сброс состояния (новое подключение / реконнект)."""
        self._staging.fill(0)
        self._next = None
        self._ok = False
        self._staged_time = 0
        self._staged_cps = 0

    @property
    def commits(self) -> int:
        """Число успешно собранных свипов."""
        return self._commits

    @property
    def drops(self) -> int:
        """Число дропнутых (битых) свипов."""
        return self._drops

    def _feed_hist(self, data: bytes) -> SweepResult | None:
        if len(data) < 2:
            return None
        offset = int.from_bytes(data[0:2], 'little')
        if offset == 0:
            self._next = 0
            self._ok = True
        if self._next is None or not self._ok:
            return None
        if offset != self._next:
            self._drops += 1
            self._ok = False
            return None
        bin_count = (len(data) - 2) // 4
        for i in range(bin_count):
            ch = offset + i
            if ch < CHANNELS:
                v = int.from_bytes(data[2 + i * 4 : 6 + i * 4], 'little')
                self._staging[ch] = v
        self._next = offset + bin_count
        if self._next >= CHANNELS:
            self._commits += 1
            result = SweepResult(
                bins=self._staging.copy(),
                total_counts=int(self._staging.sum()),
                total_time_sec=self._staged_time,
                cps=self._staged_cps,
            )
            self._next = None
            self._ok = False
            return result
        return None

    def _feed_stat(self, data: bytes) -> None:
        if len(data) < 4:
            return
        self._staged_time = int.from_bytes(data[0:4], 'little')
        if len(data) >= 10:
            self._staged_cps = int.from_bytes(data[6:10], 'little')
