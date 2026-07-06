from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


SERIES_DEFAULT_GAP_SEC: float = 0.0


class SeriesPhase(Enum):
    IDLE = "idle"            # серия ещё не стартовала (started_at is None)
    RECORDING = "recording"  # идёт запись current_index-й дорожки
    PAUSE = "pause"          # межтрековая пауза после current_index
    DONE = "done"            # все дорожки записаны


@dataclass
class RecordingSeries:
    durations_sec: list[float]
    out_dir: Path | str
    file_prefix: str = "series"
    gap_sec: float = SERIES_DEFAULT_GAP_SEC
    stop_on_error: bool = True     # ошибка серии → полный стоп (переход в DONE с _errored=True)

    _started_at: Optional[float] = field(default=None, init=False)
    _current_index: int = field(default=0, init=False)
    _phase: SeriesPhase = field(default=SeriesPhase.IDLE, init=False)
    _phase_started_at: float = field(default=0.0, init=False)
    _errored: bool = field(default=False, init=False)

    def __post_init__(self):
        if not self.durations_sec:
            raise ValueError("durations_sec must be a non-empty list")
        if any(d <= 0.0 for d in self.durations_sec):
            raise ValueError("all durations must be greater than zero")
        if self.gap_sec < 0.0:
            raise ValueError("gap_sec must be non-negative")
        if not self.file_prefix:
            raise ValueError("file_prefix must be a non-empty string")
        self.out_dir = Path(self.out_dir)

    @property
    def n_tracks(self) -> int:
        return len(self.durations_sec)

    @property
    def phase(self) -> SeriesPhase:
        return self._phase

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def started_at(self) -> Optional[float]:
        return self._started_at

    @property
    def is_running(self) -> bool:
        return self._phase in (SeriesPhase.RECORDING, SeriesPhase.PAUSE)

    @property
    def is_done(self) -> bool:
        return self._phase == SeriesPhase.DONE

    @property
    def errored(self) -> bool:
        return self._errored

    @property
    def current_file(self) -> Optional[Path]:
        if self._phase != SeriesPhase.RECORDING:
            return None
        return self.out_dir / f"{self.file_prefix}_{self._current_index:03d}"

    @property
    def current_duration_sec(self) -> Optional[float]:
        if self._phase == SeriesPhase.IDLE or self._phase == SeriesPhase.DONE:
            return None
        if self._phase == SeriesPhase.RECORDING:
            return self.durations_sec[self._current_index]
        else:  # PAUSE
            return self.gap_sec

    def remaining_sec(self, now: float) -> Optional[float]:
        if self._phase == SeriesPhase.IDLE or self._phase == SeriesPhase.DONE:
            return None
        phase_duration = self.current_duration_sec
        assert phase_duration is not None
        elapsed = now - self._phase_started_at
        return max(0.0, phase_duration - elapsed)

    def start(self, now: float) -> None:
        if self._phase != SeriesPhase.IDLE:
            raise RuntimeError("series already started")
        self._started_at = now
        self._current_index = 0
        self._phase = SeriesPhase.RECORDING
        self._phase_started_at = now
        self._errored = False

    def tick(self, now: float) -> SeriesPhase:
        if self._phase == SeriesPhase.IDLE or self._phase == SeriesPhase.DONE:
            return self._phase

        while True:
            if self._phase == SeriesPhase.RECORDING:
                phase_duration = self.durations_sec[self._current_index]
                elapsed = now - self._phase_started_at
                if elapsed < phase_duration:
                    break
                # Переход из RECORDING в PAUSE или DONE
                if self.gap_sec > 0.0 and self._current_index < self.n_tracks - 1:
                    self._phase = SeriesPhase.PAUSE
                    self._phase_started_at += phase_duration
                else:
                    self._phase_started_at += phase_duration
                    self._current_index += 1
                    if self._current_index >= self.n_tracks:
                        self._phase = SeriesPhase.DONE
                        break
                    else:
                        self._phase = SeriesPhase.RECORDING
            elif self._phase == SeriesPhase.PAUSE:
                elapsed = now - self._phase_started_at
                if elapsed < self.gap_sec:
                    break
                # Переход из PAUSE в RECORDING
                self._phase_started_at += self.gap_sec
                self._current_index += 1
                if self._current_index >= self.n_tracks:
                    self._phase = SeriesPhase.DONE
                    break
                else:
                    self._phase = SeriesPhase.RECORDING
            else:
                assert False, "unreachable"

        return self._phase

    def advance(self, now: float) -> SeriesPhase:
        return self.tick(now)

    def mark_error(self, now: float) -> None:
        if self.stop_on_error:
            self._errored = True
            self._phase = SeriesPhase.DONE
        else:
            self.advance(now)

    def reset(self) -> None:
        self._started_at = None
        self._current_index = 0
        self._phase = SeriesPhase.IDLE
        self._phase_started_at = 0.0
        self._errored = False

    def progress(self) -> tuple[int, int]:
        if self._phase == SeriesPhase.IDLE:
            return (0, self.n_tracks)
        elif self._phase == SeriesPhase.RECORDING:
            return (self._current_index, self.n_tracks)
        elif self._phase == SeriesPhase.PAUSE:
            return (self._current_index + 1, self.n_tracks)
        else:  # DONE
            return (self.n_tracks, self.n_tracks)
