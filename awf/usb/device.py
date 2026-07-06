"""Qt-free обёртка над pyserial: reader-поток + shproto-декодер.

Замечание #209 (Live-USB collector, Фаза 6). UI поверх — `awf/ui/tab_live.py`.
Cumulative→delta логика — `awf/usb/collector.py` (замечание #210).
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import serial

from awf.usb.shproto import ShprotoDecoder, ShprotoEncoder


class SerialSpectraDevice:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        *,
        on_frame: Optional[Callable[[int, bytes], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        read_chunk_size: int = 512,
        read_timeout_s: float = 0.1,
    ) -> None:
        self._port_url = port
        self._baudrate = baudrate
        self._on_frame = on_frame
        self._on_error = on_error
        self._read_chunk = read_chunk_size
        self._read_timeout = read_timeout_s
        self._ser: Optional[serial.SerialBase] = None
        self._decoder = ShprotoDecoder()
        self._frames: queue.Queue[tuple[int, bytes]] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._thread is not None and self._thread.is_alive()

    def open(self) -> None:
        if self.is_open:
            return
        self._ser = serial.serial_for_url(
            self._port_url, baudrate=self._baudrate, timeout=self._read_timeout
        )
        self._stop.clear()
        self._decoder = ShprotoDecoder()
        self._frames = queue.Queue()
        self._thread = threading.Thread(target=self._reader_loop, name="ShprotoReader", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        self._thread = None

    def send_frame(self, cmd: int, payload: bytes = b"") -> int:
        if not self.is_open:
            raise RuntimeError("device not open")
        frame = ShprotoEncoder.encode(cmd, payload)
        with self._write_lock:
            written = self._ser.write(frame)
        return len(frame) if written is None else written

    def read_frame(self, timeout: float = 1.0) -> Optional[tuple[int, bytes]]:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_frames(self) -> list[tuple[int, bytes]]:
        frames = []
        while True:
            try:
                frames.append(self._frames.get_nowait())
            except queue.Empty:
                break
        return frames

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._ser.read(self._read_chunk)
            except (serial.SerialException, OSError) as e:
                if self._on_error is not None:
                    try:
                        self._on_error(e)
                    except Exception:
                        pass
                return
            if not data:
                continue
            self._decoder.feed(data)
            for frame in self._decoder.frames():
                self._frames.put(frame)
                if self._on_frame is not None:
                    try:
                        self._on_frame(*frame)
                    except Exception:
                        pass
