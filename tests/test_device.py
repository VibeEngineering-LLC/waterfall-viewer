from __future__ import annotations

import threading
import time

import pytest

from awf.usb.device import SerialSpectraDevice
from awf.usb.shproto import ShprotoEncoder, SHPROTO_START, SHPROTO_ESC, SHPROTO_FINISH


LOOP_URL = "loop://"


def _wait_for(predicate, timeout=1.0, interval=0.01):
    """Активно ждать выполнения предиката. True — успех, False — таймаут."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_is_open_flag_transitions():
    """Устройство создано → is_open is False. dev.open() → is_open is True. dev.close() → is_open is False."""
    dev = SerialSpectraDevice(LOOP_URL)
    assert dev.is_open is False

    dev.open()
    assert dev.is_open is True

    dev.close()
    assert dev.is_open is False

    # Идемпотентность close()
    dev.close()
    assert dev.is_open is False


def test_open_close_open_reuses_port():
    """open() → close() → open() подряд не падает, is_open в конце True."""
    dev = SerialSpectraDevice(LOOP_URL)
    dev.open()
    dev.close()
    dev.open()
    assert dev.is_open is True
    dev.close()


def test_send_frame_before_open_raises():
    """Без open() вызов dev.send_frame(0x10, b"") бросает RuntimeError."""
    dev = SerialSpectraDevice(LOOP_URL)
    with pytest.raises(RuntimeError):
        dev.send_frame(0x10, b"")


def test_send_and_read_frame_roundtrip():
    """Открыть SerialSpectraDevice(LOOP_URL). dev.send_frame(0x10, b"hello"). dev.read_frame(timeout=1.0) → (0x10, b"hello")."""
    dev = SerialSpectraDevice(LOOP_URL)
    dev.open()
    try:
        dev.send_frame(0x10, b"hello")
        result = dev.read_frame(timeout=1.0)
        assert result == (0x10, b"hello")
    finally:
        dev.close()


def test_read_frame_timeout_returns_none():
    """Открыть устройство, ничего не слать, dev.read_frame(timeout=0.05) → None."""
    dev = SerialSpectraDevice(LOOP_URL)
    dev.open()
    try:
        result = dev.read_frame(timeout=0.05)
        assert result is None
    finally:
        dev.close()


def test_two_frames_preserve_order():
    """Отправить два разных кадра подряд: (0x21, b"first"), (0x22, b"second"). Прочитать оба через read_frame → порядок сохранён."""
    dev = SerialSpectraDevice(LOOP_URL)
    dev.open()
    try:
        dev.send_frame(0x21, b"first")
        dev.send_frame(0x22, b"second")
        result1 = dev.read_frame(timeout=1.0)
        result2 = dev.read_frame(timeout=1.0)
        assert result1 == (0x21, b"first")
        assert result2 == (0x22, b"second")
    finally:
        dev.close()


def test_on_frame_callback_invoked():
    """Callback вызывается при получении кадра."""
    received: list[tuple[int, bytes]] = []
    event = threading.Event()

    def cb(cmd, payload):
        received.append((cmd, payload))
        event.set()

    dev = SerialSpectraDevice(LOOP_URL, on_frame=cb)
    dev.open()
    try:
        dev.send_frame(0x33, b"abc")
        assert event.wait(1.0)
        assert received[0] == (0x33, b"abc")
    finally:
        dev.close()


def test_callback_exception_does_not_kill_reader():
    """Callback бросает RuntimeError("boom") на первый кадр. Оба должны попасть в dev.drain_frames()."""
    call_count = 0

    def cb(cmd, payload):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")

    dev = SerialSpectraDevice(LOOP_URL, on_frame=cb)
    dev.open()
    try:
        dev.send_frame(0x44, b"first")
        dev.send_frame(0x45, b"second")
        assert _wait_for(lambda: dev._frames.qsize() >= 2, timeout=1.0)
        frames = dev.drain_frames()
        assert len(frames) == 2
        assert dev.is_open is True
    finally:
        dev.close()


def test_send_frame_returns_bytes_written():
    """dev.send_frame(0x40, b"xy") возвращает количество отправленных байт."""
    dev = SerialSpectraDevice(LOOP_URL)
    dev.open()
    try:
        expected = ShprotoEncoder.encode(0x40, b"xy")
        n = dev.send_frame(0x40, b"xy")
        assert n == len(expected)
    finally:
        dev.close()


def test_marker_bytes_in_payload_roundtrip():
    """Payload с маркерами должен корректно передаваться и приниматься."""
    payload = bytes([SHPROTO_START, SHPROTO_ESC, SHPROTO_FINISH, 0x00, 0xFF])
    dev = SerialSpectraDevice(LOOP_URL)
    dev.open()
    try:
        dev.send_frame(0x50, payload)
        result = dev.read_frame(timeout=1.0)
        assert result == (0x50, payload)
    finally:
        dev.close()


def test_drain_frames_returns_all_pending():
    """dev.drain_frames() возвращает все накопленные кадры."""
    dev = SerialSpectraDevice(LOOP_URL, on_frame=None)
    dev.open()
    try:
        dev.send_frame(0x10, b"a")
        dev.send_frame(0x20, b"bc")
        dev.send_frame(0x30, b"")
        assert _wait_for(lambda: dev._frames.qsize() >= 3, timeout=1.0)
        frames = dev.drain_frames()
        assert len(frames) == 3
        assert frames[0] == (0x10, b"a")
        assert frames[1] == (0x20, b"bc")
        assert frames[2] == (0x30, b"")
        frames = dev.drain_frames()
        assert len(frames) == 0
    finally:
        dev.close()


def test_close_stops_reader_thread():
    """dev.close() останавливает reader-поток."""
    dev = SerialSpectraDevice(LOOP_URL)
    dev.open()
    thread = dev._thread
    assert thread is not None and thread.is_alive()
    dev.close()
    assert _wait_for(lambda: not thread.is_alive(), timeout=2.0)
    assert dev.is_open is False
